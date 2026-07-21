"""Build process-mining event logs from SAE attribution datasets.

Converts the partitioned Parquet dataset written by main.py --save_attributions
into an event log for process discovery (inductive miner variants, Petri nets).

Pipeline: load -> filter -> activity mapping -> dedupe/recurrence -> ordering -> export.
Filtering, activity mapping and ordering are pluggable: add a function to the
FILTERS / ACTIVITY_MAPPERS / ORDERINGS registries and select it by name.

Attribution sign convention (Marks et al., "Sparse Feature Circuits: Discovering
and Editing Interpretable Causal Graphs in Language Models", arXiv:2403.19647):
values approximate metric(patched) - metric(clean), so task-critical features
carry *negative* attributions under a higher-is-better metric. All
magnitude-based filtering therefore uses |attribution|.
"""

import argparse
import json
import logging

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger("sae_event_log")


# ---------------------------------------------------------------------------
# Filters: (df, **params) -> row subset. Decide which attribution events survive.

def filter_abs_threshold(df, tau=0.01, **_):
    """Keep events with |attribution| >= tau."""
    return df[df["attribution"].abs() >= tau]


def filter_topk_per_layer(df, k=20, **_):
    """Keep the k largest-|attribution| events per (case, layer).

    Per-layer rather than a global cut: attribution magnitudes vary
    systematically across layers, and a global threshold silently drops
    entire layers.
    """
    ranked = df.loc[df["attribution"].abs().sort_values(ascending=False).index]
    return ranked.groupby(["case_id", "layer"], sort=False).head(k)


FILTERS = {
    "abs_threshold": filter_abs_threshold,
    "topk_per_layer": filter_topk_per_layer,
}


# ---------------------------------------------------------------------------
# Activity mappers: (df, **params) -> df with an 'activity' column.

def activity_feature(df, **_):
    """Activity = layer/feature, position-agnostic."""
    df["activity"] = "L" + df["layer"].astype(str) + "/F" + df["feature_idx"].astype(str)
    return df


def activity_feature_role(df, roles=None, **_):
    """Activity = layer/feature @ token role.

    `roles` maps token_idx -> role name (e.g. {13: "S2", 16: "END"}); assumes
    templatic, length-aligned prompts like MIB IOI. Unmapped positions get
    'OTHER'. Default when roles is None: 'END' for the last attributed
    position of each case (where last-position logit-diff metrics are
    computed), 'MID' elsewhere.
    """
    if roles is None:
        last = df.groupby("case_id")["token_idx"].transform("max")
        role = df["token_idx"].eq(last).map({True: "END", False: "MID"})
    else:
        roles = {int(k): v for k, v in roles.items()}  # JSON keys arrive as str
        role = df["token_idx"].map(roles).fillna("OTHER")
    df["activity"] = ("L" + df["layer"].astype(str) + "/F"
                      + df["feature_idx"].astype(str) + "@" + role)
    return df


ACTIVITY_MAPPERS = {
    "feature": activity_feature,
    "feature_role": activity_feature_role,
}


# ---------------------------------------------------------------------------
# Orderings: (df, **params) -> df with a 'time' column (and optionally
# 'time_start' for interval semantics). Define what "before" means.

def ordering_layer(df, **_):
    """time = layer index. Same-layer events tie; miners that totally order a
    log break ties arbitrarily, so use 'layer_interval' when same-layer events
    should be seen as concurrent."""
    df["time"] = df["layer"].astype(float)
    return df


def ordering_layer_interval(df, span=0.9, **_):
    """Interval events spanning [layer, layer + span]. Overlapping intervals
    let lifecycle/interval-aware inductive miner variants treat same-layer
    events as concurrent."""
    df["time_start"] = df["layer"].astype(float)
    df["time"] = df["layer"].astype(float) + span
    return df


ORDERINGS = {
    "layer": ordering_layer,
    "layer_interval": ordering_layer_interval,
}


# ---------------------------------------------------------------------------
# Pipeline

def load_attributions(path, counterfactual=None):
    """Load the partitioned attribution dataset written by main.py.

    Attributions are float32 as written; the astype is a harmless guard for any
    legacy float16 dataset generated before that fix.
    """
    dataset = ds.dataset(path, partitioning="hive")
    filt = (ds.field("counterfactual") == counterfactual) if counterfactual else None
    df = dataset.to_table(filter=filt).to_pandas()
    df["attribution"] = df["attribution"].astype("float32")
    df["counterfactual"] = df["counterfactual"].astype(str)
    return df


def _add_case_id(df):
    """Case = prompt; counterfactual is appended only when the log mixes
    several (mine per counterfactual to keep causal questions separate)."""
    df["case_id"] = "p" + df["prompt_idx"].astype(str)
    if df["counterfactual"].nunique() > 1:
        df["case_id"] += "~" + df["counterfactual"].astype(str)
    return df


def _dedupe(df):
    """One event per (case, activity): keep the max-|attribution| occurrence."""
    idx = (df.assign(_mag=df["attribution"].abs())
             .groupby(["case_id", "activity"])["_mag"].idxmax())
    return df.loc[idx]


def _recurring(df, min_case_frac):
    """Drop activities occurring in fewer than min_case_frac of cases —
    activities that never recur across cases are unminable noise."""
    per_activity = df.groupby("activity")["case_id"].nunique()
    keep = per_activity[per_activity >= min_case_frac * df["case_id"].nunique()].index
    return df[df["activity"].isin(keep)]


def build_event_log(source, counterfactual=None,
                    filter_name="topk_per_layer", filter_params=None,
                    activity_name="feature", activity_params=None,
                    ordering_name="layer", ordering_params=None,
                    dedupe=True, min_case_frac=0.0):
    """Turn raw attributions into a sorted event log DataFrame.

    source: path to the partitioned Parquet dataset, or an equivalent
    DataFrame (schema as in main.py's raw_schema).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        if counterfactual is not None:
            df = df[df["counterfactual"] == counterfactual]
    else:
        df = load_attributions(source, counterfactual)

    # SAE error nodes (feature_idx == -1, the Marks et al. reconstruction residual)
    # are a faithfulness-accounting quantity, not a mineable activity: they are
    # dense and large-magnitude, so leaving them in would let an "L{n}/F-1" event
    # win a top-k slot in nearly every case and pollute the DFG. Drop before mining.
    df = df[df["feature_idx"] != -1].copy()

    df = _add_case_id(df)
    df = FILTERS[filter_name](df, **(filter_params or {})).copy()
    df = ACTIVITY_MAPPERS[activity_name](df, **(activity_params or {}))
    if dedupe:
        df = _dedupe(df)
    if min_case_frac > 0:
        df = _recurring(df, min_case_frac)
    df = ORDERINGS[ordering_name](df, **(ordering_params or {}))

    return (df.sort_values(["case_id", "time", "layer", "token_idx", "feature_idx"])
              .reset_index(drop=True))


def to_pm4py(log):
    """Rename columns to pm4py conventions and turn the float ordering key
    into a synthetic timestamp (seconds from a fixed epoch), so the log can be
    fed to pm4py without importing it here."""
    out = log.rename(columns={"case_id": "case:concept:name",
                              "activity": "concept:name"})
    epoch = pd.Timestamp("2000-01-01")
    out["time:timestamp"] = epoch + pd.to_timedelta(out.pop("time"), unit="s")
    if "time_start" in out.columns:
        out["start_timestamp"] = epoch + pd.to_timedelta(out.pop("time_start"), unit="s")
    return out


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAE attribution -> process-mining event log.")
    parser.add_argument("--attributions_dir", type=str, default="sae_attributions",
                        help="Partitioned Parquet dataset produced by main.py --save_attributions.")
    parser.add_argument("--counterfactual", type=str, default=None,
                        help="Restrict to one counterfactual type (recommended: mine per counterfactual).")
    parser.add_argument("--filter", type=str, default="topk_per_layer", choices=FILTERS,
                        help="Event filtering strategy.")
    parser.add_argument("--filter_params", type=str, default="{}",
                        help='JSON kwargs for the filter, e.g. \'{"k": 20}\'.')
    parser.add_argument("--activity", type=str, default="feature", choices=ACTIVITY_MAPPERS,
                        help="Activity mapping strategy.")
    parser.add_argument("--activity_params", type=str, default="{}",
                        help='JSON kwargs for the activity mapper, e.g. \'{"roles": {"13": "S2"}}\'.')
    parser.add_argument("--ordering", type=str, default="layer", choices=ORDERINGS,
                        help="Event ordering strategy.")
    parser.add_argument("--ordering_params", type=str, default="{}",
                        help="JSON kwargs for the ordering.")
    parser.add_argument("--min_case_frac", type=float, default=0.0,
                        help="Keep only activities occurring in at least this fraction of cases.")
    parser.add_argument("--no_dedupe", action="store_true",
                        help="Keep every (case, activity) occurrence instead of the max-|attribution| one.")
    parser.add_argument("--pm4py", action="store_true",
                        help="Write pm4py column names and real timestamps.")
    parser.add_argument("--output", type=str, default="sae_event_log.parquet",
                        help="Output Parquet path for the event log.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.info(f"Parameters: {vars(args)}")

    log = build_event_log(
        args.attributions_dir, counterfactual=args.counterfactual,
        filter_name=args.filter, filter_params=json.loads(args.filter_params),
        activity_name=args.activity, activity_params=json.loads(args.activity_params),
        ordering_name=args.ordering, ordering_params=json.loads(args.ordering_params),
        dedupe=not args.no_dedupe, min_case_frac=args.min_case_frac,
    )
    if args.pm4py:
        log = to_pm4py(log)

    log.to_parquet(args.output, index=False)
    case_col = "case:concept:name" if args.pm4py else "case_id"
    act_col = "concept:name" if args.pm4py else "activity"
    logger.info(f"Event log written to {args.output}: {len(log)} events, "
                f"{log[case_col].nunique()} cases, {log[act_col].nunique()} activities.")


if __name__ == "__main__":
    main()
