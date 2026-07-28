"""
Calibrated feature selection for SAE attribution event logs — final design.

Pipeline (per counterfactual type):

  raw attributions (long format)
    │  add_abs_and_sign          keep sign; never overwrite raw values
    │  data_quality_report       fp16 floor, zero fractions, group sizes
    ▼
  Stage 1: mad_screen            per-(prompt, layer) robust noise floor,
    │                            c_g = Phi^{-1}(1 - q / n_g), MAD shrunk
    │                            toward the layer level (permissive DENOISER,
    │                            controls E[false positives] = q per group)
    │  gaussianity_report        PPCC / QQ-RMSE / tail calibration on the
    │                            null bulk — validates the c arithmetic
    ▼
  Stage 2: recurrence_test       replication across prompts is the TEST:
    │                            Binomial tail on (blocked) recurrence,
    │                            BH-FDR across the feature alphabet,
    │                            + effect-size floor + sign consistency
    ▼
  select_features / build_event_log
                                 final activity alphabet + filtered events

Input: the hive-partitioned parquet dataset written by main.py, partitioned
by counterfactual and layer, with columns prompt_idx, token_idx, token_id,
feature_idx, attribution(float16). Use load_counterfactual() to read one CF;
the CLI (see bottom) runs the whole pipeline per CF and writes the feature
table and event log back in the SAME hive layout, partitioned by
counterfactual: {out_dir}/features|eventlog/counterfactual=<cf>/*.parquet.
Part 2 (partial_order.py) and edge_attribution.py --nodes read those dirs.
Optional columns if regenerated: is_error (bool) for SAE error nodes; a block
column (e.g. template_id) for correlated-prompt blocking — NOT present in the
current data, so recurrence trials default to individual prompts until it is
added at generation time.

Statistical roles, stated once:
  * Stage 1 multiplicity family = the n_g ACTIVE features observed in one
    (prompt, layer) group. Population is conditional-on-active by
    construction (the generator masks (f_clean>0)|(f_cf>0), and inactive
    features have IE == 0 exactly).
  * Stage 2 trials = prompts (or template blocks if block_col is given —
    templated prompts are correlated trials, blocking restores honesty).
    Stage 2 multiplicity family = the union feature alphabet, BH-corrected.
  * Significance alone is not importance: with thousands of trials the
    binomial detects arbitrarily small recurrence rates, so selection also
    requires a minimum recurrence fraction and (optionally) sign
    consistency. Downstream faithfulness ablations remain the arbiter of
    sufficiency; this module only makes the candidate alphabet calibrated.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

GROUP_COLS = ["prompt_idx", "layer"]
FEATURE_COLS = ["layer", "feature_idx"]

FP16_SUBNORMAL_MAX = 6.1e-5   # below this, stored fp16 loses relative precision
FP16_MIN_POSITIVE = 6.0e-8    # below this, stored fp16 underflows to zero


# ===========================================================================
# Data loading — partitioned parquet dataset produced by main.py
# ===========================================================================
#
# On-disk layout (hive-partitioned by counterfactual and layer):
#   {attributions_dir}/counterfactual=<cf>/layer=<L>/*.parquet
# File columns: prompt_idx(int32), token_idx(int16), token_id(int32),
#               feature_idx(int32), attribution(float16)
# Partition columns recovered on read: counterfactual(str), layer(int)
#
# The pipeline runs PER counterfactual (one event log / net per CF type), so
# load one counterfactual at a time. Zero-attribution rows (gradient 0 or
# Delta-f 0, plus fp16 underflow) are dropped by default: they carry no
# causal signal and inflate memory ~an order of magnitude, but the fraction
# dropped is reported so it can be inspected.

def list_counterfactuals(attributions_dir: str) -> list[str]:
    """Enumerate available counterfactual partitions on disk."""
    return sorted(
        d.split("=", 1)[1]
        for d in os.listdir(attributions_dir)
        if d.startswith("counterfactual=")
    )


def load_counterfactual(
    attributions_dir: str,
    counterfactual: str,
    layers: list[int] | None = None,
    drop_zeros: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one counterfactual partition into a long-format dataframe.

    Uses pyarrow dataset filtering so only the requested counterfactual
    (and optionally a subset of layers) is read from disk. Recovers the
    partition columns, casts fp16 attributions to float64, and — unless
    disabled — drops exact-zero attributions.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(attributions_dir, format="parquet", partitioning="hive")
    flt = ds.field("counterfactual") == counterfactual
    if layers is not None:
        flt = flt & ds.field("layer").isin(layers)

    want = columns or ["prompt_idx", "token_idx", "token_id",
                       "feature_idx", "attribution", "layer", "counterfactual"]
    have = set(dataset.schema.names)
    want = [c for c in want if c in have]

    table = dataset.to_table(filter=flt, columns=want)
    df = table.to_pandas()

    df["attribution"] = df["attribution"].astype(np.float64)
    for c in ("prompt_idx", "token_idx", "token_id", "feature_idx", "layer"):
        if c in df:
            df[c] = df[c].astype(np.int64)

    if drop_zeros:
        n0 = len(df)
        df = df[df["attribution"] != 0.0].reset_index(drop=True)
        df.attrs["zero_frac_dropped"] = (n0 - len(df)) / max(n0, 1)
    return df


# ===========================================================================
# Preparation & data quality
# ===========================================================================

def add_abs_and_sign(df: pd.DataFrame) -> pd.DataFrame:
    """Add abs_attr and sign columns without destroying the raw values."""
    out = df.copy()
    out["abs_attr"] = out["attribution"].abs().astype(np.float64)
    out["sign"] = np.sign(out["attribution"]).astype(np.int8)
    return out


def data_quality_report(df: pd.DataFrame) -> pd.Series:
    """Flags measurement artifacts that would corrupt null calibration.

    subnormal_frac: share of nonzero |IE| in the fp16 subnormal range —
    these carry quantization noise; if large, trim the low tail before any
    distributional fitting, and store float32 in the next data generation.
    """
    nz = df.loc[df["abs_attr"] > 0, "abs_attr"]
    sizes = df.groupby(GROUP_COLS).size()
    return pd.Series({
        "rows": float(len(df)),
        "zero_frac": float((df["abs_attr"] == 0).mean()),
        "subnormal_frac": float((nz < FP16_SUBNORMAL_MAX).mean()),
        "group_size_median": float(sizes.median()),
        "group_size_q10": float(sizes.quantile(0.1)),
        "group_size_q90": float(sizes.quantile(0.9)),
    })


# ===========================================================================
# Stage 1 — per-(prompt, layer) calibrated screen
# ===========================================================================

def mad_screen(
    df: pd.DataFrame,
    group_cols: list[str] = GROUP_COLS,
    q: float = 1.0,
    n0: float = 50.0,
    pool_col: str = "layer",
    exclude_subnormals_from_fit: bool = True,
) -> pd.DataFrame:
    """Permissive denoising screen with per-group calibration.

    Threshold per group: exp(median + c_g * MAD_shrunk), with
      c_g        = Phi^{-1}(1 - q / n_g)   (q expected FPs per group,
                                            n_g = group's OBSERVED size)
      MAD_shrunk = (n_g * MAD_g + n0 * MAD_pool) / (n_g + n0)
                   (scale borrows strength across the pool level;
                    centering stays strictly per-group)

    Adds: kept_screen (bool), screen_threshold, c_group, n_group.
    """
    from scipy.stats import norm

    df = df if "abs_attr" in df else add_abs_and_sign(df)
    if "is_error" in df:
        fit_df = df[~df["is_error"]]
    else:
        fit_df = df

    stats = []
    for key, g in fit_df.groupby(group_cols):
        vals = g.loc[g["abs_attr"] > 0, "abs_attr"].to_numpy()
        if exclude_subnormals_from_fit:
            fit_vals = vals[vals >= FP16_SUBNORMAL_MAX]
            if fit_vals.size < 30:          # too little left; fall back
                fit_vals = vals
        else:
            fit_vals = vals
        if fit_vals.size == 0:
            continue
        logs = np.log(fit_vals)
        med = np.median(logs)
        mad = np.median(np.abs(logs - med)) * 1.4826
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update({"_med": med, "_mad": mad, "n_group": int(vals.size)})
        stats.append(rec)
    st = pd.DataFrame(stats)

    pool = st.groupby(pool_col)["_mad"].median().rename("_mad_pool")
    st = st.join(pool, on=pool_col)
    st["_mad_shrunk"] = (
        (st["n_group"] * st["_mad"] + n0 * st["_mad_pool"])
        / (st["n_group"] + n0)
    )
    st["c_group"] = norm.ppf(1.0 - (q / st["n_group"]).clip(upper=0.5))
    st["screen_threshold"] = np.exp(st["_med"] + st["c_group"] * st["_mad_shrunk"])

    df = df.join(
        st.set_index(group_cols)[["screen_threshold", "c_group", "n_group"]],
        on=group_cols,
    )
    df["kept_screen"] = df["abs_attr"] > df["screen_threshold"]
    if "is_error" in df:
        df.loc[df["is_error"], "kept_screen"] = False
    return df


def gaussianity_report(
    df: pd.DataFrame,
    group_cols: list[str] = GROUP_COLS,
    bulk_z: float = 2.5,
) -> pd.DataFrame:
    """Effect-size diagnostics of the Gaussian-bulk assumption per group.

    ppcc >= ~0.99 and qq_rmse <~ 0.05: bulk fit is comfortable.
    tail2 near 1.0: the false-positive arithmetic behind c is trustworthy.
    tail3 >> 1 with healthy ppcc/tail2: that's signal, not a broken null.
    If tail2 is inflated across groups: raise c (lower q) or switch to the
    sign-flip permutation null (see recurrence_test docstring).
    """
    from scipy.stats import norm

    def _stats(g: pd.DataFrame) -> pd.Series:
        vals = g.loc[g["abs_attr"] > 0, "abs_attr"].to_numpy()
        vals = vals[vals >= FP16_SUBNORMAL_MAX] if vals.size > 100 else vals
        logs = np.log(vals)
        med = np.median(logs)
        mad = np.median(np.abs(logs - med)) * 1.4826
        if mad == 0 or logs.size < 30:
            return pd.Series({"n": logs.size, "ppcc": np.nan,
                              "qq_rmse": np.nan, "tail2": np.nan, "tail3": np.nan})
        z = (logs - med) / mad
        bulk = np.sort(z[np.abs(z) <= bulk_z])
        p_lo, p_hi = norm.cdf(-bulk_z), norm.cdf(bulk_z)
        probs = p_lo + (np.arange(1, bulk.size + 1) - 0.5) / bulk.size * (p_hi - p_lo)
        theo = norm.ppf(probs)
        n = z.size
        return pd.Series({
            "n": n,
            "ppcc": float(np.corrcoef(bulk, theo)[0, 1]),
            "qq_rmse": float(np.sqrt(np.mean((bulk - theo) ** 2))),
            "tail2": float((z > 2).sum() / (n * (1 - norm.cdf(2)))),
            "tail3": float((z > 3).sum() / (n * (1 - norm.cdf(3)))),
        })

    return df.groupby(group_cols).apply(_stats).reset_index()


# ===========================================================================
# Stage 2 — blocked binomial recurrence test with FDR + floors
# ===========================================================================

def recurrence_test(
    df: pd.DataFrame,
    kept_col: str = "kept_screen",
    feature_cols: list[str] = FEATURE_COLS,
    case_col: str = "prompt_idx",
    block_col: str | None = None,
    q: float = 1.0,
    fdr: float = 0.05,
    min_recurrence_frac: float = 0.10,
    min_sign_consistency: float | None = 0.8,
    min_median_abs: float | None = None,
) -> pd.DataFrame:
    """Replication test across cases; the inferential heart of the pipeline.

    Null model: a feature passes an independent case's screen w.p.
    p0 = mean(q / n_g). If block_col is given (STRONGLY recommended for
    templated datasets: prompts sharing a template are correlated trials),
    trials become blocks; a feature 'recurs in a block' if it passes in >= 1
    of the block's cases, and the null pass probability per block is
    1 - (1 - p0)^(mean cases per block).

    Selection = BH-significant at `fdr`
                AND recurrence_frac >= min_recurrence_frac  (importance floor;
                    with thousands of trials, significance alone detects
                    arbitrarily tiny recurrence rates)
                AND sign_consistency >= min_sign_consistency (a genuine causal
                    contributor should push the metric the same way each
                    time; under the null, the score is sign-symmetric, so
                    this doubles as a distribution-free check)
                AND median_abs >= min_median_abs (optional magnitude floor).

    Returns one row per feature with all statistics, floors applied in the
    'selected' column. 'significant' is BH-only, kept separate so the two
    criteria can be reported independently.

    Note on overdispersion: even with blocking, residual correlation can
    inflate recurrence among nulls. Diagnostic: the negative-control run
    (shuffled feature identities) should select ~0 features; if it does
    not, replace the binomial with a beta-binomial fit on the sub-floor
    features' pass counts.
    """
    from scipy.stats import binom

    if kept_col not in df:
        raise ValueError("Run mad_screen first.")

    p0_case = float((q / df.groupby(GROUP_COLS)[case_col].size()).clip(upper=0.5).mean())

    passes = df[df[kept_col]]
    if block_col is not None:
        n_trials = df[block_col].nunique()
        cases_per_block = df.groupby(block_col)[case_col].nunique().mean()
        p0 = 1.0 - (1.0 - p0_case) ** cases_per_block
        trial_col = block_col
    else:
        n_trials = df[case_col].nunique()
        p0, trial_col = p0_case, case_col

    obs = (
        passes.groupby(feature_cols)
        .agg(
            n_recurring=(trial_col, "nunique"),
            n_pass_rows=("abs_attr", "size"),
            median_abs=("abs_attr", "median"),
            sign_consistency=("sign", lambda s: abs(s.mean())),
        )
        .reset_index()
    )
    obs["recurrence_frac"] = obs["n_recurring"] / n_trials
    obs["p_value"] = binom.sf(obs["n_recurring"] - 1, n_trials, p0)

    # Benjamini-Hochberg over the feature alphabet
    m = len(obs)
    obs = obs.sort_values("p_value").reset_index(drop=True)
    crit = (np.arange(1, m + 1) / m) * fdr
    below = obs["p_value"].to_numpy() <= crit
    k_max = np.max(np.nonzero(below)[0]) if below.any() else -1
    obs["significant"] = obs.index <= k_max

    sel = obs["significant"] & (obs["recurrence_frac"] >= min_recurrence_frac)
    if min_sign_consistency is not None:
        sel &= obs["sign_consistency"] >= min_sign_consistency
    if min_median_abs is not None:
        sel &= obs["median_abs"] >= min_median_abs
    obs["selected"] = sel
    obs.attrs["null_pass_prob"] = p0
    obs.attrs["n_trials"] = n_trials
    return obs


# ===========================================================================
# Orchestration & export
# ===========================================================================

def select_features(
    df: pd.DataFrame,
    q: float = 1.0,
    block_col: str | None = None,
    **recurrence_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run the full pipeline. Returns (screened df, feature table, summary)."""
    df = add_abs_and_sign(df) if "abs_attr" not in df.columns else df
    df = mad_screen(df, q=q)
    feats = recurrence_test(df, q=q, block_col=block_col, **recurrence_kwargs)
    summary = pd.Series({
        "groups": df.groupby(GROUP_COLS).ngroups,
        "median_c": float(df["c_group"].median()),
        "screen_pass_rows": int(df["kept_screen"].sum()),
        "alphabet_tested": len(feats),
        "bh_significant": int(feats["significant"].sum()),
        "selected": int(feats["selected"].sum()),
    })
    return df, feats, summary


def build_event_log(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """Filter screened rows to the selected alphabet: event-log-ready table.

    An event = an occurrence of a selected feature that passed its group's
    screen. Downstream stages add role mapping and (partial-order) edges.
    """
    alphabet = feats.loc[feats["selected"], feature_cols]
    events = df[df["kept_screen"]].merge(alphabet, on=feature_cols, how="inner")
    keep = [c for c in ["prompt_idx", "layer", "token_idx", "token_id",
                        "feature_idx", "attribution", "abs_attr", "sign"]
            if c in events.columns]
    return events[keep].sort_values(["prompt_idx", "layer", "token_idx"])


def negative_control(df: pd.DataFrame, seed: int = 0, **kwargs) -> pd.Series:
    """Null-simulation control: keep every feature's presence pattern and
    every group's fitted scale, but replace magnitudes with draws from the
    group's own robust-fitted lognormal null (random signs). A calibrated
    pipeline should select ~0 features on this data.

    Why not a permutation control? Any shuffle of identities (or values)
    conserves the observed magnitude multiset, hence the ~q + signal
    screen-passing values per group. When passes are signal-dominated, the
    effective null pass rate under permutation is (passes)/(rows) — an
    order of magnitude above the design rate q/n — and ids active in more
    prompts own more rows, so redistributed passes concentrate on exactly
    the persistent features. Permutation controls therefore 'fail' by
    construction on signal-bearing data; only null simulation tests the
    claim actually made (no signal => ~0 selections). Report both this
    control and that reasoning in the paper.
    """
    rng = np.random.default_rng(seed)

    def _resample(g: pd.Series) -> np.ndarray:
        logs = np.log(g[g > 0].to_numpy())
        med = np.median(logs)
        sigma = np.median(np.abs(logs - med)) * 1.4826
        draws = np.exp(rng.normal(med, sigma, size=len(g)))
        return draws * rng.choice([-1.0, 1.0], size=len(g))

    null_df = df.copy()
    null_df["attribution"] = (
        null_df.groupby(GROUP_COLS)["attribution"]
        .transform(lambda s: _resample(s.abs()))
    )
    null_df = null_df.drop(columns=[c for c in ("abs_attr", "sign") if c in null_df])
    _, feats, summary = select_features(null_df, **kwargs)
    return summary


# ===========================================================================
# CLI — run the pipeline on the real partitioned attribution dataset
# ===========================================================================

def _write_hive(df: pd.DataFrame, root: str, counterfactual: str) -> str:
    """Write one CF as {root}/counterfactual=<cf>/*.parquet.

    Same hive layout main.py writes for the raw attributions, so downstream
    readers that open a directory as one pyarrow dataset (edge_attribution's
    --nodes, event_log.load_attributions) see the counterfactual as a column
    and can read every CF at once. delete_matching keeps a re-run from
    appending duplicate rows to the CF's partition.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_to_dataset(
        pa.Table.from_pandas(df.assign(counterfactual=str(counterfactual)),
                             preserve_index=False),
        root_path=root,
        partition_cols=["counterfactual"],
        existing_data_behavior="delete_matching",
        compression="zstd",
    )
    return os.path.join(root, f"counterfactual={counterfactual}")


def run_counterfactual(
    attributions_dir: str,
    counterfactual: str,
    out_dir: str,
    q: float = 1.0,
    block_col: str | None = None,
    min_recurrence_frac: float = 0.10,
    min_sign_consistency: float | None = 0.8,
    run_control: bool = True,
) -> pd.Series:
    """Load one CF, run selection, write feature table + event log as hive
    partitions ({out_dir}/features|eventlog/counterfactual=<cf>/), and
    (optionally) the null-simulation control. Returns the summary."""
    print(f"[{counterfactual}] loading ...")
    df = load_counterfactual(attributions_dir, counterfactual, drop_zeros=True)
    print(f"[{counterfactual}] {len(df):,} nonzero rows "
          f"(dropped {df.attrs.get('zero_frac_dropped', 0):.1%} zeros)")

    df = add_abs_and_sign(df)
    print("=== data quality ===")
    print(data_quality_report(df).to_string())

    dfs, feats, summary = select_features(
        df, q=q, block_col=block_col,
        min_recurrence_frac=min_recurrence_frac,
        min_sign_consistency=min_sign_consistency,
    )
    print("\n=== gaussianity (median over groups) ===")
    gr = gaussianity_report(df)
    print(gr[["ppcc", "qq_rmse", "tail2", "tail3"]].median().to_string())

    print("\n=== selection summary ===")
    print(summary.to_string())

    if run_control:
        print("\n=== negative control (null simulation; expect ~0 selected) ===")
        ctrl = negative_control(
            df, q=q, block_col=block_col,
            min_recurrence_frac=min_recurrence_frac,
            min_sign_consistency=min_sign_consistency,
        )
        print(f"selected under null: {int(ctrl['selected'])}")

    os.makedirs(out_dir, exist_ok=True)
    log = build_event_log(dfs, feats)
    feats_path = _write_hive(feats, os.path.join(out_dir, "features"), counterfactual)
    log_path = _write_hive(log, os.path.join(out_dir, "eventlog"), counterfactual)
    print(f"\nwrote {feats_path}\nwrote {log_path}")
    print(f"event log: {len(log):,} events | "
          f"{log.groupby(FEATURE_COLS).ngroups} activities")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Calibrated SAE feature selection on the partitioned "
                    "attribution dataset (one event log per counterfactual).")
    ap.add_argument("--attributions_dir", default="sae_attributions",
                    help="Root of the hive-partitioned parquet dataset.")
    ap.add_argument("--out_dir", default="event_logs",
                    help="Root for the hive-partitioned outputs: "
                         "{out_dir}/features|eventlog/counterfactual=<cf>/.")
    ap.add_argument("--counterfactual", default=None,
                    help="CF partition to process; default = all found.")
    ap.add_argument("--q", type=float, default=1.0,
                    help="Expected false positives per group (Stage-1 c).")
    ap.add_argument("--block_col", default=None,
                    help="Correlated-trial block column (e.g. template_id); "
                         "absent from current data — add at generation time.")
    ap.add_argument("--min_recurrence_frac", type=float, default=0.10)
    ap.add_argument("--min_sign_consistency", type=float, default=0.8)
    ap.add_argument("--no_control", action="store_true",
                    help="Skip the null-simulation negative control.")
    args = ap.parse_args()

    cfs = ([args.counterfactual] if args.counterfactual
           else list_counterfactuals(args.attributions_dir))
    print(f"counterfactuals to process: {cfs}")

    for cf in cfs:
        run_counterfactual(
            args.attributions_dir, cf, args.out_dir,
            q=args.q, block_col=args.block_col,
            min_recurrence_frac=args.min_recurrence_frac,
            min_sign_consistency=args.min_sign_consistency,
            run_control=not args.no_control,
        )


if __name__ == "__main__":
    main()
