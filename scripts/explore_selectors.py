"""EXPLORATORY (CLAUDE.md §2: exempt from the TDD gate, no reproducibility
contract). One tidy per-node table joining what BOTH selectors thought of every
node, so the difference sets can be characterised.

Answers two questions that the sweep cannot, because the sweep only measures
outcomes:

  1. The reverse experiment: for the nodes Marks' threshold selects, what is
     their statistical standing? Per-prompt (how often they clear the Stage-1
     noise floor) and across prompts (recurrence, binomial p, BH verdict, sign
     consistency). Nothing is recomputed -- recurrence_test already produces all
     of it; this just joins it onto the threshold's verdict.

  2. What characterises the difference sets -- nodes HA chose and Marks did not,
     and vice versa -- by effect size, recurrence, position, site, kind, and
     whether they are SAE error nodes.

    python -m scripts.explore_selectors --node-dir outputs_n100/node_attribution/<hash>

Reads only; costs seconds and no GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node-dir", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/pythia_ha_rc.yaml"))
    ap.add_argument("--node-threshold", type=float, default=0.1)
    ap.add_argument("--aggregation", default="none")
    ap.add_argument("--out", type=Path, default=Path("docs/selector-difference-sets.csv"))
    args = ap.parse_args()

    from src.attribution.nodes import read_node_hive
    from src.calibrated_feature_selection import calibrated_feature_cols, select_calibrated
    from src.selection.aggregate import aggregate
    from src.selection.threshold import select_by_threshold
    from src.utils.initialization import load_config

    cfg = load_config(args.config, [f"selection.aggregation={args.aggregation}",
                                    f"selection.node_threshold={args.node_threshold}"])
    key = calibrated_feature_cols(cfg)

    raw = read_node_hive(args.node_dir)
    print(f"raw rows {len(raw):,} over {raw.example_idx.nunique()} examples")

    nodes = aggregate(raw, args.aggregation)
    marks = select_by_threshold(nodes, args.node_threshold).rename(
        columns={"selected": "marks"})
    # select_calibrated returns the full recurrence table, not just the verdict.
    feats = select_calibrated(raw, cfg).rename(columns={"selected": "ha"})

    stats = [c for c in ("n_recurring", "recurrence_frac", "p_value", "significant",
                         "sign_consistency", "median_abs", "n_pass_rows") if c in feats]
    t = marks.merge(feats[key + ["ha"] + stats], on=key, how="outer")
    t["marks"] = t["marks"].fillna(False).astype(bool)
    t["ha"] = t["ha"].fillna(False).astype(bool)
    t["is_error"] = t["feature_idx"] == -1
    t["abs_ie"] = t["ie"].abs()
    t["set"] = t.apply(
        lambda r: "both" if r.marks and r.ha else
                  "marks_only" if r.marks else
                  "ha_only" if r.ha else "neither", axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"wrote {len(t):,} rows to {args.out}\n")

    print("=== contingency ===")
    print(pd.crosstab(t["marks"], t["ha"], margins=True), "\n")

    sets = t[t["set"] != "neither"]
    print("=== what distinguishes the difference sets ===")
    cols = [c for c in ("abs_ie", "recurrence_frac", "median_abs", "sign_consistency",
                        "n_recurring", "site_ordinal") if c in sets]
    print(sets.groupby("set")[cols].median().round(4), "\n")

    print("=== REVERSE EXPERIMENT: statistical standing of the threshold's picks ===")
    mk = t[t["marks"]]
    if "recurrence_frac" in mk:
        print(f"  threshold selected {len(mk)} nodes; of those:")
        print(f"    reached the recurrence test at all : "
              f"{mk['recurrence_frac'].notna().sum()}")
        print(f"    BH-significant                     : {int(mk['significant'].sum(min_count=1) or 0)}")
        print(f"    also selected by HA                : {int(mk['ha'].sum())}")
        print(f"    median recurrence fraction         : {mk['recurrence_frac'].median():.3f}")
        print(f"    median sign consistency            : {mk['sign_consistency'].median():.3f}")
        print("\n  recurrence_frac distribution of the threshold's picks:")
        print(mk["recurrence_frac"].describe().round(3).to_string())

    print("\n=== where each set lives (site x kind) ===")
    if "kind" in sets:
        print(pd.crosstab(sets["site_ordinal"], sets["set"]))
    if "token_idx" in sets:
        print("\n=== which token positions ===")
        print(pd.crosstab(sets["token_idx"], sets["set"]))
    print("\n=== SAE error nodes ===")
    print(pd.crosstab(sets["is_error"], sets["set"]))


if __name__ == "__main__":
    main()
