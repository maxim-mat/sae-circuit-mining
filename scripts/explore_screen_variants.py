"""EXPLORATORY, read-only, CPU-only (~1 min): is the recurrence test keeping
the nodes that carry the |IE| mass, or discarding them?

For each Stage-1 variant -- scale {log, linear} x method {mad, z} x scope
{group, global} x Stage 2 {on, off}, 16 cells -- this splits the tested alphabet into the features the recurrence test SELECTS and
the ones it REJECTS, and reports the average |IE| of each side. If the rejected
side carries the larger average |IE|, the test is inverted with respect to the
statistic faithfulness rewards -- which is the hypothesis this script exists to
settle.

Reads the cached n=100 node hive; writes nothing but the CSV named below.
Not part of the pipeline: no config, no manifest, no wandb.

    python scripts/explore_screen_variants.py [node_dir] [out_csv]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.attribution.nodes import read_node_hive                      # noqa: E402
from src.calibrated_feature_selection import (                        # noqa: E402
    GLOBAL_KEY, add_abs_and_sign, mad_screen, recurrence_test, screen_group_cols)
from src.selection.aggregate import aggregate                         # noqa: E402
from src.selection.threshold import select_by_threshold               # noqa: E402

NODE_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs_n100/node_attribution/77c6178e"
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "docs/screen-variants.csv"
KEY = ["site_ordinal", "feature_idx", "token_idx"]     # aggregation="none" node
MARKS_THRESHOLD = 0.1

# The calibrated arm's shipped defaults (configs/pythia_ha_rc.yaml / schema).
SCREEN_Q, SCREEN_N0, POOL = 1.0, 50.0, "site_ordinal"
FDR_Q, MIN_REC, MIN_SIGN = 0.05, 0.10, 0.8


def stats(nodes: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    """|IE| summary of one node subset, on the SAME quantity the circuit uses:
    the example-mean IE per (site, feature, position)."""
    sub = nodes.loc[mask, "abs_ie"]
    return {
        "set": label,
        "n_nodes": int(len(sub)),
        "mean_abs_ie": float(sub.mean()) if len(sub) else np.nan,
        "median_abs_ie": float(sub.median()) if len(sub) else np.nan,
        "p90_abs_ie": float(sub.quantile(0.9)) if len(sub) else np.nan,
        "sum_abs_ie": float(sub.sum()),
    }


def main() -> None:
    raw = read_node_hive(NODE_DIR)
    nodes = aggregate(raw, "none")
    nodes["abs_ie"] = nodes["ie"].abs()
    marks = select_by_threshold(nodes.rename(columns={"abs_ie": "_a"}), MARKS_THRESHOLD)
    marks_set = set(map(tuple, marks.loc[marks["selected"], KEY].to_numpy()))
    print(f"{len(nodes)} nodes, {len(marks_set)} above Marks' {MARKS_THRESHOLD} "
          f"threshold, {raw['example_idx'].nunique()} examples\n")

    raw = add_abs_and_sign(raw)
    rows = []
    grid = [(sc, m, sp, r) for sc in ("log", "linear") for m in ("mad", "z")
            for sp in ("group", "global") for r in (True, False)]
    screens: dict[tuple, pd.DataFrame] = {}
    for scale, method, scope, recur in grid:
            t0 = time.time()
            if (scale, method, scope) not in screens:
                screens[(scale, method, scope)] = mad_screen(
                    raw, q=SCREEN_Q, n0=SCREEN_N0, pool_col=POOL,
                    scale=scale, method=method, scope=scope)
            screened = screens[(scale, method, scope)]
            feats = recurrence_test(
                screened, feature_cols=KEY, q=SCREEN_Q, fdr=FDR_Q,
                screen_group_cols=screen_group_cols(method, scope) or [GLOBAL_KEY],
                min_recurrence_frac=MIN_REC, min_sign_consistency=MIN_SIGN)
            if not recur:
                feats["selected"] = True
            tab = nodes.copy()
            idx = pd.MultiIndex.from_frame(tab[KEY])
            tested = pd.Series(idx.isin(pd.MultiIndex.from_frame(feats[KEY])),
                               index=tab.index)
            tab["selected"] = idx.isin(
                pd.MultiIndex.from_frame(feats.loc[feats["selected"], KEY]))
            in_marks = pd.Series(
                [tuple(r) in marks_set for r in tab[KEY].to_numpy()], index=tab.index)

            tag = f"{scale}/{method}/{scope}/{'rec' if recur else 'norec'}"
            for label, mask in [
                ("selected (recurrence PASS)", tab["selected"]),
                ("rejected (recurrence FAIL, tested)", tested & ~tab["selected"]),
                ("all tested (alphabet)", tested),
                # The margin: where this selector and Marks' threshold disagree.
                ("marks-only (dropped by the test)", in_marks & ~tab["selected"]),
                ("selector-only (added by the test)", tab["selected"] & ~in_marks),
                ("all nodes", pd.Series(True, index=tab.index)),
            ]:
                rows.append({"variant": tag, "scale": scale, "method": method,
                             "scope": scope, "recurrence": recur,
                             **stats(tab, mask, label),
                             "screen_pass_rows": int(screened["kept_screen"].sum()),
                             "seconds": round(time.time() - t0, 1)})

            r = {d["set"].split(" ")[0]: d for d in rows[-6:]}
            rows[-1]["overlap_with_marks"] = int((tab["selected"] & in_marks).sum())
            sel, rej = r["selected"], r["rejected"]
            mo, so = r["marks-only"], r["selector-only"]
            print(f"[{tag:26s}] PASS {sel['n_nodes']:5d} nodes mean|IE| "
                  f"{sel['mean_abs_ie']:.4f} | FAIL {rej['n_nodes']:6d} mean|IE| "
                  f"{rej['mean_abs_ie']:.4f} | FAIL/PASS {rej['mean_abs_ie'] / sel['mean_abs_ie']:.2f}"
                  f" || margin: marks-only {mo['n_nodes']:3d} @ {mo['mean_abs_ie']:.4f}"
                  f" vs selector-only {so['n_nodes']:5d} @ {so['mean_abs_ie']:.4f}"
                  f"  [{time.time() - t0:.0f}s]")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
