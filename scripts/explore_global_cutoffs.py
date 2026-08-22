"""EXPLORATORY, read-only, CPU (~2 min): sweep the GLOBAL linear z-screen's
cutoff and measure how close its node set gets to Marks' threshold.

The global linear z-screen IS a constant threshold on per-row |IE| -- its cutoff
is the directly comparable quantity to Marks' node_threshold, so the sweep is
parameterised by the cutoff itself and the screen_q that produces it is derived
(and printed, since that is what the pipeline takes). Stage 2 is off, so the
node set is exactly "features that cleared the constant cutoff at least once".

Reports set overlap as JACCARD, not just cardinality: two selectors can agree on
size and disagree on every node.

Writes figures/global-cutoff-sweep.png and docs/global-cutoff-sweep.csv.

    python scripts/explore_global_cutoffs.py [cutoffs...]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, ".")
from src.attribution.nodes import read_node_hive                      # noqa: E402
from src.calibrated_feature_selection import (                        # noqa: E402
    FP16_SUBNORMAL_MAX, add_abs_and_sign, mad_screen)
from src.selection.aggregate import aggregate                         # noqa: E402

NODE_DIR = "outputs_n100/node_attribution/77c6178e"
KEY = ["site_ordinal", "feature_idx", "token_idx"]
T_MARKS = 0.1
CUTOFFS = [0.2603, 0.15, 0.10, 0.075, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015]
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1a1a19"


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def main() -> None:
    cutoffs = [float(x) for x in sys.argv[1:]] or CUTOFFS
    raw = add_abs_and_sign(read_node_hive(NODE_DIR))
    nodes = aggregate(raw, "none")
    nodes["abs_ie"] = nodes["ie"].abs()
    ranked = nodes.sort_values("abs_ie", ascending=False)
    marks84 = set(map(tuple, ranked.loc[ranked["abs_ie"] > T_MARKS, KEY].to_numpy()))

    # The global z-screen's own fit population, so q can be inverted exactly.
    # mad_screen fits mu/sd on the subnormal-trimmed values but sets n_group from
    # every nonzero row, and c = Phi^-1(1 - q/n_group). Invert with both counts.
    fit = raw.loc[raw["abs_attr"] >= FP16_SUBNORMAL_MAX, "abs_attr"].to_numpy()
    mu, sd = fit.mean(), fit.std(ddof=1)
    n_fit = int((raw["abs_attr"] > 0).sum())

    rows = []
    for cut in cutoffs:
        q = float(n_fit * norm.sf((cut - mu) / sd))
        # The global linear z-screen IS |IE| > cutoff, so the set can be taken
        # directly -- which also escapes the q parameterisation's float64 ceiling
        # (q/n_g < 1e-16 makes c infinite). Verified against mad_screen wherever
        # the equivalent q is representable.
        if q / n_fit > 1e-15:
            sc = mad_screen(raw, q=q, scale="linear", method="z", scope="global")
            got = float(sc["screen_threshold"].iloc[0])
            assert abs(got - cut) < 1e-4 * cut, (cut, got)   # tail precision
            assert int(sc["kept_screen"].sum()) == int((raw["abs_attr"] > cut).sum())
        passed = raw.loc[raw["abs_attr"] > cut, KEY].drop_duplicates()
        sel = set(map(tuple, passed.to_numpy()))
        matched = set(map(tuple, ranked.head(len(sel))[KEY].to_numpy()))
        rows.append({
            "cutoff": cut, "screen_q": q,
            "rows_passing": int((raw["abs_attr"] > cut).sum()), "nodes": len(sel),
            # the node-level constant threshold that selects the same COUNT
            "equiv_node_threshold": float(ranked["abs_ie"].iloc[min(len(sel), len(ranked)) - 1]),
            "n_shared_with_marks84": len(sel & marks84),
            "jaccard_vs_marks84": jaccard(sel, marks84),
            "jaccard_vs_size_matched_marks": jaccard(sel, matched),
        })
        print(f"cutoff {cut:.4f}  q={q:11.4g}  rows={rows[-1]['rows_passing']:7d}  "
              f"nodes={len(sel):6d}  equiv node thr={rows[-1]['equiv_node_threshold']:.4f}  "
              f"shared84={len(sel & marks84):3d}  J(marks84)={rows[-1]['jaccard_vs_marks84']:.3f}  "
              f"J(size-matched)={rows[-1]['jaccard_vs_size_matched_marks']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("docs/global-cutoff-sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(out["cutoff"], out["nodes"], color=BLUE, linewidth=2, marker="o",
                 markersize=5, label="global linear z-screen")
    axes[0].axhline(len(marks84), color=INK, linestyle="--", linewidth=1.5)
    axes[0].annotate(f"Marks at 0.1 = {len(marks84)} nodes", (cutoffs[-1], len(marks84)),
                     fontsize=9, color=INK, va="bottom",
                     xytext=(0, 4), textcoords="offset points")
    axes[0].set_yscale("log"); axes[0].set_xscale("log")
    axes[0].set_ylabel("nodes selected")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].plot(out["cutoff"], out["jaccard_vs_marks84"], color=BLUE, linewidth=2,
                 marker="o", markersize=5, label="vs Marks at 0.1 (84 nodes)")
    axes[1].plot(out["cutoff"], out["jaccard_vs_size_matched_marks"], color=ORANGE,
                 linewidth=2, marker="o", markersize=5, label="vs Marks, size-matched")
    axes[1].set_xscale("log"); axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Jaccard similarity of node sets")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.set_xlabel("screen cutoff on per-row |IE|")
        ax.grid(color="#e6e6e3", linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Global linear z-screen, no recurrence test: cutoff vs agreement "
                 "with Marks", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/global-cutoff-sweep.png", dpi=160, facecolor="#fcfcfb")
    print("\nwrote figures/global-cutoff-sweep.png, docs/global-cutoff-sweep.csv")


if __name__ == "__main__":
    main()
