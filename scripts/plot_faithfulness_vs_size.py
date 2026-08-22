"""EXPLORATORY: faithfulness vs circuit size, Marks' constant node threshold
against the global linear z-screen (no recurrence test).

Both selectors are constant thresholds -- Marks' on the example-mean |IE| of a
node, the screen's on per-row |IE| -- so plotting them against node count is the
only comparison that is not confounded by circuit size. Numbers come from the
evaluation manifests written by the runs recorded in docs/faithfulness-vs-size.csv
(n=100, eval 40 examples, edges off, mean ablation).

    python scripts/plot_faithfulness_vs_size.py
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, ".")
CSV = "docs/faithfulness-vs-size.csv"
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1a1a19"


def main() -> None:
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, col, name in zip(axes, ("f_23_mean", "f_all_mean"),
                             ("latter 2/3 of the circuit (paper section 3.2)",
                              "whole circuit (their ablation.py default)")):
        for arm, colour in (("marks", ORANGE), ("global_z", BLUE)):
            d = df[df["arm"] == arm].sort_values("nodes")
            ax.plot(d["nodes"], d[col], color=colour, linewidth=2, marker="o",
                    markersize=5,
                    label="Marks |mean IE| > T" if arm == "marks"
                          else "global linear z-screen, no recurrence")
        ax.set_xscale("log")
        ax.set_xlabel("nodes in the circuit")
        ax.set_title(name, fontsize=11, color=INK)
        ax.grid(color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("faithfulness")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    d84 = df[(df["arm"] == "marks") & (df["nodes"] == 84)].iloc[0]
    axes[0].annotate("Marks' published\noperating point (84)", (84, d84["f_23_mean"]),
                     fontsize=9, color=INK, xytext=(10, -28),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))
    fig.suptitle("At equal circuit size the constant node threshold wins everywhere",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/faithfulness-vs-size.png", dpi=160, facecolor="#fcfcfb")
    print("wrote figures/faithfulness-vs-size.png")


if __name__ == "__main__":
    main()
