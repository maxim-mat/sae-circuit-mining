"""EXPLORATORY, read-only, CPU (~1 min): what does Marks' one constant threshold
mean inside each (prompt, position, layer) group?

Treats T = 0.1 as a test statistic evaluated against each group's OWN fitted
null, and reports the z-score and the implied per-group false-positive rate.
A constant threshold is a constant only in |IE| units -- in probability units it
is a different test in every group, and this quantifies by how much. The global
z-test is the degenerate case (one group), drawn as a reference line.

Writes figures/marks-threshold-zscores.png, figures/marks-threshold-pvalues.png
and docs/marks-threshold-zscores.csv. Not part of the pipeline.

    python scripts/explore_threshold_zscores.py
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
    FP16_SUBNORMAL_MAX, add_abs_and_sign)

NODE_DIR = "outputs_n100/node_attribution/77c6178e"
T_MARKS = 0.1                      # Marks' node threshold, in |IE| units
MIN_GROUP = 30                     # a null fitted on fewer values is noise
GRAINS = {"prompt x layer": ["example_idx", "site_ordinal"],
          "prompt x layer x position": ["example_idx", "site_ordinal", "token_idx"]}
# dataviz reference palette, categorical slots 1-4 (validated: CVD dE 9.1)
C = {"linear|prompt x layer": "#2a78d6", "linear|prompt x layer x position": "#eb6834",
     "log|prompt x layer": "#1baf7a", "log|prompt x layer x position": "#eda100"}
# Figure 1 splits by scale into panels, so colour there follows the GRAIN only.
GRAIN_C = {"prompt x layer": "#2a78d6", "prompt x layer x position": "#eb6834"}
INK, MUTED = "#1a1a19", "#8a8a85"


def fit(df: pd.DataFrame, group_cols: list[str], scale: str) -> pd.DataFrame:
    """Per-group Gaussian null (mean/sd -- a z-test), on |IE| or log|IE|."""
    v = df.loc[df["abs_attr"] >= FP16_SUBNORMAL_MAX, group_cols + ["abs_attr"]].copy()
    if scale == "log":
        v["abs_attr"] = np.log(v["abs_attr"])
    g = v.groupby(group_cols)["abs_attr"]
    out = pd.DataFrame({"mu": g.mean(), "sd": g.std(ddof=1), "n": g.size()})
    return out[(out["n"] >= MIN_GROUP) & (out["sd"] > 0)]


def main() -> None:
    raw = add_abs_and_sign(read_node_hive(NODE_DIR))
    rows, series = [], {}
    for scale in ("linear", "log"):
        t = np.log(T_MARKS) if scale == "log" else T_MARKS
        allv = raw.loc[raw["abs_attr"] >= FP16_SUBNORMAL_MAX, "abs_attr"].to_numpy()
        allv = np.log(allv) if scale == "log" else allv
        z_glob = (t - allv.mean()) / allv.std(ddof=1)
        for grain, cols in GRAINS.items():
            st = fit(raw, cols, scale)
            z = (t - st["mu"]) / st["sd"]
            p = norm.sf(z)
            series[f"{scale}|{grain}"] = (z.to_numpy(), p, z_glob, norm.sf(z_glob))
            rows.append({
                "scale": scale, "grain": grain, "groups": len(st),
                "z_global": z_glob, "p_global": norm.sf(z_glob),
                "z_median": z.median(), "z_q10": z.quantile(0.1),
                "z_q90": z.quantile(0.9), "z_min": z.min(), "z_max": z.max(),
                "p_median": float(np.median(p)),
                "p_q10": float(np.quantile(p, 0.1)), "p_q90": float(np.quantile(p, 0.9)),
                "frac_groups_p_above_1e-3": float((p > 1e-3).mean()),
                "frac_groups_p_below_1e-6": float((p < 1e-6).mean()),
            })
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    tab.to_csv("docs/marks-threshold-zscores.csv", index=False)

    # --- Figure 1: the z Marks' constant threshold lands on, per group --------
    # ECDF, not a histogram: the linear-null z reaches 2019, so binning either
    # hides the tail or piles it into one clipped bar.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, scale in zip(axes, ("linear", "log")):
        for grain in GRAINS:
            z, _, z_glob, _ = series[f"{scale}|{grain}"]
            zs = np.sort(z)
            ax.plot(zs, np.arange(1, zs.size + 1) / zs.size, linewidth=2,
                    color=GRAIN_C[grain], label=f"per {grain}")
        ax.axvline(z_glob, color=INK, linestyle="--", linewidth=1.6)
        ax.annotate(f"one global null, z = {z_glob:.2f}", (z_glob, 0.03),
                    xycoords=("data", "axes fraction"), ha="left", va="bottom",
                    fontsize=9, color=INK, xytext=(6, 0), textcoords="offset points")
        ax.axvspan(-1, 0, color="#f2d7d7", alpha=0.7, linewidth=0)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_title(f"null fitted on {'|IE|' if scale == 'linear' else 'log |IE|'}",
                     fontsize=11, color=INK)
        ax.set_xlabel("z of Marks' threshold 0.1 within the group")
        ax.grid(color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("fraction of groups")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    axes[0].annotate("below the group's own noise mean", (-1, 0.5),
                     fontsize=8, color="#a33", ha="left", va="center",
                     xytext=(4, 0), textcoords="offset points")
    fig.suptitle("One constant threshold is a different test in every group "
                 "(x is symlog above 10)", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/marks-threshold-zscores.png", dpi=160,
                facecolor="#fcfcfb")

    # --- Figure 2: implied per-group false-positive rate ----------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key, (_, p, _, p_glob) in series.items():
        scale, grain = key.split("|")
        pc = np.clip(p, 1e-16, 1.0)
        ax.plot(np.sort(pc), np.arange(1, pc.size + 1) / pc.size, linewidth=2,
                color=C[key], label=f"{scale} null, per {grain}")
    for key in ("linear|prompt x layer", "log|prompt x layer"):
        p_glob = series[key][3]
        ax.axvline(max(p_glob, 1e-16), color=INK, linestyle="--", linewidth=1.5)
        ax.annotate(f"global {key.split('|')[0]}", (max(p_glob, 1e-16), 0.5),
                    xycoords=("data", "axes fraction"), rotation=90, fontsize=8,
                    color=INK, ha="right", va="center",
                    xytext=(-4, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("implied per-group false-positive rate at |IE| > 0.1 (clipped at 1e-16)")
    ax.set_ylabel("fraction of groups")
    ax.grid(color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("What Marks' 0.1 costs, in probability units", fontsize=13,
                 color=INK, loc="left")
    fig.tight_layout()
    fig.savefig("figures/marks-threshold-pvalues.png", dpi=160, facecolor="#fcfcfb")
    print("\nwrote figures/marks-threshold-zscores.png, "
          "figures/marks-threshold-pvalues.png, docs/marks-threshold-zscores.csv")


if __name__ == "__main__":
    main()
