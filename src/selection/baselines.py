"""Selectors that are not Marks' threshold and not the hypothesis test.

`random` is the floor: a faithfulness number at k nodes means nothing until you
know what k nodes drawn at random score. Marks use neuron circuits for this role
in Fig. 3; a size-matched random feature set is the tighter control, since it
holds the node budget fixed and varies only which nodes.

`screen_magnitude` decomposes the calibrated arm. It keeps Stage 1 (the MAD
screen) and drops Stage 2 (the recurrence test), ranking the surviving features
by summed |IE| instead. If it tracks the calibrated arm, the screen is doing the
work; if it tracks the threshold arm, the recurrence test is what distinguishes
them. Either answer is more informative than the two-arm comparison alone.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_random(nodes: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Mark exactly k uniformly-drawn nodes as selected.

    Seeded off cfg.seed so a given (k, seed) is reproducible and lands in the
    manifest like any other selection.
    """
    out = nodes.copy()
    k = min(k, len(out))
    picks = np.random.default_rng(seed).choice(len(out), size=k, replace=False)
    out["selected"] = False
    out.iloc[picks, out.columns.get_loc("selected")] = True
    logger.info("Random baseline: %d of %d nodes (seed %d)", k, len(out), seed)
    return out


def select_screen_magnitude(nodes_raw: pd.DataFrame, nodes: pd.DataFrame,
                            feature_cols: list[str], k: int,
                            screen_q: float, screen_n0: float,
                            pool_col: str, scale: str = "log",
                            method: str = "mad",
                            scope: str = "group") -> pd.DataFrame:
    """Stage-1 screen, then top-k by summed |IE| over the rows that passed it.

    Deliberately NOT a calibrated test: no null, no BH, no recurrence. The
    statistic is magnitude, which is what faithfulness and minimality reward,
    applied only to rows the noise floor let through.
    """
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    screened = mad_screen(add_abs_and_sign(nodes_raw), q=screen_q, n0=screen_n0,
                          pool_col=pool_col, scale=scale, method=method,
                          scope=scope)
    passed = screened[screened["kept_screen"]]
    score = (passed.groupby(feature_cols)["abs_attr"].sum()
             .rename("screen_magnitude").reset_index())
    keep = score.nlargest(min(k, len(score)), "screen_magnitude")[feature_cols]
    keep["selected"] = True

    out = nodes.merge(keep, on=feature_cols, how="left")
    out["selected"] = out["selected"].fillna(False).astype(bool)
    logger.info("Screen+magnitude: %d of %d nodes (%d cleared the screen)",
                int(out["selected"].sum()), len(out), len(score))
    return out
