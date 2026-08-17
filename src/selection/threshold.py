"""Marks-style node selection: keep nodes whose mean IE exceeds a threshold."""
from __future__ import annotations

import pandas as pd


def select_by_threshold(nodes: pd.DataFrame, node_threshold: float) -> pd.DataFrame:
    """Add a boolean `selected` column for |ie| > node_threshold (strict,
    matching Marks' `effects[submod].abs() > node_threshold`)."""
    out = nodes.copy()
    out["selected"] = out["ie"].abs() > node_threshold
    return out
