"""Aggregation of node IEs across token positions and examples.

Follows Marks et al. App. A.2 (arXiv:2403.19647):

  mode="sum"  (their non-templatic path) -- signed sum over token positions,
              then the example-wise mean. Positions cancel, which is intended.
  mode="none" (their templatic path) -- positions stay distinct nodes; only the
              example-wise mean is taken. Requires length-aligned examples, which
              data.use_min_length_only guarantees.

The example mean divides by the TOTAL number of examples, not the number in
which the node appeared: a node absent from an example had zero effect there.
"""
from __future__ import annotations

import pandas as pd

KEY_COLS = ["site_ordinal", "block", "kind", "feature_idx"]


def aggregate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in ("none", "sum"):
        raise ValueError(f"Unknown aggregation {mode!r}; expected 'none' or 'sum'")
    if df.empty:
        return df.assign(ie=pd.Series(dtype="float64"))

    n_examples = df["example_idx"].nunique()
    if mode == "sum":
        per_example = (df.groupby(KEY_COLS + ["example_idx"], sort=False,
                                  observed=True)["attribution"]
                         .sum().reset_index())
        keys = KEY_COLS
    else:
        per_example = df
        keys = KEY_COLS + ["token_idx"]

    # observed=True: KEY_COLS includes `kind`, which read_node_hive leaves
    # categorical for memory. Without this the groupby materialises every
    # (kind x site x feature x position) combination that never occurred.
    out = (per_example.groupby(keys, sort=False, observed=True)["attribution"].sum()
           / n_examples)
    out = out.rename("ie").reset_index()
    if mode == "sum":
        out["token_idx"] = -1
    return out
