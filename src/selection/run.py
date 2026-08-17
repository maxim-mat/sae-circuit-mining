"""The selection stage: aggregate node IEs, then pick the circuit's node set.

Reads the raw node hive from the node-attribution stage and writes
`nodes.parquet` -- the aggregated node table plus a boolean `selected` column.
Both selectors produce that same contract so downstream stages never learn
which one ran.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.attribution.nodes import read_node_hive
from src.calibrated_feature_selection import FEATURE_COLS, select_calibrated
from src.selection.aggregate import aggregate
from src.selection.threshold import select_by_threshold

logger = logging.getLogger(__name__)


def run_selection(cfg, node_dir: Path, out_dir: Path) -> dict:
    """Aggregate and select. Returns manifest metadata; caching is the caller's."""
    node_dir, out_dir = Path(node_dir), Path(out_dir)
    started = time.time()

    raw = read_node_hive(node_dir)
    nodes = aggregate(raw, cfg.selection.aggregation)

    if cfg.selection.method == "threshold":
        nodes = select_by_threshold(nodes, cfg.selection.node_threshold)
    elif cfg.selection.method == "calibrated":
        # The calibrated selector tests the raw rows, so its verdict comes back
        # keyed by (site_ordinal, feature_idx) and is joined onto the aggregate.
        feats = select_calibrated(raw, cfg)
        nodes = nodes.merge(feats[FEATURE_COLS + ["selected"]], on=FEATURE_COLS,
                            how="left")
        nodes["selected"] = nodes["selected"].fillna(False).astype(bool)
    else:
        raise ValueError(
            f"Unknown selection method {cfg.selection.method!r}; "
            f"expected 'threshold' or 'calibrated'")

    nodes.to_parquet(out_dir / "nodes.parquet", compression="zstd", index=False)
    n_selected = int(nodes["selected"].sum())
    logger.info("Selection: %d of %d nodes selected", n_selected, len(nodes))
    return {
        "n_nodes": len(nodes),
        "n_selected": n_selected,
        "selected_frac": n_selected / len(nodes) if len(nodes) else 0.0,
        "runtime_s": round(time.time() - started, 2),
    }
