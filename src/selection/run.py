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


def _attach_example_meta(raw: pd.DataFrame, node_dir: Path, block_col: str) -> pd.DataFrame:
    """Join the per-example template map so `block_col` names a real column.

    Blocking makes the recurrence test's trials the BLOCKS, not the examples, so
    a dataset with one template collapses to a single trial and the test can
    conclude nothing. That is the correct answer to "does this generalise across
    templates" from single-template data -- but it is not the question a
    single-frame dataset like Marks' SVA can answer, so refuse loudly rather
    than silently return an empty circuit.
    """
    meta_path = Path(node_dir) / "examples.parquet"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"selection.block_col={block_col!r} needs {meta_path}, which this node "
            f"hive predates. Recompute node attribution (run.force=true)."
        )
    meta = pd.read_parquet(meta_path)
    if block_col not in meta.columns:
        raise ValueError(
            f"selection.block_col={block_col!r} not in {sorted(meta.columns)}. "
            f"Set data.template_field so the loader records it."
        )
    n_blocks = meta[block_col].nunique()
    if n_blocks < 2:
        raise ValueError(
            f"selection.block_col={block_col!r} has {n_blocks} distinct value(s) "
            f"over {len(meta)} examples, so blocking leaves {n_blocks} trial(s) and "
            f"the recurrence test has no power. This is the expected shape for a "
            f"single-template dataset (Marks' SVA is one syntactic frame): leave "
            f"block_col unset and treat the template as a stratifier, or make the "
            f"unit of replication the dataset structure instead."
        )
    logger.info("Blocking on %s: %d blocks over %d examples",
                block_col, n_blocks, len(meta))
    return raw.merge(meta[["example_idx", block_col]], on="example_idx", how="left")


def run_selection(cfg, node_dir: Path, out_dir: Path) -> dict:
    """Aggregate and select. Returns manifest metadata; caching is the caller's."""
    node_dir, out_dir = Path(node_dir), Path(out_dir)
    started = time.time()

    raw = read_node_hive(node_dir)
    nodes = aggregate(raw, cfg.selection.aggregation)

    block_col = cfg.selection.block_col
    if block_col:
        raw = _attach_example_meta(raw, node_dir, block_col)

    # Per-stage diagnostics from the calibrated path. Without these you cannot
    # tell whether a large circuit came from a permissive Stage-1 screen or a
    # permissive Stage-2 test, which is the first question to ask of a bad
    # selection -- so they go into the manifest and to wandb.
    stage_summary: dict = {}

    if cfg.selection.method == "threshold":
        nodes = select_by_threshold(nodes, cfg.selection.node_threshold)
    elif cfg.selection.method == "calibrated":
        # The calibrated selector tests the raw rows, so its verdict comes back
        # keyed by (site_ordinal, feature_idx) and is joined onto the aggregate.
        feats = select_calibrated(raw, cfg)
        nodes = nodes.merge(feats[FEATURE_COLS + ["selected"]], on=FEATURE_COLS,
                            how="left")
        nodes["selected"] = nodes["selected"].fillna(False).astype(bool)
        summary = feats.attrs.get("summary")
        if summary is not None:
            stage_summary = {f"screen_{k}": float(v) for k, v in summary.items()}
            logger.info("Calibrated stages: %s", dict(summary))
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
        **stage_summary,
    }
