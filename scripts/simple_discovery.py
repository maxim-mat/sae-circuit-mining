"""Per-row node filtering -> event log -> Petri-net discovery.

Unlike SFC, which averages each node's IE over examples and thresholds the mean
(building the global circuit in the same step), this keeps every
(example, position, feature) row whose OWN |IE| clears the threshold. A case is
one prompt and its events are the rows that survived in it.

    python -m scripts.simple_discovery

Edit the variables below, then run. Writes outputs/nets/<EXPERIMENT_NAME>/
with net.pnml (net + initial/final marking), net.svg (figure) and
manifest.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pm4py

from src.discovery.event_log import ACTIVITY, CASE, build_event_log
from src.utils.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "rc-n100-row-0.1-ilp"
NODE_DIR = Path("outputs_sweep/node_attribution/77c6178e")   # rc_train, n=100
NODE_THRESHOLD = 0.1
ACTIVITY_COLS = ["site_ordinal", "feature_idx"]
# Trace order. Site order linearises attn || mlp within a block; see
# Suite.parallel_attn before reading sequence into those pairs.
ORDER_COLS = ["example_idx", "site_ordinal", "token_idx"]
DISCOVER = pm4py.discover_petri_net_ilp
DISCOVER_KWARGS: dict = {}          # ILP: alpha (1.0 = no noise filtering)
OUT_ROOT = Path("outputs/nets")


def read_nodes(node_dir: Path) -> pd.DataFrame:
    """The node hive, read without src.attribution.nodes.read_node_hive.

    That module imports torch, whose bundled OpenMP runtime aborts scipy's
    LAPACK -- and with it pm4py's ILP solver -- with OMP Error #15 on the
    Windows anaconda env. Keeps read_node_hive's int cast: hive keys come back
    categorical, and site_ordinal would then sort as strings ("10" < "3").
    """
    return pd.read_parquet(Path(node_dir) / "nodes").astype({"site_ordinal": "int16"})


def filter_rows(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rows whose own |IE| > threshold (strict, as Marks), SAE error nodes
    (feature_idx == -1) dropped. No aggregation over examples or positions."""
    return raw[(raw["feature_idx"] != -1) & (raw["attribution"].abs() > threshold)]


def main():
    out_dir = OUT_ROOT / EXPERIMENT_NAME
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already holds a run; change EXPERIMENT_NAME.")

    raw = read_nodes(NODE_DIR)
    events = filter_rows(raw, NODE_THRESHOLD).sort_values(ORDER_COLS, kind="stable")
    log = build_event_log(events, activity_cols=ACTIVITY_COLS)
    logger.info("Event log: %d events, %d cases, %d activities",
                len(log), log[CASE].nunique(), log[ACTIVITY].nunique())

    net, im, fm = DISCOVER(log, **DISCOVER_KWARGS)

    # PNML over pickle: it carries both markings, is a fraction of a pickled
    # pm4py object graph for nets this size, survives pm4py upgrades, and ProM
    # (eST-Miner, conformance) reads it directly.
    out_dir.mkdir(parents=True, exist_ok=True)
    pm4py.write_pnml(net, im, fm, str(out_dir / "net.pnml"))
    # SVG: vector, so large nets stay legible when zoomed. Needs Graphviz.
    pm4py.save_vis_petri_net(net, im, fm, str(out_dir / "net.svg"))

    params = {
        "node_dir": str(NODE_DIR),
        # Provenance of the attributions: dataset, n, suite, IG settings.
        "node_attribution": json.loads((NODE_DIR / "manifest.json").read_text())["params"],
        "node_threshold": NODE_THRESHOLD,
        "activity_cols": ACTIVITY_COLS,
        "order_cols": ORDER_COLS,
        "discovery": DISCOVER.__name__,
        "discovery_kwargs": DISCOVER_KWARGS,
        "pm4py_version": pm4py.__version__,
    }
    write_manifest(out_dir, "discovery", params, {
        "n_rows_raw": len(raw),
        "n_cases_raw": int(raw["example_idx"].nunique()),
        "n_events": len(log),
        "n_cases": int(log[CASE].nunique()),
        "n_activities": int(log[ACTIVITY].nunique()),
        "n_places": len(net.places),
        "n_transitions": len(net.transitions),
        "n_arcs": len(net.arcs),
    })
    logger.info("Net: %d places, %d transitions -> %s",
                len(net.places), len(net.transitions), out_dir)


if __name__ == "__main__":
    main()
