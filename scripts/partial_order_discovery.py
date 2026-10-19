"""Per-row node filtering + edge thresholding -> partial orders -> Petri net.

Supersedes src/partial_order.py, which built edges from virtual weights
(decoder . encoder, the direct residual path only) and fed linearisations to
the inductive miner. This reads measured edge IEs from an edge hive and mines
the partial orders directly with ILP².

One partial order per prompt:
  * events: the nodes surviving the per-row filter of simple_discovery,
    collapsed to one event per (site, feature). Edge IEs are summed over
    positions (src/attribution/edges.py), so positions cannot be told apart.
  * order: i < j iff the edge i -> j has |edge_ie| > EDGE_THRESHOLD and both
    endpoints are events of that prompt. Edges from SAE-error nodes and into
    the metric node are dropped. With edges.template="marks" the edges are
    direct effects along the residual chain, so the order between distant
    events runs through intermediate events: a filtered-out intermediate
    leaves its neighbours unordered (concurrent). "total_effect" edges with
    pairs="all" order distant events directly.

The edge hive only scores edges into the nodes of the selection it was run
against, so a per-row event outside that selection has no incoming arcs.

    python -m scripts.partial_order_discovery

Edit the variables below, then run. Writes outputs/nets/<EXPERIMENT_NAME>/
with net.pnml (net + initial/final marking), net.svg (figure) and
manifest.json.
"""
from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pm4py

from scripts.simple_discovery import filter_rows, read_nodes
from src.discovery.ilp2 import discover_ilp2
from src.utils.manifest import write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "rc-n100-row-0.1-edge-0.01-ilp2"
NODE_DIR = Path("outputs_sweep/node_attribution/77c6178e")   # rc_train, n=100
# Must be scored on the same examples as NODE_DIR (checked). None on disk yet
# matches 77c6178e: the existing hives are 5-example smoke runs.
EDGE_DIR = Path("outputs/edge_attribution/aa132bc3")
NODE_THRESHOLD = 0.1
EDGE_THRESHOLD = 0.01               # Marks' edge threshold (edges.threshold)
DISCOVER = discover_ilp2
DISCOVER_KWARGS: dict = {}          # ILP²: add_start_stop
OUT_ROOT = Path("outputs/nets")

EVENT = ["example_idx", "site_ordinal", "feature_idx"]


def read_edges(edge_dir: Path) -> pd.DataFrame:
    """The edge hive, read without src.attribution.edges.read_edge_hive, which
    imports torch (see read_nodes). dst_ordinal is the partition key and comes
    back categorical."""
    return pd.read_parquet(Path(edge_dir) / "edges").astype({"dst_ordinal": "int16"})


def check_same_examples(node_params: dict, edge_params: dict) -> None:
    """Raise unless both hives were computed on the same prompts, in the same
    order: example_idx is only an index into load_examples(data, seed)."""
    def key(p):
        return p["suite"]["name"], p["seed"]
    shared = node_params["data"].keys() & edge_params["data"].keys()
    data_diff = {k: (node_params["data"][k], edge_params["data"][k])
                 for k in shared if node_params["data"][k] != edge_params["data"][k]}
    if data_diff or key(node_params) != key(edge_params):
        raise ValueError(
            "node and edge hives were built from different examples, so their "
            f"example_idx do not match: data {data_diff}, (suite, seed) "
            f"{key(node_params)} vs {key(edge_params)}")


def build_partial_orders(events: pd.DataFrame, edges: pd.DataFrame,
                         threshold: float) -> list[tuple[list[str], list[tuple[int, int]]]]:
    """One (labels, arcs) partial order per prompt with events, in example
    order: event i is labelled "site:feature"; arc (i, j) is an edge with
    |edge_ie| > threshold (strict) between two of the prompt's events."""
    ev = events[EVENT].drop_duplicates().sort_values(EVENT, kind="stable")
    ev = ev.assign(i=ev.groupby("example_idx").cumcount(),
                   label=ev["site_ordinal"].astype(str) + ":" + ev["feature_idx"].astype(str))
    kept = edges[(edges["src_feature"] != -1) & (edges["dst_ordinal"] != -1)
                 & (edges["dst_feature"] != -1) & (edges["edge_ie"].abs() > threshold)]

    def endpoint(side):
        return ev[EVENT + ["i"]].rename(columns={
            "site_ordinal": f"{side}_ordinal", "feature_idx": f"{side}_feature", "i": side})
    arcs = (kept.merge(endpoint("src"), on=["example_idx", "src_ordinal", "src_feature"])
                .merge(endpoint("dst"), on=["example_idx", "dst_ordinal", "dst_feature"]))
    arcs_by_ex = {ex: list(zip(g["src"], g["dst"])) for ex, g in arcs.groupby("example_idx")}
    return [(g["label"].tolist(), arcs_by_ex.get(ex, []))
            for ex, g in ev.groupby("example_idx", sort=True)]


def main():
    out_dir = OUT_ROOT / EXPERIMENT_NAME
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already holds a run; change EXPERIMENT_NAME.")

    node_params = json.loads((NODE_DIR / "manifest.json").read_text())["params"]
    edge_params = json.loads((EDGE_DIR / "manifest.json").read_text())["params"]
    check_same_examples(node_params, edge_params)

    raw = read_nodes(NODE_DIR)
    events = filter_rows(raw, NODE_THRESHOLD)
    pos = build_partial_orders(events, read_edges(EDGE_DIR), EDGE_THRESHOLD)
    n_events = sum(len(labels) for labels, _ in pos)
    n_arcs = sum(len(arcs) for _, arcs in pos)
    n_isolated = sum(len(labels) - len({e for a in arcs for e in a}) for labels, arcs in pos)
    logger.info("%d partial orders, %d events (%d without arcs), %d arcs, %d activities",
                len(pos), n_events, n_isolated, n_arcs,
                len({lab for labels, _ in pos for lab in labels}))

    net, im, fm = DISCOVER(pos, **DISCOVER_KWARGS)

    out_dir.mkdir(parents=True, exist_ok=True)
    pm4py.write_pnml(net, im, fm, str(out_dir / "net.pnml"))
    pm4py.save_vis_petri_net(net, im, fm, str(out_dir / "net.svg"))   # needs Graphviz

    params = {
        "node_dir": str(NODE_DIR),
        "node_attribution": node_params,
        "edge_dir": str(EDGE_DIR),
        "edge_attribution": edge_params,
        "node_threshold": NODE_THRESHOLD,
        "edge_threshold": EDGE_THRESHOLD,
        "discovery": DISCOVER.__name__,
        "discovery_kwargs": DISCOVER_KWARGS,
        "pm4py_version": pm4py.__version__,
        "scipy_version": version("scipy"),
    }
    write_manifest(out_dir, "discovery", params, {
        "n_rows_raw": len(raw),
        "n_partial_orders": len(pos),
        "n_events": n_events,
        "n_events_without_arcs": n_isolated,
        "n_arcs": n_arcs,
        "n_places": len(net.places),
        "n_transitions": len(net.transitions),
        "n_net_arcs": len(net.arcs),
    })
    logger.info("Net: %d places, %d transitions -> %s",
                len(net.places), len(net.transitions), out_dir)


if __name__ == "__main__":
    main()
