"""Assemble nodes + edges into a circuit, and persist it.

Pruning order matters: nodes are filtered by `selected`, edges by
|edge_ie| > edge_threshold, and only THEN are edges dropped whose endpoints did
not survive. Pruning nodes after edges would leave dangling edges pointing at
features that are not in the circuit.

The metric node "y" (dst_ordinal = -1) has no row in the node table and is never
pruned -- it is the circuit's sink by construction (Marks et al., arXiv:2403.19647).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

Y_ORDINAL = -1


@dataclass
class Circuit:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    total_effect: float
    suite: str = ""
    params: dict = field(default_factory=dict)


def build_circuit(nodes: pd.DataFrame, edges: pd.DataFrame, total_effect: float,
                  edge_threshold: float, suite: str = "", params: dict | None = None
                  ) -> Circuit:
    kept_nodes = nodes[nodes["selected"]].reset_index(drop=True)
    kept = set(zip(kept_nodes["site_ordinal"], kept_nodes["feature_idx"]))

    kept_edges = edges[edges["edge_ie"].abs() > edge_threshold]
    alive = kept_edges.apply(
        lambda e: (e["src_ordinal"], e["src_feature"]) in kept
        and (e["dst_ordinal"] == Y_ORDINAL
             or (e["dst_ordinal"], e["dst_feature"]) in kept),
        axis=1) if len(kept_edges) else pd.Series(dtype=bool)
    kept_edges = kept_edges[alive].reset_index(drop=True) if len(kept_edges) else kept_edges

    return Circuit(nodes=kept_nodes, edges=kept_edges, total_effect=float(total_effect),
                   suite=suite, params=params or {})


def save_circuit(circuit: Circuit, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "nodes": circuit.nodes.to_dict(orient="records"),
        "edges": circuit.edges.to_dict(orient="records"),
        "total_effect": circuit.total_effect,
        "suite": circuit.suite,
        "params": circuit.params,
    }, indent=2, default=str))


def load_circuit(path) -> Circuit:
    data = json.loads(Path(path).read_text())
    return Circuit(
        nodes=pd.DataFrame(data["nodes"]),
        edges=pd.DataFrame(data["edges"]),
        total_effect=data["total_effect"],
        suite=data.get("suite", ""),
        params=data.get("params", {}),
    )


def run_circuit(cfg, node_dir: Path, selection_dir: Path, edge_dir: Path,
                out_dir: Path) -> dict:
    """Aggregate per-example edges, join with the selected nodes, save circuit.json.

    Edge IEs are averaged over the SAME denominator as node IEs -- the total
    example count -- so an edge present in half the examples carries half the
    effect (src/selection/aggregate.py states the reasoning).
    """
    started = time.time()
    nodes = pd.read_parquet(Path(selection_dir) / "nodes.parquet")
    totals = pd.read_parquet(Path(node_dir) / "total_effects.parquet")

    edge_path = Path(edge_dir) / "edges"
    if edge_path.exists():
        raw_edges = pd.read_parquet(edge_path)
        n_examples = max(len(totals), 1)
        key = ["src_ordinal", "src_feature", "dst_ordinal", "dst_feature"]
        edges = (raw_edges.groupby(key, sort=False)["edge_ie"].sum() / n_examples
                 ).rename("edge_ie").reset_index()
    else:
        logger.warning("No edge hive at %s; building a node-only circuit", edge_path)
        edges = pd.DataFrame(columns=["src_ordinal", "src_feature", "dst_ordinal",
                                      "dst_feature", "edge_ie"])

    circuit = build_circuit(nodes, edges, float(totals["total_effect"].mean()),
                            cfg.edges.threshold, suite=cfg.suite.name)
    save_circuit(circuit, Path(out_dir) / "circuit.json")
    logger.info("Circuit: %d nodes, %d edges", len(circuit.nodes), len(circuit.edges))
    return {
        "n_nodes": len(circuit.nodes),
        "n_edges": len(circuit.edges),
        "total_effect": circuit.total_effect,
        "runtime_s": round(time.time() - started, 2),
    }
