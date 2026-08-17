import pandas as pd
import pytest
from src.circuit.build import build_circuit, save_circuit, load_circuit


def _nodes():
    return pd.DataFrame({
        "site_ordinal": [1, 1, 3], "block": [0, 0, 1], "kind": ["attn"] * 3,
        "feature_idx": [10, 11, 20], "token_idx": [-1] * 3,
        "ie": [0.5, 0.01, 0.4], "selected": [True, False, True]})


def _edges():
    return pd.DataFrame({
        "src_ordinal": [1, 1], "src_feature": [10, 11],
        "dst_ordinal": [3, 3], "dst_feature": [20, 20],
        "edge_ie": [0.2, 0.001]})


def test_edges_below_threshold_are_dropped():
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    assert len(c.edges) == 1
    assert c.edges.iloc[0]["src_feature"] == 10


def test_unselected_nodes_are_dropped():
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    assert set(c.nodes["feature_idx"]) == {10, 20}


def test_no_dangling_edges_after_node_pruning():
    """An edge whose endpoint was pruned must not survive."""
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.0)
    kept = set(zip(c.nodes["site_ordinal"], c.nodes["feature_idx"]))
    for _, e in c.edges.iterrows():
        assert (e["src_ordinal"], e["src_feature"]) in kept
        assert (e["dst_ordinal"], e["dst_feature"]) in kept


def test_roundtrip(tmp_path):
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    save_circuit(c, tmp_path / "c.json")
    back = load_circuit(tmp_path / "c.json")
    assert back.total_effect == pytest.approx(1.0)
    pd.testing.assert_frame_equal(
        back.nodes.reset_index(drop=True), c.nodes.reset_index(drop=True))
