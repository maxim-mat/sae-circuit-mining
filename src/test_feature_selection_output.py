"""Self-check: Part-1 writes its event log in the hive layout that Part-2 and
edge_attribution's --nodes expect. Run: python src/test_feature_selection_output.py"""
import os
import tempfile

import pandas as pd

from calibrated_feature_selection import _write_hive


def make_log():
    return pd.DataFrame({
        "prompt_idx":  [0, 0, 1],
        "layer":       [3, 5, 5],
        "token_idx":   [1, 2, 2],
        "feature_idx": [10, -1, 7],      # -1 = SAE reconstruction-error node
        "attribution": [1.0, -2.0, 0.5],
    })


def test_event_log_reads_as_node_set():
    from edge_attribution import load_node_set

    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "eventlog")
        part = _write_hive(make_log(), root, "abc_counterfactual")
        assert os.path.isdir(part), part

        nodes = load_node_set(root, include_error=False)
        assert nodes == {"abc_counterfactual": {3: [10], 5: [7]}}, nodes
        with_err = load_node_set(root, include_error=True)
        assert with_err["abc_counterfactual"][5] == [-1, 7], with_err


def test_rerun_replaces_partition():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "eventlog")
        _write_hive(make_log(), root, "abc_counterfactual")
        _write_hive(make_log(), root, "abc_counterfactual")
        _write_hive(make_log(), root, "other_counterfactual")
        df = pd.read_parquet(root)
        assert len(df) == 2 * len(make_log()), len(df)
        assert set(df["counterfactual"].astype(str)) == {
            "abc_counterfactual", "other_counterfactual"}


if __name__ == "__main__":
    test_event_log_reads_as_node_set()
    test_rerun_replaces_partition()
    print("all feature-selection output checks passed")
