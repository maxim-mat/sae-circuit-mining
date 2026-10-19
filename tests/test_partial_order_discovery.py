import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("pm4py")
from scripts.partial_order_discovery import (  # noqa: E402
    build_partial_orders, check_same_examples)


def _events():
    # example 0: feature 3:7 at two positions, 4:2, 6:5. example 1: only 3:7.
    return pd.DataFrame({
        "example_idx":  [0, 0, 0, 0, 1],
        "site_ordinal": [3, 3, 4, 6, 3],
        "token_idx":    [1, 2, 2, 2, 1],
        "feature_idx":  [7, 7, 2, 5, 7],
        "attribution":  [0.5, 0.4, -0.3, 0.2, 0.6],
    })


def _edges():
    return pd.DataFrame({
        "example_idx": [0, 0, 0, 0, 0, 0, 1],
        "src_ordinal": [3, 4, 3, 3, 3, 6, 3],
        "src_feature": [7, 2, 7, -1, 7, 5, 7],
        "dst_ordinal": [4, 6, 6, 4, 6, -1, 4],
        "dst_feature": [2, 5, 9, 2, 5, -1, 2],
        "edge_ie":     [0.05, -0.02, 0.9, 0.9, 0.01, 0.9, 0.9],
    })


def _arcs(po):
    labels, arcs = po
    return {(labels[i], labels[j]) for i, j in arcs}


def test_one_event_per_feature_and_prompt_whatever_its_positions():
    pos = build_partial_orders(_events(), _edges(), 0.01)
    assert sorted(pos[0][0]) == ["3:7", "4:2", "6:5"]
    assert pos[1][0] == ["3:7"]


def test_arcs_need_both_endpoints_as_events_and_abs_ie_strictly_over_threshold():
    pos = build_partial_orders(_events(), _edges(), 0.01)
    # 3:7->6:9 has no event at 6:9;
    # 3:7->6:5 is exactly at the threshold; error source and metric destination are not events.
    assert _arcs(pos[0]) == {("3:7", "4:2"), ("4:2", "6:5")}


def test_a_prompt_whose_endpoint_was_filtered_has_no_arcs():
    pos = build_partial_orders(_events(), _edges(), 0.01)
    assert pos[1][1] == []


def test_mined_from_the_built_orders():
    from src.discovery.ilp2 import discover_ilp2
    net, _, _ = discover_ilp2(build_partial_orders(_events(), _edges(), 0.01))
    assert {t.label for t in net.transitions} == {"3:7", "4:2", "6:5"}


def _params(**data):
    return {"suite": {"name": "pythia-marks", "device": "cuda"},
            "data": {"path": "rc.json", "num_examples": 100, **data}, "seed": 42}


def test_accepts_hives_built_from_the_same_examples():
    edge = _params()
    edge["suite"]["device"] = "cpu"              # irrelevant to which prompts
    check_same_examples(_params(), edge)


@pytest.mark.parametrize("edge", [_params(num_examples=5), {**_params(), "seed": 0}])
def test_rejects_hives_built_from_different_examples(edge):
    with pytest.raises(ValueError, match="example_idx"):
        check_same_examples(_params(), edge)


def test_script_does_not_import_torch():
    probe = "import sys, scripts.partial_order_discovery; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False"
