from pathlib import Path
import pytest
from src.pipeline import STAGE_ORDER, resolve_stages


def test_stage_order_matches_the_declared_pipeline():
    assert STAGE_ORDER == ["node_attribution", "selection", "edge_attribution",
                           "circuit", "evaluation"]


def test_single_stage_selection_returns_just_that_stage():
    assert resolve_stages(STAGE_ORDER, only_stage="selection") == ["selection"]


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        resolve_stages(STAGE_ORDER, only_stage="nope")


def test_disabled_stages_are_dropped_in_order():
    assert resolve_stages(["node_attribution", "selection"], only_stage=None) == \
        ["node_attribution", "selection"]
