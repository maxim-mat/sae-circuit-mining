import pandas as pd
from src.selection.threshold import select_by_threshold


def test_threshold_is_on_absolute_value():
    nodes = pd.DataFrame({"site_ordinal": [0, 0], "feature_idx": [1, 2],
                          "ie": [-0.5, 0.05]})
    out = select_by_threshold(nodes, 0.1)
    assert out.set_index("feature_idx")["selected"].to_dict() == {1: True, 2: False}


def test_threshold_is_strict():
    nodes = pd.DataFrame({"site_ordinal": [0], "feature_idx": [1], "ie": [0.1]})
    assert not select_by_threshold(nodes, 0.1)["selected"].iloc[0]
