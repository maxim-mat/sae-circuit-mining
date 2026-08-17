import pandas as pd
import pytest
from src.attribution.edges import candidate_endpoints


def _nodes():
    return pd.DataFrame({
        "site_ordinal": [1, 1, 3, 3],
        "feature_idx": [10, 11, 20, 21],
        "selected":    [True, False, True, False],
    })


def test_marks_mode_filters_downstream_only():
    """attribution.py:jvp loops over surviving DOWNSTREAM nodes but keeps every
    nonzero upstream feature -- upstream is NOT thresholded."""
    up, down = candidate_endpoints(_nodes(), 1, 3, filter_mode="marks")
    assert set(down) == {20}
    assert set(up) == {10, 11}


def test_both_mode_filters_both_endpoints():
    up, down = candidate_endpoints(_nodes(), 1, 3, filter_mode="both")
    assert set(down) == {20}
    assert set(up) == {10}


def test_unknown_filter_mode_raises():
    with pytest.raises(ValueError):
        candidate_endpoints(_nodes(), 1, 3, filter_mode="neither")
