import pandas as pd
import pytest
from src.selection.aggregate import aggregate


def _df(rows):
    return pd.DataFrame(rows, columns=[
        "example_idx", "site_ordinal", "block", "kind",
        "token_idx", "feature_idx", "attribution"])


def test_sum_mode_sums_signed_over_positions_then_means_over_examples():
    """Marks aggregation='sum': signed sum over sequence, then batch mean.
    Cancellation across positions is the whole point -- +0.5 and -0.45 is a
    0.05 node, not a 0.5 node."""
    df = _df([
        [0, 3, 1, "resid", 0, 7, 0.5],
        [0, 3, 1, "resid", 1, 7, -0.45],
        [1, 3, 1, "resid", 0, 7, 0.1],
    ])
    out = aggregate(df, "sum")
    assert len(out) == 1
    assert out.iloc[0]["ie"] == pytest.approx((0.05 + 0.1) / 2)


def test_none_mode_keeps_positions_as_distinct_nodes():
    df = _df([
        [0, 3, 1, "resid", 0, 7, 0.5],
        [0, 3, 1, "resid", 1, 7, -0.45],
        [1, 3, 1, "resid", 0, 7, 0.1],
    ])
    out = aggregate(df, "none").set_index("token_idx")
    assert len(out) == 2
    assert out.loc[0, "ie"] == pytest.approx((0.5 + 0.1) / 2)
    assert out.loc[1, "ie"] == pytest.approx(-0.45 / 2)   # absent = zero effect


def test_example_mean_divides_by_all_examples_not_present_ones():
    """A node firing in 1 of 2 examples has half the mean effect."""
    df = _df([
        [0, 3, 1, "resid", 0, 7, 1.0],
        [1, 3, 1, "resid", 0, 9, 1.0],
    ])
    out = aggregate(df, "sum").set_index("feature_idx")
    assert out.loc[7, "ie"] == pytest.approx(0.5)
    assert out.loc[9, "ie"] == pytest.approx(0.5)


def test_error_nodes_survive_aggregation():
    df = _df([[0, 3, 1, "resid", 0, -1, 0.3]])
    assert (aggregate(df, "sum")["feature_idx"] == -1).any()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        aggregate(_df([]), "median")
