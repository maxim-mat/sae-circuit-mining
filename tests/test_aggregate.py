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


def test_hive_categorical_partitions_do_not_explode_the_row_count(tmp_path):
    """Regression: pyarrow returns hive partition keys as Categoricals, and
    groupby(observed=False) then materialises the full cartesian product of
    category levels. On a real 5-example run that turned 96,574 rows into
    58,194,948. src.attribution.nodes.read_node_hive casts them back."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.attribution.nodes import NODE_SCHEMA, read_node_hive

    rows = {
        "example_idx": [0, 0, 1], "site_ordinal": [1, 3, 1], "block": [0, 1, 0],
        "kind": ["attn", "mlp", "attn"], "token_idx": [0, 1, 0],
        "token_id": [5, 6, 5], "feature_idx": [7, 9, 7],
        "attribution": [0.5, 0.25, 0.5],
    }
    pq.write_to_dataset(pa.Table.from_pydict(rows, schema=NODE_SCHEMA),
                        root_path=str(tmp_path / "nodes"),
                        partition_cols=["kind", "site_ordinal"])

    straight = pd.read_parquet(tmp_path / "nodes")
    assert isinstance(straight["kind"].dtype, pd.CategoricalDtype), \
        "if pyarrow stops returning Categoricals this guard is moot"
    assert len(aggregate(straight, "none")) > 2, "the trap should still be real"

    fixed = read_node_hive(tmp_path)
    out = aggregate(fixed, "none")
    assert len(out) == 2, f"expected the 2 real nodes, got {len(out)}"
    assert set(out["kind"]) == {"attn", "mlp"}
