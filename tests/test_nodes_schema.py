import pyarrow.dataset as ds
from src.attribution.nodes import NODE_SCHEMA, TOTAL_EFFECT_SCHEMA


def test_schema_carries_typed_site_identity_not_a_layer_int():
    names = set(NODE_SCHEMA.names)
    assert {"site_ordinal", "block", "kind"} <= names
    assert "layer" not in names, "typed sites replace the integer layer column"


def test_schema_has_no_aggregation_baked_in():
    """Marks aggregate AFTER edges; node attribution stores raw
    per-(example, position, feature) rows."""
    assert {"example_idx", "token_idx", "feature_idx"} <= set(NODE_SCHEMA.names)


def test_total_effect_schema_is_one_row_per_example():
    assert set(TOTAL_EFFECT_SCHEMA.names) == {
        "example_idx", "metric_clean", "metric_patch", "total_effect"}


def test_counterfactual_is_config_not_a_column():
    """One run, one counterfactual: the choice belongs in the manifest and the
    cache key, not in a partition column that invites mixing."""
    assert "counterfactual" not in set(NODE_SCHEMA.names)
    assert "counterfactual" not in set(TOTAL_EFFECT_SCHEMA.names)
