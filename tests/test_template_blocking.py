"""The prompt -> template map, and what blocking on it does.

Recorded unconditionally at attribution time; used only when selection asks.
"""
import pandas as pd
import pytest

from src.attribution.nodes import EXAMPLE_SCHEMA
from src.selection.run import _attach_example_meta


def _write(tmp_path, rows):
    pd.DataFrame(rows).to_parquet(tmp_path / "examples.parquet", index=False)


def _raw():
    return pd.DataFrame({"example_idx": [0, 0, 1, 2], "attribution": [1.0, 2.0, 3.0, 4.0]})


def test_schema_is_one_row_per_example():
    assert set(EXAMPLE_SCHEMA.names) == {"example_idx", "template_id", "template_key"}


def test_block_column_is_joined_onto_every_row(tmp_path):
    _write(tmp_path, {"example_idx": [0, 1, 2], "template_id": [0, 1, 0],
                      "template_key": ["a", "b", "a"]})
    out = _attach_example_meta(_raw(), tmp_path, "template_id")
    assert out["template_id"].tolist() == [0, 0, 1, 0]
    assert len(out) == len(_raw()), "the join must not duplicate or drop node rows"


def test_single_template_refuses_rather_than_silently_losing_power(tmp_path):
    """Marks' SVA is one syntactic frame. Blocking on it leaves one trial, so
    the recurrence test can conclude nothing -- fail loudly instead."""
    _write(tmp_path, {"example_idx": [0, 1, 2], "template_id": [0, 0, 0],
                      "template_key": ["", "", ""]})
    with pytest.raises(ValueError, match="1 trial|no power|stratifier"):
        _attach_example_meta(_raw(), tmp_path, "template_id")


def test_missing_sidecar_names_the_remedy(tmp_path):
    with pytest.raises(FileNotFoundError, match="run.force"):
        _attach_example_meta(_raw(), tmp_path, "template_id")


def test_unknown_block_column_lists_what_is_available(tmp_path):
    _write(tmp_path, {"example_idx": [0, 1], "template_id": [0, 1],
                      "template_key": ["a", "b"]})
    with pytest.raises(ValueError, match="template_field"):
        _attach_example_meta(_raw(), tmp_path, "no_such_col")
