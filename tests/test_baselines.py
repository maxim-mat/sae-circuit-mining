"""The floor (random) and the decomposition (screen + magnitude)."""
import pandas as pd
import pytest

from src.selection.baselines import select_random, select_screen_magnitude


def _nodes(n=50):
    return pd.DataFrame({
        "site_ordinal": [i % 5 for i in range(n)],
        "feature_idx": list(range(n)),
        "token_idx": [0] * n,
        "ie": [0.01 * i for i in range(n)],
    })


def test_random_selects_exactly_k():
    out = select_random(_nodes(), k=12, seed=42)
    assert int(out["selected"].sum()) == 12


def test_random_is_reproducible_from_the_seed():
    a = select_random(_nodes(), k=12, seed=42)["selected"].tolist()
    b = select_random(_nodes(), k=12, seed=42)["selected"].tolist()
    c = select_random(_nodes(), k=12, seed=43)["selected"].tolist()
    assert a == b
    assert a != c, "a different seed must draw a different set"


def test_random_k_above_the_node_count_is_clamped():
    out = select_random(_nodes(10), k=999, seed=1)
    assert int(out["selected"].sum()) == 10


def test_random_ignores_effect_size():
    """The point of the floor: it must not preferentially pick strong nodes."""
    picked = select_random(_nodes(200), k=100, seed=7)
    strong = picked[picked["ie"] > picked["ie"].median()]["selected"].mean()
    weak = picked[picked["ie"] <= picked["ie"].median()]["selected"].mean()
    assert abs(strong - weak) < 0.25, "random must not correlate with |ie|"


def test_screen_magnitude_ranks_by_summed_abs_ie():
    raw = pd.DataFrame({
        "example_idx": [0, 1, 0, 1, 0, 1],
        "site_ordinal": [1, 1, 1, 1, 1, 1],
        "token_idx": [0, 0, 0, 0, 0, 0],
        "feature_idx": [7, 7, 9, 9, 11, 11],
        "attribution": [5.0, 5.0, 0.001, 0.001, -4.0, -4.0],
    })
    nodes = pd.DataFrame({"site_ordinal": [1, 1, 1], "feature_idx": [7, 9, 11],
                          "token_idx": [0, 0, 0], "ie": [5.0, 0.001, -4.0]})
    out = select_screen_magnitude(raw, nodes, ["site_ordinal", "feature_idx",
                                               "token_idx"], k=2,
                                  screen_q=1.0, screen_n0=50.0,
                                  pool_col="site_ordinal")
    chosen = set(out.loc[out["selected"], "feature_idx"])
    assert 9 not in chosen, "the tiny feature must not outrank the large ones"
    assert len(chosen) <= 2
