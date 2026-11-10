import numpy as np
import pandas as pd
import pytest

from src.selection.destinations import destination_masks

T = 0.1
N = 4


def _raw():
    rows = []
    for ex in range(N):
        rows.append((ex, 0, 0, 1, 0.3))                       # A: strong at token 0 ...
        rows.append((ex, 0, 1, 1, 0.01))                      # ... weak at token 1
        rows.append((ex, 1, 2, 2, 1.0 if ex < 2 else -1.0))  # B: cancels in the mean
        rows.append((ex, 1, 2, 3, 0.05))                      # C: small but constant
    rows.append((3, 0, 2, 1, 0.0))                            # A active but unmoved
    rows.append((0, 0, 1, -1, 0.5))                           # error node, one example
    return pd.DataFrame(rows, columns=["example_idx", "site_ordinal", "token_idx",
                                       "feature_idx", "attribution"])


def _rows(raw, mask):
    return set(map(tuple, raw.loc[mask, ["example_idx", "site_ordinal", "token_idx",
                                         "feature_idx"]].to_numpy().tolist()))


def test_case_keeps_each_instance_on_its_own_ie():
    raw = _raw()
    got = _rows(raw, destination_masks(raw, {"c": {"rule": "case", "threshold": T}}, N)["c"])
    assert got == ({(ex, 0, 0, 1) for ex in range(N)} | {(ex, 1, 2, 2) for ex in range(N)}
                   | {(0, 0, 1, -1)})


def test_marks_summed_alphabet_keeps_every_moved_instance_of_a_kept_activity():
    raw = _raw()
    spec = {"rule": "marks", "threshold": T, "aggregation": "sum"}
    got = _rows(raw, destination_masks(raw, {"m": spec}, N)["m"])
    # A: mean of per-example sums 0.31 passes; both of its moved positions are kept, the
    # unmoved row is not. B cancels to 0, C is 0.05. The error node's mean is 0.5 / 4.
    assert got == ({(ex, 0, t, 1) for ex in range(N) for t in (0, 1)} | {(0, 0, 1, -1)})


def test_marks_by_position_is_its_own_alphabet():
    raw = _raw()
    spec = {"rule": "marks", "threshold": T, "aggregation": "none"}
    got = _rows(raw, destination_masks(raw, {"m": spec}, N)["m"])
    assert got == {(ex, 0, 0, 1) for ex in range(N)} | {(0, 0, 1, -1)}


def test_marks_plus_case_is_the_intersection():
    raw = _raw()
    methods = {"m": {"rule": "marks", "threshold": T, "aggregation": "sum"},
               "c": {"rule": "case", "threshold": T},
               "mc": {"rule": "marks+case", "threshold": T, "aggregation": "sum"}}
    masks = destination_masks(raw, methods, N)
    assert (masks["mc"] == (masks["m"] & masks["c"])).all()
    assert masks["mc"].sum() == N + 1


def test_mean_divides_by_all_examples_not_the_ones_present():
    raw = _raw()
    spec = {"rule": "marks", "threshold": 0.2, "aggregation": "sum"}
    # The error node is present in 1 of 4 examples: 0.5 / 4 = 0.125 < 0.2.
    assert not destination_masks(raw, {"m": spec}, N)["m"][raw.feature_idx == -1].any()


def test_recurrence_keeps_large_effects_that_cancel():
    raw = _raw()
    spec = {"rule": "recurrence", "threshold": T, "aggregation": "sum", "min_frac": 0.5}
    got = _rows(raw, destination_masks(raw, {"r": spec}, N)["r"])
    assert {(ex, 1, 2, 2) for ex in range(N)} <= got
    assert not any(f == 3 for *_, f in got)


def test_mass_keeps_the_smallest_set_holding_the_share():
    raw = _raw()
    spec = {"rule": "mass", "aggregation": "sum", "mass": 0.6}
    got = _rows(raw, destination_masks(raw, {"s": spec}, N)["s"])
    # |means|: A 0.31, error 0.125, C 0.05, B 0 -> A alone holds 0.31 / 0.485 = 64%.
    assert got == {(ex, 0, t, 1) for ex in range(N) for t in (0, 1)}


def test_ttest_keeps_consistent_small_effects_and_drops_cancelling_ones():
    raw = _raw()
    spec = {"rule": "ttest_bh", "aggregation": "sum", "fdr": 0.05}
    got = _rows(raw, destination_masks(raw, {"t": spec}, N)["t"])
    assert {(ex, 1, 2, 3) for ex in range(N)} <= got
    assert not any(f == 2 for *_, f in got)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="nope"):
        destination_masks(_raw(), {"x": {"rule": "nope"}}, N)
