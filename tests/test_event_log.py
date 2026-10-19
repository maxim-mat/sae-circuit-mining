import pandas as pd
import pytest

from src.discovery.event_log import build_event_log


def _events():
    # Two cases, deliberately NOT sorted by site/feature: the caller's row
    # order is the trace order.
    return pd.DataFrame({
        "example_idx":  [0, 0, 0, 1, 1],
        "site_ordinal": [6, 3, 9, 3, 6],
        "feature_idx":  [11, 7, 2, 7, 11],
        "attribution":  [0.5, -0.2, 0.1, 0.3, 0.4],
    })


def test_trace_order_is_the_input_row_order_within_each_case():
    log = build_event_log(_events())
    traces = (log.sort_values(["case:concept:name", "time:timestamp"])
                 .groupby("case:concept:name")["concept:name"].apply(list))
    assert traces["0"] == ["6:11", "3:7", "9:2"]
    assert traces["1"] == ["3:7", "6:11"]


def test_activity_is_built_from_the_given_columns():
    log = build_event_log(_events(), activity_cols=["feature_idx"])
    assert list(log["concept:name"]) == ["11", "7", "2", "7", "11"]


def test_timestamps_are_datetimes_strictly_increasing_within_a_case():
    log = build_event_log(_events())
    assert pd.api.types.is_datetime64_any_dtype(log["time:timestamp"])
    assert log.groupby("case:concept:name")["time:timestamp"].apply(
        lambda t: t.is_monotonic_increasing and t.is_unique).all()


def test_other_columns_pass_through_as_event_attributes():
    log = build_event_log(_events())
    assert list(log["attribution"]) == [0.5, -0.2, 0.1, 0.3, 0.4]


def test_pm4py_ilp_miner_accepts_the_log():
    pm4py = pytest.importorskip("pm4py")
    net, _, _ = pm4py.discover_petri_net_ilp(build_event_log(_events()))
    labels = {t.label for t in net.transitions if t.label}
    assert labels == {"6:11", "3:7", "9:2"}
