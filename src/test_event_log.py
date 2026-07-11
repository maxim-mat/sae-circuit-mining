"""Self-check for event_log.py on synthetic data. Run: python src/test_event_log.py"""
import pandas as pd

from event_log import build_event_log, to_pm4py


def make_df():
    # Two cases, two layers. p0 has a duplicate activity (L1/F12 at two tokens);
    # L1/F13 and L1/F99 each occur in only one case.
    return pd.DataFrame({
        "counterfactual": ["cf1"] * 9,
        "prompt_idx":  [0, 0, 0, 0, 0, 1, 1, 1, 1],
        "token_idx":   [0, 1, 1, 2, 2, 0, 1, 2, 2],
        "token_id":    [7] * 9,
        "feature_idx": [10, 11, 12, 12, 13, 10, 11, 99, 12],
        "attribution": [-0.5, 0.2, -0.9, 0.3, 0.05, -0.4, 0.3, 0.01, -0.6],
        "layer":       [0, 0, 1, 1, 1, 0, 0, 1, 1],
    })


def test_topk_per_layer():
    log = build_event_log(make_df(), filter_params={"k": 1})
    assert len(log) == 4, log
    assert set(log["activity"]) == {"L0/F10", "L1/F12"}


def test_threshold_and_dedupe():
    log = build_event_log(make_df(), filter_name="abs_threshold",
                          filter_params={"tau": 0.25})
    # p0 L1/F12 occurs at two tokens (-0.9, 0.3); dedupe keeps the -0.9 one.
    row = log[(log["case_id"] == "p0") & (log["activity"] == "L1/F12")]
    assert len(row) == 1 and abs(row["attribution"].iloc[0] + 0.9) < 1e-6

    log_all = build_event_log(make_df(), filter_name="abs_threshold",
                              filter_params={"tau": 0.25}, dedupe=False)
    assert len(log_all) == len(log) + 1  # the duplicate survives


def test_min_case_frac():
    log = build_event_log(make_df(), min_case_frac=1.0)
    # Only activities present in both cases survive.
    assert set(log["activity"]) == {"L0/F10", "L0/F11", "L1/F12"}


def test_ordering():
    log = build_event_log(make_df(), ordering_name="layer_interval")
    assert (log["time_start"] == log["layer"]).all()
    assert (log["time"] == log["layer"] + 0.9).all()
    # Sorted by time within each case.
    for _, case in log.groupby("case_id"):
        assert case["time"].is_monotonic_increasing


def test_feature_role():
    log = build_event_log(make_df(), activity_name="feature_role")
    ends = log[log["activity"].str.endswith("@END")]
    assert (ends["token_idx"] == 2).all() and len(ends) > 0
    assert log["activity"].str.endswith("@MID").any()

    log = build_event_log(make_df(), activity_name="feature_role",
                          activity_params={"roles": {"2": "END", "1": "S2"}})
    assert log["activity"].str.endswith("@S2").any()
    assert log["activity"].str.endswith("@OTHER").any()  # token 0 unmapped


def test_to_pm4py():
    out = to_pm4py(build_event_log(make_df(), ordering_name="layer_interval"))
    assert {"case:concept:name", "concept:name", "time:timestamp", "start_timestamp"} <= set(out.columns)
    assert pd.api.types.is_datetime64_any_dtype(out["time:timestamp"])


if __name__ == "__main__":
    test_topk_per_layer()
    test_threshold_and_dedupe()
    test_min_case_frac()
    test_ordering()
    test_feature_role()
    test_to_pm4py()
    print("all event_log checks passed")
