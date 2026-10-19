"""Selected nodes -> a sequential event log for process discovery (ILP, eST, ...).

Case = prompt, event = one row of the input. Everything that decides WHICH
rows are events (selection, per-row cutoffs, dropping error nodes at
feature_idx == -1) and in WHAT ORDER (site order, causal linear extensions,
tie-breaking between concurrent attn/mlp sites) is the caller's job and must be
done before calling build_event_log: the input's row order within a case IS the
trace order.
"""
from __future__ import annotations

import pandas as pd

CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"

_BASE_TIME = pd.Timestamp("2020-01-01")


def build_event_log(events: pd.DataFrame, case_col: str = "example_idx",
                    activity_cols: list[str] = ("site_ordinal", "feature_idx"),
                    ) -> pd.DataFrame:
    """Tag rows with pm4py's standard XES columns, keeping every input column.

    The activity label is `activity_cols` joined with ":" (e.g. "3:1234" for
    site 3, feature 1234). Leave `token_idx` out of it and a feature active at
    several positions becomes a repeated activity in the trace; put it in and
    each position is its own activity. Timestamps are synthetic, one second
    apart in row order, so they only fix the order within a case. The result
    goes straight into pm4py (discover_petri_net_ilp, write_xes for ProM's
    eST-Miner).
    """
    activity_cols = list(activity_cols)
    labels = events[activity_cols].astype(str)
    return events.assign(**{
        CASE: events[case_col].astype(str),
        ACTIVITY: labels.iloc[:, 0].str.cat(labels.iloc[:, 1:], sep=":"),
        TIMESTAMP: _BASE_TIME + pd.to_timedelta(
            events.groupby(case_col, sort=False).cumcount(), unit="s"),
    })
