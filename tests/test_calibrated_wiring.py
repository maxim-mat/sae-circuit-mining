"""The calibrated selector's two rates are distinct knobs; conflating them
silently changes what gets selected. This path had never executed before the
first HA run, which is how the mix-up survived."""
import pandas as pd
import pytest

import src.calibrated_feature_selection as cfs
from src.utils.config import Config


def _cfg():
    cfg = Config()
    cfg.suite.name = "pythia-marks"
    cfg.data.path = "x.json"
    cfg.selection.method = "calibrated"
    cfg.selection.screen_q = 0.7      # Stage 1: false positives per group
    cfg.selection.fdr_q = 0.02        # Stage 2: BH level
    cfg.selection.min_recurrence = 0.3
    return cfg


def test_screen_q_and_fdr_q_reach_different_parameters(monkeypatch):
    seen = {}

    def fake_select_features(df, q=1.0, block_col=None, **kw):
        seen.update({"q": q, "block_col": block_col, **kw})
        feats = pd.DataFrame({"site_ordinal": [], "feature_idx": [], "selected": []})
        return df, feats, pd.Series(dtype=float)

    monkeypatch.setattr(cfs, "select_features", fake_select_features)
    cfs.select_calibrated(pd.DataFrame(), _cfg())

    assert seen["q"] == 0.7, "Stage-1 screen rate must come from selection.screen_q"
    assert seen["fdr"] == 0.02, "Stage-2 BH level must come from selection.fdr_q"
    assert seen["min_recurrence_frac"] == 0.3


def test_the_two_rates_are_not_the_same_field():
    """Guard the schema itself: a single field for both rates is the bug."""
    cfg = Config()
    assert cfg.selection.screen_q != cfg.selection.fdr_q
