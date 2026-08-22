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
    cfg.selection.screen_n0 = 12.0
    cfg.selection.min_sign_consistency = 0.6
    cfg.selection.min_median_abs = 0.05
    cfg.selection.screen_pool_col = "kind"
    cfg.selection.screen_scale = "linear"
    cfg.selection.screen_method = "z"
    cfg.selection.screen_scope = "global"
    cfg.selection.recurrence = False
    return cfg


def test_screen_q_and_fdr_q_reach_different_parameters(monkeypatch):
    seen = {}

    def fake_select_features(df, q=1.0, block_col=None, n0=50.0,
                             pool_col="site_ordinal", **kw):
        kw.update(n0=n0, pool_col=pool_col)
        seen.update({"q": q, "block_col": block_col, **kw})
        feats = pd.DataFrame({"site_ordinal": [], "feature_idx": [], "selected": []})
        return df, feats, pd.Series(dtype=float)

    monkeypatch.setattr(cfs, "select_features", fake_select_features)
    cfs.select_calibrated(pd.DataFrame(), _cfg())

    assert seen["q"] == 0.7, "Stage-1 screen rate must come from selection.screen_q"
    assert seen["fdr"] == 0.02, "Stage-2 BH level must come from selection.fdr_q"
    assert seen["min_recurrence_frac"] == 0.3
    assert seen["n0"] == 12.0, "the MAD shrinkage pseudo-count must be searchable"
    assert seen["min_sign_consistency"] == 0.6
    assert seen["min_median_abs"] == 0.05,         "the effect-size floor is the knob that reintroduces magnitude"
    assert seen["pool_col"] == "kind"
    assert seen["scale"] == "linear", "the screen's scale must be searchable too"
    assert seen["method"] == "z", "the screen's estimator must be searchable too"
    assert seen["scope"] == "global", "the screen's scope must be searchable too"
    assert seen["recurrence"] is False, "Stage 2 must be switchable off"


def test_global_scope_fits_one_null_for_everything():
    """scope='global' is Marks' assumption: a single floor, so a quiet group
    no longer gets a proportionally quiet bar."""
    import numpy as np
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    rng = np.random.default_rng(4)
    rows = []
    for ex, scale_ in enumerate([0.01, 1.0]):        # two very unequal prompts
        v = np.abs(rng.normal(0.0, scale_, size=150))
        v[:3] = 10 * scale_          # a real signal, ON THAT PROMPT'S OWN SCALE
        rows.append(pd.DataFrame({"example_idx": ex, "site_ordinal": 0,
                                  "token_idx": 0,
                                  "feature_idx": np.arange(v.size),
                                  "attribution": v}))
    raw = add_abs_and_sign(pd.concat(rows, ignore_index=True))

    glob = mad_screen(raw, scope="global", exclude_subnormals_from_fit=False)
    assert glob["screen_threshold"].nunique() == 1
    assert glob["n_group"].iloc[0] == len(raw), "the global group is every row"

    per = mad_screen(raw, scope="group", exclude_subnormals_from_fit=False)
    assert per.groupby("example_idx")["screen_threshold"].first().nunique() == 2
    # The quiet prompt's own signal survives a per-group floor and is erased by
    # a global one set mostly by the loud prompt. That is the whole trade-off.
    assert per[per["example_idx"] == 0]["kept_screen"].sum() >= 3
    assert glob[glob["example_idx"] == 0]["kept_screen"].sum() == 0


def test_recurrence_off_keeps_the_whole_screen_alphabet():
    import numpy as np
    from src.calibrated_feature_selection import select_features

    rng = np.random.default_rng(5)
    n_ex, n_feat = 12, 40
    raw = pd.DataFrame({
        "example_idx": np.repeat(np.arange(n_ex), n_feat),
        "site_ordinal": 0,
        "token_idx": 0,
        "feature_idx": np.tile(np.arange(n_feat), n_ex),
        "attribution": np.exp(rng.normal(0.0, 1.0, size=n_ex * n_feat)),
    })
    _, on, _ = select_features(raw, recurrence=True)
    _, off, _ = select_features(raw, recurrence=False)
    assert off["selected"].all(), "Stage 1 alone selects the whole alphabet"
    assert len(off) == len(on), "same alphabet either way; only the verdict differs"
    assert int(off["selected"].sum()) >= int(on["selected"].sum())


def test_z_screen_is_per_token_and_uses_mean_and_sd():
    """scale='linear', method='z' is the textbook screen: one z-test per
    (prompt, layer, TOKEN) group, mean + c*sd, no robustness, no log."""
    import numpy as np
    from scipy.stats import norm
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    rng = np.random.default_rng(1)
    rows = []
    for token_idx, scale_ in enumerate([0.01, 1.0]):        # two very different tokens
        v = np.abs(rng.normal(0.0, scale_, size=120))
        rows.append(pd.DataFrame({"example_idx": 0, "site_ordinal": 0,
                                  "token_idx": token_idx,
                                  "feature_idx": np.arange(v.size),
                                  "attribution": v}))
    raw = add_abs_and_sign(pd.concat(rows, ignore_index=True))

    # n0=0: no shrinkage toward the pool, so the arithmetic is checkable exactly.
    out = mad_screen(raw, method="z", scale="linear", n0=0.0,
                     exclude_subnormals_from_fit=False)
    thr = out.groupby("token_idx")["screen_threshold"].first()
    assert thr.nunique() == 2, "a per-token screen must give each token its own floor"

    v0 = raw.loc[raw["token_idx"] == 0, "abs_attr"].to_numpy()
    c = norm.ppf(1.0 - 1.0 / v0.size)
    assert thr.loc[0] == pytest.approx(v0.mean() + c * v0.std(ddof=1))

    # The MAD screen pools the tokens: one floor for the whole (prompt, layer).
    mad = mad_screen(raw, method="mad", scale="linear", n0=0.0,
                     exclude_subnormals_from_fit=False)
    assert mad.groupby("token_idx")["screen_threshold"].first().nunique() == 1


def test_tiny_screen_q_is_refused_rather_than_selecting_nothing():
    """q below ~1e-13 makes 1 - q/n_g round to 1.0, so every group's floor is
    +inf and the screen keeps zero rows -- a silent empty circuit."""
    import numpy as np
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    rng = np.random.default_rng(3)
    raw = pd.DataFrame({"example_idx": 0, "site_ordinal": 0, "token_idx": 0,
                        "feature_idx": np.arange(200),
                        "attribution": np.exp(rng.normal(size=200))})
    with pytest.raises(ValueError, match="too small"):
        mad_screen(add_abs_and_sign(raw), q=1e-30)


def test_z_screen_needs_token_idx():
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    raw = pd.DataFrame({"example_idx": [0], "site_ordinal": [0],
                        "feature_idx": [0], "attribution": [1.0]})
    with pytest.raises(ValueError, match="token_idx"):
        mad_screen(add_abs_and_sign(raw), method="z")


def test_recurrence_null_follows_the_screens_grouping():
    """p0 = q/n_g is the SCREEN's pass rate. Per-token screening makes groups
    ~10x smaller, so a p0 still computed over (prompt, layer) would understate
    the null pass rate and call noise significant."""
    import numpy as np
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen, recurrence_test

    rng = np.random.default_rng(2)
    n_tok, n_feat = 5, 60
    raw = pd.DataFrame({
        "example_idx": np.repeat(np.arange(8), n_tok * n_feat),
        "site_ordinal": 0,
        "token_idx": np.tile(np.repeat(np.arange(n_tok), n_feat), 8),
        "feature_idx": np.tile(np.arange(n_feat), 8 * n_tok),
        "attribution": np.exp(rng.normal(0.0, 1.0, size=8 * n_tok * n_feat)),
    })
    cols = ["site_ordinal", "feature_idx", "token_idx"]
    screened = mad_screen(add_abs_and_sign(raw), method="z", scale="log")
    coarse = recurrence_test(screened, feature_cols=cols)
    fine = recurrence_test(screened, feature_cols=cols,
                           screen_group_cols=["example_idx", "site_ordinal", "token_idx"])
    assert fine.attrs["null_pass_prob"] > coarse.attrs["null_pass_prob"]


def test_linear_screen_thresholds_raw_ie_not_its_log():
    """scale='linear' must put the noise floor in the same units as Marks'
    node_threshold: med(|IE|) + c * MAD(|IE|), no exponential anywhere."""
    import numpy as np
    from scipy.stats import norm
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    rng = np.random.default_rng(0)
    vals = np.exp(rng.normal(-2.0, 1.0, size=200))       # lognormal null bulk
    raw = pd.DataFrame({"example_idx": 0, "site_ordinal": 0,
                        "feature_idx": np.arange(vals.size), "attribution": vals})

    lin = mad_screen(add_abs_and_sign(raw), scale="linear")
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    c = norm.ppf(1.0 - 1.0 / vals.size)                  # one group: no shrinkage
    assert lin["screen_threshold"].iloc[0] == pytest.approx(med + c * mad)

    log = mad_screen(add_abs_and_sign(raw), scale="log")
    assert log["screen_threshold"].iloc[0] != pytest.approx(med + c * mad)
    assert lin["kept_screen"].sum() != log["kept_screen"].sum()


def test_unknown_screen_scale_is_refused():
    from src.calibrated_feature_selection import add_abs_and_sign, mad_screen

    raw = pd.DataFrame({"example_idx": [0], "site_ordinal": [0],
                        "feature_idx": [0], "attribution": [1.0]})
    with pytest.raises(ValueError, match="scale"):
        mad_screen(add_abs_and_sign(raw), scale="linlog")


def test_the_two_rates_are_not_the_same_field():
    """Guard the schema itself: a single field for both rates is the bug."""
    cfg = Config()
    assert cfg.selection.screen_q != cfg.selection.fdr_q


def test_tested_unit_matches_what_a_node_is():
    """Regression: with aggregation='none' a node is (site, feature, position).
    Testing recurrence over (site, feature) alone and broadcasting the verdict
    to all positions inflated the circuit 338 nodes vs 125 on rc_train n=100,
    and made node counts incomparable to the threshold selector's."""
    from src.calibrated_feature_selection import calibrated_feature_cols

    cfg = _cfg()
    cfg.selection.aggregation = "none"
    assert calibrated_feature_cols(cfg) == ["site_ordinal", "feature_idx", "token_idx"]

    cfg.selection.aggregation = "sum"
    assert calibrated_feature_cols(cfg) == ["site_ordinal", "feature_idx"], \
        "positions are summed away, so there is no position to test over"
