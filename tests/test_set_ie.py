import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from src.attribution.nodes import (all_set_ies, exact_set_ies, moving_nodes, nde,
                                   pair_reachable, reachable_pairs, set_ie)
from src.models.suites import Site, Suite

D = torch.float64


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(n_layers=2, d_model=8, n_heads=2, d_head=4, d_mlp=16,
                                  n_ctx=6, d_vocab=7, act_fn="gelu",
                                  parallel_attn_mlp=True, dtype=D)
    model = HookedTransformer(cfg)
    sites = []
    for b in range(2):
        for i, kind in enumerate(("attn", "mlp", "resid")):
            hook = f"blocks.{b}.hook_{kind}_{'post' if kind == 'resid' else 'out'}"
            sites.append(Site(3 * b + i + 1, b, kind, "", hook, hook))
    suite = Suite("tiny", "", True, tuple(sites))
    saes = {s.sae_id: _SAE(cfg.device) for s in sites}
    clean = torch.tensor([[1, 2, 3, 4, 5, 6]])
    patch = torch.tensor([[1, 2, 0, 4, 5, 6]])
    return model, saes, suite, clean, patch


class _SAE:
    def __init__(self, device):
        self.W_enc = torch.randn(8, 12, dtype=D, device=device)
        self.W_dec = torch.randn(12, 8, dtype=D, device=device)
        self.b_dec = torch.randn(8, dtype=D, device=device)

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def _metric(logits, **_):
    return logits[0, -1, 0] - logits[0, -1, 1]


def _moved_feature(model, sae, hook, clean, patch, tok):
    """The feature whose activation the patch moves most at `tok`."""
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
    return int((sae.encode(pc[hook][0]) - sae.encode(cc[hook][0]))[tok].abs().argmax())


def _exact_patch(model, saes, suite, nodes, clean, patch):
    """m(clean | do(a = a_patch for a in nodes)) - m(clean), by real patching."""
    by_ord = {s.ordinal: s for s in suite.sites}
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
        add = {}
        for o, t, f in nodes:
            s = by_ord[o]
            sae = saes[s.sae_id]
            step = sae.encode(pc[s.hook][0])[t, f] - sae.encode(cc[s.hook][0])[t, f]
            add.setdefault(s.hook, torch.zeros_like(cc[s.hook][0]))[t] += step * sae.W_dec[f]
        patched = model.run_with_hooks(
            clean, fwd_hooks=[(h, lambda x, hook: x + add[hook.name]) for h in add])
        return float(_metric(patched) - _metric(model(clean)))


def _nodes(setup):
    model, saes, suite, clean, patch = setup
    # attn_0 and mlp_0 at the same position: parallel in this model, so neither
    # reaches the other.
    return [(o, 2, _moved_feature(model, saes[h], h, clean, patch, 2))
            for o, h in ((1, "blocks.0.hook_attn_out"), (2, "blocks.0.hook_mlp_out"))]


def test_ig_set_ie_converges_to_the_exact_joint_patch(setup):
    model, saes, suite, clean, patch = setup
    nodes = _nodes(setup)
    exact = _exact_patch(model, saes, suite, nodes, clean, patch)
    approx = set_ie(model, saes, suite, nodes, clean, patch, _metric, steps=500)
    assert approx == pytest.approx(exact, rel=1e-2)


def test_non_ig_method_warns_and_is_the_sum_of_singleton_ies(setup):
    model, saes, suite, clean, patch = setup
    nodes = _nodes(setup)
    with pytest.warns(UserWarning, match="sum of"):
        joint = set_ie(model, saes, suite, nodes, clean, patch, _metric, method="attrib")
    with pytest.warns(UserWarning):
        parts = [set_ie(model, saes, suite, [n], clean, patch, _metric, method="attrib")
                 for n in nodes]
    assert joint == pytest.approx(sum(parts))


def test_rejects_nodes_without_a_position(setup):
    model, saes, suite, clean, patch = setup
    with pytest.raises(ValueError, match="token_idx"):
        set_ie(model, saes, suite, [(1, -1, 0)], clean, patch, _metric)


def test_reachability_follows_the_model_wiring(setup):
    model, _, suite, clean, _ = setup
    reach = reachable_pairs(model, suite, clean)
    attn0, mlp0, resid0, attn1 = 1, 2, 3, 4
    assert ((attn0, 2), (mlp0, 2)) not in reach      # parallel attn || mlp
    assert ((attn0, 2), (resid0, 2)) in reach
    assert ((mlp0, 2), (resid0, 4)) not in reach     # no attention in between
    assert ((mlp0, 2), (attn1, 4)) in reach          # attention moves it forward
    assert ((resid0, 4), (attn1, 2)) not in reach    # causal mask
    assert ((mlp0, 2), (mlp0, 3)) not in reach       # same site never


def test_all_set_ies_keeps_only_unreachable_sets(setup):
    model, saes, suite, clean, patch = setup
    a, m = _nodes(setup)
    r = (3, 2, 0)                                     # resid_0 is downstream of both
    out = all_set_ies(model, saes, suite, [a, m, r], 2, clean, patch, _metric, steps=4)
    assert list(out) == [(a, m)]
    assert out[(a, m)] == pytest.approx(
        set_ie(model, saes, suite, [a, m], clean, patch, _metric, steps=4))


def _exact_nde(model, saes, suite, nodes, block, clean, patch):
    """m(clean | do(A = patch, B = clean)) - m(clean), by real patching: B's
    nodes are re-read from the recomputed activation and reset to clean."""
    by_ord = {s.ordinal: s for s in suite.sites}
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
        add = {}
        for o, t, f in nodes:
            s = by_ord[o]
            sae = saes[s.sae_id]
            step = sae.encode(pc[s.hook][0])[t, f] - sae.encode(cc[s.hook][0])[t, f]
            add.setdefault(s.hook, torch.zeros_like(cc[s.hook][0]))[t] += step * sae.W_dec[f]

        def hook(x, hook):
            x = x.clone()
            for o, t, f in block:
                s = by_ord[o]
                if s.hook != hook.name:
                    continue
                sae, xc = saes[s.sae_id], cc[s.hook][0]
                if f == -1:
                    err = lambda v: v - sae.decode(sae.encode(v))
                    x[0, t] += err(xc)[t] - err(x[0])[t]
                else:
                    x[0, t] += (sae.encode(xc)[t, f] - sae.encode(x[0])[t, f]) * sae.W_dec[f]
            return x + add[hook.name] if hook.name in add else x

        hooks = {by_ord[o].hook for o, _, _ in nodes} | {by_ord[o].hook for o, _, _ in block}
        out = model.run_with_hooks(clean, fwd_hooks=[(h, hook) for h in hooks])
        return float(_metric(out) - _metric(model(clean)))


@pytest.mark.parametrize("blocked_feature", ["feature", "error"])
def test_nde_converges_to_the_exact_blocked_patch(setup, blocked_feature):
    model, saes, suite, clean, patch = setup
    a = _nodes(setup)[:1]                                   # attn_0 at token 2
    f = (-1 if blocked_feature == "error"
         else _moved_feature(model, saes["blocks.0.hook_resid_post"],
                             "blocks.0.hook_resid_post", clean, patch, 2))
    block = [(3, 2, f)]                                     # resid_0, downstream of a
    exact = _exact_nde(model, saes, suite, a, block, clean, patch)
    assert abs(exact - _exact_patch(model, saes, suite, a, clean, patch)) > 1e-3
    # Blocking re-encodes a site, putting ReLU kinks on the path, so the
    # left-Riemann sum converges slower (~1/steps) than for set_ie alone.
    approx = nde(model, saes, suite, a, block, clean, patch, _metric, steps=8000)
    assert approx == pytest.approx(exact, rel=1e-2)


def test_blocking_an_unreachable_node_changes_nothing(setup):
    model, saes, suite, clean, patch = setup
    a, m = _nodes(setup)                                    # mlp_0 is parallel to attn_0
    assert nde(model, saes, suite, [a], [m], clean, patch, _metric) == pytest.approx(
        set_ie(model, saes, suite, [a], clean, patch, _metric))


def test_nde_rejects_a_node_both_patched_and_blocked(setup):
    model, saes, suite, clean, patch = setup
    a = _nodes(setup)[0]
    with pytest.raises(ValueError, match="both"):
        nde(model, saes, suite, [a], [a], clean, patch, _metric)


# ---------------------------------------------------------------------------
# Exact IE by real patching: do(node = its patch value), re-read from the
# recomputed activation, batched over sets.
# ---------------------------------------------------------------------------

def _ie(model, clean, out):
    return float(_metric(out) - _metric(model(clean)))


def _nodes_of(setup):
    model, saes, suite, clean, patch = setup
    df = moving_nodes(model, saes, suite, clean, patch)
    return df, list(zip(df.site_ordinal, df.token_idx, df.feature_idx))


def test_moving_nodes_are_exactly_those_whose_patch_value_differs(setup):
    model, saes, suite, clean, patch = setup
    df, _ = _nodes_of(setup)
    assert (df.token_idx >= 2).all()        # clean and patch agree before token 2
    feats = df[df.feature_idx >= 0]
    assert (feats.f_clean != feats.f_patch).all()
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
    for s in suite.sites:
        sae = saes[s.sae_id]
        moved = int((sae.encode(cc[s.hook][0]) != sae.encode(pc[s.hook][0])).sum())
        assert (feats.site_ordinal == s.ordinal).sum() == moved
    assert ((df.feature_idx == -1) & (df.site_ordinal == 3) & (df.token_idx == 2)).any()
    assert not ((df.feature_idx == -1) & (df.token_idx < 2)).any()


def test_exact_ie_matches_real_patching_for_singles_and_antichain_pairs(setup):
    model, saes, suite, clean, patch = setup
    df, _ = _nodes_of(setup)
    a, m = _nodes(setup)                                   # parallel attn_0 || mlp_0
    s1, s2 = [(1, 2, int(f)) for f in                       # siblings: same site, token
              df[(df.site_ordinal == 1) & (df.token_idx == 2) & (df.feature_idx >= 0)]
              .feature_idx[:2]]
    nodes = [a, m, s1, s2]
    singles = exact_set_ies(model, saes, suite, nodes, torch.tensor([[0], [1]]),
                            clean, patch, _metric)
    pairs = exact_set_ies(model, saes, suite, nodes, torch.tensor([[0, 1], [2, 3]]),
                          clean, patch, _metric)
    assert singles.tolist() == pytest.approx(
        [_exact_patch(model, saes, suite, [n], clean, patch) for n in (a, m)])
    assert pairs.tolist() == pytest.approx(
        [_exact_patch(model, saes, suite, p, clean, patch) for p in ([a, m], [s1, s2])])


def test_error_node_patches_the_reconstruction_error(setup):
    model, saes, suite, clean, patch = setup
    h, t = "blocks.0.hook_resid_post", 2
    sae = saes[h]
    err = lambda v: v - sae.decode(sae.encode(v))
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
        add = torch.zeros_like(cc[h])
        add[0, t] = err(pc[h][0])[t] - err(cc[h][0])[t]
        exact = _ie(model, clean, model.run_with_hooks(
            clean, fwd_hooks=[(h, lambda x, hook: x + add)]))
    ie = exact_set_ies(model, saes, suite, [(3, t, -1)], torch.tensor([[0]]),
                       clean, patch, _metric)
    assert ie.item() == pytest.approx(exact)


def test_reachable_pair_holds_the_downstream_node_at_its_patch_value(setup):
    model, saes, suite, clean, patch = setup
    a = _nodes(setup)[0]                                   # attn_0, token 2
    ha, hr = "blocks.0.hook_attn_out", "blocks.0.hook_resid_post"
    v = (3, 2, _moved_feature(model, saes[hr], hr, clean, patch, 2))  # downstream of a
    sa, sr = saes[ha], saes[hr]
    with torch.no_grad():
        _, cc = model.run_with_cache(clean)
        _, pc = model.run_with_cache(patch)
        add = torch.zeros_like(cc[ha])
        add[0, 2] = (sa.encode(pc[ha][0])[2, a[2]] - sa.encode(cc[ha][0])[2, a[2]]) * sa.W_dec[a[2]]
        fp_v = sr.encode(pc[hr][0])[2, v[2]]

        def clamp(x, hook):
            x = x.clone()
            x[0, 2] += (fp_v - sr.encode(x[0])[2, v[2]]) * sr.W_dec[v[2]]
            return x

        exact = _ie(model, clean, model.run_with_hooks(
            clean, fwd_hooks=[(ha, lambda x, hook: x + add), (hr, clamp)]))
    ie = exact_set_ies(model, saes, suite, [a, v], torch.tensor([[0, 1]]),
                       clean, patch, _metric)
    assert ie.item() == pytest.approx(exact)
    # Adding v's clean->patch move as a fixed delta is a different intervention.
    assert abs(exact - _exact_patch(model, saes, suite, [a, v], clean, patch)) > 1e-6


def test_batching_does_not_change_the_result(setup):
    model, saes, suite, clean, patch = setup
    _, nodes = _nodes_of(setup)
    g = torch.Generator().manual_seed(0)
    pairs = torch.combinations(torch.arange(len(nodes)), 2)
    pairs = pairs[torch.randperm(len(pairs), generator=g)[:40]]
    one = exact_set_ies(model, saes, suite, nodes, pairs, clean, patch, _metric,
                        batch_size=1)
    many = exact_set_ies(model, saes, suite, nodes, pairs, clean, patch, _metric,
                         batch_size=len(pairs))
    torch.testing.assert_close(one, many)


def test_pair_reachable_agrees_with_reachable_pairs(setup):
    model, _, suite, clean, _ = setup
    _, nodes = _nodes_of(setup)
    reach = reachable_pairs(model, suite, clean)
    pairs = torch.combinations(torch.arange(len(nodes)), 2)
    flags = pair_reachable(nodes, pairs, reach, n_tokens=clean.shape[1])
    expected = [((nodes[i][:2], nodes[j][:2]) in reach or (nodes[j][:2], nodes[i][:2]) in reach)
                for i, j in pairs.tolist()]
    assert flags.tolist() == expected
    assert any(expected) and not all(expected)
