"""Self-check for the edge-attribution math in src/attribution/edges.py.

Validates the seed -> autograd.grad -> contract pipeline against a closed-form
Jacobian on a synthetic linear graph, independent of the transformer. The graph
mirrors the real one: an upstream feature leaf f_u and residual leaf res_u feed
x_i = f_u @ W_dec_u + res_u; a linear block M maps to the downstream site; a
linear encoder W_enc_d reads downstream features g_d. For linear pieces the
Jacobian dd/du is exactly J = W_dec_u @ M @ W_enc_d, so the edge has a closed
form to check against.
"""
import torch

from src.models.suites import Site


def _edges_via_autograd(f_u, res_u, W_dec_u, M, W_enc_d, W_dec_d, grad_f_j, direction):
    """Replicates edge_pass's seed/grad/contract steps on the synthetic graph."""
    x_i = f_u @ W_dec_u + res_u
    y_j = x_i @ M
    g_j = y_j @ W_enc_d
    yres_j = y_j - g_j @ W_dec_d
    grad_res_j = torch.randn_like(y_j)  # dm/dresidual seed for the error node
    return x_i, y_j, g_j, yres_j, grad_res_j


def test_feature_edge_matches_closed_form():
    torch.manual_seed(0)
    seq, d_model, n_u, n_d = 3, 8, 5, 4
    W_dec_u = torch.randn(n_u, d_model)
    M = torch.randn(d_model, d_model)
    W_enc_d = torch.randn(d_model, n_d)
    W_dec_d = torch.randn(n_d, d_model)
    grad_f_j = torch.randn(seq, n_d)          # dm/dd (constant, from node pass)
    df_u = torch.randn(seq, n_u)              # (f_cf - f_clean) upstream
    dres_u = torch.randn(seq, d_model)

    # The metric's sign is folded in upstream now (Task 5); both signs of the
    # incoming gradient must still contract correctly.
    for direction in (1.0, -1.0):
        f_u = torch.randn(seq, n_u, requires_grad=True)
        res_u = torch.randn(seq, d_model, requires_grad=True)
        x_i, y_j, g_j, yres_j, grad_res_j = _edges_via_autograd(
            f_u, res_u, W_dec_u, M, W_enc_d, W_dec_d, grad_f_j, direction)

        J = W_dec_u @ M @ W_enc_d      # [n_u, n_d] = dd/du
        Jr = M @ W_enc_d               # [d_model, n_d] = dd/d(res_u)

        for d in range(n_d):
            seed = (grad_f_j[:, d] * g_j[:, d]).sum()
            gfi, gri = torch.autograd.grad(seed, [f_u, res_u], retain_graph=True)
            up_vals = direction * (gfi * df_u).sum(dim=0)        # feature edges
            err_val = (direction * (gri * dres_u).sum()).item()  # upstream-error edge

            # Closed forms.
            weighted = (grad_f_j[:, d] * df_u.T).sum(dim=1)      # sum_t dm/dd * df_u  -> [n_u]
            expected_feat = direction * J[:, d] * weighted
            assert torch.allclose(up_vals, expected_feat, atol=1e-4), (d, up_vals, expected_feat)

            expected_err = direction * (grad_f_j[:, d][:, None] * Jr[:, d][None, :] * dres_u).sum().item()
            assert abs(err_val - expected_err) < 1e-3, (d, err_val, expected_err)


def test_zero_delta_gives_zero_edge():
    torch.manual_seed(1)
    seq, d_model, n_u, n_d = 2, 6, 3, 2
    f_u = torch.randn(seq, n_u, requires_grad=True)
    res_u = torch.randn(seq, d_model, requires_grad=True)
    W_dec_u, M, W_enc_d = torch.randn(n_u, d_model), torch.randn(d_model, d_model), torch.randn(d_model, n_d)
    g_j = (f_u @ W_dec_u + res_u) @ M @ W_enc_d
    grad_f_j = torch.randn(seq, n_d)
    df_u = torch.zeros(seq, n_u)  # no upstream change -> no edge
    seed = (grad_f_j[:, 0] * g_j[:, 0]).sum()
    gfi, = torch.autograd.grad(seed, [f_u], retain_graph=True)
    assert torch.allclose((gfi * df_u).sum(dim=0), torch.zeros(n_u))


class FakeSAE:
    """Linear stand-in for an SAE: encode/decode are matmuls, so the autograd
    graph through the feature basis stays intact."""
    def __init__(self, d_model, d_sae):
        self.W_enc = torch.randn(d_model, d_sae)
        self.W_dec = torch.randn(d_sae, d_model)

        class Cfg:
            pass
        self.cfg = Cfg()
        self.cfg.d_sae = d_sae

    def encode(self, x):
        return x @ self.W_enc

    def decode(self, f):
        return f @ self.W_dec


class FakeTLModel:
    """Mimics HookedTransformer's hook protocol: a registered hook is called as
    fn(tensor, hook=hook_point) with the second argument passed by KEYWORD. Sites
    are chained through fixed linear maps so downstream really depends on upstream.
    """
    def __init__(self, hooks_in_order, seq, d_model):
        self.order = list(hooks_in_order)
        self.x0 = torch.randn(1, seq, d_model, requires_grad=True)
        self.M = {h: torch.randn(d_model, d_model) for h in self.order}
        self.hooks = []

    def add_hook(self, name, fn, dir="fwd"):
        self.hooks.append((name, fn))

    def reset_hooks(self):
        self.hooks = []

    def zero_grad(self, set_to_none=True):
        pass

    def __call__(self, tokens):
        x, outs = self.x0, []
        for h in self.order:
            x = x @ self.M[h]
            for name, fn in self.hooks:
                if name == h:
                    x = fn(x, hook=h)  # keyword, exactly as transformer_lens does
            outs.append(x)
        return torch.stack(outs).sum(0)


def _fake_sites(n=3):
    return [Site(ordinal=i, block=i, kind="resid", release="r",
                 sae_id=f"s{i}", hook=f"h{i}") for i in range(n)]


def test_passes_run_under_transformer_lens_hook_protocol():
    """Regression: hooks must take their 2nd arg as `hook` (TL passes it by keyword).
    Naming it anything else raised TypeError at the first model call in node_pass."""
    from src.attribution.edges import edge_pass, node_pass

    torch.manual_seed(0)
    seq, d_model, d_sae = 4, 6, 5
    sites = _fake_sites(3)
    model = FakeTLModel([s.hook for s in sites], seq, d_model)
    saes = {s.sae_id: FakeSAE(d_model, d_sae) for s in sites}
    tokens = torch.zeros(1, seq, dtype=torch.long)
    metric_fn = lambda logits, example, model, **_: logits.sum()

    grad_f, grad_res, df, dres = node_pass(
        model, saes, sites, tokens, tokens, metric_fn, None)

    for s in sites:
        assert grad_f[s.ordinal].shape == (seq, d_sae)
        assert grad_res[s.ordinal].shape == (seq, d_model)
        assert df[s.ordinal].shape == (seq, d_sae)
        assert dres[s.ordinal].shape == (seq, d_model)
        assert torch.isfinite(grad_f[s.ordinal]).all()

    # The fake ignores tokens, so clean == patch and df is exactly zero (no edges
    # by construction -- see test_zero_delta_gives_zero_edge). Feed nonzero deltas.
    from src.attribution.edge_template import EdgeSpec
    spec = EdgeSpec(sites[0], sites[2])
    survivors = [0, 2, 4]
    edges = edge_pass(model, saes, spec, tokens,
                      grad_f[2], grad_res[2],
                      torch.randn_like(df[0]), torch.randn_like(dres[0]),
                      survivors, survivors, include_error=False)
    assert edges, "edge_pass returned no edges through the fake chain"
    assert all(u in survivors and d in survivors for u, d, _ in edges)


def test_stop_gradient_cuts_the_indirect_path():
    """A stop-gradient on the only intermediate site must zero the edge: that is
    what makes a `marks` edge a DIRECT effect rather than a total one."""
    from src.attribution.edges import edge_pass, node_pass
    from src.attribution.edge_template import EdgeSpec

    torch.manual_seed(3)
    seq, d_model, d_sae = 4, 6, 5
    sites = _fake_sites(3)
    model = FakeTLModel([s.hook for s in sites], seq, d_model)
    saes = {s.sae_id: FakeSAE(d_model, d_sae) for s in sites}
    tokens = torch.zeros(1, seq, dtype=torch.long)
    metric_fn = lambda logits, example, model, **_: logits.sum()

    grad_f, grad_res, df, dres = node_pass(
        model, saes, sites, tokens, tokens, metric_fn, None)
    d_f, d_res = torch.randn_like(df[0]), torch.randn_like(dres[0])
    survivors = [0, 2, 4]

    direct = edge_pass(model, saes, EdgeSpec(sites[0], sites[2], stop=(sites[1],)),
                       tokens, grad_f[2], grad_res[2], d_f, d_res,
                       survivors, survivors, include_error=False)
    total = edge_pass(model, saes, EdgeSpec(sites[0], sites[2]), tokens,
                      grad_f[2], grad_res[2], d_f, d_res,
                      survivors, survivors, include_error=False)
    # site 1 is the ONLY path from 0 to 2 in the fake chain, so stopping it
    # removes every edge; without it the total-effect edges are nonzero.
    assert direct == []
    assert total


def test_detach_hook_blocks_gradient_but_preserves_value():
    """Stop-gradient must be a value-preserving identity in the forward pass --
    if it changed activations it would change the edge being measured."""
    from src.attribution.edges import _detach_site_hook

    x = torch.randn(3, 4, requires_grad=True)
    captured = {}

    def fake_hook_point(t):
        out = _detach_site_hook(t, hook=None)
        captured["out"] = out
        return out

    y = fake_hook_point(x * 2)
    assert torch.equal(captured["out"], (x * 2).detach())
    y.sum().backward()
    assert x.grad is None or torch.all(x.grad == 0)
