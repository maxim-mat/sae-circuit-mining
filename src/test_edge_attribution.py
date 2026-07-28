"""Self-check for the edge-attribution math in edge_attribution.py.

Validates the seed -> autograd.grad -> contract pipeline against a closed-form
Jacobian on a synthetic linear graph, independent of the transformer. The graph
mirrors the real one: an upstream feature leaf f_u and residual leaf res_u feed
x_i = f_u @ W_dec_u + res_u; a linear block M maps to the downstream site; a
linear encoder W_enc_d reads downstream features g_d. For linear pieces the
Jacobian dd/du is exactly J = W_dec_u @ M @ W_enc_d, so the edge has a closed
form to check against.

Run: python src/test_edge_attribution.py
"""
import torch


def _edges_via_autograd(f_u, res_u, W_dec_u, M, W_enc_d, W_dec_d, grad_f_j, direction):
    """Replicates edge_attribution.edge_pass's seed/grad/contract steps on the
    synthetic graph. Returns (feature_edges[u,d], error_edges[d]) given fixed
    upstream deltas df_u, dres_u supplied by the caller via closures is avoided —
    deltas are passed in the asserts below."""
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
            up_vals = direction * (gfi * df_u).sum(dim=0)        # script's feature edges
            err_val = (direction * (gri * dres_u).sum()).item()  # script's upstream-error edge

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


def _stub_gpu_deps():
    """edge_attribution imports sae_lens/transformer_lens at module scope but only
    uses them in load_model_and_saes, which the fake-model check never calls. Stub
    them when absent so this check runs off the GPU box too."""
    import sys
    import types
    try:
        import sae_lens  # noqa: F401
        import transformer_lens  # noqa: F401
        return
    except ImportError:
        pass
    for name in ("sae_lens", "sae_lens.loading",
                 "sae_lens.loading.pretrained_saes_directory", "transformer_lens"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sae_lens"].SAE = object
    sys.modules["sae_lens.loading.pretrained_saes_directory"].get_pretrained_saes_directory = dict
    sys.modules["transformer_lens"].HookedTransformer = object


class FakeSAE:
    """Linear stand-in for an SAE: encode/decode are matmuls, so the autograd
    graph through the feature basis stays intact."""
    def __init__(self, d_model, d_sae):
        self.W_enc = torch.randn(d_model, d_sae)
        self.W_dec = torch.randn(d_sae, d_model)

    def encode(self, x):
        return x @ self.W_enc

    def decode(self, f):
        return f @ self.W_dec


class FakeTLModel:
    """Mimics HookedTransformer's hook protocol: a registered hook is called as
    fn(tensor, hook=hook_point) with the second argument passed by KEYWORD. Layers
    are chained through fixed linear maps so downstream really depends on upstream.
    """
    def __init__(self, layers, seq, d_model):
        self.order = sorted(layers)
        self.x0 = torch.randn(1, seq, d_model, requires_grad=True)
        self.M = {L: torch.randn(d_model, d_model) for L in self.order}
        self.hooks = []

    def add_hook(self, name, fn, dir="fwd"):
        self.hooks.append((name, fn))

    def reset_hooks(self):
        self.hooks = []

    def zero_grad(self, set_to_none=True):
        pass

    def __call__(self, tokens):
        x, outs = self.x0, []
        for L in self.order:
            x = x @ self.M[L]
            for name, fn in self.hooks:
                if name == L:
                    x = fn(x, hook=L)  # keyword, exactly as transformer_lens does
            outs.append(x)
        return torch.stack(outs).sum(0)


def _fake_setup(layers=(0, 1, 2), seq=4, d_model=6, d_sae=5):
    model = FakeTLModel(layers, seq, d_model)
    sae = {L: FakeSAE(d_model, d_sae) for L in layers}
    hook_points = {L: L for L in layers}               # hook_point name == layer here
    w_dec_t = {L: sae[L].W_dec.T for L in layers}
    tokens = torch.zeros(1, seq, dtype=torch.long)
    return model, sae, hook_points, w_dec_t, tokens


def test_passes_run_under_transformer_lens_hook_protocol():
    """Regression: hooks must take their 2nd arg as `hook` (TL passes it by keyword).
    Naming it anything else raised TypeError at the first model call in node_pass."""
    _stub_gpu_deps()
    from edge_attribution import edge_pass, node_pass

    torch.manual_seed(0)
    layers = (0, 1, 2)
    model, sae, hook_points, w_dec_t, tokens = _fake_setup(layers)
    metric_fn = lambda logits, tokens, item, model: (logits.sum(), 1)

    grad_f, grad_res, df, dres, direction = node_pass(
        model, sae, hook_points, w_dec_t, list(layers), tokens, tokens, metric_fn, None)

    for L in layers:
        assert grad_f[L].shape == (4, 5), (L, grad_f[L].shape)   # [seq, d_sae]
        assert grad_res[L].shape == (4, 6), (L, grad_res[L].shape)
        assert df[L].shape == (4, 5) and dres[L].shape == (4, 6)
        assert torch.isfinite(grad_f[L]).all()

    # The fake ignores tokens, so clean == cf and df[0] is exactly zero (no edges by
    # construction — see test_zero_delta_gives_zero_edge). Feed nonzero deltas instead.
    survivors = [0, 2, 4]
    edges = edge_pass(model, sae, hook_points, tokens, 0, 2,
                      grad_f[2], grad_res[2], torch.randn_like(df[0]), torch.randn_like(dres[0]),
                      survivors, survivors, direction, include_error=False)
    assert edges, "edge_pass returned no edges through the fake chain"
    assert all(u in survivors and d in survivors for u, d, _ in edges)


if __name__ == "__main__":
    test_feature_edge_matches_closed_form()
    test_zero_delta_gives_zero_edge()
    test_passes_run_under_transformer_lens_hook_protocol()
    print("all edge_attribution checks passed")
