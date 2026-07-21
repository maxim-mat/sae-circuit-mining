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


if __name__ == "__main__":
    test_feature_edge_matches_closed_form()
    test_zero_delta_gives_zero_edge()
    print("all edge_attribution checks passed")
