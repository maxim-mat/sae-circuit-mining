"""Checks the SAE error-node decomposition used by node attribution.

Property (Marks et al. 2024, arXiv:2403.19647): at each site, the sum of
feature attributions plus the reconstruction-error attribution equals the
residual-stream indirect effect grad_x . (x_cf - x_clean).

The `direction` multiplier is gone (Task 5 folds the sign into the metric), so
the decomposition is now over the raw signed effects.
"""
import torch


class LinearSAE:
    """Minimal stand-in mirroring the sae_lens encode/decode shapes we use."""
    def __init__(self, d_model, d_sae):
        self.W_enc = torch.randn(d_model, d_sae)
        self.W_dec = torch.randn(d_sae, d_model)  # [d_sae, d_model], as in res_jb SAEs
        self.b_dec = torch.randn(d_model)
        class Cfg: pass
        self.cfg = Cfg(); self.cfg.d_sae = d_sae

    def encode(self, x):  # x: [1, seq, d_model]
        return torch.relu((x - self.b_dec) @ self.W_enc)

    def decode(self, f):  # f: [1, seq, d_sae]
        return f @ self.W_dec + self.b_dec


def test_decomposition_sums_to_residual_ie():
    torch.manual_seed(0)
    seq, d_model, d_sae = 5, 8, 32

    acts_clean = torch.randn(seq, d_model)
    acts_cf = torch.randn(seq, d_model)
    grad_acts = torch.randn(seq, d_model)

    sae = LinearSAE(d_model, d_sae)
    f_clean = sae.encode(acts_clean.unsqueeze(0))[0]
    f_corrupt = sae.encode(acts_cf.unsqueeze(0))[0]

    # --- Exactly the ops src/attribution/nodes.py performs ---
    W_dec_T = sae.W_dec.T if sae.W_dec.shape[0] == sae.cfg.d_sae else sae.W_dec
    grad_f = torch.matmul(grad_acts, W_dec_T)
    feat_attr = grad_f * (f_corrupt - f_clean)                       # [seq, d_sae]

    err_clean = acts_clean - sae.decode(f_clean.unsqueeze(0))[0]
    err_cf = acts_cf - sae.decode(f_corrupt.unsqueeze(0))[0]
    err_attr = (grad_acts * (err_cf - err_clean)).sum(dim=-1)        # [seq]

    # Residual-stream IE per token.
    residual_ie = (grad_acts * (acts_cf - acts_clean)).sum(dim=-1)

    reconstructed = feat_attr.sum(dim=-1) + err_attr
    assert torch.allclose(reconstructed, residual_ie, atol=1e-4), \
        (reconstructed - residual_ie).abs().max()
