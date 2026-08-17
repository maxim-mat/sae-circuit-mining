"""Integrated-gradients residual-stream gradient at a single hook point."""
from __future__ import annotations

import torch


def integrated_gradient_grad_x(model, clean_tokens, hook_point, acts_clean, acts_cf,
                               metric_fn, example, steps):
    """Mean residual-stream gradient of the metric along the clean->counterfactual
    interpolation path at one hook point.

    Integrated-gradients node attribution, following the reference implementation
    _pe_ig in Marks et al. 2024, "Sparse Feature Circuits" (arXiv:2403.19647):
    only this hook's activation is interpolated, so all other layers see the clean
    input. The interpolated activation is (1-a)*acts_clean + a*acts_cf for
    a = step/steps, step in [0, steps); this is exactly decode(interp features) +
    interp residual because decode is affine and coefficients sum to 1, so the
    residual-stream gradient projected by W_dec.T / read directly recovers the
    feature / error gradients. The `steps` points are batched into one pass.

    Returns [seq, d_model]; steps=1 reduces to attribution patching (gradient at
    the clean point).
    """
    alphas = torch.arange(steps, device=clean_tokens.device, dtype=acts_clean.dtype) / steps
    x_interp = acts_clean.unsqueeze(0) + alphas.view(-1, 1, 1) * (acts_cf - acts_clean).unsqueeze(0)
    x_interp = x_interp.detach().requires_grad_()  # [steps, seq, d_model] leaf

    def ig_hook(tensor, hook):
        return x_interp

    logits = model.run_with_hooks(
        clean_tokens.repeat(steps, 1), fwd_hooks=[(hook_point, ig_hook)]
    )
    # Sum per-step metrics: d(sum)/dx_interp[i] isolates the gradient at alpha_i.
    total_metric = 0
    for i in range(steps):
        m_i = metric_fn(logits=logits[i:i+1], example=example, model=model,
                        tokens=clean_tokens)
        total_metric = total_metric + m_i
    model.zero_grad(set_to_none=True)
    total_metric.backward()
    return x_interp.grad.mean(dim=0)  # [seq, d_model]
