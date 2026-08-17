"""Checks integrated_gradient_grad_x with a fake model, so it does not need
GPT-2/SAEs.

Validates the path interpolation, per-step gradient isolation, and averaging by
comparing against closed-form integrated gradients for known metrics.
"""
import torch

from src.attribution.ig import integrated_gradient_grad_x


class FakeModel:
    """run_with_hooks calls the injected hook to get x_interp and returns it as
    'logits', so gradients flow back to x_interp exactly as in the real path."""
    def run_with_hooks(self, batch, fwd_hooks):
        _, hook = fwd_hooks[0]
        return hook(None, None)

    def zero_grad(self, set_to_none=True):
        pass


def make_metric(kind, W):
    # Metrics now return a bare signed scalar (Task 5); no direction tuple.
    def metric_fn(logits, example, model, **_):
        row = logits[0]  # [seq, d]
        if kind == "linear":
            return (row * W).sum()
        return 0.5 * (row ** 2).sum()  # quadratic
    return metric_fn


def test_linear_metric_recovers_constant_gradient():
    # Linear metric -> gradient is W at every interpolation point -> mean = W.
    torch.manual_seed(0)
    seq, d, steps = 3, 4, 10
    acts_clean, acts_cf, W = torch.randn(seq, d), torch.randn(seq, d), torch.randn(seq, d)
    tokens = torch.zeros(1, seq, dtype=torch.long)
    grad = integrated_gradient_grad_x(FakeModel(), tokens, "hp", acts_clean, acts_cf,
                                      make_metric("linear", W), None, steps)
    assert torch.allclose(grad, W, atol=1e-5), (grad - W).abs().max()


def test_quadratic_metric_averages_interpolation_points():
    # Quadratic metric -> grad at alpha is the interp point -> mean = mean over path.
    torch.manual_seed(1)
    seq, d, steps = 3, 4, 8
    acts_clean, acts_cf = torch.randn(seq, d), torch.randn(seq, d)
    tokens = torch.zeros(1, seq, dtype=torch.long)
    grad = integrated_gradient_grad_x(FakeModel(), tokens, "hp", acts_clean, acts_cf,
                                      make_metric("quad", None), None, steps)
    mean_alpha = (steps - 1) / (2 * steps)  # mean of {0,1,...,steps-1}/steps
    expected = acts_clean + mean_alpha * (acts_cf - acts_clean)
    assert torch.allclose(grad, expected, atol=1e-5), (grad - expected).abs().max()


def test_steps_one_is_clean_point():
    # steps=1 -> only alpha=0 -> gradient evaluated at acts_clean (== attrib).
    torch.manual_seed(2)
    seq, d = 3, 4
    acts_clean, acts_cf = torch.randn(seq, d), torch.randn(seq, d)
    tokens = torch.zeros(1, seq, dtype=torch.long)
    grad = integrated_gradient_grad_x(FakeModel(), tokens, "hp", acts_clean, acts_cf,
                                      make_metric("quad", None), None, 1)
    assert torch.allclose(grad, acts_clean, atol=1e-5), (grad - acts_clean).abs().max()
