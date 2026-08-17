import torch
import pytest
from src.data.loaders import Example
from src.metrics import get_metric


class FakeModel:
    def to_single_token(self, s):
        return {" is": 5, " are": 7, " Mary": 5, " John": 7}[s]


def test_sva_logit_diff_is_patch_minus_clean_at_last_position():
    """Marks' paired metric: logit(patch_answer) - logit(clean_answer)."""
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 5] = 2.0    # clean answer " is"
    logits[0, -1, 7] = 3.0    # patch answer " are"
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() == pytest.approx(1.0)


def test_sva_metric_reads_only_the_last_position():
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 7] = 99.0    # earlier position must be ignored
    logits[0, -1, 5] = 1.0
    logits[0, -1, 7] = 1.0
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() == pytest.approx(0.0)


def test_metrics_return_a_bare_scalar_not_a_direction_tuple():
    logits = torch.zeros(1, 2, 10)
    ex = Example("the boy", " is", "the boys", " are")
    out = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("no_such_metric")
