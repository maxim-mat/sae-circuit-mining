import torch
import pytest
from src.data.loaders import Example
from src.metrics import get_metric


class FakeModel:
    def to_single_token(self, s):
        return {" is": 5, " are": 7, " Mary": 5, " John": 7}[s]


def test_sva_logit_diff_is_clean_minus_patch_at_last_position():
    """Marks' notebook orientation: logit(clean_answer) - logit(patch_answer),
    so that higher = the behaviour of interest (Wang et al. arXiv:2211.00593)."""
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 5] = 2.0    # clean answer " is"
    logits[0, -1, 7] = 3.0    # patch answer " are"
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() == pytest.approx(-1.0)


def test_a_correct_model_scores_positive_on_a_clean_input():
    """The sign that matters in practice: preferring the clean answer must be
    positive, or F(M) lands below F(empty) and every IE reads backwards."""
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 5] = 3.0    # clean answer " is" preferred
    logits[0, -1, 7] = 1.0    # patch answer " are"
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() > 0


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
