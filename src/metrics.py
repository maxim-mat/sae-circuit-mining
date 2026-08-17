"""Task metrics.

Every metric returns a bare scalar with the sign already folded in, so that the
indirect effect grad(m) . delta is literally the IE of the metric being
reported. The previous (value, direction) convention multiplied the effect
AFTER the gradient, which is only equivalent for direction = +/-1 and has no
counterpart in Marks et al. (arXiv:2403.19647) -- they put the sign inside the
metric, before backprop.
"""
from __future__ import annotations

import torch


def sva_logit_diff(logits, example, model, **_):
    """Marks' paired subject-verb agreement metric:
    logit(patch_answer) - logit(clean_answer), read at the final position."""
    correct = model.to_single_token(example.patch_answer)
    clean = model.to_single_token(example.clean_answer)
    return logits[0, -1, correct] - logits[0, -1, clean]


def mib_logit_diff(logits, example, model, **_):
    """MIB-bench: logit(correct choice) - logit(incorrect choice) at the final
    position. Answers must be single tokens (assumes the GPT-2 leading space)."""
    choices, answer_key = example.meta.get("choices"), example.meta.get("answerKey")
    if choices is None or answer_key is None:
        raise ValueError("mib_logit_diff requires 'choices' and 'answerKey' in meta")
    if len(choices) != 2:
        raise ValueError(f"mib_logit_diff expects exactly 2 choices, got {len(choices)}")
    if answer_key not in (0, 1):
        raise ValueError(f"mib_logit_diff expects answerKey in {{0,1}}, got {answer_key!r}")
    try:
        correct = model.to_single_token(" " + choices[answer_key])
        wrong = model.to_single_token(" " + choices[1 - answer_key])
    except Exception as e:
        raise ValueError(
            f"mib_logit_diff requires single-token answers; "
            f"{choices[answer_key]!r}/{choices[1-answer_key]!r} are not"
        ) from e
    return logits[0, -1, correct] - logits[0, -1, wrong]


def neg_next_token_loss(logits, example, model, tokens=None, **_):
    """Negated next-token cross-entropy: higher is better, sign folded in."""
    if tokens is None:
        raise ValueError("neg_next_token_loss requires `tokens`")
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = tokens[0, 1:].contiguous()
    return -torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), reduction="sum",
    )


METRICS = {
    "sva_logit_diff": sva_logit_diff,
    "mib_logit_diff": mib_logit_diff,
    "neg_next_token_loss": neg_next_token_loss,
}


def get_metric(name: str):
    if name not in METRICS:
        raise KeyError(f"Unknown metric {name!r}; known: {sorted(METRICS)}")
    return METRICS[name]
