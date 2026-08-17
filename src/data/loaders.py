"""Dataset loaders.

The subject-verb agreement loader ports load_examples from Marks et al.'s
feature-circuits repo (arXiv:2403.19647). Its filters are not incidental
hygiene: use_min_length_only forces every example to the same token length,
which is what makes token positions comparable across examples and therefore
what makes aggregation="none" (position-preserving circuits) meaningful at all.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Example:
    clean_prefix: str
    clean_answer: str
    patch_prefix: str | None = None
    patch_answer: str | None = None
    meta: dict = field(default_factory=dict)


def load_sva_examples(path, num_examples: int, model, use_min_length_only: bool = True,
                      max_length: int | None = None, seed: int = 12,
                      shuffle: bool = True) -> list[Example]:
    """Load Marks' subject-verb agreement stimuli (jsonl, despite the .json name).

    Drops examples with multi-token answers, unequal clean/patch prefix lengths,
    or tokenization mismatches at the prefix|answer boundary. With
    use_min_length_only, keeps only examples of the shortest length seen,
    restarting collection whenever a shorter one appears.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if shuffle:
        random.Random(seed).shuffle(lines)

    examples: list[Example] = []
    min_length = float("inf")
    tok = model.tokenizer

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed line in %s: %s", path, e)
            continue

        clean_prefix = tok(data["clean_prefix"]).input_ids
        patch_prefix = tok(data["patch_prefix"]).input_ids
        clean_answer = tok(data["clean_answer"]).input_ids
        patch_answer = tok(data["patch_answer"]).input_ids
        clean_full = tok(data["clean_prefix"] + data["clean_answer"]).input_ids
        patch_full = tok(data["patch_prefix"] + data["patch_answer"]).input_ids

        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and clean_answer and clean_answer[0] == bos:
            clean_answer = clean_answer[1:]
        if bos is not None and patch_answer and patch_answer[0] == bos:
            patch_answer = patch_answer[1:]

        if len(clean_answer) != 1 or len(patch_answer) != 1:
            continue
        if len(clean_prefix) != len(patch_prefix):
            continue
        # Boundary check: the answer must tokenize as its own trailing token(s)
        # rather than merging with the prefix's last token. Marks compare
        # lengths (`clean_full.shape[1] != prefix + answer`), not token ids.
        if len(clean_full) != len(clean_prefix) + len(clean_answer):
            continue
        if len(patch_full) != len(patch_prefix) + len(patch_answer):
            continue
        if max_length is not None and len(clean_prefix) > max_length:
            continue

        if use_min_length_only:
            length = len(clean_prefix)
            if length < min_length:
                examples = []
                min_length = length
            elif length > min_length:
                continue

        examples.append(Example(
            clean_prefix=data["clean_prefix"], clean_answer=data["clean_answer"],
            patch_prefix=data["patch_prefix"], patch_answer=data["patch_answer"],
            meta={k: v for k, v in data.items() if k not in
                  ("clean_prefix", "clean_answer", "patch_prefix", "patch_answer")},
        ))
        if len(examples) >= num_examples:
            break
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def select_counterfactual(item: dict, name: str | None) -> dict:
    """Pick the one counterfactual this run is scoped to.

    A run answers one causal question, so there is no default and no
    'first available' fallback -- picking silently would attribute against a
    different intervention than the config names.
    """
    available = sorted(k for k, v in item.items()
                       if isinstance(v, dict) and "prompt" in v)
    if name is None:
        raise ValueError(
            f"data.counterfactual is required for loader='mib'. "
            f"Available in this dataset: {available}"
        )
    if name not in available:
        raise ValueError(
            f"Counterfactual {name!r} not found. Available: {available}"
        )
    return item[name]


def load_mib_examples(path: str, num_examples: int, model, counterfactual: str,
                      split: str = "train",
                      text_column: str | None = None) -> list[Example]:
    """Load a MIB-bench dataset, scoped to a single counterfactual.

    The counterfactual name is NOT stored per row -- it is a config field, so it
    lands in the stage manifest and the cache key, and two counterfactuals
    produce two independent hives rather than two partitions of one.
    """
    from datasets import load_dataset

    dataset = load_dataset(path=path, split=split, streaming=False)
    column = text_column or next(
        (c for c in ("prompt", "text", "sentence", "content") if c in dataset.column_names),
        None,
    )
    if column is None:
        raise ValueError(
            f"Could not detect a text column in {dataset.column_names}; "
            f"set data.text_column explicitly."
        )

    examples: list[Example] = []
    for idx, item in enumerate(dataset):
        cf = select_counterfactual(item, counterfactual)
        examples.append(Example(
            clean_prefix=item[column], clean_answer="",
            patch_prefix=cf["prompt"], patch_answer="",
            meta={"source_idx": idx, "template": item.get("template", ""),
                  "choices": item.get("choices"), "answerKey": item.get("answerKey")},
        ))
        if len(examples) >= num_examples:
            break
    logger.info("Loaded %d examples from %s (counterfactual=%s)",
                len(examples), path, counterfactual)
    return examples


def load_examples(cfg, model, seed: int) -> list[Example]:
    """Dispatch on cfg.loader."""
    if cfg.loader == "sva":
        # SVA stimuli carry exactly one patch_prefix per example, so
        # cfg.counterfactual is not applicable here.
        return load_sva_examples(cfg.path, cfg.num_examples, model,
                                 use_min_length_only=cfg.use_min_length_only,
                                 max_length=cfg.max_length, seed=seed)
    if cfg.loader == "mib":
        return load_mib_examples(cfg.path, cfg.num_examples, model,
                                 counterfactual=cfg.counterfactual,
                                 split=cfg.split, text_column=cfg.text_column)
    raise ValueError(f"Unknown loader {cfg.loader!r}; expected 'sva' or 'mib'")
