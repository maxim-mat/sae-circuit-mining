"""Ablation-based faithfulness, ported from Marks et al.'s ablation.py
(arXiv:2403.19647).

Faithfulness is NOT metric_patch - metric_clean: that quantity is the total
effect, which becomes the circuit's "y" node. It is

    (F(C) - F(empty)) / F(M) - F(empty))

where F(M) runs the unmodified model, F(C) runs with everything OUTSIDE the
circuit ablated, and F(empty) ablates every feature at every site.

Ablation replaces a feature's activation with a reference value rather than
deleting it: "mean" uses the batch mean at that site (their default), "resample"
permutes it across the batch, "zero" writes zero.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from src.circuit.build import load_circuit
from src.data.loaders import load_sva_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes

logger = logging.getLogger(__name__)

ERROR_IDX = -1


def faithfulness(f_c: float, f_m: float, f_empty: float) -> float:
    """(F(C) - F(0)) / (F(M) - F(0)), per Marks et al. ablation.py.

    Not clamped: a circuit scoring above the full model is a real result.
    """
    denom = f_m - f_empty
    if denom == 0:
        raise ZeroDivisionError(
            "F(M) == F(empty): the metric does not distinguish the full model "
            "from the fully-ablated one, so faithfulness is undefined."
        )
    return (f_c - f_empty) / denom


def _ablation_values(feats: torch.Tensor, style: str) -> torch.Tensor:
    """Reference activations to substitute for ablated features.

    feats is [batch, seq, d_sae] over the example batch at one site.
    """
    if style == "mean":
        return feats.mean(dim=0, keepdim=True).expand_as(feats)
    if style == "resample":
        perm = torch.randperm(feats.shape[0], device=feats.device)
        return feats[perm]
    if style == "zero":
        return torch.zeros_like(feats)
    raise ValueError(f"Unknown ablation {style!r}; expected 'mean', 'resample' or 'zero'")


def run_with_ablations(model, saes, suite, examples, keep_mask, metric_fn,
                       ablation: str, handle_errors: str, device: str,
                       prepend_bos: bool = True) -> float:
    """Mean metric with every feature outside `keep_mask` ablated.

    keep_mask maps site_ordinal -> set of kept feature indices (ERROR_IDX for the
    reconstruction-error term). Pass None to keep everything (that is F(M)).
    handle_errors: "keep" never ablates the error term, "remove" always does,
    "default" lets the mask decide.
    """
    if handle_errors not in ("default", "keep", "remove"):
        raise ValueError(f"Unknown handle_errors {handle_errors!r}")

    # Ablation references are batch statistics, so every example is run in one
    # padded-free batch: the loaders guarantee a single token length when
    # data.use_min_length_only is set, which is also what makes positions align.
    token_batch = torch.cat(
        [model.to_tokens(ex.clean_prefix, prepend_bos=prepend_bos) for ex in examples]
    ).to(device)

    site_by_hook = {site.hook: site for site in suite.sites}

    def make_hook(site):
        sae = saes[site.sae_id]

        def hook(tensor, hook):                     # tensor: [batch, seq, d_model]
            feats = sae.encode(tensor)
            recon = sae.decode(feats)
            err = tensor - recon

            ref = _ablation_values(feats, ablation)
            if keep_mask is None:
                new_feats = feats
            else:
                kept = keep_mask.get(site.ordinal, set())
                mask = torch.zeros(feats.shape[-1], dtype=torch.bool, device=feats.device)
                idx = [f for f in kept if f != ERROR_IDX]
                if idx:
                    mask[torch.tensor(idx, device=feats.device)] = True
                new_feats = torch.where(mask, feats, ref)

            keep_err = (handle_errors == "keep" or
                        (handle_errors == "default" and
                         (keep_mask is None or ERROR_IDX in keep_mask.get(site.ordinal, set()))))
            new_err = err if keep_err else _ablation_values(err, ablation)
            return sae.decode(new_feats) + new_err

        return hook

    with torch.no_grad():
        logits = model.run_with_hooks(
            token_batch,
            fwd_hooks=[(site.hook, make_hook(site)) for site in suite.sites],
        )
        scores = [
            float(metric_fn(logits=logits[i:i + 1], example=ex, model=model,
                            tokens=token_batch[i:i + 1]))
            for i, ex in enumerate(examples)
        ]
    return sum(scores) / len(scores)


def run_evaluation(cfg, circuit_path: Path, out_dir: Path) -> dict:
    """F(C), F(M), F(empty) and the faithfulness ratio, written to out_dir."""
    import json

    started = time.time()
    torch.manual_seed(cfg.seed)
    circuit = load_circuit(circuit_path)

    suite = build_suite(cfg.suite.name)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    prepend_bos = getattr(next(iter(saes.values())).cfg, "prepend_bos", True)

    data_path = cfg.evaluation.data_path or cfg.data.path
    examples = load_sva_examples(
        data_path, cfg.evaluation.num_examples, model,
        use_min_length_only=cfg.data.use_min_length_only,
        max_length=cfg.data.max_length, seed=cfg.seed)
    if not examples:
        raise ValueError(f"No evaluation examples loaded from {data_path!r}")

    keep_mask: dict[int, set[int]] = {}
    for ordinal, feature in zip(circuit.nodes["site_ordinal"], circuit.nodes["feature_idx"]):
        keep_mask.setdefault(int(ordinal), set()).add(int(feature))

    common = dict(ablation=cfg.evaluation.ablation,
                  handle_errors=cfg.evaluation.handle_errors,
                  device=cfg.suite.device, prepend_bos=prepend_bos)
    f_m = run_with_ablations(model, saes, suite, examples, None, metric_fn, **common)
    f_c = run_with_ablations(model, saes, suite, examples, keep_mask, metric_fn, **common)
    f_empty = run_with_ablations(model, saes, suite, examples, {}, metric_fn, **common)

    result = {
        "faithfulness": faithfulness(f_c, f_m, f_empty),
        "F_C": f_c, "F_M": f_m, "F_empty": f_empty,
        "n_examples": len(examples),
        "runtime_s": round(time.time() - started, 2),
    }
    (Path(out_dir) / "faithfulness.json").write_text(json.dumps(result, indent=2))
    logger.info("Faithfulness %.4f (F_C=%.4f F_M=%.4f F_empty=%.4f)",
                result["faithfulness"], f_c, f_m, f_empty)
    return result
