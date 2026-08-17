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
import math
import time
from pathlib import Path

import torch

from src.circuit.build import load_circuit
from src.data.loaders import load_sva_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes, prepend_bos_for

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


def resolve_start_layer(suite, start_layer: int) -> int:
    """First block whose sites take part in the ablation.

    Marks et al. §3.2: "we ignore the first 1/3 of our circuit, and only evaluate
    the latter 2/3" -- early features key on specific tokens, and the train and
    test splits do not share tokens, so scoring the early segment measures
    vocabulary overlap rather than circuit quality. start_layer = -1 reproduces
    that default; any other value is used as given.
    """
    if start_layer != -1:
        return start_layer
    n_blocks = max(site.block for site in suite.sites) + 1
    return math.ceil(n_blocks / 3)


def run_with_ablations(model, saes, suite, examples, keep_mask, metric_fn,
                       ablation: str, handle_errors: str, device: str,
                       prepend_bos: bool = True, start_layer: int = -1,
                       invert: bool = False) -> float:
    """Mean metric with everything outside `keep_mask` ablated.

    keep_mask maps site_ordinal -> set of (token_idx, feature_idx) pairs, with
    feature_idx = ERROR_IDX for the reconstruction-error term and token_idx = -1
    meaning "every position" (which is what aggregation="sum" produces). Pass
    None to keep everything -- that is F(M).

    The mask is POSITION-RESOLVED. Keying it by feature alone would keep a
    feature at all positions when the circuit selected it at one, scoring a
    circuit larger than the one being reported: on rc_train n=100 that turned 84
    selected nodes into 486 kept ones.

    handle_errors: "keep" never ablates the error term, "remove" always does,
    "default" lets the mask decide. Sites before start_layer are left untouched.
    """
    if handle_errors not in ("default", "keep", "remove"):
        raise ValueError(f"Unknown handle_errors {handle_errors!r}")
    first_block = resolve_start_layer(suite, start_layer)
    ablated_sites = [s for s in suite.sites if s.block >= first_block]

    # Ablation references are batch statistics, so every example is run in one
    # padded-free batch: the loaders guarantee a single token length when
    # data.use_min_length_only is set, which is also what makes positions align.
    token_batch = torch.cat(
        [model.to_tokens(ex.clean_prefix, prepend_bos=prepend_bos) for ex in examples]
    ).to(device)

    def make_hook(site):
        sae = saes[site.sae_id]

        def hook(tensor, hook):                     # tensor: [batch, seq, d_model]
            feats = sae.encode(tensor)
            recon = sae.decode(feats)
            err = tensor - recon

            ref = _ablation_values(feats, ablation)
            seq = feats.shape[1]
            if keep_mask is None:
                new_feats = feats
                err_keep = torch.ones(seq, dtype=torch.bool, device=feats.device)
            else:
                kept = keep_mask.get(site.ordinal, set())
                # [seq, d_sae], broadcast against feats [batch, seq, d_sae].
                mask = torch.zeros(seq, feats.shape[-1], dtype=torch.bool,
                                   device=feats.device)
                err_keep = torch.zeros(seq, dtype=torch.bool, device=feats.device)
                for tok, feat in kept:
                    if tok >= seq:
                        continue          # guarded and reported by the caller
                    rows = slice(None) if tok < 0 else tok
                    if feat == ERROR_IDX:
                        err_keep[rows] = True
                    else:
                        mask[rows, feat] = True
                if invert:                 # M \ C: ablate the circuit, keep the rest
                    mask = ~mask
                    err_keep = ~err_keep
                new_feats = torch.where(mask, feats, ref)

            if handle_errors == "keep":
                err_keep = torch.ones_like(err_keep)
            elif handle_errors == "remove":
                err_keep = torch.zeros_like(err_keep)
            new_err = torch.where(err_keep.unsqueeze(-1), err,
                                  _ablation_values(err, ablation))
            return sae.decode(new_feats) + new_err

        return hook

    with torch.no_grad():
        logits = model.run_with_hooks(
            token_batch,
            fwd_hooks=[(site.hook, make_hook(site)) for site in ablated_sites],
        )
        scores = [
            float(metric_fn(logits=logits[i:i + 1], example=ex, model=model,
                            tokens=token_batch[i:i + 1]))
            for i, ex in enumerate(examples)
        ]
    return sum(scores) / len(scores)


def build_keep_mask(nodes) -> dict[int, set[tuple[int, int]]]:
    """Circuit node table -> {site_ordinal: {(token_idx, feature_idx)}}.

    token_idx is carried through, not collapsed: under aggregation="none" the
    node IS the (position, feature) pair. aggregation="sum" writes token_idx=-1,
    which the mask reads as "all positions".
    """
    mask: dict[int, set[tuple[int, int]]] = {}
    tokens = nodes["token_idx"] if "token_idx" in nodes else [-1] * len(nodes)
    for ordinal, token, feature in zip(nodes["site_ordinal"], tokens,
                                       nodes["feature_idx"]):
        mask.setdefault(int(ordinal), set()).add((int(token), int(feature)))
    return mask


def minimality_scores(model, saes, suite, examples, keep_mask, metric_fn,
                      f_c: float, **common) -> dict:
    """Per-node leave-one-out: |F(C \\ {v}) - F(C)| for every v in C.

    An approximation of Wang et al. 2023 (arXiv:2211.00593) section 4.2, which
    asks for SOME subset K with |F(C\\(K u {v})) - F(C\\K)| large and needs an
    exponential search. K = {} is the cheap lower bound: it understates the
    necessity of any node with a redundant backup, since the backup covers for
    it while the rest of the circuit is intact.

    Costs one forward pass per node. Marks do not measure this -- their criteria
    are interpretability, faithfulness and completeness (section 3.2) -- so it is
    reported for the selector comparison, not for the reproduction.
    """
    scores = {}
    for ordinal, nodes in keep_mask.items():
        for node in nodes:
            reduced = {o: set(v) for o, v in keep_mask.items()}
            reduced[ordinal].discard(node)
            f_without = run_with_ablations(model, saes, suite, examples, reduced,
                                           metric_fn, **common)
            scores[(ordinal, node[0], node[1])] = abs(f_without - f_c)
    return scores


def run_evaluation(cfg, circuit_path: Path, out_dir: Path) -> dict:
    """F(C), F(M), F(empty) and the faithfulness ratio, written to out_dir."""
    import json

    started = time.time()
    torch.manual_seed(cfg.seed)
    circuit = load_circuit(circuit_path)

    suite = build_suite(cfg.suite.name)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    prepend_bos = prepend_bos_for(saes)

    data_path = cfg.evaluation.data_path or cfg.data.path
    examples = load_sva_examples(
        data_path, cfg.evaluation.num_examples, model,
        use_min_length_only=cfg.data.use_min_length_only,
        max_length=cfg.data.max_length, seed=cfg.seed)
    if not examples:
        raise ValueError(f"No evaluation examples loaded from {data_path!r}")

    keep_mask = build_keep_mask(circuit.nodes)

    common = dict(ablation=cfg.evaluation.ablation,
                  handle_errors=cfg.evaluation.handle_errors,
                  device=cfg.suite.device, prepend_bos=prepend_bos,
                  start_layer=cfg.evaluation.start_layer)
    f_m = run_with_ablations(model, saes, suite, examples, None, metric_fn, **common)
    f_c = run_with_ablations(model, saes, suite, examples, keep_mask, metric_fn, **common)
    f_empty = run_with_ablations(model, saes, suite, examples, {}, metric_fn, **common)
    # Completeness (Marks section 3.2): the faithfulness of the circuit's COMPLEMENT.
    # Ideal is 0 -- if ablating just the circuit destroys the behaviour, the
    # circuit captured everything that mattered. High completeness means the
    # model routes around the circuit, i.e. the circuit is missing something.
    f_complement = run_with_ablations(model, saes, suite, examples, keep_mask,
                                      metric_fn, invert=True, **common)

    # Position indices only transfer if the eval split tokenises to the same
    # length the circuit was discovered at; use_min_length_only makes that true
    # within a split but not necessarily across train/test.
    seq_len = model.to_tokens(examples[0].clean_prefix,
                              prepend_bos=prepend_bos).shape[1]
    stale = sum(1 for nodes in keep_mask.values() for tok, _ in nodes if tok >= seq_len)
    if stale:
        logger.warning("%d circuit nodes sit at token positions >= the eval "
                       "sequence length (%d) and were dropped from the mask; the "
                       "train and test splits do not share a token length.",
                       stale, seq_len)

    minimality = {}
    if cfg.evaluation.minimality:
        minimality = minimality_scores(model, saes, suite, examples, keep_mask,
                                       metric_fn, f_c, **common)

    result = {
        "faithfulness": faithfulness(f_c, f_m, f_empty),
        "completeness": faithfulness(f_complement, f_m, f_empty),
        "F_C": f_c, "F_M": f_m, "F_empty": f_empty,
        "F_complement": f_complement,
        "n_nodes": len(circuit.nodes),
        "n_examples": len(examples),
        "n_nodes_stale": stale,
        "start_layer": resolve_start_layer(suite, cfg.evaluation.start_layer),
        "runtime_s": round(time.time() - started, 2),
    }
    if minimality:
        vals = sorted(minimality.values())
        # Wang et al. call a node's contribution nontrivial at >=1% of F(M).
        floor = 0.01 * abs(f_m)
        result.update({
            "minimality_min": vals[0],
            "minimality_median": vals[len(vals) // 2],
            "minimality_mean": sum(vals) / len(vals),
            "minimality_frac_nontrivial": sum(v >= floor for v in vals) / len(vals),
        })
        (Path(out_dir) / "minimality.json").write_text(json.dumps(
            {f"{o}:{t}:{f}": v for (o, t, f), v in minimality.items()}, indent=2))
        logger.info("Minimality: median %.4f, %.0f%% of nodes above the 1%% floor",
                    result["minimality_median"],
                    100 * result["minimality_frac_nontrivial"])
    (Path(out_dir) / "faithfulness.json").write_text(json.dumps(result, indent=2))
    logger.info("Faithfulness %.4f | completeness %.4f (ideal 0) "
                "[F_C=%.4f F_M\\C=%.4f F_M=%.4f F_empty=%.4f]",
                result["faithfulness"], result["completeness"],
                f_c, f_complement, f_m, f_empty)
    return result
