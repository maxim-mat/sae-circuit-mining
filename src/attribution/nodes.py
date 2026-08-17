"""Node indirect effects: one row per (example, position, feature) per site.

Ported from the monolithic loop that used to live in src/main.py, whose IE math
was verified line-by-line against _pe_attrib / _pe_ig in Marks et al. 2024,
"Sparse Feature Circuits" (arXiv:2403.19647) -- see docs/spec.md section 2. The
math is unchanged here; what changed is the surroundings:

  * sites come from the typed Suite table, so attn/mlp/embed dictionaries can
    coexist with resid ones (columns site_ordinal/block/kind, no `layer`);
  * one counterfactual per run, chosen at load time, so the per-example inner
    loop over counterfactuals is gone (spec D4);
  * metrics carry their own sign, so the post-hoc `direction` multiplier is gone;
  * the counterfactual forward's logits are no longer discarded -- the same
    metric is evaluated on them to give total_effect = metric_patch -
    metric_clean, which is the circuit's "y" node (spec D3).

No aggregation happens here. Marks aggregate after edges, so the hive stores raw
signed per-position effects (spec D2).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.attribution.ig import integrated_gradient_grad_x
from src.data.loaders import load_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes

logger = logging.getLogger(__name__)

NODE_SCHEMA = pa.schema([
    ("example_idx", pa.int32()),
    ("site_ordinal", pa.int16()),
    ("block", pa.int16()),
    ("kind", pa.string()),
    ("token_idx", pa.int16()),
    ("token_id", pa.int32()),
    ("feature_idx", pa.int32()),
    ("attribution", pa.float32()),
])

TOTAL_EFFECT_SCHEMA = pa.schema([
    ("example_idx", pa.int32()),
    ("metric_clean", pa.float32()),
    ("metric_patch", pa.float32()),
    ("total_effect", pa.float32()),
])

_EMPTY_BATCH = {name: [] for name in NODE_SCHEMA.names}
CHUNK_SIZE = 10          # examples between parquet flushes

_NODE_DTYPES = {"example_idx": "int32", "site_ordinal": "int16", "block": "int16",
                "kind": "object", "token_idx": "int16", "token_id": "int32",
                "feature_idx": "int32", "attribution": "float32"}


def read_node_hive(node_dir: Path) -> pd.DataFrame:
    """Read the node hive with partition columns restored to their real dtypes.

    pyarrow hands hive partition keys back as pandas Categoricals. That is a trap
    for every groupby downstream: with observed=False (the pandas default)
    grouping on a Categorical materialises the FULL CARTESIAN PRODUCT of category
    levels, inventing a zero-valued row for every (kind x site x feature x
    position) combination that never occurred. On a 5-example run that turned
    96,574 real rows into 58,194,948. Cast them back here, once, rather than
    hunting observed=True through every consumer.
    """
    df = pd.read_parquet(Path(node_dir) / "nodes")
    return df.astype({c: t for c, t in _NODE_DTYPES.items() if c in df.columns})


def _flush(batch: dict, out_dir: Path) -> int:
    """Write the accumulated rows as a hive partition; returns rows written."""
    n = len(batch["example_idx"])
    if n == 0:
        return 0
    table = pa.Table.from_pydict(batch, schema=NODE_SCHEMA)
    pq.write_to_dataset(table, root_path=str(out_dir / "nodes"),
                        partition_cols=["kind", "site_ordinal"], compression="zstd")
    for v in batch.values():
        v.clear()
    return n


def run_node_attribution(cfg, out_dir: Path) -> dict:
    """Compute node IEs for every site and write the hive into `out_dir`.

    Returns manifest metadata; the caller (src.pipeline via
    src.utils.manifest.resolve_stage_output) owns the caching protocol, so this
    function never looks for or writes a manifest itself.
    """
    out_dir = Path(out_dir)
    started = time.time()
    torch.manual_seed(cfg.seed)

    suite = build_suite(cfg.suite.name)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    examples = load_examples(cfg.data, model, cfg.seed)
    if not examples:
        raise ValueError(
            f"No examples survived loading from {cfg.data.path!r}; nothing to attribute."
        )

    # The SAEs were trained on activations produced with this BOS convention;
    # dropping it leaves position 0 off-distribution. A candidate divergence
    # from Marks, who tokenize with the bare HF tokenizer (docs/spec.md Task 13).
    prepend_bos = getattr(next(iter(saes.values())).cfg, "prepend_bos", True)

    batch = {name: list(_EMPTY_BATCH[name]) for name in NODE_SCHEMA.names}
    totals = {name: [] for name in TOTAL_EFFECT_SCHEMA.names}
    rows_written = 0
    skipped = 0

    for ex_idx, example in enumerate(examples):
        clean_tokens = model.to_tokens(example.clean_prefix, prepend_bos=prepend_bos
                                       ).to(cfg.suite.device)
        patch_tokens = model.to_tokens(example.patch_prefix, prepend_bos=prepend_bos
                                       ).to(cfg.suite.device)

        # Counterfactual patching requires clean/patch positions to correspond.
        # The loaders length-match on the tokenizer; if BPE still makes them
        # differ, positions misalign and the metric-bearing final position would
        # be silently dropped, so skip rather than emit corrupt data.
        if patch_tokens.shape[1] != clean_tokens.shape[1]:
            logger.warning(
                "Example %d: token-length mismatch (clean=%d, patch=%d); skipping.",
                ex_idx, clean_tokens.shape[1], patch_tokens.shape[1])
            skipped += 1
            continue

        model.zero_grad()
        acts_clean_by_hook: dict = {}
        acts_patch_by_hook: dict = {}

        def make_fwd_hook(hook_name, target_dict):
            def fwd_hook(tensor, hook):
                if target_dict is acts_clean_by_hook:
                    tensor.retain_grad()
                target_dict[hook_name] = tensor
                return tensor
            return fwd_hook

        for site in suite.sites:
            model.add_hook(site.hook, make_fwd_hook(site.hook, acts_clean_by_hook),
                           dir="fwd")
        clean_logits = model(clean_tokens)
        metric_clean = metric_fn(logits=clean_logits, example=example, model=model,
                                 tokens=clean_tokens)
        # 'ig' recomputes gradients along an interpolation path per site, so it
        # does not use the clean-point gradient.
        if cfg.attribution.method == "attrib":
            metric_clean.backward()
        model.reset_hooks()

        for site in suite.sites:
            model.add_hook(site.hook, make_fwd_hook(site.hook, acts_patch_by_hook),
                           dir="fwd")
        with torch.no_grad():
            patch_logits = model(patch_tokens)
            metric_patch = metric_fn(logits=patch_logits, example=example, model=model,
                                     tokens=patch_tokens)
        model.reset_hooks()

        totals["example_idx"].append(ex_idx)
        totals["metric_clean"].append(float(metric_clean))
        totals["metric_patch"].append(float(metric_patch))
        totals["total_effect"].append(float(metric_patch) - float(metric_clean))

        clean_token_ids = clean_tokens[0]
        for site in suite.sites:
            sae = saes[site.sae_id]
            acts_clean = acts_clean_by_hook[site.hook][0]
            acts_cf = acts_patch_by_hook[site.hook][0]

            f_clean = sae.encode(acts_clean.unsqueeze(0))[0]
            f_corrupt = sae.encode(acts_cf.unsqueeze(0))[0]

            # Residual-stream gradient of the metric at this hook point.
            if cfg.attribution.method == "ig":
                grad_acts = integrated_gradient_grad_x(
                    model, clean_tokens, site.hook, acts_clean, acts_cf,
                    metric_fn, example, cfg.attribution.ig_steps)
            else:
                grad_acts = acts_clean_by_hook[site.hook].grad[0]

            W_dec_T = sae.W_dec.T if sae.W_dec.shape[0] == sae.cfg.d_sae else sae.W_dec
            grad_f = torch.matmul(grad_acts, W_dec_T)
            attributions = grad_f * (f_corrupt - f_clean)

            # Lossless storage mask: both inactive => delta zero => effect zero.
            raw_passing = ((f_clean > 0.0) | (f_corrupt > 0.0)).nonzero(as_tuple=False)
            if raw_passing.numel() > 0:
                token_indices = raw_passing[:, 0]
                feature_indices = raw_passing[:, 1]
                vals = attributions[token_indices, feature_indices]

                t_cpu = token_indices.cpu().numpy()
                f_cpu = feature_indices.cpu().numpy()
                # float32: the fp16 subnormal floor flushes small IEs to zero and
                # quantizes the null tail that calibrates hypothesis-test
                # selection. zstd absorbs the size cost.
                v_cpu = vals.detach().cpu().numpy()

                n = len(t_cpu)
                batch["example_idx"].extend([ex_idx] * n)
                batch["site_ordinal"].extend([site.ordinal] * n)
                batch["block"].extend([site.block] * n)
                batch["kind"].extend([site.kind] * n)
                batch["token_idx"].extend(t_cpu.tolist())
                batch["token_id"].extend([int(clean_token_ids[t]) for t in t_cpu])
                batch["feature_idx"].extend(f_cpu.tolist())
                batch["attribution"].extend(v_cpu.tolist())

            # SAE reconstruction-error node: the residual no feature captures.
            # Stored at feature_idx = -1 so a site's feature attributions plus
            # the error attribution sum to the residual-stream IE (guarded by
            # tests/test_error_node.py). Dense: one value per token.
            err_clean = acts_clean - sae.decode(f_clean.unsqueeze(0))[0]
            err_cf = acts_cf - sae.decode(f_corrupt.unsqueeze(0))[0]
            err_attr = (grad_acts * (err_cf - err_clean)).sum(dim=-1)
            err_cpu = err_attr.detach().cpu().numpy()

            n_err = len(err_cpu)
            batch["example_idx"].extend([ex_idx] * n_err)
            batch["site_ordinal"].extend([site.ordinal] * n_err)
            batch["block"].extend([site.block] * n_err)
            batch["kind"].extend([site.kind] * n_err)
            batch["token_idx"].extend(range(n_err))
            batch["token_id"].extend([int(clean_token_ids[t]) for t in range(n_err)])
            batch["feature_idx"].extend([-1] * n_err)
            batch["attribution"].extend(err_cpu.tolist())

        for tensor in acts_clean_by_hook.values():
            if tensor.grad is not None:
                tensor.grad.zero_()
        acts_clean_by_hook.clear()
        acts_patch_by_hook.clear()

        if (ex_idx + 1) % CHUNK_SIZE == 0:
            rows_written += _flush(batch, out_dir)

    rows_written += _flush(batch, out_dir)
    pq.write_table(pa.Table.from_pydict(totals, schema=TOTAL_EFFECT_SCHEMA),
                   out_dir / "total_effects.parquet", compression="zstd")

    n_examples = len(totals["example_idx"])
    mean_total_effect = (sum(totals["total_effect"]) / n_examples) if n_examples else 0.0
    logger.info("Node attribution: %d rows over %d examples (%d skipped)",
                rows_written, n_examples, skipped)
    return {
        "row_count": rows_written,
        "n_examples": n_examples,
        "n_skipped": skipped,
        "mean_total_effect": mean_total_effect,
        "runtime_s": round(time.time() - started, 2),
    }
