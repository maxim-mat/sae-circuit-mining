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

import itertools
import logging
import time
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.attribution.ig import integrated_gradient_grad_x
from src.data.loaders import load_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes, prepend_bos_for

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

# prompt -> template map, one row per example. Written unconditionally so the
# choice of whether to block on it stays a selection-time decision rather than
# an attribution-time one -- re-deciding it must never cost an attribution run.
#
# `template_key` is the raw value from Example.meta[data.template_field];
# `template_id` is a small int assigned in first-appearance order, stable within
# a run. With template_field unset every example lands in template 0, which is
# the honest description of Marks' SVA data: one syntactic frame throughout.
EXAMPLE_SCHEMA = pa.schema([
    ("example_idx", pa.int32()),
    ("template_id", pa.int16()),
    ("template_key", pa.string()),
])

_EMPTY_BATCH = {name: [] for name in NODE_SCHEMA.names}
CHUNK_SIZE = 10          # examples between parquet flushes

# `kind` stays CATEGORICAL: as object it is one Python str per row, 71MB of a
# 101MB frame at n=100 and 7 of 10.1GB at n=10k. The integer partition keys are
# cast back to ints because they are grouped on all over the selection code;
# `kind` is only ever a groupby key in src.selection.aggregate, which passes
# observed=True. See read_node_hive.
_NODE_DTYPES = {"example_idx": "int32", "site_ordinal": "int16", "block": "int16",
                "kind": "category", "token_idx": "int16", "token_id": "int32",
                "feature_idx": "int32", "attribution": "float32"}


def read_node_hive(node_dir: Path) -> pd.DataFrame:
    """Read the node hive with partition columns restored to their real dtypes.

    pyarrow hands hive partition keys back as pandas Categoricals. That is a trap
    for every groupby downstream: with observed=False (the pandas default)
    grouping on a Categorical materialises the FULL CARTESIAN PRODUCT of category
    levels, inventing a zero-valued row for every (kind x site x feature x
    position) combination that never occurred. On a 5-example run that turned
    96,574 real rows into 58,194,948.

    The integer keys are therefore cast back to ints, since GROUP_COLS and
    FEATURE_COLS are grouped on throughout the selection code. `kind` is the one
    exception -- it is 70% of the frame as object, and the only groupby that uses
    it as a key is src.selection.aggregate, which passes observed=True.
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
    # getting it wrong leaves position 0 off-distribution. pythia-marks declares
    # prepend_bos=False, which is also what Marks' bare-HF-tokenizer loader does.
    prepend_bos = prepend_bos_for(saes)

    batch = {name: list(_EMPTY_BATCH[name]) for name in NODE_SCHEMA.names}
    totals = {name: [] for name in TOTAL_EFFECT_SCHEMA.names}
    examples_meta = {name: [] for name in EXAMPLE_SCHEMA.names}
    template_ids: dict[str, int] = {}
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

        key = ("" if cfg.data.template_field is None
               else str(example.meta.get(cfg.data.template_field, "")))
        examples_meta["example_idx"].append(ex_idx)
        examples_meta["template_id"].append(
            template_ids.setdefault(key, len(template_ids)))
        examples_meta["template_key"].append(key)

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
    pq.write_table(pa.Table.from_pydict(examples_meta, schema=EXAMPLE_SCHEMA),
                   out_dir / "examples.parquet", compression="zstd")

    n_examples = len(totals["example_idx"])
    mean_total_effect = (sum(totals["total_effect"]) / n_examples) if n_examples else 0.0
    logger.info("Node attribution: %d rows over %d examples (%d skipped)",
                rows_written, n_examples, skipped)
    return {
        "row_count": rows_written,
        "n_examples": n_examples,
        "n_skipped": skipped,
        "n_templates": len(template_ids),
        "mean_total_effect": mean_total_effect,
        "runtime_s": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Set IE: the joint effect of patching several nodes at once.
#
# A node is (site_ordinal, token_idx, feature_idx), feature_idx = -1 being the
# SAE error. For an antichain A (no node reaches another),
#     IE(m; A) = m(x_clean | do(a = a_patch  for all a in A)) - m(x_clean),
# estimated with integrated gradients along ONE path on which only A's nodes
# move from clean to patch (Marks et al., arXiv:2403.19647; IG as in Hanna et
# al., arXiv:2403.17806). A linear estimator would make IE(A) the sum of its
# members' IEs by construction, which is why only "ig" is meaningful here.
#
# NOT comparable with the hive's per-node IEs: integrated_gradient_grad_x moves
# a site's whole activation, so a hive IE is a feature's share of that joint
# move. For Delta(A) = IE(A) - sum IE(a), take the individual terms from
# set_ie([a]) as well.
# ---------------------------------------------------------------------------

def set_ie(model, saes, suite, nodes, clean_tokens, patch_tokens, metric_fn,
           example=None, method: str = "ig", steps: int = 10) -> float:
    """IE of patching every node in `nodes` jointly, for one example: nde with
    nothing blocked.

    `nodes` must be an antichain (see reachable_pairs; all_set_ies enforces it).
    The patch is added to each site's recomputed activation, which is exact only
    when no node in the set lies downstream of another.
    """
    if method != "ig":
        warnings.warn(f"set IE with method={method!r} is the sum of the members' "
                      "individual IEs; only 'ig' captures their interaction",
                      stacklevel=2)
    return nde(model, saes, suite, nodes, [], clean_tokens, patch_tokens, metric_fn,
               example, method, steps)


def nde(model, saes, suite, nodes, block, clean_tokens, patch_tokens, metric_fn,
        example=None, method: str = "ig", steps: int = 10) -> float:
    """Natural direct effect of patching `nodes` (A) with `block` (B) held clean:

        NDE(m; A, B) = m(x_clean | do(a = a_patch for a in A, b = b_clean for b in B))
                       - m(x_clean),     one example.

    set_ie(A) - nde(A, B) is the natural indirect effect of A through B (Pearl,
    "Direct and Indirect Effects", UAI 2001). IG along the path on which only A
    moves; each blocked node is re-read from its site's recomputed activation and
    reset to its clean value, so nothing downstream sees it change. Blocking a
    node that A cannot reach changes nothing. `method` other than "ig" takes the
    gradient at the clean point (still a valid first-order NDE, not a sum).
    """
    if method != "ig":
        steps = 1                      # gradient at the clean point
    if not nodes:
        raise ValueError("nde needs at least one node to patch")
    if any(t < 0 for _, t, _ in [*nodes, *block]):
        raise ValueError("set IE / NDE need position-resolved nodes (token_idx >= 0); "
                         "aggregation='sum' nodes have token_idx = -1")
    if set(nodes) & set(block):
        raise ValueError(f"nodes both patched and blocked: {set(nodes) & set(block)}")

    by_ord = {s.ordinal: s for s in suite.sites}
    by_hook = {s.hook: s for s in suite.sites}
    hooks = sorted({by_ord[o].hook for o, _, _ in [*nodes, *block]})
    with torch.no_grad():
        _, clean = model.run_with_cache(clean_tokens, names_filter=hooks)
        _, patch = model.run_with_cache(patch_tokens, names_filter=hooks)

    # Per site: the residual-space move that takes A's nodes there from clean to
    # patch. Decode is affine, so decode(f with A patched) - decode(f) is exactly
    # the sum of the features' moves.
    moves, fc_of, errc_of = {}, {}, {}
    for hook in hooks:
        sae = saes[by_hook[hook].sae_id]
        xc, xp = clean[hook][0], patch[hook][0]
        fc, fp = sae.encode(xc.unsqueeze(0))[0], sae.encode(xp.unsqueeze(0))[0]
        fc_of[hook] = fc
        errc_of[hook] = xc - sae.decode(fc.unsqueeze(0))[0]
        here = [(t, f) for o, t, f in nodes if by_ord[o].hook == hook]
        if not here:
            continue
        f_moved = fc.clone()
        move = torch.zeros_like(xc)
        for t, f in here:
            if f == -1:
                move[t] += (xp - sae.decode(fp.unsqueeze(0))[0])[t] - errc_of[hook][t]
            else:
                f_moved[t, f] = fp[t, f]
        moves[hook] = (move + sae.decode(f_moved.unsqueeze(0))[0]
                       - sae.decode(fc.unsqueeze(0))[0]).detach()
    blocked = {h: [(t, f) for o, t, f in block if by_ord[o].hook == h] for h in hooks}

    ref = next(iter(moves.values()))
    alphas = torch.arange(steps, device=ref.device, dtype=ref.dtype) / steps
    # Zero leaves added at each patched site carry d(metric)/d(activation) at
    # every alpha; same left-Riemann path as integrated_gradient_grad_x.
    leaves = {h: torch.zeros(steps, *m.shape, dtype=m.dtype, device=m.device,
                             requires_grad=True) for h, m in moves.items()}

    def hook_fn(tensor, hook):
        h = hook.name
        out = tensor
        if blocked[h]:
            # Read B from the incoming activation, BEFORE adding A's same-site
            # move: same-site features are siblings, and re-encoding after the
            # move would let A leak into B through the encoder.
            sae = saes[by_hook[h].sae_id]
            f = sae.encode(tensor)
            f_reset = f.clone()
            fix = torch.zeros_like(tensor)
            for t, feat in blocked[h]:
                if feat == -1:
                    fix[:, t] += errc_of[h][t] - (tensor - sae.decode(f))[:, t]
                else:
                    f_reset[:, t, feat] = fc_of[h][t, feat]
            out = tensor + fix + sae.decode(f_reset) - sae.decode(f)
        if h in moves:
            out = out + alphas.view(-1, 1, 1) * moves[h] + leaves[h]
        return out

    logits = model.run_with_hooks(clean_tokens.repeat(steps, 1),
                                  fwd_hooks=[(h, hook_fn) for h in hooks])
    total = sum(metric_fn(logits=logits[i:i + 1], example=example, model=model,
                          tokens=clean_tokens) for i in range(steps))
    model.zero_grad(set_to_none=True)
    total.backward()
    return float(sum((leaves[h].grad.mean(dim=0) * m).sum() for h, m in moves.items()))


def reachable_pairs(model, suite, tokens) -> set:
    """All ((site_ordinal, token), (site_ordinal, token)) pairs (u, v) where u's
    activation can change v's.

    Read off the Jacobian's sparsity, so it follows the model's real wiring:
    causal masking, per-token MLPs and parallel attn || mlp blocks all give exact
    zero gradients, while GELU, softmax and LayerNorm never do -- the pattern is
    structural, not input-dependent. One backward per (site, token); a random
    projection keeps components from cancelling. Same-site pairs are never
    reachable: features of one SAE are siblings, not ancestors.
    """
    acts, leaves = {}, {}

    def hook_fn(tensor, hook):
        leaves[hook.name] = torch.zeros_like(tensor, requires_grad=True)
        acts[hook.name] = tensor + leaves[hook.name]
        return acts[hook.name]

    model.run_with_hooks(tokens[:1], fwd_hooks=[(s.hook, hook_fn) for s in suite.sites])
    seq = tokens.shape[1]
    reach = set()
    for v in suite.sites:
        ups = [u for u in suite.sites if u.ordinal < v.ordinal]
        if not ups:
            continue
        for tv in range(seq):
            out = (acts[v.hook][0, tv] * torch.randn_like(acts[v.hook][0, tv])).sum()
            grads = torch.autograd.grad(out, [leaves[u.hook] for u in ups],
                                        retain_graph=True, allow_unused=True)
            for u, g in zip(ups, grads):
                if g is None:
                    continue
                for tu in g[0].abs().sum(dim=-1).nonzero().flatten().tolist():
                    reach.add(((u.ordinal, tu), (v.ordinal, tv)))
    return reach


def all_set_ies(model, saes, suite, candidates, n, clean_tokens, patch_tokens,
                metric_fn, example=None, method: str = "ig", steps: int = 10,
                reach: set | None = None) -> dict:
    """set_ie of every size-n subset of `candidates` whose nodes are pairwise
    unreachable, for one example. Returns {node tuple: IE}.

    Pass `reach` (from reachable_pairs) when looping over examples of one token
    length: it depends only on the model and the sequence length.
    """
    # ponytail: enumerates C(len(candidates), n) subsets; restrict candidates to
    # selected nodes, or batch sets into one forward if this gets slow.
    if reach is None:
        reach = reachable_pairs(model, suite, clean_tokens)
    out = {}
    for subset in itertools.combinations(candidates, n):
        if any((u[:2], v[:2]) in reach or (v[:2], u[:2]) in reach
               for u, v in itertools.combinations(subset, 2)):
            continue
        out[subset] = set_ie(model, saes, suite, list(subset), clean_tokens,
                             patch_tokens, metric_fn, example, method, steps)
    return out
