"""Feature-edge indirect effects over a selected node set.

Third stage: run AFTER node selection has pruned the alphabet -- edges are only
computed for surviving downstream nodes, which is where the cost saving comes
from.

Method -- stop-gradient edge attribution from Marks et al. 2024, "Sparse Feature
Circuits" (arXiv:2403.19647), their `jvp`. For an upstream feature u at site i
and a downstream feature d at site j:

    IE(u -> d) = (dm/dd) * (dd/du) * (f_patch - f_clean)_u

  * dm/dd (`grad_f` at j) comes from a node pass (metric backward at the clean
    point) -- decode is affine so grad_f = grad_resid @ W_dec.T.
  * dd/du is the Jacobian of the downstream feature w.r.t. the upstream feature
    through the frozen network. We get (dm/dd)*(dd/du) in one backward by seeding
    autograd with (grad_f[d] * g[d]).sum() and differentiating w.r.t. a leaf
    upstream feature tensor, then contracting with the upstream delta.

The upstream activation at site i is replaced by decode(f_i) + res_i with f_i,
res_i as leaves (an identity substitution -- value unchanged -- that re-routes
the graph through the feature basis so dd/du is defined). This mirrors Marks'
`set_activation(x_hat + res)`.

Which triples get scored is `edge_template`'s business, not this module's. With
edges.template="marks" every EdgeSpec carries stop-gradients on the intermediate
sites, making each edge a DIRECT effect; with "total_effect" there are none and
the gradient flows through the whole frozen network ("eventually-follows"
semantics). Direct and total edges must never be summed into one graph.

Positions are summed on both endpoints (downstream positions in the seed,
upstream positions in the contraction), matching Marks' aggregation into a
task-level edge. Position-resolved edges would need a 4-D schema and are not
implemented -- see docs/spec.md.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.attribution.edge_template import Y_SITE, EdgeSpec, build_template
from src.data.loaders import load_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes

logger = logging.getLogger(__name__)

ERROR_IDX = -1  # SAE reconstruction-error node, same convention as nodes.py.

EDGE_SCHEMA = pa.schema([
    ("example_idx", pa.int32()),
    ("src_ordinal", pa.int16()),
    ("src_feature", pa.int32()),
    # dst_ordinal = -1 is the metric ("y") node; its dst_feature is unused and
    # written as -1 too.
    ("dst_ordinal", pa.int16()),
    ("dst_feature", pa.int32()),
    ("edge_ie", pa.float32()),
])

CHUNK_SIZE = 10          # examples between parquet flushes

_EDGE_DTYPES = {"example_idx": "int32", "src_ordinal": "int16", "src_feature": "int32",
                "dst_ordinal": "int16", "dst_feature": "int32", "edge_ie": "float32"}


def read_edge_hive(edge_dir) -> pd.DataFrame:
    """Read the edge hive with partition columns restored to their real dtypes.
    See src.attribution.nodes.read_node_hive for why the Categorical matters."""
    df = pd.read_parquet(Path(edge_dir) / "edges")
    return df.astype({c: t for c, t in _EDGE_DTYPES.items() if c in df.columns})


def _detach_site_hook(tensor, hook):
    """Stop-gradient at one site: a value-preserving identity in the forward
    pass that is a fresh autograd leaf, so nothing differentiates back through
    it. The TransformerLens equivalent of Marks' `submodule.stop_grad()`.

    requires_grad_() keeps the substituted tensor differentiable in its own
    right, so a graph whose only remaining path runs through this site still
    backwards cleanly (yielding zero) instead of raising.
    """
    return tensor.detach().requires_grad_()


def candidate_endpoints(nodes, src_ordinal: int, dst_ordinal: int,
                        filter_mode: str) -> tuple[list[int], list[int]]:
    """Which (upstream, downstream) features to score for this site pair.

    filter_mode="marks" reproduces attribution.py:jvp -- the downstream side is
    gated by node selection (one backward per surviving downstream node) while
    every upstream feature is kept, because the upstream contraction is free
    once that backward has run. filter_mode="both" also gates upstream: cheaper,
    strictly fewer edges, and NOT comparable to Marks' edge counts.
    """
    if filter_mode not in ("marks", "both"):
        raise ValueError(f"Unknown filter_mode {filter_mode!r}; expected 'marks' or 'both'")
    src = nodes[nodes["site_ordinal"] == src_ordinal]
    dst = nodes[nodes["site_ordinal"] == dst_ordinal]
    down = dst.loc[dst["selected"], "feature_idx"].tolist()
    up = (src.loc[src["selected"], "feature_idx"].tolist()
          if filter_mode == "both" else src["feature_idx"].tolist())
    return up, down


def node_pass(model, saes, sites, clean_tokens, patch_tokens, metric_fn, example):
    """One clean fwd+bwd and one patch fwd. Returns per-site-ordinal dicts:
      grad_f[o]   = dm/df                    [seq, d_sae]
      grad_res[o] = dm/dresidual             [seq, d_model]
      df[o]       = (f_patch - f_clean)      [seq, d_sae]
      dres[o]     = (res_patch - res_clean)  [seq, d_model]

    The metric carries its own sign (see src/metrics.py), so there is no
    direction multiplier to return.
    """
    acts_clean = {}

    def cap_grad(hook_name):
        # TL calls hooks as h(tensor, hook=hook_point): 2nd arg is keyword-named `hook`.
        def h(tensor, hook):
            tensor.retain_grad()
            acts_clean[hook_name] = tensor
            return tensor
        return h

    model.zero_grad(set_to_none=True)
    for site in sites:
        model.add_hook(site.hook, cap_grad(site.hook), dir="fwd")
    clean_logits = model(clean_tokens)
    metric_fn(logits=clean_logits, example=example, model=model,
              tokens=clean_tokens).backward()
    model.reset_hooks()

    acts_patch = {}

    def cap(hook_name):
        def h(tensor, hook):
            acts_patch[hook_name] = tensor
            return tensor
        return h

    for site in sites:
        model.add_hook(site.hook, cap(site.hook), dir="fwd")
    with torch.no_grad():
        model(patch_tokens)
    model.reset_hooks()

    grad_f, grad_res, df, dres = {}, {}, {}, {}
    for site in sites:
        sae = saes[site.sae_id]
        x_clean = acts_clean[site.hook][0]
        x_patch = acts_patch[site.hook][0]
        g_res = acts_clean[site.hook].grad[0]
        w_dec_t = sae.W_dec.T if sae.W_dec.shape[0] == sae.cfg.d_sae else sae.W_dec

        grad_res[site.ordinal] = g_res
        grad_f[site.ordinal] = g_res @ w_dec_t
        f_clean = sae.encode(x_clean.unsqueeze(0))[0]
        f_patch = sae.encode(x_patch.unsqueeze(0))[0]
        df[site.ordinal] = (f_patch - f_clean).detach()
        res_clean = x_clean - sae.decode(f_clean.unsqueeze(0))[0]
        res_patch = x_patch - sae.decode(f_patch.unsqueeze(0))[0]
        dres[site.ordinal] = (res_patch - res_clean).detach()

    for site in sites:
        acts_clean[site.hook].grad = None
    return grad_f, grad_res, df, dres


def edge_pass(model, saes, spec: EdgeSpec, clean_tokens,
              grad_f_j, grad_res_j, df_i, dres_i, up_feats, down_feats,
              include_error: bool):
    """Edges from every candidate upstream feature to every surviving downstream
    feature for one example, with spec.stop sites stop-gradiented.

    Returns a list of (src_feature, dst_feature, edge_ie).
    """
    store_i, store_j = {}, {}
    src_sae = saes[spec.src.sae_id]
    dst_sae = saes[spec.dst.sae_id]

    def sub_hook(tensor, hook):
        x = tensor[0]
        f_val = src_sae.encode(x.unsqueeze(0))[0]
        recon_val = src_sae.decode(f_val.unsqueeze(0))[0]
        f = f_val.detach().requires_grad_()
        res = (x - recon_val).detach().requires_grad_()
        store_i["f"], store_i["res"] = f, res
        recon = src_sae.decode(f.unsqueeze(0))[0]
        return (recon + res).unsqueeze(0)

    def cap_hook(tensor, hook):
        store_j["y"] = tensor[0]
        return tensor

    model.reset_hooks()
    model.add_hook(spec.src.hook, sub_hook, dir="fwd")
    for stop_site in spec.stop:
        model.add_hook(stop_site.hook, _detach_site_hook, dir="fwd")
    model.add_hook(spec.dst.hook, cap_hook, dir="fwd")
    model(clean_tokens)
    model.reset_hooks()

    y_j = store_j["y"]
    g_j = dst_sae.encode(y_j.unsqueeze(0))[0]
    yres_j = y_j - dst_sae.decode(g_j.unsqueeze(0))[0]
    f_i, res_i = store_i["f"], store_i["res"]

    feature_ups = [u for u in up_feats if u != ERROR_IDX]
    up_has_error = include_error and ERROR_IDX in up_feats
    # One tensor gather beats len(up_feats) scalar .item() calls per downstream
    # node: with filter_mode="marks" the upstream side is unfiltered, so that
    # loop ran into the thousands per backward.
    up_index = (torch.tensor(feature_ups, device=df_i.device, dtype=torch.long)
                if feature_ups else None)
    edges = []
    for d in down_feats:
        if d == ERROR_IDX:
            if not include_error:
                continue
            seed = (grad_res_j * yres_j).sum()
        else:
            seed = (grad_f_j[:, d] * g_j[:, d]).sum()
        gfi, gri = torch.autograd.grad(seed, [f_i, res_i], retain_graph=True,
                                       allow_unused=True)
        # (dm/dd)*(dd/du) is in gfi/gri; contract with the upstream delta and
        # sum positions.
        if up_index is not None and gfi is not None:
            up_vals = (gfi * df_i).sum(dim=0)[up_index]      # [n_up]
            nz = up_vals.nonzero(as_tuple=True)[0]
            if nz.numel():
                us = up_index[nz].tolist()
                vs = up_vals[nz].tolist()
                edges.extend((u, d, v) for u, v in zip(us, vs))
        if up_has_error and gri is not None:
            v = (gri * dres_i).sum().item()
            if v != 0.0:
                edges.append((ERROR_IDX, d, v))
    return edges


def _flush(batch: dict, out_dir: Path) -> int:
    n = len(batch["example_idx"])
    if n == 0:
        return 0
    pq.write_to_dataset(pa.Table.from_pydict(batch, schema=EDGE_SCHEMA),
                        root_path=str(out_dir / "edges"),
                        partition_cols=["dst_ordinal"], compression="zstd")
    for v in batch.values():
        v.clear()
    return n


def run_edge_attribution(cfg, selection_dir: Path, out_dir: Path) -> dict:
    """Score the configured edge template over the selected node set."""
    selection_dir, out_dir = Path(selection_dir), Path(out_dir)
    started = time.time()
    torch.manual_seed(cfg.seed)

    nodes = pd.read_parquet(selection_dir / "nodes.parquet")
    suite = build_suite(cfg.suite.name)
    template = build_template(suite, cfg.edges)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    examples = load_examples(cfg.data, model, cfg.seed)
    prepend_bos = getattr(next(iter(saes.values())).cfg, "prepend_bos", True)

    batch = {name: [] for name in EDGE_SCHEMA.names}
    total = 0

    for ex_idx, example in enumerate(examples):
        clean_tokens = model.to_tokens(example.clean_prefix, prepend_bos=prepend_bos
                                       ).to(cfg.suite.device)
        patch_tokens = model.to_tokens(example.patch_prefix, prepend_bos=prepend_bos
                                       ).to(cfg.suite.device)
        if patch_tokens.shape[1] != clean_tokens.shape[1]:
            logger.warning("Example %d: token-length mismatch; skipping.", ex_idx)
            continue

        grad_f, grad_res, df, dres = node_pass(
            model, saes, suite.sites, clean_tokens, patch_tokens, metric_fn, example)

        for spec in template:
            if spec.dst.ordinal == Y_SITE.ordinal:
                # circuit.py assigns edges[resid_last]["y"] = effects[resid_last]
                # directly: the edge into the metric IS the node's own IE, so
                # there is no jvp to run.
                up, _ = candidate_endpoints(nodes, spec.src.ordinal, spec.src.ordinal,
                                            cfg.edges.filter_mode)
                ie = (grad_f[spec.src.ordinal] * df[spec.src.ordinal]).sum(dim=0)
                found = []
                for u in up:
                    if u == ERROR_IDX:
                        if not cfg.edges.include_error:
                            continue
                        v = (grad_res[spec.src.ordinal] * dres[spec.src.ordinal]).sum().item()
                    else:
                        v = ie[u].item()
                    if v != 0.0:
                        found.append((u, ERROR_IDX, v))
            else:
                up, down = candidate_endpoints(nodes, spec.src.ordinal,
                                               spec.dst.ordinal, cfg.edges.filter_mode)
                if not down:
                    continue
                found = edge_pass(
                    model, saes, spec, clean_tokens,
                    grad_f[spec.dst.ordinal], grad_res[spec.dst.ordinal],
                    df[spec.src.ordinal], dres[spec.src.ordinal],
                    up, down, cfg.edges.include_error)

            for u, d, v in found:
                batch["example_idx"].append(ex_idx)
                batch["src_ordinal"].append(spec.src.ordinal)
                batch["src_feature"].append(u)
                batch["dst_ordinal"].append(spec.dst.ordinal)
                batch["dst_feature"].append(d)
                batch["edge_ie"].append(v)
            total += len(found)

        if (ex_idx + 1) % CHUNK_SIZE == 0:
            _flush(batch, out_dir)

    _flush(batch, out_dir)
    logger.info("Edge attribution: %d edges scored over %d specs", total, len(template))
    return {
        "n_edges_scored": total,
        "n_specs": len(template),
        "runtime_s": round(time.time() - started, 2),
    }
