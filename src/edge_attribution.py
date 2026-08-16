"""Compute SAE feature-edge indirect effects, on the fly, over a set of
already-selected nodes.

Second pass of the pipeline: run this AFTER node selection (Part 1) has pruned
the feature alphabet — edges are only computed between surviving nodes, which is
where the ~90% cost saving comes from. This script does NOT care how nodes
survived; it just reads a node set.

Method — stop-gradient edge attribution from Marks et al. 2024, "Sparse Feature
Circuits" (arXiv:2403.19647), their `jvp`. For an upstream feature u (layer i)
and a downstream feature d (layer j > i):

    IE(u -> d) = direction * (dm/dd) * (dd/du) * (f_cf - f_clean)_u

  * dm/dd  (`grad_f` at j) comes from a node pass (metric backward at the clean
    point), exactly as in main.py — decode is affine so grad_f = grad_resid @ W_dec.T.
  * dd/du is the Jacobian of the downstream feature w.r.t. the upstream feature
    through the frozen network. We get (dm/dd)*(dd/du) in one backward by seeding
    autograd with (grad_f[d] * g[d]).sum() and differentiating w.r.t. a leaf
    upstream feature tensor, then contracting with the upstream delta.

The upstream activation at layer i is replaced by decode(f_i) + res_i with f_i,
res_i as leaves (an identity substitution — value unchanged — that re-routes the
graph through the feature basis so dd/du is defined). This mirrors Marks'
`set_activation(x_hat + res)`.

Scope of this version:
  * Edges for *every* upstream<downstream surviving-layer pair by default
    (`--pairs consecutive` restricts to adjacent surviving layers — the
    directly-follows / DFG view). Non-adjacent edges here are TOTAL-effect edges:
    no SAE reconstruction is inserted at intermediate layers, so dd/du flows
    through the frozen network end to end. This deliberately differs from Marks'
    stop-gradient DIRECT edges — it matches the process-mining "eventually
    follows" relation, where an edge is the whole causal effect of u on d
    (transitive and non-exclusive; do NOT naively sum direct and total edges).
  * Positions are summed (downstream positions in the seed, upstream positions in
    the contraction), matching Marks' aggregation into a task-level edge. Keeping
    token roles is a future extension.

Self-check: python src/test_edge_attribution.py
"""

import argparse
import logging
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pds
import torch
from datasets import load_dataset
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from transformer_lens import HookedTransformer

from metrics import get_metric_function

logger = logging.getLogger("sae_edge_attribution")

ERROR_IDX = -1  # SAE reconstruction-error node, same convention as main.py.


def parse_args():
    p = argparse.ArgumentParser(description="SAE feature-edge indirect effects over a selected node set.")
    p.add_argument("--release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset", type=str, required=True,
                   help="HuggingFace dataset path (must carry counterfactual columns, e.g. mib-bench/ioi).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="prompt")
    p.add_argument("--num_prompts", type=int, default=None)
    p.add_argument("--nodes", type=str, required=True,
                   help="Parquet with the surviving node set: columns counterfactual, layer, feature_idx "
                        "(e.g. a Part-1 selection or an event_log output).")
    p.add_argument("--output_dir", type=str, default="sae_edge_attributions")
    p.add_argument("--chunk_size", type=int, default=10)
    p.add_argument("--pairs", choices=["all", "consecutive"], default="all",
                   help="Which layer pairs get edges. 'all' (default) = every upstream<downstream "
                        "surviving-layer pair (eventually-follows / total-effect edges); 'consecutive' "
                        "= only adjacent surviving layers (directly-follows / DFG).")
    p.add_argument("--include_error", action="store_true",
                   help="Include the reconstruction-error node (feature_idx=-1) as a valid edge endpoint.")
    return p.parse_args()


def load_node_set(path, include_error):
    """Read the surviving node set into {counterfactual: {layer: sorted[feature_idx]}}."""
    df = pds.dataset(path, partitioning="hive").to_table(
        columns=["counterfactual", "layer", "feature_idx"]
    ).to_pandas()
    df["counterfactual"] = df["counterfactual"].astype(str)
    if not include_error:
        df = df[df["feature_idx"] != ERROR_IDX]
    nodes = {}
    for (cf, layer), grp in df.groupby(["counterfactual", "layer"]):
        nodes.setdefault(str(cf), {})[int(layer)] = sorted(set(int(f) for f in grp["feature_idx"]))
    return nodes


def load_model_and_saes(release, device):
    """Load gpt2-small and its per-layer residual SAEs. Returns model plus
    dicts keyed by integer layer: sae, hook_point, W_dec_T (d_model x d_sae)."""
    release_saes = get_pretrained_saes_directory()[release].saes_map
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sorted_items = sorted(release_saes.items(),
                          key=lambda x: int(x[0].split(".")[1]) if "." in x[0] else 0)
    sae, hook_points, w_dec_t = {}, {}, {}
    for sid, hook_point in sorted_items:
        layer = int(sid.split(".")[1])
        s = SAE.from_pretrained(release=release, sae_id=sid, device=device)[0]
        sae[layer] = s
        hook_points[layer] = hook_point
        w_dec_t[layer] = s.W_dec.T if s.W_dec.shape[0] == s.cfg.d_sae else s.W_dec
    prepend_bos = getattr(list(sae.values())[0].cfg, "prepend_bos", True)
    return model, sae, hook_points, w_dec_t, prepend_bos


def node_pass(model, sae, hook_points, w_dec_t, layers, clean_tokens, cf_tokens, metric_fn, item):
    """One clean fwd+bwd and one CF fwd. Returns per-layer dicts:
      grad_f[layer]  = dm/df at j  (direction NOT applied)  [seq, d_sae]
      grad_res[layer]= dm/dresidual                          [seq, d_model]
      df[layer]      = (f_cf - f_clean)                      [seq, d_sae]
      dres[layer]    = (res_cf - res_clean)                  [seq, d_model]
    and the metric direction (+/-1).
    """
    resid_clean = {}

    def cap_grad(layer):
        # TL calls hooks as h(tensor, hook=hook_point): 2nd arg is keyword-named `hook`.
        def h(tensor, hook):
            tensor.retain_grad()
            resid_clean[layer] = tensor
            return tensor
        return h

    model.zero_grad(set_to_none=True)
    for L in layers:
        model.add_hook(hook_points[L], cap_grad(L), dir="fwd")
    clean_logits = model(clean_tokens)
    metric_tensor, direction = metric_fn(logits=clean_logits, tokens=clean_tokens, item=item, model=model)
    metric_tensor.backward()
    model.reset_hooks()

    resid_cf = {}
    for L in layers:
        model.add_hook(hook_points[L], (lambda layer: (lambda t, hook: resid_cf.__setitem__(layer, t) or t))(L), dir="fwd")
    with torch.no_grad():
        model(cf_tokens)
    model.reset_hooks()

    grad_f, grad_res, df, dres = {}, {}, {}, {}
    for L in layers:
        s = sae[L]
        x_clean = resid_clean[L][0]
        x_cf = resid_cf[L][0]
        g_res = resid_clean[L].grad[0]
        grad_res[L] = g_res
        grad_f[L] = g_res @ w_dec_t[L]
        f_clean = s.encode(x_clean.unsqueeze(0))[0]
        f_cf = s.encode(x_cf.unsqueeze(0))[0]
        df[L] = (f_cf - f_clean).detach()
        res_clean = x_clean - s.decode(f_clean.unsqueeze(0))[0]
        res_cf = x_cf - s.decode(f_cf.unsqueeze(0))[0]
        dres[L] = (res_cf - res_clean).detach()

    for L in layers:
        resid_clean[L].grad = None
    return grad_f, grad_res, df, dres, direction


def edge_pass(model, sae, hook_points, clean_tokens, up_L, down_L,
              grad_f_j, grad_res_j, df_i, dres_i, survivors_i, survivors_j,
              direction, include_error):
    """Edges from every surviving node at up_L to every surviving node at down_L,
    for one prompt. Returns list of (up_feature, down_feature, attribution)."""
    store_i, store_j = {}, {}

    def sub_hook(tensor, hook):
        x = tensor[0]
        f_val = sae[up_L].encode(x.unsqueeze(0))[0]
        recon_val = sae[up_L].decode(f_val.unsqueeze(0))[0]
        f = f_val.detach().requires_grad_()
        res = (x - recon_val).detach().requires_grad_()
        store_i["f"], store_i["res"] = f, res
        recon = sae[up_L].decode(f.unsqueeze(0))[0]
        return (recon + res).unsqueeze(0)

    def cap_hook(tensor, hook):
        store_j["y"] = tensor[0]
        return tensor

    model.reset_hooks()
    model.add_hook(hook_points[up_L], sub_hook, dir="fwd")
    model.add_hook(hook_points[down_L], cap_hook, dir="fwd")
    model(clean_tokens)
    model.reset_hooks()

    y_j = store_j["y"]
    g_j = sae[down_L].encode(y_j.unsqueeze(0))[0]
    yres_j = y_j - sae[down_L].decode(g_j.unsqueeze(0))[0]
    f_i, res_i = store_i["f"], store_i["res"]

    up_feats = [u for u in survivors_i if u != ERROR_IDX]
    up_has_error = include_error and ERROR_IDX in survivors_i
    edges = []
    for d in survivors_j:
        if d == ERROR_IDX:
            if not include_error:
                continue
            seed = (grad_res_j * yres_j).sum()
        else:
            seed = (grad_f_j[:, d] * g_j[:, d]).sum()
        gfi, gri = torch.autograd.grad(seed, [f_i, res_i], retain_graph=True, allow_unused=True)
        # (dm/dd)*(dd/du) is in gfi/gri; contract with the upstream delta, sum positions.
        if up_feats:
            up_vals = direction * (gfi * df_i).sum(dim=0)  # [d_sae]
            for u in up_feats:
                v = up_vals[u].item()
                if v != 0.0:
                    edges.append((u, d, v))
        if up_has_error and gri is not None:
            v = (direction * (gri * dres_i).sum()).item()
            if v != 0.0:
                edges.append((ERROR_IDX, d, v))
    return edges


def _layer_pairs(layers, all_pairs):
    layers = sorted(layers)
    if all_pairs:
        return [(a, b) for k, a in enumerate(layers) for b in layers[k + 1:]]
    return list(zip(layers, layers[1:]))  # consecutive surviving layers


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.info(f"Parameters: {vars(args)}")

    node_set = load_node_set(args.nodes, args.include_error)
    logger.info(f"Loaded node set for counterfactuals: {list(node_set)}")

    dataset = load_dataset(path=args.dataset, split=args.split, streaming=False)
    metric_fn = get_metric_function(args.dataset)
    model, sae, hook_points, w_dec_t, prepend_bos = load_model_and_saes(args.release, args.device)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    schema = pa.schema([
        ("counterfactual", pa.string()),
        ("prompt_idx", pa.int32()),
        ("template_id", pa.int16()),
        ("up_layer", pa.int16()),
        ("up_feature", pa.int32()),
        ("down_layer", pa.int16()),
        ("down_feature", pa.int32()),
        ("attribution", pa.float32()),
    ])
    batch = {k: [] for k in schema.names}
    template_ids = {}
    total = 0

    def flush():
        if not batch["prompt_idx"]:
            return
        logger.info(f"Flushing {len(batch['prompt_idx'])} edges...")
        pq.write_to_dataset(pa.Table.from_pydict(batch, schema=schema),
                            root_path=args.output_dir,
                            partition_cols=["counterfactual", "down_layer"],
                            compression="zstd")
        for v in batch.values():
            v.clear()

    for p_idx, row in enumerate(dataset):
        if args.num_prompts is not None and p_idx >= args.num_prompts:
            break
        counterfactuals = {k: v for k, v in row.items() if isinstance(v, dict) and "prompt" in v}
        if not counterfactuals:
            continue

        template_id = template_ids.setdefault(row.get("template", ""), len(template_ids))
        clean_tokens = model.to_tokens(row[args.text_column], prepend_bos=prepend_bos).to(args.device)

        for cf_name, cf_item in counterfactuals.items():
            layer_survivors = node_set.get(cf_name)
            if not layer_survivors:
                continue
            cf_tokens = model.to_tokens(cf_item["prompt"], prepend_bos=prepend_bos).to(args.device)
            if cf_tokens.shape[1] != clean_tokens.shape[1]:
                logger.warning(f"Prompt {p_idx} cf '{cf_name}': token-length mismatch; skipping.")
                continue

            layers = sorted(layer_survivors)
            grad_f, grad_res, df, dres, direction = node_pass(
                model, sae, hook_points, w_dec_t, layers, clean_tokens, cf_tokens, metric_fn, row
            )

            for up_L, down_L in _layer_pairs(layers, args.pairs == "all"):
                edges = edge_pass(
                    model, sae, hook_points, clean_tokens, up_L, down_L,
                    grad_f[down_L], grad_res[down_L], df[up_L], dres[up_L],
                    layer_survivors[up_L], layer_survivors[down_L], direction, args.include_error,
                )
                for u, d, v in edges:
                    batch["counterfactual"].append(cf_name)
                    batch["prompt_idx"].append(p_idx)
                    batch["template_id"].append(template_id)
                    batch["up_layer"].append(up_L)
                    batch["up_feature"].append(u)
                    batch["down_layer"].append(down_L)
                    batch["down_feature"].append(d)
                    batch["attribution"].append(v)
                total += len(edges)

        if (p_idx + 1) % args.chunk_size == 0:
            flush()

    flush()
    logger.info(f"Edges written: {total}. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
