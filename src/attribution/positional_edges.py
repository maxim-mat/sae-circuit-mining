"""Position-resolved edge attribution, stored once per filtering method.

For one example, an upstream node u = (site i, position p, feature) and a
downstream node d = (site j, position q, feature) get

    IE(u -> d) = left[q, d] * (dd/du at the clean input, intermediates frozen) * (u_patch - u_clean)

which is Marks et al.'s `jvp` (arXiv:2403.19647) with the gradients their
`_pe_ig` returns: `left` is dm/dd averaged over the integrated-gradients path
that moves site j alone (src.attribution.ig), the Jacobian is taken on the
clean input. One backward per downstream instance yields its edges from every
upstream (position, feature) and the upstream error node at once. Which site
pairs are scored, with which stop-gradients, is edge_template.marks_template.
Edges into the metric are the last residual site's node IEs, as in Marks'
circuit.py.

Nothing is aggregated or thresholded here: every non-zero edge of every computed
destination is stored per example and position. Which destinations get a
backward pass is the filtering method's business (src.selection.destinations).
Edge values do not depend on the method, so one pass over the union of the
methods' destinations serves all of them; each method still gets its own store,
outputs/<dataset>/attributions/edges/<counterfactual>/<method>/, whose manifest
records the method and the node run it read.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from src.attribution.edge_template import Y_SITE, marks_template
from src.attribution.edges import _detach_site_hook
from src.attribution.ig import integrated_gradient_grad_x
from src.data.loaders import load_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes, prepend_bos_for
from src.selection.destinations import destination_masks
from src.utils.config import Config
from src.utils.manifest import SCHEMA_VERSION, _changed_params, write_manifest

logger = logging.getLogger(__name__)

STAGE = "positional_edges"
ERROR_IDX = -1          # SAE error node, as in the node hive
CHUNK_SIZE = 10         # examples between parquet flushes
LOG_EVERY = 100         # examples between progress lines

EDGE_SCHEMA = pa.schema([
    ("example_idx", pa.int32()),
    ("src_ordinal", pa.int16()),
    ("src_token", pa.int16()),
    ("src_feature", pa.int32()),
    # dst_ordinal = -1 is the metric node; its token and feature are -1 too.
    ("dst_ordinal", pa.int16()),
    ("dst_token", pa.int16()),
    ("dst_feature", pa.int32()),
    ("edge_ie", pa.float32()),
])
_DTYPES = {f.name: f.type.to_pandas_dtype() for f in EDGE_SCHEMA}


def edge_dir(node_dir, method: str) -> Path:
    """outputs/<dataset>/attributions/nodes/<cf> -> .../edges/<cf>/<method>."""
    node_dir = Path(node_dir)
    return node_dir.parent.parent / "edges" / node_dir.name / method


def read_edges(out_dir) -> pd.DataFrame:
    """One method's edge store, with the partition column back to its dtype."""
    return pd.read_parquet(Path(out_dir) / "edges").astype(_DTYPES)


def _w_dec_t(sae):
    return sae.W_dec.T if sae.W_dec.shape[0] == sae.cfg.d_sae else sae.W_dec


def _cache(model, sites, tokens) -> dict:
    acts = {}

    def cap(name):
        def hook_fn(tensor, hook):
            acts[name] = tensor[0].detach()
            return tensor
        return hook_fn

    for site in sites:
        model.add_hook(site.hook, cap(site.hook), dir="fwd")
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        model.reset_hooks()
    return acts


def _contract(grad_f, grad_res, delta_f, delta_res):
    """(positions, features, values) of every non-zero upstream edge; feature -1 is
    the upstream error node."""
    out = []
    if grad_f is not None:
        vals = grad_f * delta_f
        p, u = vals.nonzero(as_tuple=True)
        out.append((p, u, vals[p, u]))
    if grad_res is not None:
        vals = (grad_res * delta_res).sum(dim=-1)
        p = vals.nonzero(as_tuple=True)[0]
        out.append((p, torch.full_like(p, ERROR_IDX), vals[p]))
    return out


def _spec_edges(model, saes, spec, clean_tokens, grad_x, delta_f, delta_res, dests):
    """Edges into each (position, feature) in `dests` at spec.dst from spec.src."""
    src_sae, dst_sae = saes[spec.src.sae_id], saes[spec.dst.sae_id]
    store = {}

    def sub_hook(tensor, hook):
        # Identity in value; re-routes the graph through leaf features and error
        # so d(dst)/d(feature) exists. Marks: set_activation(x_hat + res).
        x = tensor[0]
        f_val = src_sae.encode(x.unsqueeze(0))[0]
        f = f_val.detach().requires_grad_()
        res = (x - src_sae.decode(f_val.unsqueeze(0))[0]).detach().requires_grad_()
        store["f"], store["res"] = f, res
        return (src_sae.decode(f.unsqueeze(0))[0] + res).unsqueeze(0)

    def cap_hook(tensor, hook):
        store["y"] = tensor[0]
        return tensor

    model.reset_hooks()
    model.add_hook(spec.src.hook, sub_hook, dir="fwd")
    for stop_site in spec.stop:
        model.add_hook(stop_site.hook, _detach_site_hook, dir="fwd")
    model.add_hook(spec.dst.hook, cap_hook, dir="fwd")
    try:
        model(clean_tokens)
    finally:
        model.reset_hooks()

    y = store["y"]
    g = dst_sae.encode(y.unsqueeze(0))[0]
    y_res = y - dst_sae.decode(g.unsqueeze(0))[0]
    left_f = grad_x @ _w_dec_t(dst_sae)
    rows = []
    # ponytail: one backward per destination; batch them (is_grads_batched) if slow.
    for q, d in dests:
        seed = (grad_x[q] * y_res[q]).sum() if d == ERROR_IDX else left_f[q, d] * g[q, d]
        if not seed.requires_grad:              # no path from src to this destination
            continue
        grad_f, grad_res = torch.autograd.grad(seed, [store["f"], store["res"]],
                                               retain_graph=True, allow_unused=True)
        for p, u, v in _contract(grad_f, grad_res, delta_f, delta_res):
            rows.append((spec.src.ordinal, p, u, spec.dst.ordinal, q, d, v))
    return rows


def example_edges(model, saes, specs, clean_tokens, patch_tokens, metric_fn, example,
                  destinations: dict, ig_steps: int) -> dict:
    """Every non-zero edge of one example, as numpy columns of EDGE_SCHEMA (minus
    example_idx).

    destinations: {dst site ordinal: [(position, feature), ...]}; edges into the
    metric are always computed.
    """
    sites = {s.ordinal: s for spec in specs for s in (spec.src, spec.dst)
             if s.ordinal != Y_SITE.ordinal}
    clean = _cache(model, sites.values(), clean_tokens)
    patch = _cache(model, sites.values(), patch_tokens)

    delta_f, delta_res = {}, {}
    for o, site in sites.items():
        sae = saes[site.sae_id]
        xc, xp = clean[site.hook], patch[site.hook]
        fc, fp = sae.encode(xc.unsqueeze(0))[0], sae.encode(xp.unsqueeze(0))[0]
        delta_f[o] = (fp - fc).detach()
        delta_res[o] = ((xp - sae.decode(fp.unsqueeze(0))[0])
                        - (xc - sae.decode(fc.unsqueeze(0))[0])).detach()

    left = {}

    def ig_grad(site):
        if site.ordinal not in left:
            left[site.ordinal] = integrated_gradient_grad_x(
                model, clean_tokens, site.hook, clean[site.hook], patch[site.hook],
                metric_fn, example, ig_steps).detach()
        return left[site.ordinal]

    rows = []
    for spec in specs:
        o = spec.src.ordinal
        if spec.dst.ordinal == Y_SITE.ordinal:
            grad_x = ig_grad(spec.src)
            grad_f = grad_x @ _w_dec_t(saes[spec.src.sae_id])
            for p, u, v in _contract(grad_f, grad_x, delta_f[o], delta_res[o]):
                rows.append((o, p, u, Y_SITE.ordinal, ERROR_IDX, ERROR_IDX, v))
        elif destinations.get(spec.dst.ordinal):
            rows += _spec_edges(model, saes, spec, clean_tokens, ig_grad(spec.dst),
                                delta_f[o], delta_res[o], destinations[spec.dst.ordinal])

    names = [n for n in EDGE_SCHEMA.names if n != "example_idx"]
    if not rows:
        return {n: np.empty(0, dtype=_DTYPES[n]) for n in names}
    cols = {n: [] for n in names}
    for src_o, p, u, dst_o, q, d, v in rows:
        k = len(p)
        cols["src_ordinal"].append(np.full(k, src_o))
        cols["src_token"].append(p.cpu().numpy())
        cols["src_feature"].append(u.cpu().numpy())
        cols["dst_ordinal"].append(np.full(k, dst_o))
        cols["dst_token"].append(np.full(k, q))
        cols["dst_feature"].append(np.full(k, d))
        cols["edge_ie"].append(v.detach().float().cpu().numpy())
    return {n: np.concatenate(cols[n]).astype(_DTYPES[n]) for n in names}


def _dst_keys(site, token, feature) -> np.ndarray:
    return ((np.asarray(site, np.int64) << 40) | ((np.asarray(token, np.int64) + 1) << 24)
            | (np.asarray(feature, np.int64) + 1))


def _read_node_rows(node_dir: Path) -> pd.DataFrame:
    hive = ds.dataset(node_dir / "nodes", format="parquet", partitioning="hive")
    cols = ["example_idx", "site_ordinal", "token_idx", "token_id", "feature_idx", "attribution"]
    return hive.to_table(columns=cols).to_pandas().astype(
        {"example_idx": "int32", "site_ordinal": "int16", "token_idx": "int16",
         "token_id": "int32", "feature_idx": "int32", "attribution": "float32"})


def _check_prompts(node_dir: Path, raw: pd.DataFrame, tokens: dict) -> None:
    """Refuse unless example i here is prompt i of the node run.

    The node hive records the clean token at every position it wrote, so each
    example's clean tokens must match them, and the examples the node run kept
    must be exactly the ones whose clean and patch lengths agree here.
    """
    recorded = raw[["example_idx", "token_idx", "token_id"]].drop_duplicates()
    if set(recorded["example_idx"].unique()) != set(tokens):
        raise RuntimeError(f"The node run at {node_dir} kept different examples than the prompts "
                           f"loaded now; its config no longer reproduces its prompts.")
    for ex_idx, rows in recorded.groupby("example_idx", sort=False):
        clean = tokens[ex_idx][0][0].cpu().numpy()
        pos, ids = rows["token_idx"].to_numpy(), rows["token_id"].to_numpy()
        if pos.max() >= clean.shape[0] or (clean[pos] != ids).any():
            raise RuntimeError(f"Example {ex_idx}: prompt differs from the one the node run at "
                               f"{node_dir} attributed; refusing to mix them.")


def _normalised(params: dict) -> dict:
    """As it reads back from manifest.json, so comparisons are like for like."""
    return json.loads(json.dumps(params, default=str))


def _flush(batches: dict, outs: dict) -> None:
    for name, parts in batches.items():
        if not parts:
            continue
        cols = {n: np.concatenate([p[n] for p in parts]) for n in EDGE_SCHEMA.names}
        if len(cols["example_idx"]):
            pq.write_to_dataset(pa.Table.from_pydict(cols, schema=EDGE_SCHEMA),
                                root_path=str(outs[name] / "edges"),
                                partition_cols=["dst_ordinal"], compression="zstd")
        parts.clear()


def run_positional_edges(node_dir, methods: dict[str, dict], force: bool = False,
                         device: str | None = None) -> dict[str, Path]:
    """Compute and store the edges of every method in `methods` for one finished
    node run. Returns {method: output dir} for the methods computed now.

    A method whose store already holds a finished run with the same parameters is
    skipped; one with different parameters is refused unless `force`.
    """
    started = time.time()
    node_dir = Path(node_dir)
    node_manifest = json.loads((node_dir / "manifest.json").read_text())
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config),
                                              node_manifest["params"]))
    if device:
        cfg.suite.device = device
    node_ref = {"path": node_dir.as_posix(), "params": node_manifest["params"],
                "git_commit": node_manifest.get("git_commit"),
                "created_utc": node_manifest.get("created_utc")}

    todo = {}
    for name, spec in methods.items():
        out = edge_dir(node_dir, name)
        params = _normalised({"node_run": node_ref, "method": spec,
                              "edges": {"template": "marks", "ig_steps": cfg.attribution.ig_steps}})
        manifest = out / "manifest.json"
        if manifest.is_file() and not force:
            old = json.loads(manifest.read_text())
            if old.get("params") == params and old.get("schema_version") == SCHEMA_VERSION:
                logger.info("%s: finished with these parameters; skipping", out)
                continue
            changed = "; ".join(_changed_params(old.get("params", {}), params)) or "schema version"
            raise RuntimeError(f"{out} already holds a finished run with different parameters "
                               f"({changed}). Pass force to overwrite it.")
        todo[name] = (out, params)
    if not todo:
        return {}

    torch.manual_seed(cfg.seed)                 # as the node stage does
    suite = build_suite(cfg.suite.name)
    specs = marks_template(suite)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    examples = load_examples(cfg.data, model, cfg.seed)    # same data config and seed
    expected = node_manifest["n_examples"] + node_manifest.get("n_skipped", 0)
    if len(examples) != expected:
        raise RuntimeError(f"Loaded {len(examples)} examples but the node run at {node_dir} had "
                           f"{expected}; example indices would not line up.")
    prepend_bos = prepend_bos_for(saes)
    tokens = {}
    for ex_idx, example in enumerate(examples):
        clean = model.to_tokens(example.clean_prefix, prepend_bos=prepend_bos).to(cfg.suite.device)
        patch = model.to_tokens(example.patch_prefix, prepend_bos=prepend_bos).to(cfg.suite.device)
        if patch.shape[1] == clean.shape[1]:
            tokens[ex_idx] = (clean, patch)
        else:                                   # the node run skipped it too (checked below)
            logger.warning("Example %d: token-length mismatch (clean=%d, patch=%d); skipping.",
                           ex_idx, clean.shape[1], patch.shape[1])
    n_skipped = len(examples) - len(tokens)

    raw = _read_node_rows(node_dir)
    _check_prompts(node_dir, raw, tokens)
    n_examples = raw["example_idx"].nunique()
    masks = destination_masks(raw, {n: methods[n] for n in todo}, n_examples)
    union = masks.any(axis=1).to_numpy()
    dests = raw.loc[union, ["example_idx", "site_ordinal", "token_idx", "feature_idx"]].join(masks[union])
    del raw
    dests["key"] = _dst_keys(dests["site_ordinal"], dests["token_idx"], dests["feature_idx"])
    by_example = dict(tuple(dests.groupby("example_idx", sort=False)))
    n_destinations = {name: int(masks[name].sum()) for name in todo}
    del masks

    outs = {name: out for name, (out, _) in todo.items()}
    for out in outs.values():
        if out.exists():
            shutil.rmtree(out)                  # half-written hives would duplicate rows
        out.mkdir(parents=True)
    batches = {name: [] for name in todo}
    n_edges = dict.fromkeys(todo, 0)

    for ex_idx, example in enumerate(examples):
        if ex_idx not in tokens:
            continue
        clean_tokens, patch_tokens = tokens[ex_idx]
        mine = by_example.get(ex_idx)
        destinations = ({} if mine is None else
                        {int(o): list(zip(g["token_idx"].tolist(), g["feature_idx"].tolist()))
                         for o, g in mine.groupby("site_ordinal", sort=False)})
        cols = example_edges(model, saes, specs, clean_tokens, patch_tokens, metric_fn,
                             example, destinations, cfg.attribution.ig_steps)
        cols["example_idx"] = np.full(len(cols["edge_ie"]), ex_idx, dtype=np.int32)
        edge_keys = _dst_keys(cols["dst_ordinal"], cols["dst_token"], cols["dst_feature"])
        into_metric = cols["dst_ordinal"] == Y_SITE.ordinal
        for name in todo:
            wanted = mine.loc[mine[name], "key"].to_numpy() if mine is not None else []
            keep = into_metric | np.isin(edge_keys, wanted)
            batches[name].append({n: v[keep] for n, v in cols.items()})
            n_edges[name] += int(keep.sum())

        if (ex_idx + 1) % CHUNK_SIZE == 0:
            _flush(batches, outs)
        if (ex_idx + 1) % LOG_EVERY == 0:
            elapsed = time.time() - started
            logger.info("%s: %d/%d examples, %.2f s/example", node_dir, ex_idx + 1, len(examples),
                        elapsed / (ex_idx + 1))
    _flush(batches, outs)

    runtime = round(time.time() - started, 2)
    for name, (out, params) in todo.items():
        write_manifest(out, STAGE, params, {"n_edges": n_edges[name],
                                            "n_destinations": n_destinations[name],
                                            "n_examples": len(tokens), "n_skipped": n_skipped,
                                            "runtime_s": runtime})
        logger.info("%s: %d edges into %d destinations over %d examples (%d skipped)",
                    out, n_edges[name], n_destinations[name], len(tokens), n_skipped)
    return outs
