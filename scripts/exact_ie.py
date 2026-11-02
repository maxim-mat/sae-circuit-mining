"""Exact IE by real patching: every single node (--size 1) or every pair (--size 2).

For each example, list the nodes whose patch value differs from their clean
value (src.attribution.nodes.moving_nodes; all others are dropped entirely),
then run each set of --size of them as one real intervention on the clean
prompt, do(node = its patch value), re-read from the recomputed activation
(exact_set_ies). Batching only puts independent sets side by side.

Output, under --out:
  metadata.json                  config, git commit, args, environment
  ex{idx:05d}/nodes.parquet      node id = row order (site, token, feature, f_clean, f_patch)
  ex{idx:05d}/singles.parquet    node, ie (float32)                     [--size 1]
  ex{idx:05d}/pairs.parquet      i, j, ie (float16), reachable          [--size 2]
Delta(i, j) = IE(i, j) - IE(i) - IE(j) is a join of the two runs' outputs.
An example whose output file already exists is skipped, so a killed run resumes.

Usage:
    python -m scripts.exact_ie --size 1
    python -m scripts.exact_ie --size 2 --batch-size 16384
    python -m scripts.exact_ie --size 2 --n-examples 1 --out outputs/exact_ie/timing
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.attribution.nodes import exact_set_ies, moving_nodes, pair_reachable, reachable_pairs
from src.data.loaders import load_examples
from src.metrics import get_metric
from src.models.suites import build_suite, load_model_and_saes, prepend_bos_for
from src.utils.initialization import load_config
from src.utils.manifest import _git_commit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/pythia_marks_rc.yaml"))
    p.add_argument("--size", type=int, choices=(1, 2), required=True)
    p.add_argument("--n-examples", type=int, default=1000)
    # Halved automatically on CUDA OOM, so start high to fill VRAM. Each row
    # holds full logits (seq x d_vocab fp32, ~1.2 MB on Pythia), which is what
    # bounds it.
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--out", type=Path, default=None,
                   help="Default: outputs/exact_ie/size{size}")
    p.add_argument("overrides", nargs="*", help="Config dotlist overrides")
    return p.parse_args()


def _write(table: pa.Table, path: Path) -> None:
    """Write then rename, so a run killed mid-write leaves no file that the
    resume check would mistake for a finished example."""
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    out = args.out or Path(f"outputs/exact_ie/size{args.size}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config, [f"data.num_examples={args.n_examples}", *args.overrides])
    torch.manual_seed(cfg.seed)
    suite = build_suite(cfg.suite.name)
    model, saes = load_model_and_saes(suite, cfg.suite.device, cfg.suite.dtype)
    metric_fn = get_metric(cfg.attribution.metric)
    examples = load_examples(cfg.data, model, cfg.seed)
    if not examples:
        raise ValueError(f"No examples survived loading from {cfg.data.path!r}")
    if len(examples) < args.n_examples:
        logger.warning("Asked for %d examples, %d survived loading (use_min_length_only=%s)",
                       args.n_examples, len(examples), cfg.data.use_min_length_only)
    prepend_bos = prepend_bos_for(saes)

    (out / "metadata.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "config": dataclasses.asdict(cfg),
        "n_examples_loaded": len(examples),
        "prepend_bos": prepend_bos,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "python": sys.version,
        "device": (torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"),
    }, indent=2, default=str))

    reach_by_len: dict[int, set] = {}
    for ex_idx, example in enumerate(examples):
        dest = out / f"ex{ex_idx:05d}"
        target = dest / ("singles.parquet" if args.size == 1 else "pairs.parquet")
        if target.exists():
            continue
        clean = model.to_tokens(example.clean_prefix, prepend_bos=prepend_bos).to(cfg.suite.device)
        patch = model.to_tokens(example.patch_prefix, prepend_bos=prepend_bos).to(cfg.suite.device)
        if clean.shape[1] != patch.shape[1]:
            logger.warning("Example %d: clean/patch lengths differ (%d vs %d); skipped",
                           ex_idx, clean.shape[1], patch.shape[1])
            continue

        started = time.time()
        dest.mkdir(exist_ok=True)
        df = moving_nodes(model, saes, suite, clean, patch)
        _write(pa.Table.from_pandas(df, preserve_index=False), dest / "nodes.parquet")
        nodes = list(zip(df.site_ordinal.tolist(), df.token_idx.tolist(), df.feature_idx.tolist()))
        n = len(nodes)

        if args.size == 1:
            sets = torch.arange(n).unsqueeze(1)
        else:
            # ponytail: all pairs materialised at once (~1.5 GB int64 at 13.5k
            # nodes); stream rows of the triangle if host RAM gets tight.
            sets = torch.triu_indices(n, n, offset=1).T
        ies = exact_set_ies(model, saes, suite, nodes, sets, clean, patch, metric_fn,
                            example=example, batch_size=args.batch_size)

        if args.size == 1:
            table = pa.table({"node": pa.array(range(n), pa.int32()),
                              "ie": pa.array(ies.float().numpy())})
        else:
            if clean.shape[1] not in reach_by_len:   # depends only on model + length
                reach_by_len[clean.shape[1]] = reachable_pairs(model, suite, clean)
            reachable = pair_reachable(nodes, sets, reach_by_len[clean.shape[1]],
                                       n_tokens=clean.shape[1])
            table = pa.table({"i": pa.array(sets[:, 0].int().numpy()),
                              "j": pa.array(sets[:, 1].int().numpy()),
                              "ie": pa.array(ies.half().numpy()),
                              "reachable": pa.array(reachable.numpy())})
        _write(table, target)
        secs = time.time() - started
        logger.info("[%d/%d] %d nodes, %d sets, %.1fs (%.0f sets/s)",
                    ex_idx + 1, len(examples), n, len(sets), secs, len(sets) / max(secs, 1e-9))


if __name__ == "__main__":
    main()
