"""Manifest-based caching for pipeline stages.

Each stage writes its outputs to outputs/<stage>/<param_hash8>/ (node attribution:
outputs/<dataset>/attributions/nodes/<counterfactual>/) alongside a manifest.json
recording the resolved config subtree that produced them, plus provenance (git
commit, timestamp, row counts). A stage that finds a manifest matching its
parameters reuses the directory instead of recomputing.

The manifest is the reproducibility record for the hive: it answers "what
produced this data" without needing the shell history that created it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Which config subtrees each stage's output depends on. A stage is re-run when
# any of these change; changes elsewhere (e.g. logging) reuse the cache.
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "node_attribution": ("suite", "data", "attribution", "seed"),
    "selection":        ("suite", "data", "attribution", "selection", "seed"),
    "edge_attribution": ("suite", "data", "attribution", "selection", "edges", "seed"),
    "circuit":          ("suite", "data", "attribution", "selection", "edges", "seed"),
    "evaluation":       ("suite", "data", "attribution", "selection", "edges",
                         "evaluation", "seed"),
}


def stage_params(cfg, stage: str) -> dict:
    """The resolved config subtree `stage`'s output depends on."""
    if stage not in STAGE_DEPS:
        raise KeyError(f"Unknown stage {stage!r}; known: {sorted(STAGE_DEPS)}")
    container = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    return {k: container[k] for k in STAGE_DEPS[stage]}


def param_hash(params: dict) -> str:
    """Stable 8-char hash of a parameter dict, independent of key order."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def stage_dir(output_root: str, stage: str, params: dict) -> Path:
    """Where `stage` writes: node attribution by dataset and counterfactual,
    every other stage by parameter hash."""
    if stage == "node_attribution":
        data = params["data"]
        dataset = Path(data["path"]).stem.removesuffix("_train")
        return (Path(output_root) / dataset / "attributions" / "nodes"
                / (data["counterfactual"] or "default"))
    return Path(output_root) / stage / param_hash(params)


def _changed_params(old: dict, new: dict, prefix: str = "") -> list[str]:
    """One 'dotted.key: old -> new' line per differing leaf."""
    changed = []
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changed += _changed_params(a, b, f"{prefix}{key}.")
        elif a != b:
            changed.append(f"{prefix}{key}: {a!r} -> {b!r}")
    return changed


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def write_manifest(path: Path, stage: str, params: dict, extra: dict[str, Any]) -> None:
    """Write manifest.json. Call this LAST, after outputs are complete --
    its presence is what marks the directory as a usable cache entry."""
    manifest = {
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        "params": params,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    (Path(path) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def find_cached(output_root: str, stage: str, params: dict) -> Path | None:
    """Return the directory of a completed run with matching params, else None.

    Scans sibling directories rather than trusting the hash alone, so a hash
    collision or a schema bump cannot silently serve the wrong data.
    """
    if stage == "node_attribution":
        candidates = [stage_dir(output_root, stage, params)]
    else:
        root = Path(output_root) / stage
        if not root.is_dir():
            return None
        candidates = sorted(root.iterdir())
    for candidate in candidates:
        mpath = candidate / "manifest.json"
        if not mpath.is_file():
            continue           # crashed or in-progress run
        try:
            m = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Unreadable manifest at %s: %s", mpath, e)
            continue
        if (m.get("stage") == stage
                and m.get("schema_version") == SCHEMA_VERSION
                and m.get("params") == params):
            return candidate
    return None


def resolve_stage_output(cfg, stage: str, compute_fn):
    """Run `stage` unless an identical run already exists. THE cache entry point.

    compute_fn(out_dir: Path) -> dict is called only on a miss, and its returned
    dict is merged into the manifest as summary metadata (row counts, timings).
    Returns (output_dir, was_cached).

    The manifest is stamped only after compute_fn returns, so an interrupted run
    leaves an unmanifested directory that will never be served as a hit.

    Any pre-existing directory is CLEARED before computing. Hive writers append
    (pq.write_to_dataset adds files rather than replacing them), so recomputing
    into a populated directory duplicates every row -- and since the manifest is
    rewritten afterwards with the new run's counts, the result looks correct and
    is not. That is reachable both via run.force and by resuming after a crash
    that left a half-written hive.
    """
    params = stage_params(cfg, stage)
    root = cfg.run.output_root

    if not cfg.run.force:
        hit = find_cached(root, stage, params)
        if hit is not None:
            logger.info("[%s] reusing cached output %s (params unchanged)", stage, hit)
            return hit, True

    out_dir = stage_dir(root, stage, params)
    # The node path carries no hash, so a run with other parameters lands on the
    # same directory; clearing it would silently destroy a finished hive.
    if (stage == "node_attribution" and not cfg.run.force
            and (out_dir / "manifest.json").is_file()):
        old = json.loads((out_dir / "manifest.json").read_text())["params"]
        changed = "; ".join(_changed_params(old, params)) or "schema version"
        raise RuntimeError(
            f"{out_dir} already holds a finished run with different parameters "
            f"({changed}). Pass run.force=true to overwrite it."
        )
    if out_dir.exists():
        logger.info("[%s] clearing %s before recompute", stage, out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    logger.info("[%s] computing into %s", stage, out_dir)
    extra = compute_fn(out_dir) or {}
    write_manifest(out_dir, stage, params, extra)
    return out_dir, False
