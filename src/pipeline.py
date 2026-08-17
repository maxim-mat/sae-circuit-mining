"""Stage sequencing for circuit discovery.

Each stage is a plain function taking (cfg, input_dir, out_dir). Caching is the
stage's own responsibility via src.utils.manifest -- there is no Stage base
class, registry, or plugin protocol, because five hard-coded calls in order is
the whole requirement.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.manifest import find_cached, resolve_stage_output, stage_params
from src.utils.tracking import finish_tracking, init_tracking, log_stage

logger = logging.getLogger(__name__)

STAGE_ORDER = ["node_attribution", "selection", "edge_attribution",
               "circuit", "evaluation"]

# Manifest keys that are provenance rather than stage summary scalars.
_PROVENANCE = {"stage", "schema_version", "params", "git_commit", "created_utc"}


def resolve_stages(configured: list[str], only_stage: str | None) -> list[str]:
    if only_stage is None:
        return [s for s in STAGE_ORDER if s in configured]
    if only_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {only_stage!r}; known: {STAGE_ORDER}")
    return [only_stage]


def _enabled_stages(cfg) -> list[str]:
    stages = list(cfg.run.stages)
    if not cfg.edges.enabled:
        stages = [s for s in stages if s != "edge_attribution"]
    if not cfg.evaluation.enabled:
        stages = [s for s in stages if s != "evaluation"]
    return stages


def _input_dir(cfg, dirs: dict, stage: str) -> Path:
    """The output directory of a prerequisite stage.

    Present in `dirs` during a full run; on a single-stage run it is recovered
    from the cache, which is also the check that the prerequisite was ever run
    with these parameters.
    """
    if stage in dirs:
        return dirs[stage]
    hit = find_cached(cfg.run.output_root, stage, stage_params(cfg, stage))
    if hit is None:
        raise RuntimeError(
            f"Stage needs the output of {stage!r}, which has not been computed "
            f"for these parameters. Run it first (--stage {stage})."
        )
    return hit


def _summary(out_dir: Path, was_cached: bool) -> dict:
    """Stage scalars from the manifest, so a cache hit logs the same numbers a
    fresh run would."""
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    metrics = {k: v for k, v in manifest.items() if k not in _PROVENANCE}
    metrics["was_cached"] = was_cached
    return metrics


def run_pipeline(cfg, only_stage: str | None = None) -> dict[str, Path]:
    """Run the configured stages in order. Returns {stage: output_dir}."""
    from src.attribution.edges import run_edge_attribution
    from src.attribution.nodes import run_node_attribution
    from src.circuit.build import run_circuit
    from src.eval.faithfulness import run_evaluation
    from src.selection.run import run_selection

    stages = resolve_stages(_enabled_stages(cfg), only_stage)
    logger.info("Pipeline stages: %s", stages)
    init_tracking(cfg)
    dirs: dict[str, Path] = {}

    try:
        for stage in stages:
            if stage == "node_attribution":
                fn = lambda out: run_node_attribution(cfg, out)
            elif stage == "selection":
                node_dir = _input_dir(cfg, dirs, "node_attribution")
                fn = lambda out, d=node_dir: run_selection(cfg, d, out)
            elif stage == "edge_attribution":
                sel_dir = _input_dir(cfg, dirs, "selection")
                fn = lambda out, d=sel_dir: run_edge_attribution(cfg, d, out)
            elif stage == "circuit":
                node_dir = _input_dir(cfg, dirs, "node_attribution")
                sel_dir = _input_dir(cfg, dirs, "selection")
                edge_dir = (_input_dir(cfg, dirs, "edge_attribution")
                            if cfg.edges.enabled else sel_dir)
                fn = lambda out, n=node_dir, s=sel_dir, e=edge_dir: run_circuit(
                    cfg, n, s, e, out)
            elif stage == "evaluation":
                circuit_dir = _input_dir(cfg, dirs, "circuit")
                fn = lambda out, c=circuit_dir: run_evaluation(
                    cfg, c / "circuit.json", out)
            else:
                raise ValueError(f"Unknown stage {stage!r}")

            out_dir, was_cached = resolve_stage_output(cfg, stage, fn)
            dirs[stage] = out_dir
            log_stage(stage, _summary(out_dir, was_cached))
    finally:
        finish_tracking()

    return dirs
