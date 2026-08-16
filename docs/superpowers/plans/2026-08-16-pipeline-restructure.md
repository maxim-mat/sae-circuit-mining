# Configurable Circuit-Discovery Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo into a configurable, cached, wandb-tracked pipeline (node attribution → filtering → edge IE → circuit → metrics) that runs on both `gpt2-small-res-jb` and Marks et al.'s pythia-70m-deduped SAE suite, and reproduce one published subject–verb agreement circuit with a faithfulness number.

**Architecture:** A thin `main.py` loads an OmegaConf structured config and calls `run_pipeline(cfg)`, which invokes five stage functions in sequence. Each stage writes a parquet hive plus a `manifest.json` recording the resolved config subtree that produced it, and skips recomputation when an existing manifest matches. SAE sites are described by an explicit per-suite `Site` table rather than an integer parsed from the SAE id, which is what lets attn/mlp/embed dictionaries coexist with resid ones.

**Tech Stack:** PyTorch, TransformerLens, sae_lens, pyarrow, pandas, OmegaConf, wandb, pytest. No Lightning (no training loop to wrap), no `dictionary_learning` (Marks' Pythia SAEs are in the sae_lens catalog).

**Spec:** `docs/spec.md`

## Global Constraints

- Python: `C:\Users\Maxim\anaconda3\python.exe` — the plain `python` on PATH lacks the deps.
- **No new dependencies** beyond `omegaconf` and `wandb`. Marks' Pythia SAEs load via the existing `sae_lens` path; do not add `dictionary_learning`.
- Reproducibility artifacts are mandatory and must never be trimmed: seeds, resolved-config logging, run metadata (data version, git commit, environment), structured error handling around I/O, docstrings on any function used by more than one module.
- Reference implementation for all method questions: `saprmarks/feature-circuits` @ `7fbd82b`. Cite paper as arXiv:2403.19647 at the point of implementation.
- Tests live in `tests/`, run with `pytest`. Tests that need a real model are marked `@pytest.mark.slow` and excluded from the default run.
- Every stage's output directory is `outputs/<stage>/<param_hash8>/`, containing `manifest.json`.
- Commit after every task. Never commit data files or `outputs/`.

---

## Starting cold

Execution was deliberately deferred when this plan was written — nothing in it
has been implemented. If you are picking this up in a fresh session, this
section is everything you need that is not already in the repo.

**Read first, in order:** `docs/spec.md` (goal, verification results, decisions
D1–D5), then this plan. Do not re-derive the reference-implementation findings —
they cost the bulk of a session and are recorded in the spec.

**Repo state at plan time** (commit `b297d97`): `src/event_log.py`,
`src/discover.py`, `src/test_event_log.py` were already deleted;
`src/edge_attribution.py`, `src/main.py`, `src/partial_order.py` had uncommitted
modifications; `src/sae_layers.py` was untracked. The pipeline described here
does not exist yet — `main.py` still contains the monolithic attribution loop.

**Reference implementation:** clone or browse `saprmarks/feature-circuits` at
commit `7fbd82b`. The files that matter are `attribution.py` (`_pe_attrib`,
`_pe_ig`, `jvp`), `circuit.py` (`get_circuit`, the edge template, aggregation),
`activation_utils.py` (`SparseAct.__matmul__` — how effects contract),
`ablation.py` (faithfulness), `data_loading_utils.py` (`load_examples` filters),
`dictionary_loading_utils.py` (which submodules get dictionaries).

**Questions already answered — do not re-open:**

| Question | Answer | Where |
|---|---|---|
| Is our node-attribution math right? | Yes, verified line by line | spec §2 |
| Where do Marks' Pythia SAEs come from? | sae_lens catalog, not `dictionary_learning` | spec §3 |
| Do they sum IEs over token positions? | Only for non-templatic data; templatic keeps positions | spec §2 |
| Do they filter both edge endpoints? | No — downstream only, on the *unaggregated* effect | spec §2 |
| What edges do they compute? | A fixed per-layer template, not layer pairs | spec §2 |
| What is faithfulness? | Ablation-based, not `metric_patch - metric_clean` | spec §2 |
| Lightning? | No — no training loop exists to wrap | spec R5 |

**Decisions the user made explicitly**, which are not up for renegotiation
without asking them:

1. Both model suites stay configurable; neither is dropped (spec R3).
2. One counterfactual per run (spec D4) — long-standing, first agreed during the
   event-log design work.
3. Edge attribution aligns to Marks by default; the total-effect variant is
   preserved as opt-in for the process-mining direction (spec D5).
4. Every stage checks for an existing hive with matching parameters before
   computing (spec R4, Task 2).
5. wandb for tracking, OmegaConf for config, matching the idiom in
   `D:\Projects\trace-denoise-refactor` and `D:\Projects\lod-generation`.

**Still undecided:** execution mode (subagent-driven vs inline). Ask before
starting Task 1.

---

## File Structure

**Created:**

| Path | Responsibility |
|---|---|
| `src/utils/config.py` | OmegaConf dataclass schema; root `Config` |
| `src/utils/initialization.py` | `load_config`, `parse_args` |
| `src/utils/manifest.py` | write/find manifests; param hashing; skip-if-cached |
| `src/utils/tracking.py` | wandb run lifecycle from config |
| `src/models/suites.py` | `Site` dataclass, per-suite site tables, model+SAE loading |
| `src/data/loaders.py` | Marks jsonl SVA loader, MIB loader, common `Example` type |
| `src/attribution/ig.py` | integrated-gradients residual gradient (moved) |
| `src/attribution/nodes.py` | node IE computation + `total_effect` (moved from `main.py`) |
| `src/attribution/edge_template.py` | Marks' stop-gradient edge topology; total-effect alternative |
| `src/attribution/edges.py` | edge IE computation (moved from `src/edge_attribution.py`) |
| `src/selection/aggregate.py` | position/example aggregation (`none` \| `sum`) |
| `src/selection/threshold.py` | Marks-style `\|mean IE\| > T` |
| `src/circuit/build.py` | nodes + edges → circuit object, save/load |
| `src/eval/faithfulness.py` | ablation-based `F(C)`, `F(∅)`, `F(M)` |
| `src/pipeline.py` | `run_pipeline(cfg)` — stage sequence |
| `configs/pythia_marks_rc.yaml` | Marks reproduction experiment |
| `configs/gpt2_mib_ioi.yaml` | existing GPT-2 / MIB setup |
| `tests/*` | pytest suite |

**Modified:**

| Path | Change |
|---|---|
| `src/main.py` | gutted to a thin entry point |
| `src/metrics.py` | add SVA metric; drop the `direction` return |
| `src/partial_order.py` | column rename `layer` → `site_ordinal`, `prompt_idx` → `example_idx` |
| `src/calibrated_feature_selection.py` | column rename; expose a stage-level entry point |

**Deleted:**

| Path | Reason |
|---|---|
| `src/sae_layers.py` | superseded by `src/models/suites.py` |
| `src/edge_attribution.py` | moved to `src/attribution/edges.py` |
| `src/trace_log_converter.py` | consumed the deleted event-log format |

---

## ⚠️ Risk notice

`sae_layer_ordinal` is **HIGH risk** — gitnexus reports 10 impacted symbols, 3 broken execution flows. Direct callers: `main.py:main`, `partial_order.py:load_sae_directions`, `sae_layers.py:sorted_saes`; transitive: `edge_attribution.py:load_model_and_saes` → `edge_attribution.py:main`, `partial_order.py:run`. Task 3 replaces it. Tasks 3, 6, 8 and the rename step in Task 9 must land together before the tree is coherent again; do not stop in the middle of that run.

That figure was measured against a **stale index** (it still listed the deleted `event_log.py`). Re-run `node .gitnexus/run.cjs analyze`, then `impact({target: "sae_layer_ordinal", direction: "upstream", repo: "sae-circuit-mining"})`, before starting Task 3 — the real blast radius may be smaller now that three modules are gone.

---

## Task 1: Config schema and entry point

**Files:**
- Create: `src/utils/config.py`, `src/utils/initialization.py`, `configs/pythia_marks_rc.yaml`, `configs/gpt2_mib_ioi.yaml`
- Create: `tests/test_config.py`
- Modify: `src/main.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` and its nested dataclasses (`SuiteConfig`, `DataConfig`, `AttributionConfig`, `SelectionConfig`, `EdgeConfig`, `EvalConfig`, `LoggingConfig`, `RunConfig`); `load_config(path: Path, overrides: list[str]) -> Config`; `parse_args() -> argparse.Namespace` with `.config`, `.overrides`, `.stage`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from src.utils.config import Config
from src.utils.initialization import load_config


def test_defaults_and_yaml_and_override_precedence(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        "suite: {name: pythia-marks}\n"
        "data: {path: data/rc_train.json, num_examples: 100}\n"
        "attribution: {method: ig, ig_steps: 10}\n"
    )
    cfg = load_config(yaml, ["attribution.ig_steps=5", "selection.node_threshold=0.2"])

    assert isinstance(cfg, Config)
    assert cfg.suite.name == "pythia-marks"       # from yaml
    assert cfg.attribution.method == "ig"          # from yaml
    assert cfg.attribution.ig_steps == 5           # override beats yaml
    assert cfg.selection.node_threshold == 0.2     # override beats default
    assert cfg.seed == 42                          # schema default survives


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", [])


def test_unknown_key_rejected(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text("attribution: {no_such_key: 1}\n")
    with pytest.raises(Exception):
        load_config(yaml, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.config'`

- [ ] **Step 3: Write the schema**

```python
# src/utils/config.py
"""OmegaConf structured-config schema for the circuit-discovery pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING


@dataclass
class SuiteConfig:
    """Which model + SAE dictionaries to run against. See src/models/suites.py."""
    name: str = MISSING          # "gpt2-res" | "pythia-marks"
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class DataConfig:
    """Dataset selection and the filtering that makes token positions align."""
    path: str = MISSING          # jsonl path (Marks SVA) or HF dataset id (MIB)
    loader: str = "sva"          # "sva" | "mib"
    num_examples: int = 100
    use_min_length_only: bool = True   # required for aggregation="none"
    max_length: Optional[int] = None
    text_column: Optional[str] = None  # mib only
    split: str = "train"               # mib only
    # A run is scoped to ONE counterfactual: causal questions do not mix, and
    # this way the choice lands in the manifest and the cache key instead of
    # becoming a partition column. Required for loader="mib"; unused for "sva",
    # which has a single patch_prefix per example.
    counterfactual: Optional[str] = None


@dataclass
class AttributionConfig:
    method: str = "ig"           # "attrib" | "ig"
    ig_steps: int = 10
    metric: str = "sva_logit_diff"   # key into metrics.METRICS
    batch_size: int = 1


@dataclass
class SelectionConfig:
    aggregation: str = "none"    # "none" | "sum"  (Marks App. A.2)
    method: str = "threshold"    # "threshold" | "calibrated"
    node_threshold: float = 0.1
    # calibrated-only
    fdr_q: float = 0.05
    min_recurrence: float = 0.1
    block_col: Optional[str] = None


@dataclass
class EdgeConfig:
    enabled: bool = True
    threshold: float = 0.01
    # "marks": the fixed per-layer template with stop-gradients -> DIRECT edges.
    # "total_effect": all/consecutive site pairs, no stop-gradients -> TOTAL
    #   effect edges ("eventually-follows" semantics for process mining).
    #   Not comparable to Marks' edge counts; never mix the two in one graph.
    template: str = "marks"
    pairs: str = "consecutive"   # total_effect template only: "consecutive" | "all"
    filter_mode: str = "marks"   # "marks" = downstream-filtered only; "both" = both endpoints
    include_error: bool = True


@dataclass
class EvalConfig:
    enabled: bool = True
    data_path: Optional[str] = None    # defaults to the *_test split of data.path
    num_examples: int = 100
    ablation: str = "mean"             # "mean" | "resample" | "zero"
    handle_errors: str = "default"     # "default" | "keep" | "remove"
    start_layer: int = -1


@dataclass
class LoggingConfig:
    backend: List[str] = field(default_factory=lambda: [])   # [] | ["wandb"]
    project: str = "sae-circuit-mining"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    offline: bool = False


@dataclass
class RunConfig:
    output_root: str = "outputs"
    force: bool = False          # recompute even when a manifest matches
    stages: List[str] = field(default_factory=lambda: [
        "node_attribution", "selection", "edge_attribution", "circuit", "evaluation"
    ])


@dataclass
class Config:
    suite: SuiteConfig = field(default_factory=SuiteConfig)
    data: DataConfig = field(default_factory=DataConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    run: RunConfig = field(default_factory=RunConfig)
    seed: int = 42
```

```python
# src/utils/initialization.py
"""Config loading and CLI parsing."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from src.utils.config import Config


def load_config(config_path: Path, overrides: list[str]) -> Config:
    """Merge schema defaults <- YAML file <- CLI dotlist overrides.

    Raises FileNotFoundError if the YAML is missing, and omegaconf errors on
    keys absent from the schema (structured configs are strict by default).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(schema, file_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="SAE circuit discovery pipeline")
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to YAML configuration file")
    parser.add_argument("--stage", default=None,
                        help="Run a single stage instead of the full pipeline")
    parser.add_argument("overrides", nargs="*",
                        help="Dotlist overrides, e.g. selection.node_threshold=0.2")
    return parser.parse_args()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Write the experiment configs**

```yaml
# configs/pythia_marks_rc.yaml
# Reproduction target: Marks et al. 2025 (arXiv:2403.19647) README worked example
#   scripts/get_circuit.sh EleutherAI/pythia-70m-deduped rc_train 0.1 0.01 none
suite:
  name: pythia-marks
data:
  path: data/marks/rc_train.json
  loader: sva
  num_examples: 100
  use_min_length_only: true
attribution:
  method: ig
  ig_steps: 10
  metric: sva_logit_diff
selection:
  aggregation: none
  method: threshold
  node_threshold: 0.1
edges:
  threshold: 0.01
  template: marks
  filter_mode: marks
evaluation:
  data_path: data/marks/rc_test.json
  ablation: mean
logging:
  backend: [wandb]
  group: marks-repro
seed: 42
```

```yaml
# configs/gpt2_mib_ioi.yaml
suite:
  name: gpt2-res
data:
  path: mib-bench/ioi
  loader: mib
  num_examples: 100
  use_min_length_only: false
  counterfactual: abc          # one run, one counterfactual
attribution:
  method: ig
  ig_steps: 10
  metric: mib_logit_diff
selection:
  aggregation: sum
  method: calibrated
edges:
  threshold: 0.01
  template: total_effect
  pairs: consecutive
  filter_mode: both
evaluation:
  enabled: false
logging:
  backend: [wandb]
  group: gpt2-baseline
seed: 42
```

- [ ] **Step 6: Replace `main.py` with a thin entry point**

```python
# src/main.py
"""Entry point for the SAE circuit-discovery pipeline.

Usage:
    python -m src.main -c configs/pythia_marks_rc.yaml
    python -m src.main -c configs/pythia_marks_rc.yaml selection.node_threshold=0.2
    python -m src.main -c configs/pythia_marks_rc.yaml --stage node_attribution
"""
import logging
import sys

from omegaconf import OmegaConf

from src.utils.initialization import load_config, parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sae_circuit_mining")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    from src.pipeline import run_pipeline   # imported late: heavy deps
    run_pipeline(cfg, only_stage=args.stage)


if __name__ == "__main__":
    main()
```

Note: `src/pipeline.py` does not exist until Task 11. `main.py` will fail at
runtime until then; the import is inside `main()` so the module still imports
and tests still collect.

- [ ] **Step 7: Add package markers**

Create empty `src/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py`.

- [ ] **Step 8: Commit**

```bash
git add src/utils/config.py src/utils/initialization.py src/utils/__init__.py \
        src/__init__.py tests/__init__.py tests/test_config.py \
        configs/ src/main.py docs/spec.md docs/superpowers/
git commit -m "feat: OmegaConf structured config schema and thin entry point"
```

---

## Task 2: Manifest-based stage caching

**Files:**
- Create: `src/utils/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: `Config` from Task 1.
- Produces:
  - `param_hash(params: dict) -> str` — 8-char stable hash
  - `stage_dir(output_root: str, stage: str, params: dict) -> Path`
  - `find_cached(output_root: str, stage: str, params: dict) -> Path | None`
  - `write_manifest(path: Path, stage: str, params: dict, extra: dict) -> None`
  - `stage_params(cfg: Config, stage: str) -> dict` — the config subtree a stage depends on
  - **`resolve_stage_output(cfg, stage, compute_fn) -> tuple[Path, bool]`** — the
    single entry point every stage goes through. Returns `(output_dir, was_cached)`
    and guarantees `compute_fn` is not called on a cache hit.

**This is the redundancy mechanism.** No stage calls `find_cached` or
`write_manifest` directly — they call `resolve_stage_output`, which owns the
whole protocol: look for a matching manifest, skip if found, otherwise compute
into a fresh param-hashed directory and stamp the manifest last. Because the
manifest is written *after* the outputs, a crashed run leaves a directory with
no manifest and is never mistaken for a cache hit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import json
from src.utils.manifest import param_hash, stage_dir, find_cached, write_manifest


def test_hash_is_order_independent_and_stable():
    assert param_hash({"a": 1, "b": [2, 3]}) == param_hash({"b": [2, 3], "a": 1})
    assert param_hash({"a": 1}) != param_hash({"a": 2})
    assert len(param_hash({"a": 1})) == 8


def test_find_cached_misses_when_absent(tmp_path):
    assert find_cached(str(tmp_path), "node_attribution", {"a": 1}) is None


def test_find_cached_hits_after_write(tmp_path):
    params = {"suite": "pythia-marks", "steps": 10}
    d = stage_dir(str(tmp_path), "node_attribution", params)
    d.mkdir(parents=True)
    (d / "part-0.parquet").write_bytes(b"x")
    write_manifest(d, "node_attribution", params, {"row_count": 5})

    hit = find_cached(str(tmp_path), "node_attribution", params)
    assert hit == d
    m = json.loads((hit / "manifest.json").read_text())
    assert m["params"] == params
    assert m["stage"] == "node_attribution"
    assert m["row_count"] == 5
    assert m["git_commit"]


def test_different_params_do_not_collide(tmp_path):
    p1, p2 = {"steps": 10}, {"steps": 20}
    d1 = stage_dir(str(tmp_path), "s", p1); d1.mkdir(parents=True)
    write_manifest(d1, "s", p1, {})
    assert find_cached(str(tmp_path), "s", p2) is None


def test_incomplete_dir_is_not_a_hit(tmp_path):
    """A directory without a manifest is a crashed run, not a cache hit."""
    params = {"steps": 10}
    d = stage_dir(str(tmp_path), "s", params)
    d.mkdir(parents=True)
    assert find_cached(str(tmp_path), "s", params) is None
```

```python
# tests/test_manifest.py (continued) -- the redundancy mechanism itself
from src.utils.config import Config
from src.utils.manifest import resolve_stage_output


def _cfg(tmp_path, **over):
    cfg = Config()
    cfg.suite.name = "gpt2-res"
    cfg.data.path = "x.json"
    cfg.run.output_root = str(tmp_path)
    for k, v in over.items():
        setattr(cfg.attribution, k, v)
    return cfg


def test_second_call_with_same_params_does_not_recompute(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        (out_dir / "data.parquet").write_bytes(b"x")
        return {"row_count": 1}

    cfg = _cfg(tmp_path)
    d1, cached1 = resolve_stage_output(cfg, "node_attribution", compute)
    d2, cached2 = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 1, "compute_fn must not run on a cache hit"
    assert (d1, cached1, cached2) == (d2, False, True)


def test_changed_params_recompute_into_a_separate_dir(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    d1, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=10), "node_attribution", compute)
    d2, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=20), "node_attribution", compute)

    assert len(calls) == 2
    assert d1 != d2, "different params must not overwrite each other"
    assert d1.exists() and d2.exists(), "old hives are kept, not replaced"


def test_irrelevant_config_change_still_hits_the_cache(tmp_path):
    """logging is not in STAGE_DEPS['node_attribution'] -- changing it must not
    invalidate an expensive attribution run."""
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    cfg = _cfg(tmp_path)
    resolve_stage_output(cfg, "node_attribution", compute)
    cfg.logging.run_name = "a-different-name"
    _, cached = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 1
    assert cached is True


def test_force_recomputes_and_overwrites(tmp_path):
    calls = []

    def compute(out_dir):
        calls.append(out_dir)
        return {}

    cfg = _cfg(tmp_path)
    resolve_stage_output(cfg, "node_attribution", compute)
    cfg.run.force = True
    _, cached = resolve_stage_output(cfg, "node_attribution", compute)

    assert len(calls) == 2
    assert cached is False


def test_manifest_records_what_produced_the_hive(tmp_path):
    import json
    d, _ = resolve_stage_output(_cfg(tmp_path, ig_steps=7), "node_attribution",
                                lambda out_dir: {"row_count": 42})
    m = json.loads((d / "manifest.json").read_text())
    assert m["params"]["attribution"]["ig_steps"] == 7
    assert m["params"]["data"]["counterfactual"] is None
    assert m["row_count"] == 42
    assert m["git_commit"] and m["created_utc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.manifest'`

- [ ] **Step 3: Implement**

```python
# src/utils/manifest.py
"""Manifest-based caching for pipeline stages.

Each stage writes its outputs to outputs/<stage>/<param_hash8>/ alongside a
manifest.json recording the resolved config subtree that produced them, plus
provenance (git commit, timestamp, row counts). A stage that finds a manifest
matching its parameters reuses the directory instead of recomputing.

The manifest is the reproducibility record for the hive: it answers "what
produced this data" without needing the shell history that created it.
"""
from __future__ import annotations

import hashlib
import json
import logging
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
    return Path(output_root) / stage / param_hash(params)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def write_manifest(path: Path, stage: str, params: dict, extra: dict[str, Any]) -> None:
    """Write manifest.json. Call this LAST, after outputs are complete —
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
    root = Path(output_root) / stage
    if not root.is_dir():
        return None
    for candidate in sorted(root.iterdir()):
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
    leaves an unmanifested directory that will never be served as a hit. On
    cfg.run.force the directory is reused in place and recomputed.
    """
    params = stage_params(cfg, stage)
    root = cfg.run.output_root

    if not cfg.run.force:
        hit = find_cached(root, stage, params)
        if hit is not None:
            logger.info("[%s] reusing cached output %s (params unchanged)", stage, hit)
            return hit, True

    out_dir = stage_dir(root, stage, params)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] computing into %s", stage, out_dir)
    extra = compute_fn(out_dir) or {}
    write_manifest(out_dir, stage, params, extra)
    return out_dir, False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_manifest.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/utils/manifest.py tests/test_manifest.py
git commit -m "feat: manifest-based stage caching with param-hashed output dirs"
```

---

## Task 3: SAE suite registry and typed sites ⚠️ HIGH RISK

Replaces `src/sae_layers.py`. Breaks `edge_attribution.py` and `partial_order.py` until Tasks 8 and 9.

**Files:**
- Create: `src/models/suites.py`, `src/models/__init__.py`, `tests/test_suites.py`
- Delete: `src/sae_layers.py`

**Interfaces:**
- Consumes: `SuiteConfig` from Task 1.
- Produces:
  - `@dataclass(frozen=True) Site(ordinal:int, block:int, kind:str, release:str, sae_id:str, hook:str)`
  - `@dataclass(frozen=True) Suite(name:str, model_name:str, parallel_attn:bool, sites:tuple[Site,...])`
  - `SUITES: dict[str, Suite]`
  - `build_suite(name: str) -> Suite`
  - `load_model_and_saes(suite: Suite, device: str, dtype: str) -> tuple[HookedTransformer, dict[str, SAE]]` keyed by `Site.sae_id`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_suites.py
import pytest
from src.models.suites import build_suite, SUITES


def test_gpt2_ordinals_match_legacy_resid_ordering():
    """resid_post of block L is the same tensor as resid_pre of L+1, so only
    the final block's post-hook is a distinct site and gets ordinal 12."""
    s = build_suite("gpt2-res")
    by_id = {site.sae_id: site.ordinal for site in s.sites}
    for i in range(12):
        assert by_id[f"blocks.{i}.hook_resid_pre"] == i
    assert by_id["blocks.11.hook_resid_post"] == 12
    assert all(site.kind == "resid" for site in s.sites)


def test_pythia_has_all_19_marks_sites():
    s = build_suite("pythia-marks")
    assert len(s.sites) == 19                       # embed + 6 blocks x {attn, mlp, resid}
    kinds = [site.kind for site in s.sites]
    assert kinds.count("embed") == 1
    assert kinds.count("attn") == 6
    assert kinds.count("mlp") == 6
    assert kinds.count("resid") == 6


def test_pythia_ordinals_are_unique_strict_and_ascending():
    s = build_suite("pythia-marks")
    ordinals = [site.ordinal for site in s.sites]
    assert ordinals == sorted(ordinals)
    assert len(set(ordinals)) == len(ordinals)
    assert ordinals[0] == 0                          # embed first


def test_pythia_block_layout():
    s = build_suite("pythia-marks")
    by_ord = {site.ordinal: site for site in s.sites}
    assert by_ord[0].kind == "embed" and by_ord[0].block == -1
    for b in range(6):
        assert (by_ord[3 * b + 1].kind, by_ord[3 * b + 1].block) == ("attn", b)
        assert (by_ord[3 * b + 2].kind, by_ord[3 * b + 2].block) == ("mlp", b)
        assert (by_ord[3 * b + 3].kind, by_ord[3 * b + 3].block) == ("resid", b)


def test_parallel_attn_flag_records_concurrency():
    """Pythia runs attn and mlp in parallel; consecutive ordinals for those two
    sites are a linearisation, not a real dependency (Marks' circuit.py sets
    parallel_attn=True for pythia and skips the attn->mlp edge)."""
    assert build_suite("pythia-marks").parallel_attn is True
    assert build_suite("gpt2-res").parallel_attn is False


def test_hooks_are_transformerlens_names():
    s = build_suite("pythia-marks")
    hooks = {site.hook for site in s.sites}
    assert "blocks.0.hook_resid_pre" in hooks
    assert "blocks.3.hook_attn_out" in hooks
    assert "blocks.5.hook_mlp_out" in hooks
    assert "blocks.5.hook_resid_post" in hooks


def test_unknown_suite_raises():
    with pytest.raises(KeyError):
        build_suite("no-such-suite")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_suites.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.suites'`

- [ ] **Step 3: Implement**

```python
# src/models/suites.py
"""Model + SAE-dictionary suites, and the site table that orders them.

A "site" is one hook point with an SAE attached. Sites carry an explicit
ordinal rather than an integer parsed out of the SAE id, because typed
dictionaries (embed / attn / mlp / resid) cannot be ordered by block number
alone:

  * resid_post of block L is the same tensor as resid_pre of block L+1, so
    parsing L collides two disjoint feature spaces (the reason the previous
    sae_layer_ordinal existed at all);
  * attn and mlp are PARALLEL in Pythia, so no total order over integers
    describes them honestly. `Suite.parallel_attn` records this so downstream
    partial-order work can recover the concurrency; the consecutive ordinals
    assigned here are a linearisation for the edge code's src < dst test.
    Edges across a falsely-ordered pair score ~0.

The pythia-marks suite is the SAE set from Marks et al. 2024
("Sparse Feature Circuits", arXiv:2403.19647), served through the sae_lens
catalog as ctigges/pythia-70m-deduped__{res,att,mlp}-sm_processed rather than
the dictionary_learning .pt files their repo downloads.

Self-check: python -m pytest tests/test_suites.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass(frozen=True)
class Site:
    """One SAE-instrumented hook point."""
    ordinal: int      # strict position in execution order; the total order edges use
    block: int        # transformer block index; -1 for embed
    kind: str         # "embed" | "attn" | "mlp" | "resid"
    release: str      # sae_lens release
    sae_id: str       # sae_lens id within the release
    hook: str         # TransformerLens hook name


@dataclass(frozen=True)
class Suite:
    name: str
    model_name: str
    parallel_attn: bool
    sites: tuple[Site, ...]


def _gpt2_res_sites() -> tuple[Site, ...]:
    rel = "gpt2-small-res-jb"
    sites = [
        Site(ordinal=i, block=i, kind="resid", release=rel,
             sae_id=f"blocks.{i}.hook_resid_pre", hook=f"blocks.{i}.hook_resid_pre")
        for i in range(12)
    ]
    sites.append(Site(ordinal=12, block=11, kind="resid", release=rel,
                      sae_id="blocks.11.hook_resid_post",
                      hook="blocks.11.hook_resid_post"))
    return tuple(sites)


def _pythia_marks_sites() -> tuple[Site, ...]:
    res, att, mlp = (
        "pythia-70m-deduped-res-sm",
        "pythia-70m-deduped-att-sm",
        "pythia-70m-deduped-mlp-sm",
    )
    sites = [Site(ordinal=0, block=-1, kind="embed", release=res,
                  sae_id="blocks.0.hook_resid_pre", hook="blocks.0.hook_resid_pre")]
    for b in range(6):
        sites += [
            Site(ordinal=3 * b + 1, block=b, kind="attn", release=att,
                 sae_id=f"blocks.{b}.hook_attn_out", hook=f"blocks.{b}.hook_attn_out"),
            Site(ordinal=3 * b + 2, block=b, kind="mlp", release=mlp,
                 sae_id=f"blocks.{b}.hook_mlp_out", hook=f"blocks.{b}.hook_mlp_out"),
            Site(ordinal=3 * b + 3, block=b, kind="resid", release=res,
                 sae_id=f"blocks.{b}.hook_resid_post", hook=f"blocks.{b}.hook_resid_post"),
        ]
    return tuple(sites)


SUITES: dict[str, Suite] = {
    "gpt2-res": Suite("gpt2-res", "gpt2-small", False, _gpt2_res_sites()),
    "pythia-marks": Suite("pythia-marks", "pythia-70m-deduped", True,
                          _pythia_marks_sites()),
}


def build_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"Unknown suite {name!r}; known: {sorted(SUITES)}")
    return SUITES[name]


def load_model_and_saes(suite: Suite, device: str, dtype: str = "float32"):
    """Load the model and every SAE in the suite. Returns (model, {sae_id: SAE})."""
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch_dtype = _DTYPES[dtype]
    logger.info("Loading model %s on %s", suite.model_name, device)
    model = HookedTransformer.from_pretrained(
        suite.model_name, device=device, dtype=torch_dtype
    )

    saes = {}
    for site in suite.sites:
        logger.info("Loading SAE %s/%s", site.release, site.sae_id)
        try:
            saes[site.sae_id] = SAE.from_pretrained(
                release=site.release, sae_id=site.sae_id, device=device
            )[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAE {site.release}/{site.sae_id} for suite "
                f"{suite.name!r}. Check the sae_lens catalog entry."
            ) from e
    return model, saes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_suites.py -v`
Expected: 7 passed

- [ ] **Step 5: Delete the superseded module**

```bash
git rm src/sae_layers.py src/trace_log_converter.py
```

- [ ] **Step 6: Commit**

```bash
git add src/models/ tests/test_suites.py
git commit -m "feat!: typed SAE site tables replace integer layer ordinals

BREAKING: sae_layer_ordinal removed. Sites now carry (ordinal, block, kind);
attn/mlp concurrency is recorded via Suite.parallel_attn. Adds Marks et al.
pythia-70m-deduped suite via the sae_lens catalog."
```

---

## Task 4: Data loaders

**Files:**
- Create: `src/data/loaders.py`, `src/data/__init__.py`, `tests/test_loaders.py`

**Interfaces:**
- Consumes: `DataConfig` from Task 1.
- Produces:
  - `@dataclass Example(clean_prefix:str, clean_answer:str, patch_prefix:str|None, patch_answer:str|None, meta:dict)`
  - `load_sva_examples(path, num_examples, model, use_min_length_only=True, max_length=None, seed=12) -> list[Example]`
  - `load_examples(cfg: DataConfig, model, seed: int) -> list[Example]` — dispatches on `cfg.loader`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loaders.py
import json
import pytest
from src.data.loaders import load_sva_examples, Example


class FakeTokenizer:
    """Whitespace tokenizer: one token per word, so lengths are predictable."""
    def __call__(self, text):
        class R:
            input_ids = list(range(len(text.split())))
        return R()


class FakeModel:
    tokenizer = FakeTokenizer()


def _write(tmp_path, rows):
    p = tmp_path / "d.json"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_reads_jsonl_despite_json_extension(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is",
             "patch_prefix": "the boys", "patch_answer": " are"}]
    ex = load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                           use_min_length_only=False)
    assert len(ex) == 1
    assert isinstance(ex[0], Example)
    assert ex[0].clean_prefix == "the boy"
    assert ex[0].patch_answer == " are"


def test_drops_mismatched_prefix_lengths(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is",
             "patch_prefix": "the tall boys", "patch_answer": " are"}]
    assert load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                             use_min_length_only=False) == []


def test_drops_multi_token_answers(tmp_path):
    rows = [{"clean_prefix": "the boy", "clean_answer": " is here",
             "patch_prefix": "the boys", "patch_answer": " are"}]
    assert load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                             use_min_length_only=False) == []


def test_min_length_only_keeps_one_length_and_restarts(tmp_path):
    """Position alignment for aggregation='none' requires a single length.
    A shorter example appearing later must discard everything collected so far."""
    rows = [
        {"clean_prefix": "a b c", "clean_answer": " x",
         "patch_prefix": "d e f", "patch_answer": " y"},
        {"clean_prefix": "a b", "clean_answer": " x",
         "patch_prefix": "d e", "patch_answer": " y"},
        {"clean_prefix": "g h", "clean_answer": " x",
         "patch_prefix": "i j", "patch_answer": " y"},
        {"clean_prefix": "k l m", "clean_answer": " x",
         "patch_prefix": "n o p", "patch_answer": " y"},
    ]
    ex = load_sva_examples(_write(tmp_path, rows), 10, FakeModel(),
                           use_min_length_only=True, shuffle=False)
    assert len(ex) == 2
    assert {e.clean_prefix for e in ex} == {"a b", "g h"}


def test_respects_num_examples(tmp_path):
    rows = [{"clean_prefix": "a b", "clean_answer": " x",
             "patch_prefix": "c d", "patch_answer": " y"} for _ in range(10)]
    assert len(load_sva_examples(_write(tmp_path, rows), 3, FakeModel(),
                                 use_min_length_only=False)) == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sva_examples(tmp_path / "nope.json", 1, FakeModel())
```

```python
# tests/test_loaders.py (continued) -- one counterfactual per run
import pytest
from src.data.loaders import select_counterfactual


def test_selects_only_the_designated_counterfactual():
    """A run is scoped to one counterfactual: causal questions do not mix."""
    item = {"prompt": "clean text",
            "abc": {"prompt": "abc text"},
            "name_swap": {"prompt": "swapped text"}}
    cf = select_counterfactual(item, "abc")
    assert cf["prompt"] == "abc text"


def test_missing_counterfactual_lists_what_is_available():
    item = {"prompt": "clean", "abc": {"prompt": "x"}, "name_swap": {"prompt": "y"}}
    with pytest.raises(ValueError, match="abc.*name_swap|name_swap.*abc"):
        select_counterfactual(item, "no_such_cf")


def test_unset_counterfactual_is_an_error_not_a_silent_pick():
    """Defaulting to 'the first one' would silently answer a different causal
    question than the config names."""
    item = {"prompt": "clean", "abc": {"prompt": "x"}}
    with pytest.raises(ValueError):
        select_counterfactual(item, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.loaders'`

- [ ] **Step 3: Implement**

```python
# src/data/loaders.py
"""Dataset loaders.

The subject-verb agreement loader ports load_examples from Marks et al.'s
feature-circuits repo (arXiv:2403.19647). Its filters are not incidental
hygiene: use_min_length_only forces every example to the same token length,
which is what makes token positions comparable across examples and therefore
what makes aggregation="none" (position-preserving circuits) meaningful at all.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Example:
    clean_prefix: str
    clean_answer: str
    patch_prefix: str | None = None
    patch_answer: str | None = None
    meta: dict = field(default_factory=dict)


def load_sva_examples(path, num_examples: int, model, use_min_length_only: bool = True,
                      max_length: int | None = None, seed: int = 12,
                      shuffle: bool = True) -> list[Example]:
    """Load Marks' subject-verb agreement stimuli (jsonl, despite the .json name).

    Drops examples with multi-token answers, unequal clean/patch prefix lengths,
    or tokenization mismatches at the prefix|answer boundary. With
    use_min_length_only, keeps only examples of the shortest length seen,
    restarting collection whenever a shorter one appears.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if shuffle:
        random.Random(seed).shuffle(lines)

    examples: list[Example] = []
    min_length = float("inf")
    tok = model.tokenizer

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed line in %s: %s", path, e)
            continue

        clean_prefix = tok(data["clean_prefix"]).input_ids
        patch_prefix = tok(data["patch_prefix"]).input_ids
        clean_answer = tok(data["clean_answer"]).input_ids
        patch_answer = tok(data["patch_answer"]).input_ids
        clean_full = tok(data["clean_prefix"] + data["clean_answer"]).input_ids
        patch_full = tok(data["patch_prefix"] + data["patch_answer"]).input_ids

        bos = getattr(tok, "bos_token_id", None)
        if bos is not None and clean_answer and clean_answer[0] == bos:
            clean_answer = clean_answer[1:]
        if bos is not None and patch_answer and patch_answer[0] == bos:
            patch_answer = patch_answer[1:]

        if len(clean_answer) != 1 or len(patch_answer) != 1:
            continue
        if len(clean_prefix) != len(patch_prefix):
            continue
        if clean_prefix + clean_answer != clean_full:
            continue
        if patch_prefix + patch_answer != patch_full:
            continue
        if max_length is not None and len(clean_prefix) > max_length:
            continue

        if use_min_length_only:
            length = len(clean_prefix)
            if length < min_length:
                examples = []
                min_length = length
            elif length > min_length:
                continue

        examples.append(Example(
            clean_prefix=data["clean_prefix"], clean_answer=data["clean_answer"],
            patch_prefix=data["patch_prefix"], patch_answer=data["patch_answer"],
            meta={k: v for k, v in data.items() if k not in
                  ("clean_prefix", "clean_answer", "patch_prefix", "patch_answer")},
        ))
        if len(examples) >= num_examples:
            break
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def select_counterfactual(item: dict, name: str | None) -> dict:
    """Pick the one counterfactual this run is scoped to.

    A run answers one causal question, so there is no default and no
    'first available' fallback -- picking silently would attribute against a
    different intervention than the config names.
    """
    available = sorted(k for k, v in item.items()
                       if isinstance(v, dict) and "prompt" in v)
    if name is None:
        raise ValueError(
            f"data.counterfactual is required for loader='mib'. "
            f"Available in this dataset: {available}"
        )
    if name not in available:
        raise ValueError(
            f"Counterfactual {name!r} not found. Available: {available}"
        )
    return item[name]


def load_mib_examples(path: str, num_examples: int, model, counterfactual: str,
                      split: str = "train",
                      text_column: str | None = None) -> list[Example]:
    """Load a MIB-bench dataset, scoped to a single counterfactual.

    The counterfactual name is NOT stored per row -- it is a config field, so it
    lands in the stage manifest and the cache key, and two counterfactuals
    produce two independent hives rather than two partitions of one.
    """
    from datasets import load_dataset

    dataset = load_dataset(path=path, split=split, streaming=False)
    column = text_column or next(
        (c for c in ("prompt", "text", "sentence", "content") if c in dataset.column_names),
        None,
    )
    if column is None:
        raise ValueError(
            f"Could not detect a text column in {dataset.column_names}; "
            f"set data.text_column explicitly."
        )

    examples: list[Example] = []
    for idx, item in enumerate(dataset):
        cf = select_counterfactual(item, counterfactual)
        examples.append(Example(
            clean_prefix=item[column], clean_answer="",
            patch_prefix=cf["prompt"], patch_answer="",
            meta={"source_idx": idx, "template": item.get("template", ""),
                  "choices": item.get("choices"), "answerKey": item.get("answerKey")},
        ))
        if len(examples) >= num_examples:
            break
    logger.info("Loaded %d examples from %s (counterfactual=%s)",
                len(examples), path, counterfactual)
    return examples


def load_examples(cfg, model, seed: int) -> list[Example]:
    """Dispatch on cfg.loader."""
    if cfg.loader == "sva":
        # SVA stimuli carry exactly one patch_prefix per example, so
        # cfg.counterfactual is not applicable here.
        return load_sva_examples(cfg.path, cfg.num_examples, model,
                                 use_min_length_only=cfg.use_min_length_only,
                                 max_length=cfg.max_length, seed=seed)
    if cfg.loader == "mib":
        return load_mib_examples(cfg.path, cfg.num_examples, model,
                                 counterfactual=cfg.counterfactual,
                                 split=cfg.split, text_column=cfg.text_column)
    raise ValueError(f"Unknown loader {cfg.loader!r}; expected 'sva' or 'mib'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_loaders.py -v`
Expected: 9 passed

- [ ] **Step 5: Vendor the datasets**

```bash
mkdir -p data/marks
cd data/marks
curl -LO https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data/rc_train.json
curl -LO https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data/rc_test.json
curl -LO https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data/simple_train.json
curl -LO https://raw.githubusercontent.com/saprmarks/feature-circuits/main/data/simple_test.json
```

Add `data/` to `.gitignore` — never commit raw data.

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/test_loaders.py .gitignore
git commit -m "feat: SVA (Marks) and MIB dataset loaders"
```

---

## Task 5: Metric rework

**Files:**
- Modify: `src/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Example` from Task 4.
- Produces: `METRICS: dict[str, Callable]`; `get_metric(name) -> Callable`. Every metric has signature `(logits: Tensor, example: Example, model) -> Tensor` returning a **scalar with the sign already folded in** (higher = the behaviour of interest). The `direction` second return value is removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import torch
import pytest
from src.data.loaders import Example
from src.metrics import get_metric


class FakeModel:
    def to_single_token(self, s):
        return {" is": 5, " are": 7, " Mary": 5, " John": 7}[s]


def test_sva_logit_diff_is_patch_minus_clean_at_last_position():
    """Marks' paired metric: logit(patch_answer) - logit(clean_answer)."""
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 5] = 2.0    # clean answer " is"
    logits[0, -1, 7] = 3.0    # patch answer " are"
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() == pytest.approx(1.0)


def test_sva_metric_reads_only_the_last_position():
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 7] = 99.0    # earlier position must be ignored
    logits[0, -1, 5] = 1.0
    logits[0, -1, 7] = 1.0
    ex = Example("the boy", " is", "the boys", " are")
    m = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert m.item() == pytest.approx(0.0)


def test_metrics_return_a_bare_scalar_not_a_direction_tuple():
    logits = torch.zeros(1, 2, 10)
    ex = Example("the boy", " is", "the boys", " are")
    out = get_metric("sva_logit_diff")(logits=logits, example=ex, model=FakeModel())
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("no_such_metric")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_metric'`

- [ ] **Step 3: Rewrite `src/metrics.py`**

```python
# src/metrics.py
"""Task metrics.

Every metric returns a bare scalar with the sign already folded in, so that the
indirect effect grad(m) . delta is literally the IE of the metric being
reported. The previous (value, direction) convention multiplied the effect
AFTER the gradient, which is only equivalent for direction = +/-1 and has no
counterpart in Marks et al. (arXiv:2403.19647) -- they put the sign inside the
metric, before backprop.
"""
from __future__ import annotations

import torch


def sva_logit_diff(logits, example, model, **_):
    """Marks' paired subject-verb agreement metric:
    logit(patch_answer) - logit(clean_answer), read at the final position."""
    correct = model.to_single_token(example.patch_answer)
    clean = model.to_single_token(example.clean_answer)
    return logits[0, -1, correct] - logits[0, -1, clean]


def mib_logit_diff(logits, example, model, **_):
    """MIB-bench: logit(correct choice) - logit(incorrect choice) at the final
    position. Answers must be single tokens (assumes the GPT-2 leading space)."""
    choices, answer_key = example.meta.get("choices"), example.meta.get("answerKey")
    if choices is None or answer_key is None:
        raise ValueError("mib_logit_diff requires 'choices' and 'answerKey' in meta")
    if len(choices) != 2:
        raise ValueError(f"mib_logit_diff expects exactly 2 choices, got {len(choices)}")
    if answer_key not in (0, 1):
        raise ValueError(f"mib_logit_diff expects answerKey in {{0,1}}, got {answer_key!r}")
    try:
        correct = model.to_single_token(" " + choices[answer_key])
        wrong = model.to_single_token(" " + choices[1 - answer_key])
    except Exception as e:
        raise ValueError(
            f"mib_logit_diff requires single-token answers; "
            f"{choices[answer_key]!r}/{choices[1-answer_key]!r} are not"
        ) from e
    return logits[0, -1, correct] - logits[0, -1, wrong]


def neg_next_token_loss(logits, example, model, tokens=None, **_):
    """Negated next-token cross-entropy: higher is better, sign folded in."""
    if tokens is None:
        raise ValueError("neg_next_token_loss requires `tokens`")
    shift_logits = logits[0, :-1, :].contiguous()
    shift_labels = tokens[0, 1:].contiguous()
    return -torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), reduction="sum",
    )


METRICS = {
    "sva_logit_diff": sva_logit_diff,
    "mib_logit_diff": mib_logit_diff,
    "neg_next_token_loss": neg_next_token_loss,
}


def get_metric(name: str):
    if name not in METRICS:
        raise KeyError(f"Unknown metric {name!r}; known: {sorted(METRICS)}")
    return METRICS[name]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "refactor!: metrics return a signed scalar; drop the direction tuple"
```

---

## Task 6: Node attribution module

**Files:**
- Create: `src/attribution/ig.py`, `src/attribution/nodes.py`, `src/attribution/__init__.py`
- Create: `tests/test_ig.py` (move from `src/test_ig.py`), `tests/test_error_node.py` (move from `src/test_error_node.py`), `tests/test_nodes_schema.py`
- Delete: `src/test_ig.py`, `src/test_error_node.py`

**Interfaces:**
- Consumes: `Site`/`Suite` (Task 3), `Example` (Task 4), `get_metric` (Task 5), manifest helpers (Task 2).
- Produces:
  - `integrated_gradient_grad_x(model, clean_tokens, hook, acts_clean, acts_cf, metric_fn, example, steps) -> Tensor[seq, d_model]`
  - `run_node_attribution(cfg, out_dir: Path) -> Path` — writes the hive + `total_effects.parquet` + manifest
  - Parquet schema: `example_idx:int32, site_ordinal:int16, block:int16, kind:string, token_idx:int16, token_id:int32, feature_idx:int32, attribution:float32` (feature_idx = −1 is the SAE error node), hive-partitioned on `kind`, `site_ordinal`.
  - `total_effects.parquet`: `example_idx:int32, metric_clean:float32, metric_patch:float32, total_effect:float32`

- [ ] **Step 1: Move the existing self-checks to pytest, unchanged in substance**

`src/test_ig.py` asserts IG reduces to the analytic gradient for a linear metric, to the midpoint gradient for a quadratic, and to attribution patching at `steps=1`. `src/test_error_node.py` asserts `Σ feature_attr + err_attr == residual_ie`. Both are the correctness guarantees for this module — port them verbatim into `tests/`, adjusting only the import paths.

- [ ] **Step 2: Write the new failing test**

```python
# tests/test_nodes_schema.py
import pyarrow.dataset as ds
from src.attribution.nodes import NODE_SCHEMA, TOTAL_EFFECT_SCHEMA


def test_schema_carries_typed_site_identity_not_a_layer_int():
    names = set(NODE_SCHEMA.names)
    assert {"site_ordinal", "block", "kind"} <= names
    assert "layer" not in names, "typed sites replace the integer layer column"


def test_schema_has_no_aggregation_baked_in():
    """Marks aggregate AFTER edges; node attribution stores raw
    per-(example, position, feature) rows."""
    assert {"example_idx", "token_idx", "feature_idx"} <= set(NODE_SCHEMA.names)


def test_total_effect_schema_is_one_row_per_example():
    assert set(TOTAL_EFFECT_SCHEMA.names) == {
        "example_idx", "metric_clean", "metric_patch", "total_effect"}


def test_counterfactual_is_config_not_a_column():
    """One run, one counterfactual: the choice belongs in the manifest and the
    cache key, not in a partition column that invites mixing."""
    assert "counterfactual" not in set(NODE_SCHEMA.names)
    assert "counterfactual" not in set(TOTAL_EFFECT_SCHEMA.names)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_nodes_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.attribution.nodes'`

- [ ] **Step 4: Move IG out of `main.py` verbatim**

Copy `integrated_gradient_grad_x` from `src/main.py:109-143` into
`src/attribution/ig.py` unchanged except for the `metric_fn` call, which becomes
`metric_fn(logits=logits[i:i+1], example=example, model=model)` to match the new
metric signature from Task 5. Keep the full docstring — it records the proof
that residual-space interpolation equals Marks' SAE-space interpolation.

- [ ] **Step 5: Implement `src/attribution/nodes.py`**

Port the attribution body from `src/main.py:299-423`, with these changes:

1. Iterate `suite.sites` instead of `sorted_sae_items`; write `site.ordinal`,
   `site.block`, `site.kind` instead of `layer`.
2. **Delete the counterfactual loop** (`main.py:318`). Each `Example` already
   carries one clean/patch pair, chosen by `data.counterfactual` at load time.
   One forward on `example.clean_prefix`, one on `example.patch_prefix`, done.
   The token-length guard at `main.py:328-333` stays — it is what keeps clean
   and patch positions in correspondence.
3. Drop the `direction` multiplier — metrics now carry their own sign.
4. Capture the counterfactual forward's logits (currently discarded at
   `main.py:339-340`) and evaluate the same metric on them to get
   `metric_patch`; write `total_effect = metric_patch - metric_clean` to the
   sidecar. Marks compute exactly this in `_pe_attrib`/`_pe_ig` and use it as
   the circuit's `"y"` node.
5. Keep unchanged, because they were verified faithful: `grad_f = grad_acts @ W_dec.T`,
   `attributions = grad_f * (f_cf - f_clean)`, the error node
   `(grad_acts * (err_cf - err_clean)).sum(-1)` at `feature_idx = -1`, and the
   `(f_clean > 0) | (f_cf > 0)` storage mask.
6. The whole body becomes the `compute_fn` passed to `resolve_stage_output`,
   returning `{"row_count": ..., "n_examples": ..., "runtime_s": ...}` for the
   manifest. Do not call `find_cached`/`write_manifest` directly.

```python
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
```

- [ ] **Step 6: Run the full attribution test set**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_ig.py tests/test_error_node.py tests/test_nodes_schema.py -v`
Expected: all pass. If `test_error_node.py` fails, the `direction` removal broke
sign consistency — fix that before proceeding, it is the invariant that feature
attributions plus the error attribution reconstruct the residual-stream IE.

- [ ] **Step 7: Commit**

```bash
git add src/attribution/ tests/test_ig.py tests/test_error_node.py tests/test_nodes_schema.py
git rm src/test_ig.py src/test_error_node.py
git commit -m "feat: extract node attribution into src/attribution; add total_effect"
```

---

## Task 7: Selection stage (aggregation + both methods)

**Files:**
- Create: `src/selection/aggregate.py`, `src/selection/threshold.py`, `src/selection/__init__.py`
- Create: `tests/test_aggregate.py`, `tests/test_threshold.py`
- Modify: `src/calibrated_feature_selection.py` (column rename + stage entry point)

**Interfaces:**
- Consumes: node hive from Task 6.
- Produces:
  - `aggregate(df: DataFrame, mode: str) -> DataFrame` — `mode="none"` keys nodes by `(site_ordinal, feature_idx, token_idx)`; `mode="sum"` signed-sums over `token_idx`, keying by `(site_ordinal, feature_idx)`. Both then take the **mean over examples**. Output columns: `site_ordinal, block, kind, feature_idx, token_idx (or -1), ie`
  - `select_by_threshold(nodes: DataFrame, node_threshold: float) -> DataFrame` — adds `selected: bool`
  - `run_selection(cfg, node_dir: Path, out_dir: Path) -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregate.py
import pandas as pd
import pytest
from src.selection.aggregate import aggregate


def _df(rows):
    return pd.DataFrame(rows, columns=[
        "example_idx", "site_ordinal", "block", "kind",
        "token_idx", "feature_idx", "attribution"])


def test_sum_mode_sums_signed_over_positions_then_means_over_examples():
    """Marks aggregation='sum': signed sum over sequence, then batch mean.
    Cancellation across positions is the whole point -- +0.5 and -0.45 is a
    0.05 node, not a 0.5 node."""
    df = _df([
        [0, 3, 1, "resid", 0, 7, 0.5],
        [0, 3, 1, "resid", 1, 7, -0.45],
        [1, 3, 1, "resid", 0, 7, 0.1],
    ])
    out = aggregate(df, "sum")
    assert len(out) == 1
    assert out.iloc[0]["ie"] == pytest.approx((0.05 + 0.1) / 2)


def test_none_mode_keeps_positions_as_distinct_nodes():
    df = _df([
        [0, 3, 1, "resid", 0, 7, 0.5],
        [0, 3, 1, "resid", 1, 7, -0.45],
        [1, 3, 1, "resid", 0, 7, 0.1],
    ])
    out = aggregate(df, "none").set_index("token_idx")
    assert len(out) == 2
    assert out.loc[0, "ie"] == pytest.approx((0.5 + 0.1) / 2)
    assert out.loc[1, "ie"] == pytest.approx(-0.45 / 2)   # absent = zero effect


def test_example_mean_divides_by_all_examples_not_present_ones():
    """A node firing in 1 of 2 examples has half the mean effect."""
    df = _df([
        [0, 3, 1, "resid", 0, 7, 1.0],
        [1, 3, 1, "resid", 0, 9, 1.0],
    ])
    out = aggregate(df, "sum").set_index("feature_idx")
    assert out.loc[7, "ie"] == pytest.approx(0.5)
    assert out.loc[9, "ie"] == pytest.approx(0.5)


def test_error_nodes_survive_aggregation():
    df = _df([[0, 3, 1, "resid", 0, -1, 0.3]])
    assert (aggregate(df, "sum")["feature_idx"] == -1).any()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        aggregate(_df([]), "median")
```

```python
# tests/test_threshold.py
import pandas as pd
from src.selection.threshold import select_by_threshold


def test_threshold_is_on_absolute_value():
    nodes = pd.DataFrame({"site_ordinal": [0, 0], "feature_idx": [1, 2],
                          "ie": [-0.5, 0.05]})
    out = select_by_threshold(nodes, 0.1)
    assert out.set_index("feature_idx")["selected"].to_dict() == {1: True, 2: False}


def test_threshold_is_strict():
    nodes = pd.DataFrame({"site_ordinal": [0], "feature_idx": [1], "ie": [0.1]})
    assert not select_by_threshold(nodes, 0.1)["selected"].iloc[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_aggregate.py tests/test_threshold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.selection.aggregate'`

- [ ] **Step 3: Implement aggregation**

```python
# src/selection/aggregate.py
"""Aggregation of node IEs across token positions and examples.

Follows Marks et al. App. A.2 (arXiv:2403.19647):

  mode="sum"  (their non-templatic path) -- signed sum over token positions,
              then the example-wise mean. Positions cancel, which is intended.
  mode="none" (their templatic path) -- positions stay distinct nodes; only the
              example-wise mean is taken. Requires length-aligned examples, which
              data.use_min_length_only guarantees.

The example mean divides by the TOTAL number of examples, not the number in
which the node appeared: a node absent from an example had zero effect there.
"""
from __future__ import annotations

import pandas as pd

KEY_COLS = ["site_ordinal", "block", "kind", "feature_idx"]


def aggregate(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in ("none", "sum"):
        raise ValueError(f"Unknown aggregation {mode!r}; expected 'none' or 'sum'")
    if df.empty:
        return df.assign(ie=pd.Series(dtype="float64"))

    n_examples = df["example_idx"].nunique()
    if mode == "sum":
        per_example = (df.groupby(KEY_COLS + ["example_idx"], sort=False)["attribution"]
                         .sum().reset_index())
        keys = KEY_COLS
    else:
        per_example = df.rename(columns={"attribution": "attribution"})
        keys = KEY_COLS + ["token_idx"]

    out = (per_example.groupby(keys, sort=False)["attribution"].sum() / n_examples)
    out = out.rename("ie").reset_index()
    if mode == "sum":
        out["token_idx"] = -1
    return out
```

```python
# src/selection/threshold.py
"""Marks-style node selection: keep nodes whose mean IE exceeds a threshold."""
from __future__ import annotations

import pandas as pd


def select_by_threshold(nodes: pd.DataFrame, node_threshold: float) -> pd.DataFrame:
    """Add a boolean `selected` column for |ie| > node_threshold (strict,
    matching Marks' `effects[submod].abs() > node_threshold`)."""
    out = nodes.copy()
    out["selected"] = out["ie"].abs() > node_threshold
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_aggregate.py tests/test_threshold.py -v`
Expected: 7 passed

- [ ] **Step 5: Adapt the calibrated selector**

In `src/calibrated_feature_selection.py`, rename `GROUP_COLS = ["prompt_idx", "layer"]`
→ `["example_idx", "site_ordinal"]` and `FEATURE_COLS = ["layer", "feature_idx"]`
→ `["site_ordinal", "feature_idx"]`. Add a stage entry point with the same
signature as the threshold path so `run_selection` can dispatch on
`cfg.selection.method`:

```python
def select_calibrated(nodes_raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Hypothesis-testing selection. Consumes the RAW per-(example, position,
    feature) rows, not the aggregate -- its null model is built from the
    per-row null tail."""
```

Run `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_feature_selection_output.py -v`
after the rename (move that file from `src/` to `tests/` first) and fix fallout.

- [ ] **Step 6: Commit**

```bash
git add src/selection/ tests/test_aggregate.py tests/test_threshold.py \
        src/calibrated_feature_selection.py tests/test_feature_selection_output.py
git rm src/test_feature_selection_output.py
git commit -m "feat: selection stage with Marks aggregation and both selectors"
```

---

## Task 8: Edge IE stage, aligned to Marks

The current module scores **all** or **consecutive** resid-layer pairs and, for
non-adjacent pairs, computes **total**-effect edges (no stop-gradient). Marks
score a **fixed local template** of direct edges with stop-gradients on the
intermediates, and represent long-range influence by composition along the resid
chain. Reproduction requires the template; the total-effect variant is preserved
as opt-in for the process-mining direction.

**Files:**
- Create: `src/attribution/edge_template.py`, `tests/test_edge_template.py`, `tests/test_edge_filtering.py`
- Modify: `src/edge_attribution.py` → moved to `src/attribution/edges.py`
- Create: `tests/test_edge_attribution.py` (move from `src/`)
- Delete: `src/edge_attribution.py`, `src/test_edge_attribution.py`

**Interfaces:**
- Consumes: `Suite`/`Site` (Task 3), selection output (Task 7), `total_effects.parquet` (Task 6).
- Produces:
  - `@dataclass(frozen=True) EdgeSpec(src: Site, dst: Site, stop: tuple[Site, ...])`
  - `marks_template(suite: Suite) -> list[EdgeSpec]`
  - `total_effect_template(suite: Suite, pairs: str) -> list[EdgeSpec]`
  - `build_template(suite: Suite, cfg_edges) -> list[EdgeSpec]`
  - `candidate_endpoints(nodes, src_ordinal, dst_ordinal, filter_mode) -> tuple[list[int], list[int]]`
  - `run_edge_attribution(cfg, selection_dir: Path, out_dir: Path) -> Path`
  - parquet columns: `example_idx:int32, src_ordinal:int16, src_feature:int32, dst_ordinal:int16, dst_feature:int32, edge_ie:float32` (feature `-1` = SAE error node; `dst_ordinal = -1` = the `y` node)

- [ ] **Step 1: Write the failing template test**

```python
# tests/test_edge_template.py
import pytest
from src.models.suites import build_suite
from src.attribution.edge_template import marks_template, total_effect_template


def _as_kinds(specs):
    return [(s.src.kind, s.src.block, s.dst.kind, s.dst.block,
             tuple(sorted((x.kind, x.block) for x in s.stop))) for s in specs]


def test_marks_template_matches_circuit_py_for_one_pythia_block():
    """circuit.py, layer L (parallel_attn=True so AM is skipped):
        MR = N(mlp_L,     resid_L)
        AR = N(attn_L,    resid_L, stop=[mlp_L])
        RM = N(resid_L-1, mlp_L,   stop=[attn_L])
        RA = N(resid_L-1, attn_L)
        RR = N(resid_L-1, resid_L, stop=[mlp_L, attn_L])
    """
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    block1 = [s for s in specs if s[3] == 1 or (s[1] == 1 and s[3] == 1)]
    assert ("mlp", 1, "resid", 1, ()) in specs
    assert ("attn", 1, "resid", 1, (("mlp", 1),)) in specs
    assert ("resid", 0, "mlp", 1, (("attn", 1),)) in specs
    assert ("resid", 0, "attn", 1, ()) in specs
    assert ("resid", 0, "resid", 1, (("attn", 1), ("mlp", 1))) in specs


def test_parallel_attn_suppresses_the_attn_to_mlp_edge():
    """circuit.py: `if not parallel_attn: AM_effect = N(attn, mlp)`."""
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    assert not any(s[0] == "attn" and s[2] == "mlp" for s in specs)


def test_embed_feeds_block_zero_in_place_of_resid_minus_one():
    """circuit.py: `prev_resid = resids[layer-1] if layer > 0 else embed`."""
    specs = _as_kinds(marks_template(build_suite("pythia-marks")))
    assert ("embed", -1, "attn", 0, ()) in specs
    assert ("embed", -1, "mlp", 0, (("attn", 0),)) in specs
    assert ("embed", -1, "resid", 0, (("attn", 0), ("mlp", 0))) in specs


def test_template_has_no_long_range_edges():
    """Long-range influence is composition along the resid chain, never an edge."""
    for spec in marks_template(build_suite("pythia-marks")):
        if spec.src.kind == "embed":
            assert spec.dst.block == 0
        else:
            assert spec.dst.block - spec.src.block <= 1


def test_final_resid_connects_to_the_y_node():
    specs = marks_template(build_suite("pythia-marks"))
    y_edges = [s for s in specs if s.dst.ordinal == -1]
    assert len(y_edges) == 1
    assert y_edges[0].src.kind == "resid" and y_edges[0].src.block == 5


def test_resid_only_suite_degenerates_to_the_resid_chain():
    """gpt2-res has no attn/mlp dictionaries, so the template is resid_i ->
    resid_i+1 with nothing to stop-gradient."""
    specs = marks_template(build_suite("gpt2-res"))
    chain = [s for s in specs if s.dst.ordinal != -1]
    assert all(s.src.kind == "resid" and s.dst.kind == "resid" for s in chain)
    assert all(s.dst.ordinal - s.src.ordinal == 1 for s in chain)
    assert all(s.stop == () for s in chain)


def test_total_effect_template_is_all_pairs_without_stopgrads():
    specs = total_effect_template(build_suite("gpt2-res"), pairs="all")
    assert all(s.stop == () for s in specs)
    assert any(s.dst.ordinal - s.src.ordinal > 1 for s in specs), \
        "total-effect mode exists precisely to score non-adjacent pairs"


def test_total_effect_consecutive_is_adjacent_only():
    specs = total_effect_template(build_suite("gpt2-res"), pairs="consecutive")
    assert all(s.dst.ordinal - s.src.ordinal == 1 for s in specs)
```

```python
# tests/test_edge_filtering.py
import pandas as pd
import pytest
from src.attribution.edges import candidate_endpoints


def _nodes():
    return pd.DataFrame({
        "site_ordinal": [1, 1, 3, 3],
        "feature_idx": [10, 11, 20, 21],
        "selected":    [True, False, True, False],
    })


def test_marks_mode_filters_downstream_only():
    """attribution.py:jvp loops over surviving DOWNSTREAM nodes but keeps every
    nonzero upstream feature -- upstream is NOT thresholded."""
    up, down = candidate_endpoints(_nodes(), 1, 3, filter_mode="marks")
    assert set(down) == {20}
    assert set(up) == {10, 11}


def test_both_mode_filters_both_endpoints():
    up, down = candidate_endpoints(_nodes(), 1, 3, filter_mode="both")
    assert set(down) == {20}
    assert set(up) == {10}


def test_unknown_filter_mode_raises():
    with pytest.raises(ValueError):
        candidate_endpoints(_nodes(), 1, 3, filter_mode="neither")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_edge_template.py tests/test_edge_filtering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.attribution.edge_template'`

- [ ] **Step 3: Implement the templates**

```python
# src/attribution/edge_template.py
"""Which (upstream, downstream, stop-gradient) triples to score.

Marks et al. (arXiv:2403.19647) do not score all site pairs. circuit.py walks
layers in reverse and emits a fixed local template, applying stop-gradients to
intermediate submodules so every edge is a DIRECT effect -- the influence of u
on d that is not mediated by anything else in the layer. Long-range influence is
represented by composing edges along the residual chain, never by a long-range
edge. Reproducing their edge counts requires reproducing this topology exactly.

`total_effect_template` preserves this repo's earlier all-pairs / consecutive
behaviour, which computes TOTAL effects (no stop-gradients, gradient flows
through the whole frozen network). That matches the process-mining
"eventually-follows" relation and is transitive and non-exclusive: direct and
total edges must never be summed or mixed in one graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.suites import Site, Suite

# Sentinel destination for the metric node. circuit.py:
#   edges[f"resid_{len(resids)-1}"]["y"] = effects[resids[-1]]
Y_SITE = Site(ordinal=-1, block=-1, kind="y", release="", sae_id="", hook="")


@dataclass(frozen=True)
class EdgeSpec:
    src: Site
    dst: Site
    stop: tuple[Site, ...] = ()


def _by_kind(suite: Suite):
    embed = next((s for s in suite.sites if s.kind == "embed"), None)
    blocks = sorted({s.block for s in suite.sites if s.block >= 0})
    idx = {(s.kind, s.block): s for s in suite.sites}
    return embed, blocks, idx


def marks_template(suite: Suite) -> list[EdgeSpec]:
    """The per-layer template from circuit.py, plus the final resid -> y edge."""
    embed, blocks, idx = _by_kind(suite)
    specs: list[EdgeSpec] = []

    for b in blocks:
        attn, mlp, resid = idx.get(("attn", b)), idx.get(("mlp", b)), idx.get(("resid", b))
        if resid is None:
            continue
        if mlp is not None:
            specs.append(EdgeSpec(mlp, resid))                        # MR
        if attn is not None:
            specs.append(EdgeSpec(attn, resid, tuple(x for x in (mlp,) if x)))  # AR
            if not suite.parallel_attn and mlp is not None:
                specs.append(EdgeSpec(attn, mlp))                     # AM

        prev = idx.get(("resid", b - 1)) if b > 0 else embed
        if prev is None:
            continue
        if mlp is not None:
            specs.append(EdgeSpec(prev, mlp, tuple(x for x in (attn,) if x)))   # RM
        if attn is not None:
            specs.append(EdgeSpec(prev, attn))                        # RA
        specs.append(EdgeSpec(prev, resid,
                              tuple(x for x in (attn, mlp) if x)))    # RR

    last_resid = max((s for s in suite.sites if s.kind == "resid"),
                     key=lambda s: s.ordinal)
    specs.append(EdgeSpec(last_resid, Y_SITE))
    return specs


def total_effect_template(suite: Suite, pairs: str) -> list[EdgeSpec]:
    """All (pairs='all') or adjacent (pairs='consecutive') site pairs, no
    stop-gradients. TOTAL-effect edges; not comparable to Marks' edge counts."""
    if pairs not in ("all", "consecutive"):
        raise ValueError(f"Unknown pairs {pairs!r}; expected 'all' or 'consecutive'")
    sites = sorted(suite.sites, key=lambda s: s.ordinal)
    if pairs == "consecutive":
        return [EdgeSpec(a, b) for a, b in zip(sites, sites[1:])]
    return [EdgeSpec(a, b) for i, a in enumerate(sites) for b in sites[i + 1:]]


def build_template(suite: Suite, cfg_edges) -> list[EdgeSpec]:
    if cfg_edges.template == "marks":
        return marks_template(suite)
    if cfg_edges.template == "total_effect":
        return total_effect_template(suite, cfg_edges.pairs)
    raise ValueError(
        f"Unknown edge template {cfg_edges.template!r}; "
        f"expected 'marks' or 'total_effect'"
    )
```

- [ ] **Step 4: Implement stop-gradients and endpoint filtering in `src/attribution/edges.py`**

Move `src/edge_attribution.py` to `src/attribution/edges.py`, keeping the
verified stop-gradient seeding math intact (`tests/test_edge_attribution.py`
guards it), and make these changes:

1. Consume `EdgeSpec` from `build_template` instead of `_layer_pairs`.
2. Implement `spec.stop` by substituting each intermediate site's activation
   with its `.detach()`ed value in that edge's forward pass. The value is
   unchanged so the forward is bit-identical; no gradient flows through the
   site. This is the TransformerLens equivalent of their
   `submodule.stop_grad()`.
3. Replace `sae_layer_ordinal` / local `load_model_and_saes` with
   `src.models.suites.load_model_and_saes`.
4. Handle `dst = Y_SITE`: the edge value is the downstream node's own IE (they
   assign `effects[resids[-1]]` directly rather than running a jvp).
5. Take `cfg` instead of argparse; route through `resolve_stage_output`.

```python
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
```

- [ ] **Step 5: Add a stop-gradient regression test**

```python
# tests/test_edge_attribution.py (append to the moved file)
import torch
from src.attribution.edges import _detach_site_hook


def test_detach_hook_blocks_gradient_but_preserves_value():
    """Stop-gradient must be a value-preserving identity in the forward pass --
    if it changed activations it would change the edge being measured."""
    x = torch.randn(3, 4, requires_grad=True)
    captured = {}

    def fake_hook_point(t):
        out = _detach_site_hook(t, hook=None)
        captured["out"] = out
        return out

    y = fake_hook_point(x * 2)
    assert torch.equal(captured["out"], (x * 2).detach())
    y.sum().backward()
    assert x.grad is None or torch.all(x.grad == 0)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_edge_template.py tests/test_edge_filtering.py tests/test_edge_attribution.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/attribution/edge_template.py src/attribution/edges.py \
        tests/test_edge_template.py tests/test_edge_filtering.py \
        tests/test_edge_attribution.py
git rm src/edge_attribution.py src/test_edge_attribution.py
git commit -m "feat!: edge topology follows Marks' stop-gradient template

BREAKING: default edges are now DIRECT effects over the fixed per-layer
template (MR/AR/AM/RM/RA/RR + resid->y), not total effects over layer pairs.
The previous all-pairs total-effect behaviour is retained under
edges.template=total_effect for the process-mining direction."

---

## Task 9: Circuit construction

**Files:**
- Create: `src/circuit/build.py`, `src/circuit/__init__.py`, `tests/test_circuit.py`
- Modify: `src/partial_order.py` (column rename only)

**Interfaces:**
- Consumes: selection output (Task 7), edge output (Task 8), `total_effects.parquet` (Task 6).
- Produces:
  - `@dataclass Circuit(nodes: DataFrame, edges: DataFrame, total_effect: float, suite: str, params: dict)`
  - `build_circuit(nodes: DataFrame, edges: DataFrame, total_effect: float, edge_threshold: float) -> Circuit`
  - `save_circuit(c, path)` / `load_circuit(path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_circuit.py
import pandas as pd
import pytest
from src.circuit.build import build_circuit, save_circuit, load_circuit


def _nodes():
    return pd.DataFrame({
        "site_ordinal": [1, 1, 3], "block": [0, 0, 1], "kind": ["attn"] * 3,
        "feature_idx": [10, 11, 20], "token_idx": [-1] * 3,
        "ie": [0.5, 0.01, 0.4], "selected": [True, False, True]})


def _edges():
    return pd.DataFrame({
        "src_ordinal": [1, 1], "src_feature": [10, 11],
        "dst_ordinal": [3, 3], "dst_feature": [20, 20],
        "edge_ie": [0.2, 0.001]})


def test_edges_below_threshold_are_dropped():
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    assert len(c.edges) == 1
    assert c.edges.iloc[0]["src_feature"] == 10


def test_unselected_nodes_are_dropped():
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    assert set(c.nodes["feature_idx"]) == {10, 20}


def test_no_dangling_edges_after_node_pruning():
    """An edge whose endpoint was pruned must not survive."""
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.0)
    kept = set(zip(c.nodes["site_ordinal"], c.nodes["feature_idx"]))
    for _, e in c.edges.iterrows():
        assert (e["src_ordinal"], e["src_feature"]) in kept
        assert (e["dst_ordinal"], e["dst_feature"]) in kept


def test_roundtrip(tmp_path):
    c = build_circuit(_nodes(), _edges(), total_effect=1.0, edge_threshold=0.01)
    save_circuit(c, tmp_path / "c.json")
    back = load_circuit(tmp_path / "c.json")
    assert back.total_effect == pytest.approx(1.0)
    pd.testing.assert_frame_equal(
        back.nodes.reset_index(drop=True), c.nodes.reset_index(drop=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_circuit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.circuit.build'`

- [ ] **Step 3: Implement `src/circuit/build.py`**

`build_circuit` keeps `nodes[nodes.selected]`, keeps edges with
`|edge_ie| > edge_threshold`, then drops edges whose endpoints are not in the
kept node set (this ordering matters: pruning nodes after edges leaves
dangling edges). Serialise as JSON with the two frames in `records` orientation
plus `total_effect`, `suite`, `params`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_circuit.py -v`
Expected: 4 passed

- [ ] **Step 5: Rename columns in `partial_order.py`**

Mechanical: `layer` → `site_ordinal`, `prompt_idx` → `example_idx`,
`src_layer < dst_layer` → `src_ordinal < dst_ordinal`. This module is phase-3
work and is not wired into the pipeline yet; the rename exists so the tree has
no module referencing a column that no longer exists. Add a module-level note:

```python
# NOTE: attn and mlp sites at the same block are CONCURRENT when
# Suite.parallel_attn is True; their consecutive ordinals are a linearisation.
# The Petri-net work must read `kind`/`block` and the suite's parallel_attn
# flag rather than treating site_ordinal as a genuine total order.
```

- [ ] **Step 6: Commit**

```bash
git add src/circuit/ tests/test_circuit.py src/partial_order.py
git commit -m "feat: circuit construction from nodes + edges"
```

---

## Task 10: Faithfulness evaluation

**Files:**
- Create: `src/eval/faithfulness.py`, `src/eval/__init__.py`, `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: `Circuit` (Task 9), `Suite` (Task 3), `Example` (Task 4).
- Produces:
  - `faithfulness(f_c: float, f_m: float, f_empty: float) -> float`
  - `run_with_ablations(model, saes, suite, examples, keep_mask, metric_fn, ablation_fn, handle_errors) -> Tensor`
  - `run_evaluation(cfg, circuit_path, out_dir) -> dict` returning `{"faithfulness", "F_C", "F_M", "F_empty"}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_faithfulness.py
import pytest
from src.eval.faithfulness import faithfulness


def test_formula_matches_marks_ablation_py():
    """(F(C) - F(empty)) / (F(M) - F(empty))"""
    assert faithfulness(f_c=0.8, f_m=1.0, f_empty=0.0) == pytest.approx(0.8)
    assert faithfulness(f_c=0.5, f_m=1.0, f_empty=0.5) == pytest.approx(0.0)
    assert faithfulness(f_c=1.0, f_m=1.0, f_empty=0.5) == pytest.approx(1.0)


def test_circuit_beating_the_full_model_exceeds_one():
    """Not clamped -- values above 1 are a real, reportable outcome."""
    assert faithfulness(f_c=1.2, f_m=1.0, f_empty=0.0) == pytest.approx(1.2)


def test_degenerate_denominator_raises():
    with pytest.raises(ZeroDivisionError):
        faithfulness(f_c=0.5, f_m=1.0, f_empty=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_faithfulness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.faithfulness'`

- [ ] **Step 3: Implement**

Port `run_with_ablations` from their `ablation.py` to TransformerLens hooks: for
each site, encode the clean activation, replace features **not** in the keep
mask with the ablation value, and write back `decode(f) + res`. Ablation values
come from `ablation_fn` over the patch states: `mean` (mean over the example
batch, expanded), `resample` (random row permutation), `zero`. `handle_errors`
controls whether the SAE error term is kept (`keep`), always ablated (`remove`),
or follows the mask (`default`).

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_faithfulness.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ tests/test_faithfulness.py
git commit -m "feat: ablation-based faithfulness evaluation"
```

---

## Task 11: Pipeline orchestration and wandb

**Files:**
- Create: `src/pipeline.py`, `src/utils/tracking.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: every stage function from Tasks 6–10, manifest helpers from Task 2.
- Produces: `run_pipeline(cfg, only_stage: str | None = None) -> dict[str, Path]`; `init_tracking(cfg)` / `log_stage(name, metrics)` / `finish_tracking()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from pathlib import Path
import pytest
from src.pipeline import STAGE_ORDER, resolve_stages


def test_stage_order_matches_the_declared_pipeline():
    assert STAGE_ORDER == ["node_attribution", "selection", "edge_attribution",
                           "circuit", "evaluation"]


def test_single_stage_selection_returns_just_that_stage():
    assert resolve_stages(STAGE_ORDER, only_stage="selection") == ["selection"]


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        resolve_stages(STAGE_ORDER, only_stage="nope")


def test_disabled_stages_are_dropped_in_order():
    assert resolve_stages(["node_attribution", "selection"], only_stage=None) == \
        ["node_attribution", "selection"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Implement the orchestrator**

```python
# src/pipeline.py
"""Stage sequencing for circuit discovery.

Each stage is a plain function taking (cfg, input_dir, out_dir). Caching is the
stage's own responsibility via src.utils.manifest -- there is no Stage base
class, registry, or plugin protocol, because five hard-coded calls in order is
the whole requirement.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.utils.manifest import resolve_stage_output

logger = logging.getLogger(__name__)

STAGE_ORDER = ["node_attribution", "selection", "edge_attribution",
               "circuit", "evaluation"]


def resolve_stages(configured: list[str], only_stage: str | None) -> list[str]:
    if only_stage is None:
        return [s for s in STAGE_ORDER if s in configured]
    if only_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage {only_stage!r}; known: {STAGE_ORDER}")
    return [only_stage]
```

`run_pipeline` then, per stage, calls `resolve_stage_output(cfg, stage,
compute_fn)` — the mechanism from Task 2 — and passes the returned directory to
the next stage. It logs `was_cached` alongside the stage's summary scalars, so a
wandb run records which stages actually ran and which were served from an
existing hive. No stage touches `find_cached` or `write_manifest` itself.

- [ ] **Step 4: Implement tracking**

```python
# src/utils/tracking.py
"""wandb experiment tracking.

Deliberately not Lightning: this pipeline has no training loop, optimizer, or
DataLoader, so a LightningModule/Trainer would wrap a plain iteration and
WandbLogger without a Trainer adds nothing over wandb.log.
"""
from __future__ import annotations

import logging
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
_run = None


def init_tracking(cfg):
    """Start a wandb run if configured. No-op when logging.backend is empty."""
    global _run
    if "wandb" not in cfg.logging.backend:
        return None
    import wandb
    _run = wandb.init(
        project=cfg.logging.project, entity=cfg.logging.entity,
        name=cfg.logging.run_name, group=cfg.logging.group,
        mode="offline" if cfg.logging.offline else "online",
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
    )
    return _run


def log_stage(stage: str, metrics: dict) -> None:
    logger.info("[%s] %s", stage, metrics)
    if _run is not None:
        _run.log({f"{stage}/{k}": v for k, v in metrics.items()})


def finish_tracking() -> None:
    if _run is not None:
        _run.finish()
```

Stage summary scalars to log: `node_attribution` → `n_rows`, `n_examples`,
`mean_total_effect`, `runtime_s`; `selection` → `n_nodes`, `n_selected`,
`selected_frac`; `edge_attribution` → `n_edges_scored`, `runtime_s`;
`circuit` → `n_nodes`, `n_edges`; `evaluation` → `faithfulness`, `F_C`, `F_M`,
`F_empty`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the whole suite**

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/ -v -m "not slow"`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py src/utils/tracking.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with manifest caching and wandb"
```

---

## Task 12: Resolve the two open questions

**Files:**
- Create: `scripts/check_sae_reconstruction.py`, `docs/reference-numbers.md`

**Interfaces:**
- Consumes: `Suite` (Task 3).
- Produces: documented answers to spec Q1 and Q2.

- [ ] **Step 1: Q1 — verify the ctigges SAEs reconstruct**

The `pythia-70m-deduped-*-sm` releases are *processed* (folded/normalised)
conversions of Marks' `dictionary_learning` dictionaries. Nothing downstream is
trustworthy if they do not reconstruct.

```python
# scripts/check_sae_reconstruction.py
"""Sanity-check a suite's SAEs: variance explained and L0 per site.

Marks report per-layer variance explained in their Appendix B.1 tables
(arXiv:2403.19647); anything far below those is a loading bug, not a research
finding.
"""
```

For each site: run ~50 sequences of `NeelNanda/pile-10k` through the model,
capture the hook activation `x`, compute `x_hat = sae.decode(sae.encode(x))`,
and report `1 - Var(x - x_hat)/Var(x)` plus mean L0. Write results to
`docs/reference-numbers.md`.

Acceptance: variance explained ≥ 0.80 at every site and L0 in the tens-to-low-hundreds.
If a site fails, stop and investigate the release before continuing — do not
proceed to Task 13.

- [ ] **Step 2: Q2 — get Marks' published faithfulness numbers**

Fetch `experiments/faithfulness.ipynb` from their repo and read off the
faithfulness for `rc_train` at node 0.1 / edge 0.01 / aggregation `none`, along
with the node count. Record both in `docs/reference-numbers.md` with the source
cell. This is the acceptance target for Task 13 — without it there is nothing to
compare against.

- [ ] **Step 3: Commit**

```bash
git add scripts/check_sae_reconstruction.py docs/reference-numbers.md
git commit -m "docs: SAE reconstruction check and Marks reference numbers"
```

---

## Task 13: Reproduction run

**Files:**
- Create: `tests/test_end_to_end.py`, `docs/reproduction-report.md`

- [ ] **Step 1: Smoke test on 5 examples**

```python
# tests/test_end_to_end.py
import pytest
from pathlib import Path
from src.utils.initialization import load_config
from src.pipeline import run_pipeline


@pytest.mark.slow
def test_pipeline_runs_end_to_end_on_five_examples(tmp_path):
    cfg = load_config(Path("configs/pythia_marks_rc.yaml"), [
        "data.num_examples=5",
        "attribution.ig_steps=2",
        f"run.output_root={tmp_path}",
        "logging.backend=[]",
    ])
    out = run_pipeline(cfg)
    assert (out["circuit"] / "manifest.json").exists()
    assert out["evaluation"] is not None


@pytest.mark.slow
def test_second_run_reuses_the_cache(tmp_path):
    args = ["data.num_examples=5", "attribution.ig_steps=2",
            f"run.output_root={tmp_path}", "logging.backend=[]"]
    cfg = load_config(Path("configs/pythia_marks_rc.yaml"), args)
    first = run_pipeline(cfg)
    second = run_pipeline(cfg)
    assert first["node_attribution"] == second["node_attribution"]
```

Run: `C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_end_to_end.py -v -m slow`

- [ ] **Step 2: Full reproduction run**

```bash
C:/Users/Maxim/anaconda3/python.exe -m src.main -c configs/pythia_marks_rc.yaml
```

100 examples, IG 10 steps, aggregation `none`, node 0.1, edge 0.01 — matching
`scripts/get_circuit.sh EleutherAI/pythia-70m-deduped rc_train 0.1 0.01 none`.

- [ ] **Step 3: Compare and write the report**

`docs/reproduction-report.md` records: our faithfulness vs theirs (from Task 12),
node and edge counts vs theirs, per-site node counts, and every known
divergence from the spec (§2 gaps, edge filter mode, aggregation ordering).

Acceptance: faithfulness within 0.1 of their published value, and the edge set
built from the same topology (`edges.template=marks`) so edge counts are
comparable at all. If it misses, the report must name the suspected cause and
the next diagnostic — the ranked candidates are (a) processed-SAE mismatch,
(b) TransformerLens vs nnsight hook placement (their `attn` is
`layer.attention` output, ours is `hook_attn_out`; their `mlp` is `layer.mlp`,
ours is `hook_mlp_out`), (c) stop-gradient coverage in the edge template,
(d) IG step-grid or batching differences.

- [ ] **Step 4: Commit**

```bash
git add docs/reproduction-report.md tests/test_end_to_end.py
git commit -m "docs: Marks et al. reproduction report"
```

---

## Follow-on plans (not in scope here)

1. **Calibrated vs threshold selection** — run both selectors over identical
   node hives across a threshold sweep, compare faithfulness-vs-node-count
   curves. This is the phase-2 research question and needs its own plan; the
   config switch (`selection.method`) and the caching that makes a sweep cheap
   are delivered here.
2. **Petri-net / partial-order circuits** — phase 3. `partial_order.py` gets its
   real rework there, reading `kind`/`block`/`parallel_attn` to build a genuine
   partial order instead of treating `site_ordinal` as total.
3. **Exact-patching validation (spec G7)** — port `_pe_exact` and reproduce
   their Fig. 25 approximation-quality scatter for a handful of nodes.
