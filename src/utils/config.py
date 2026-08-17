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
    # calibrated-only. The two rates are NOT the same knob:
    #   screen_q -- Stage 1, expected false positives per (example, site) group,
    #               which sets the per-group noise floor c_g.
    #   fdr_q    -- Stage 2, the Benjamini-Hochberg level for the recurrence
    #               test across examples.
    fdr_q: float = 0.05
    screen_q: float = 1.0
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
