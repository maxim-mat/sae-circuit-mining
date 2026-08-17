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
    # None = no cap, i.e. every example surviving the filters. Marks sample:
    # their get_circuit.sh uses 100 of rc_train's 105,600 rows, of which
    # ~77,764 clear the min-length filter, so None here is ~650 GPU-hours
    # at ig_steps=10. Viable for the small structures, not for rc.
    num_examples: Optional[int] = 100
    use_min_length_only: bool = True   # required for aggregation="none"
    max_length: Optional[int] = None
    text_column: Optional[str] = None  # mib only
    split: str = "train"               # mib only
    # Which Example.meta key identifies an example's template. Recorded in the
    # examples.parquet sidecar ALWAYS; whether it is used is selection's call
    # (selection.block_col). "case" for Marks' SVA (the agreement configuration),
    # "template" for MIB. None puts every example in one template.
    template_field: Optional[str] = None
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
    # "threshold"        -- Marks: |mean IE| > node_threshold
    # "calibrated"       -- screen + recurrence test (the HA arm)
    # "screen_magnitude" -- screen + rank by sum|IE|, no significance test.
    #                       Decomposes the HA arm: if this matches "calibrated",
    #                       the Stage-1 screen is doing the work, not the test.
    # "random"           -- uniform k nodes, the floor every other arm must beat
    method: str = "threshold"
    node_threshold: float = 0.1
    # "random" and "screen_magnitude": how many nodes to select. Sweep it to
    # trace a curve at the same node counts the other arms produce.
    k: int = 100
    # calibrated-only. The two rates are NOT the same knob:
    #   screen_q -- Stage 1, expected false positives per (example, site) group,
    #               which sets the per-group noise floor c_g.
    #   fdr_q    -- Stage 2, the Benjamini-Hochberg level for the recurrence
    #               test across examples.
    fdr_q: float = 0.05
    screen_q: float = 1.0
    # Shrinkage weight for the Stage-1 scale estimate: the per-group MAD is
    # pulled toward the per-site pooled MAD as
    #     MAD_shrunk = (n_g * MAD_g + screen_n0 * MAD_pool) / (n_g + screen_n0)
    # so screen_n0 is a pseudo-count -- the pool is worth this many observations.
    # A group needs ~screen_n0 active features before its own scale outweighs
    # the pool's. Raise it to trust the pool more (stabler, less adaptive).
    screen_n0: float = 50.0
    min_recurrence: float = 0.1
    # A genuine causal contributor should push the metric the same way each
    # time; under the null the score is sign-symmetric. None disables the check.
    min_sign_consistency: Optional[float] = 0.8
    # Effect-size floor on the selected feature's median |IE|. None disables it.
    # The recurrence test counts how OFTEN a feature clears the noise floor and
    # is blind to how far it clears it, so a consistently-tiny feature outranks
    # a rarely-huge one. This is the knob that reintroduces magnitude.
    min_median_abs: Optional[float] = None
    # Level at which the Stage-1 MAD is pooled before shrinkage: "site_ordinal"
    # pools across examples at one site, "kind" pools attn/mlp/resid separately.
    screen_pool_col: str = "site_ordinal"
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
    # None = no cap. Note run_with_ablations puts every eval example in ONE
    # batched forward, so this is bounded by memory, not just time.
    num_examples: Optional[int] = 100
    ablation: str = "mean"             # "mean" | "resample" | "zero"
    # Marks' four SAE-error treatments, which are the separate lines of their
    # Fig. 3: "default" lets the circuit mask decide (features), "remove"
    # ablates every error node (features_wo_errs), "resid_only" ablates attn and
    # MLP errors but leaves resid ones to the mask (features_wo_some_errs), and
    # "keep" never ablates an error term.
    handle_errors: str = "default"     # "default" | "keep" | "remove" | "resid_only"
    start_layer: int = -1
    # Per-node leave-one-out necessity (Wang et al. arXiv:2211.00593 section 4.2,
    # approximated with K={}). Costs one forward pass per circuit node, so it is
    # opt-in. Marks do not measure this; it is for comparing selectors.
    minimality: bool = False


@dataclass
class LoggingConfig:
    """Experiment tracking. The three names form wandb's hierarchy:
    project (the whole line of work) > experiment (one comparison, wandb's
    `group`) > run_name (this arm of it, wandb's `name`). Set all three
    explicitly in a config -- a null run_name gets a random wandb codename,
    which makes two arms of the same comparison indistinguishable in the UI.
    """
    backend: List[str] = field(default_factory=lambda: [])   # [] | ["wandb"]
    project: str = "sae-circuit-mining"
    experiment: Optional[str] = None     # -> wandb group
    run_name: Optional[str] = None       # -> wandb name
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: [])
    notes: Optional[str] = None
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
