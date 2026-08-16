# Spec — Configurable circuit-discovery pipeline

**Status:** active
**Date:** 2026-08-16
**Reference implementation:** `saprmarks/feature-circuits` @ `7fbd82b`
**Paper:** Marks et al. 2025 (ICLR), *Sparse Feature Circuits*, arXiv:2403.19647

---

## 1. Goal

Three research phases, in order:

1. **Reproduce** at least one published Marks et al. circuit end-to-end, with a
   faithfulness number comparable to theirs.
2. **Test the hypothesis-testing thresholding** (`calibrated_feature_selection`)
   against Marks' plain `|mean IE| > T` on the same data — does calibrated
   selection buy anything measurable?
3. **Transition downstream** to partial-order / Petri-net circuit
   representations (out of scope for this spec; the data model must not
   foreclose it).

This spec covers the infrastructure needed for (1) and (2), and the data-model
decisions that keep (3) reachable.

---

## 2. Verification of current node attribution (completed 2026-08-16)

`src/main.py` node attribution was compared line-by-line against
`attribution.py` (`_pe_attrib`, `_pe_ig`) and `activation_utils.py`.

### Faithful

| Marks | ours | note |
|---|---|---|
| `_pe_attrib` substitutes `x_recon = x_hat + residual`, saves grads on `f`, `residual` | `retain_grad()` on raw resid; `grad_f = grad_acts @ W_dec.T` | equivalent — `decode` is affine, so `∂m/∂f_i = (∂m/∂x)·W_dec[i]`, `∂m/∂res = ∂m/∂x`. Their substitution is a numerical no-op that exists only to give autograd a handle |
| `effect = delta @ grad` → `act: delta.act*grad.act`, `resc: (delta.res*grad.res).sum(-1)` | `grad_f * (f_cf - f_clean)`; `(grad_acts*(err_cf-err_clean)).sum(-1)` | identical, incl. error node as a d_model contraction |
| `delta = patch_state - clean_state` | `f_corrupt - f_clean` | same orientation |
| error node at index `n_feats+1` of `to_tensor()` | `feature_idx = -1` | same node |
| `_pe_ig`: `alpha = step/steps`, interpolate `(f, res)` in SAE space, mean of per-step grads | interpolate `x` directly, `x_interp.grad.mean(0)` | equivalent (affine decode, coefficients sum to 1); identical α grid incl. omitted α=1 |
| per-submodule trace, others clean | one `ig_hook` per hook point | same |
| `metric.sum().backward()` over batched steps | `total_metric = Σ mᵢ; backward()` | same decoupling trick |

`raw_mask = (f_clean>0)|(f_corrupt>0)` is lossless (both zero ⟹ delta zero ⟹
effect zero) — not a divergence.

**Conclusion: the IE math is correct.** No fix required.

### Gaps (drive this spec)

| # | Gap | Consequence |
|---|---|---|
| G1 | resid-only SAEs (`gpt2-small-res-jb`) | no attn/mlp/embed nodes; no published circuit is reproducible |
| G2 | gpt2-small; paper is pythia-70m-deduped / gemma-2-2b | no published baseline to compare against |
| G3 | metric is `choices[answerKey] − choices[1−answerKey]` on the clean item; theirs is `logit(patch_answer) − logit(clean_answer)` | coincide for name-swap IOI, diverge in general |
| G4 | no `total_effect` | no `"y"` node; nothing to scale IEs against |
| G5 | no ablation-based faithfulness | the headline number of a reproduction is unavailable |
| G6 | `direction` multiplier applied *after* the gradient | no counterpart in Marks (they fold the sign into the metric); inert at `+1` today, fragile |
| G7 | no exact-patching ground truth (`_pe_exact`) | approximation quality unvalidated against real interventions |

### Aggregation semantics (established)

- Marks `aggregation="sum"` (non-templatic): nodes `.sum(dim=1)` over sequence,
  then `.mean(dim=0)` over batch; edges `.sum(dim=(1,4))` over both endpoints'
  positions. **Signed** sums.
- Marks `aggregation="none"` (templatic): no sequence sum; node identity is
  `(position, feature)`. Enabled by `load_examples(..., use_min_length_only=True)`,
  which discards every example not of the single shortest token length, forcing
  positions to align across the batch.
- Their justification for not summing on templatic data (§3.1, App. A.2, Fig. 6)
  is that the **cross-example mean** requires position `i` to mean the same thing
  in every example. It is not a claim that summing is wrong.
- Our pipeline currently thresholds per token *first* (top-k per (case, layer)),
  then takes max-|·| per activity, then sums survivors — an order of operations
  matching neither Marks variant, and one that destroys cross-position
  cancellation. Our `abs_attr = Σ|IE|` has no analogue in their method.

### Edge filtering (established)

From `circuit.py` + `attribution.py:jvp`:

- `features_by_submod = {s: effects[s].abs() > node_threshold}` — computed on
  the **raw per-(example, position, feature)** effect, before any aggregation.
- `jvp` loops one backward per surviving **downstream** node
  (`downstream_idxs = downstream_features.to_tensor().nonzero()`).
- **Upstream is not filtered**: `vjv = (upstream_act.grad @ right_vec)` is
  computed for all upstream features and every nonzero entry is kept.
- Edges are pruned only later, by `edge_threshold`.
- `node_threshold` is therefore used twice with different semantics: per-example
  gating for edge computation, and again on the *aggregated* tensor for the
  final node set (`ablation.py`: `circuit[submod.name].abs() > threshold`).

### Edge topology (established)

Marks do **not** compute edges over all upstream/downstream pairs. `circuit.py`
walks layers in reverse and emits a fixed local template per layer, with
stop-gradients on the intermediates so every edge is a **direct** effect:

```
MR = N(mlp_L,      resid_L)                    # no stopgrad
AR = N(attn_L,     resid_L,  stop=[mlp_L])
AM = N(attn_L,     mlp_L)                      # skipped when parallel_attn
RM = N(resid_L-1,  mlp_L,    stop=[attn_L])
RA = N(resid_L-1,  attn_L)
RR = N(resid_L-1,  resid_L,  stop=[mlp_L, attn_L])
```

plus `edges["resid_last"]["y"] = effects[resid_last]`. `resid_-1` is `embed`.
Long-range influence is represented by **composition along the resid chain**,
never by a long-range edge.

Our `edge_attribution.py` diverges on three axes: it filters **both** endpoints,
it scores **all** or **consecutive** resid-layer pairs rather than the template,
and for non-adjacent pairs it computes **total**-effect edges (no stop-gradient,
gradient flows through the whole frozen network) rather than direct ones. The
total-effect variant was a deliberate choice for the process-mining
"eventually-follows" reading and is worth keeping — but it is not Marks, cannot
be compared to their edge counts, and must not be the default while
reproduction is the goal.

### Faithfulness definition (established)

Not `metric_patch − metric_clean`. That quantity is `total_effect`, which
becomes the `"y"` node. Faithfulness is ablation-based (`ablation.py`):

```
faithfulness = (F(C) − F(∅)) / (F(M) − F(∅))
```

where `F(M)` is the unmodified model, `F(C)` runs with everything *outside* the
circuit mean-ablated, and `F(∅)` ablates everything. Ablation styles: `mean`
(default), `resample`, `zero`. `handle_errors ∈ {default, keep, remove}`
controls whether SAE error terms are ablated.

---

## 3. Key enabling finding

**Marks' Pythia SAEs are already in `sae_lens`** — no new dependency, no zip
download, no `dictionary_learning`:

| release | ids |
|---|---|
| `pythia-70m-deduped-res-sm` | `blocks.0.hook_resid_pre` (= their embed SAE), `blocks.{0..5}.hook_resid_post` |
| `pythia-70m-deduped-att-sm` | `blocks.{0..5}.hook_attn_out` |
| `pythia-70m-deduped-mlp-sm` | `blocks.{0..5}.hook_mlp_out` |

(`ctigges/pythia-70m-deduped__*_processed` — the same `p70d-sm` dictionaries the
paper's README links to on Neuronpedia.) All 19 sites load through the existing
`SAE.from_pretrained` path. TransformerLens supports `pythia-70m-deduped`.

**Unverified:** these are "processed" (folded/normalized) variants. Reconstruction
quality must be checked before any number is trusted.

---

## 4. Data

Marks' subject–verb agreement stimuli, `data/*.json` in their repo — **jsonl
despite the extension**. Structures: `simple`, `nounpp`, `rc`, `within_rc`;
`*_train` for discovery, `*_test` for faithfulness evaluation.

Fields: `clean_prefix`, `clean_answer`, `patch_prefix`, `patch_answer`.

Their loader filters: single-token answers, `len(clean_prefix) == len(patch_prefix)`,
tokenization round-trip (`clean_prefix + clean_answer == clean_full`), optional
`max_length`, and `use_min_length_only` (keep only the shortest length seen,
restarting collection whenever a shorter example appears).

README worked example: `rc_train`, node 0.1, edge 0.01, aggregation `none`, 100
examples.

---

## 5. Requirements

### R1 — Module structure

Node attribution moves out of `main.py` into its own module. `main.py` becomes a
thin entry point that runs the pipeline:

```
node attribution → filtering → edge IE → circuit creation → metrics
```

### R2 — Configuration

OmegaConf structured configs, matching the idiom in `trace-denoise-refactor` and
`lod-generation`: dataclass schema with `MISSING`, `load_config` merging
`structured(Config) ← YAML ← CLI dotlist overrides`, per-experiment
`configs/*.yaml`. Everything configurable — model, SAE suite, dataset,
attribution method, aggregation, thresholds, selection method.

### R3 — Dual model/SAE suites

Both must work from config alone:

- `gpt2-res` — gpt2-small + `gpt2-small-res-jb` (current behaviour, resid-only)
- `pythia-marks` — pythia-70m-deduped + the three `-sm` releases above (embed,
  attn, mlp, resid)

### R4 — Manifest-based caching

Every stage writes its outputs to a directory carrying a `manifest.json`
recording the resolved config subtree that produced it, plus provenance. Before
computing, a stage searches existing output directories for a manifest whose
parameters match; on a match it reuses rather than recomputes.

### R5 — Experiment tracking

wandb. OmegaConf → `wandb.config`; per-stage summary scalars and artifacts.
**Not** Lightning: there is no training loop, optimizer, epoch, or DataLoader
here, so `LightningModule`/`Trainer` would wrap a plain iteration and
`WandbLogger` without a `Trainer` adds nothing over `wandb.log`.

### R6 — Selection methods, side by side

Both selectable from config on identical inputs:

- `threshold` — Marks' `|mean IE| > T`
- `calibrated` — the existing hypothesis-testing selection

Phase 2 of the goal is the comparison between them.

### R7 — Preserve the Petri-net path

Node identity must retain the information needed to recover concurrency
(attn ‖ mlp in Pythia) without regenerating attributions.

---

## 6. Data model decisions

### D1 — Site identity replaces integer `layer`

`sae_layer_ordinal(sae_id) -> int` cannot express typed sites. attn and mlp are
**parallel** in Pythia (their `circuit.py` sets `parallel_attn=True` and skips
the attn→mlp edge), so no single integer honestly orders them.

Replace with an explicit per-suite site table:

```python
@dataclass(frozen=True)
class Site:
    ordinal: int   # strict position in execution order; the total order used by edge code
    block: int     # transformer block index; -1 for embed
    kind: str      # "embed" | "attn" | "mlp" | "resid"
    release: str
    sae_id: str
    hook: str
```

Parquet columns become `site_ordinal`, `block`, `kind` — the `layer` column is
dropped. `parallel_attn` is recorded in the suite and the manifest, so the
Petri-net stage can recover that `attn_b ‖ mlp_b` even though their ordinals are
consecutive. Consecutive ordinals for concurrent sites introduce a false
ordering; the resulting edges score ~0 and are harmless for phases 1–2.

Ordinals:
- `pythia-marks`: embed=0; per block `b`: attn=3b+1, mlp=3b+2, resid=3b+3
- `gpt2-res`: `resid_pre_{0..11}` = 0..11, `resid_post_11` = 12 (unchanged from
  the current `sae_layer_ordinal` result)

### D2 — Aggregation applied in the filtering stage

Node attribution writes per-(example, position, feature) rows, unaggregated —
matching Marks, where aggregation happens after edges. The filtering stage
applies `aggregation ∈ {none, sum}` and the cross-example mean.

### D3 — `total_effect` sidecar

One row per example: `metric_clean`, `metric_patch`, `total_effect`. Not
per-event. Feeds the circuit's `"y"` node and the `resid_last → y` edge.

### D4 — One counterfactual per run

A run is scoped to exactly one counterfactual. The `counterfactual` column is
removed from the parquet schema and becomes a **config field**
(`data.counterfactual`), which means it lands in the manifest and participates in
the cache key — two counterfactuals produce two independent, independently
cacheable hives rather than two partitions of one.

Rationale: causal questions do not mix. A circuit discovered against an
ABB→ABC patch answers a different question than one against a name-swap patch,
and averaging or co-storing them invites exactly the silent mixing this pipeline
is meant to avoid. Marks' own data carries one patch per clean example.

Required for `loader: mib` (validated at load time against the item's available
counterfactual keys); ignored for `loader: sva`, which has a single
`patch_prefix` per example.

### D5 — Edge topology follows Marks

Default `edges.template = "marks"`: the fixed per-layer template above, with
stop-gradients, over typed sites. `edges.template = "total_effect"` preserves
the existing all-pairs / consecutive-pairs total-effect behaviour for the
process-mining direction, opt-in and clearly labelled as non-comparable to
Marks' edge counts.

Stop-gradients are implemented by substituting each intermediate site's
activation with its `.detach()`ed value during the edge's forward pass — the
value is unchanged, so the forward is identical, but no gradient flows through
that site. This is equivalent to their `submodule.stop_grad()` under nnsight.

---

## 7. Out of scope

- Petri-net / partial-order mining (phase 3; `partial_order.py` stays as-is
  until its inputs stabilise)
- gemma-2-2b suite
- SHIFT / Bias-in-Bios experiments
- Cluster (non-templatic) circuits
- `_pe_exact` ground-truth validation (G7) — desirable, deferred

## 8. Open questions

- **Q1:** Do the `ctigges` processed SAEs reconstruct equivalently to the
  original `dictionary_learning` `.pt` dictionaries? Must be measured.
- **Q2:** What faithfulness value do Marks report for `rc_train` at node 0.1 /
  edge 0.01 / agg `none`? Needed as the acceptance target; read from
  `experiments/faithfulness.ipynb`.
