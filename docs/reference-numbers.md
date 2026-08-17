# Reference numbers

Acceptance targets for the reproduction (plan Task 12/13). Source: Marks et al.
2025 (ICLR), *Sparse Feature Circuits*, arXiv:2403.19647v3, and the reference
implementation `saprmarks/feature-circuits` @ `7fbd82b`.

---

## Q2 — Marks' published faithfulness for the RC circuit

**Answer: faithfulness 0.21 at 86 nodes, Pythia-70M, agreement across a relative
clause, T_N = 0.1 and T_E = 0.01.**

- §3.3: *"we tune our node threshold to produce a small circuit with faithfulness
  > 0.2. For Pythia, this results in a circuit with 86 nodes and faithfulness
  0.21."*
- Fig. 7 caption: *"The feature circuit for agreement across a relative clause in
  Pythia-70M, computed using T_N = 0.1 and T_E = 0.01."*

Those are the thresholds in `configs/pythia_marks_rc.yaml`, so **86 nodes /
0.21** is the target, not a curve read-off.

### Why the notebook could not answer this

`experiments/faithfulness.ipynb` was fetched and inspected. It contains **no
executed outputs** (`execute_result` count: 0) and writes its figure straight to
`faithfulness.pdf`. It also does not evaluate a single threshold: it loads
`../circuits/{model}_{dataset}_train_n100_aggnone_nodeall.pt` — circuits saved
with *no* node threshold applied — and sweeps `thresholds =
t.logspace(-4, 0, 15)`, plotting faithfulness against node count over
`datasets = ["rc", "nounpp", "simple", "within_rc"]`. So the notebook produces
Fig. 3's curve, and the single comparable number lives in §3.3 instead.

### Evaluation settings to match

| Setting | Their value | Where |
|---|---|---|
| eval examples | 40, from the `*_test` split, `use_min_length_only=True` | notebook `get_fcs` |
| ablation | mean over the batch, **position-specific** (`x.mean(dim=0).expand_as(x)`) | notebook |
| scored segment | **only the latter 2/3 of the circuit** — the first 1/3 is left unablated | §3.2 |
| faithfulness | `(m(C) − m(∅)) / (m(M) − m(∅))`, unclamped | §3.2 |

The 2/3 rule is why `EvalConfig.start_layer` exists: early features key on
specific tokens, and the train/test splits do not share tokens, so scoring the
early segment measures vocabulary overlap rather than circuit quality.
`start_layer = -1` resolves to `ceil(n_blocks / 3)` = **2** for pythia-70m
(`src/eval/faithfulness.py:resolve_start_layer`).

Note `evaluation.num_examples` defaults to 100 in our schema; set it to 40 to
match the notebook exactly.

---

## Q1 — SAE reconstruction targets

Marks' own Pythia-70M SAE evaluations (App. B.1, Tables 4–6). These are the
numbers `scripts/check_sae_reconstruction.py` must approximately reproduce for
the `ctigges/pythia-70m-deduped__*_processed` releases; a site far below its row
here is a loading bug, not a research finding.

| Site | % Variance Explained | L0 |
|---|---|---|
| Attn 0 | 92% | 128 |
| Attn 1 | 87% | 127 |
| Attn 2 | 90% | 215 |
| Attn 3 | 89% | 169 |
| Attn 4 | 83% | 132 |
| Attn 5 | 89% | 144 |
| MLP 0 | 97% | 5 |
| MLP 1 | 85% | 69 |
| MLP 2 | 99% | 88 |
| MLP 3 | 88% | 160 |
| MLP 4 | 92% | 100 |
| MLP 5 | 96% | 102 |
| Resid 0 | 92% | 59 |
| Resid 1 | 85% | 54 |
| Resid 2 | 96% | 108 |
| Resid 3 | 96% | 68 |
| Resid 4 | 88% | 61 |
| Resid 5 | 90% | 72 |

Lowest variance explained is 83% (Attn 4), which is what sets the script's 0.80
acceptance floor. **L0 varies by two orders of magnitude across sites** — MLP 0
sits at 5 — so "L0 in the tens to low hundreds" is a suite-level sanity band, not
a per-site test: compare each site against its own row above.

### Measured, 2026-08-17 — resolved, both anomalies were ours

`python -m scripts.check_sae_reconstruction --suite pythia-marks --n-seqs 64`.
Full table in `docs/sae-reconstruction.md`. **Marks' dictionaries are fine and
the `ctigges` conversion is faithful.** The two sites that initially failed the
0.80 floor were both artefacts on our side, one in the pipeline and one in the
diagnostic:

| Apparent problem | Actual cause | Status |
|---|---|---|
| `blocks.4.hook_attn_out` 0.733 vs 0.83 | variance explained was computed about a **per-sequence** mean, shrinking the denominator | **fixed** in the script; now 0.827 |
| `blocks.3.hook_resid_post` 0.790 vs 0.96 | `load_model_and_saes` applies TransformerLens weight processing the SAEs were not trained for | **outstanding**, see below |

#### The `ctigges` conversion is bit-faithful to the original dictionaries

Marks' own `dictionary_learning` `.pt` files were pulled out of
`saprmarks/pythia-70m-deduped-saes` (a 2.37 GB zip, fetched member-wise over
HTTP range requests) and their `AutoEncoder` reimplemented from its four
tensors — `encode(x) = ReLU(W_enc(x − bias) + b_enc)`, `decode(f) = W_dec f + bias`
— so no `dictionary_learning` dependency was added. Scored on identical inputs:

| Site | Marks' `.pt` | `ctigges` `_processed` |
|---|---|---|
| `blocks.4.hook_attn_out` | 0.7338 / L0 136.1 | 0.7338 / L0 136.1 |
| `blocks.0.hook_attn_out` | 0.9027 / L0 129.6 | 0.9027 / L0 129.6 |

Identical to four decimal places under both weight processings. The conversion
introduces no loss; the sae_lens catalog route in spec §3 is sound.

#### With both artefacts removed, we match their tables

128 sequences, unprocessed weights, global-mean variance:

| Site | ours | Marks | Δ | L0 ours / theirs |
|---|---|---|---|---|
| `blocks.4.hook_attn_out` | 0.8290 | 0.83 | −0.001 | 132 / 132 |
| `blocks.0.hook_attn_out` | 0.9178 | 0.92 | −0.002 | 129 / 128 |
| `blocks.5.hook_attn_out` | 0.8609 | 0.89 | −0.029 | 145 / 144 |
| `blocks.3.hook_resid_post` | 0.9432 | 0.96 | −0.017 | 69 / 68 |
| `blocks.1.hook_resid_post` | 0.8462 | 0.85 | −0.004 | 54 / 54 |
| `blocks.3.hook_mlp_out` | 0.8694 | 0.88 | −0.011 | 164 / 160 |

Residual deltas of 1–3 points are sample size and corpus (4k–16k tokens of
`NeelNanda/pile-10k` against their eval set).

### Root cause of the resid-3 gap — TransformerLens weight processing

Same SAE, same sequences, only the model's weight processing differs:

| Site | `from_pretrained` (fold_ln + center_writing_weights + center_unembed) | `from_pretrained_no_processing` | Marks |
|---|---|---|---|
| `blocks.3.hook_resid_post` | 0.790 | **0.942** | 96% |
| `blocks.2.hook_resid_post` | 0.946 | 0.952 | 96% |
| `blocks.0.hook_resid_post` | 0.908 | 0.915 | 92% |
| `blocks.4.hook_attn_out` | 0.733 | 0.734 | 83% |

**Why.** `center_writing_weights` subtracts the mean-over-`d_model` component
from every matrix that writes to the residual stream. That component is
invisible to the *model* (the next LayerNorm removes it) but perfectly visible
to an *SAE*, which was trained on the uncentered activation. Sites whose mean
component is large — resid 3 here — are mis-reconstructed by exactly that
amount. Attention sites barely move, which is why this never showed up at
`hook_attn_out`. Marks ran nnsight on the raw HF model, so unprocessed weights
are also the faithful choice for reproduction.

**Confirmed by sae_lens' own metadata** (`sae.cfg.metadata`), which is
authoritative and matches the measurement:

| Release | `model_from_pretrained_kwargs` | `prepend_bos` |
|---|---|---|
| `gpt2-small-res-jb` | `{'center_writing_weights': True}` | `True` |
| `pythia-70m-deduped-{res,att,mlp}-sm` | `None` | **`False`** |

**The two suites need opposite treatment.** Loading gpt2-small *without*
processing is catastrophic — `blocks.11.hook_resid_post` scores VE **−241**
(L0 877) — because those SAEs were trained on TransformerLens-processed
activations. Per-suite property, not a global switch. (Marks themselves never
use GPT-2; that suite is this repo's own baseline, spec R3.)

### Two consequent defects in this repo (not yet fixed)

1. **`src/models/suites.py:load_model_and_saes` hardcodes
   `HookedTransformer.from_pretrained`**, silently applying full processing to
   both suites. Every pythia number produced so far — including the 5-example
   smoke run's faithfulness 0.179 — was computed on activations the SAEs were
   not trained on.
2. **`prepend_bos` is read wrongly.** Three call sites do
   `getattr(sae.cfg, "prepend_bos", True)`, but in sae_lens 6.x that field lives
   on `sae.cfg.metadata`, so the `getattr` always falls through to its `True`
   default. The pythia SAEs declare `prepend_bos = False`. This was listed as
   divergence 4 below ("believed equivalent, unverified"); it is now known to be
   an actual defect, and fixing it aligns us with Marks, whose loader uses the
   bare HF tokenizer.

Both are gated on a HIGH-risk edit to `load_model_and_saes` (8 impacted symbols,
4 direct callers, 3 execution flows) and await confirmation.

## Known divergences to carry into the reproduction report

1. **Metric sign — resolved, no longer a divergence.** The paper (§2) defines
   `m = log P(patch_answer) − log P(clean_answer)`; their notebook's `metric_fn`
   is the negation (`clean − patch`). `sva_logit_diff` now follows the
   **notebook**, since that is the code that produced the published numbers and
   it is the orientation under which "higher is better" actually holds — the
   same convention as Wang et al. 2023, *Interpretability in the Wild*
   (arXiv:2211.00593), `logit(correct) − logit(incorrect)`. Both faithfulness (a
   ratio in which the sign cancels) and `|IE| > T` selection are invariant, so
   no reported quantity moves; only the sign of individual IEs and of
   `total_effect` does.
2. **IG interpolation direction.** The paper writes the gradient as evaluated at
   `α·a_clean + (1−α)·a_patch` with `α ∈ {0, 1/N, …, (N−1)/N}`, i.e. the
   patch end is sampled and the clean end omitted. Our implementation (verified
   line-by-line against their *code* `_pe_ig`, docs/spec.md §2) samples the clean
   end and omits the patch end. The code is authoritative; the paper's prose has
   the endpoints the other way round. Worth re-checking if node IEs differ
   systematically at early sites.
3. **Hook placement.** Their `attn` submodule is `layer.attention`'s output and
   `mlp` is `layer.mlp`'s; ours are TransformerLens `hook_attn_out` /
   `hook_mlp_out`. Believed equivalent, unverified.
4. **BOS.** We tokenize with `prepend_bos` taken from the SAE config; their
   loader uses the bare HF tokenizer.
5. **Edge positions.** We sum over token positions on both edge endpoints even
   when `selection.aggregation = "none"`, which keeps positions distinct for
   *nodes*. Marks keep positions on both in the templatic setting. Node counts
   are therefore comparable; edge counts are not exactly.
