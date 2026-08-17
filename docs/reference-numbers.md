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

### Measured, 2026-08-17 — **GATE NOT PASSED**

`python -m scripts.check_sae_reconstruction --suite pythia-marks --n-seqs 32`
(32 x 128 tokens of `NeelNanda/pile-10k`, seed 42, BOS position skipped). Full
table in `docs/sae-reconstruction.md`; the comparison against their rows:

| Site | Marks VE | ours | Δ | Marks L0 | ours |
|---|---|---|---|---|---|
| Attn 0 | 92% | 0.905 | −1.5 | 128 | 132 |
| Attn 1 | 87% | 0.819 | −5.1 | 127 | 127 |
| Attn 2 | 90% | 0.880 | −2.0 | 215 | 221 |
| Attn 3 | 89% | 0.860 | −3.0 | 169 | 172 |
| **Attn 4** | 83% | **0.733** | **−9.7** | 132 | 137 |
| Attn 5 | 89% | 0.819 | −7.2 | 144 | 168 |
| MLP 0 | 97% | 0.968 | −0.2 | 5 | 130 |
| MLP 1 | 85% | 0.834 | −1.6 | 69 | 69 |
| MLP 2 | 99% | 0.976 | −1.5 | 88 | 91 |
| MLP 3 | 88% | 0.814 | −6.6 | 160 | 136 |
| MLP 4 | 92% | 0.885 | −3.5 | 100 | 93 |
| MLP 5 | 96% | 0.919 | −4.1 | 102 | 164 |
| Resid 0 | 92% | 0.908 | −1.3 | 59 | 64 |
| Resid 1 | 85% | 0.808 | −4.2 | 54 | 63 |
| Resid 2 | 96% | 0.946 | −1.4 | 108 | 111 |
| **Resid 3** | 96% | **0.790** | **−17.0** | 68 | 67 |
| Resid 4 | 88% | 0.833 | −4.7 | 61 | 61 |
| Resid 5 | 90% | 0.860 | −4.0 | 72 | 98 |
| embed | (not tabled) | 0.964 | — | — | 5 |

**Two sites fall below the 0.80 floor: `blocks.3.hook_resid_post` (0.790) and
`blocks.4.hook_attn_out` (0.733).** Per the plan's own gate, the reproduction
run should not be trusted until this is understood.

Reading of the result:

- **Not a BOS artifact.** Re-measured with the BOS position included and it
  moves nothing (0.7900 → 0.7953 at resid 3). Ruled out.
- **Not the wrong dictionaries.** L0 tracks their tables closely at most sites,
  and at resid 3 almost exactly (67 vs 68) — the site with the *worst*
  reconstruction has the *right* sparsity. That points at a normalisation or
  folding difference in the `_processed` conversion, or at the evaluation
  distribution, rather than a loading error.
- **A uniform −1 to −5 shortfall is expected** and not alarming: this is 4k
  tokens of pile-10k against whatever corpus they evaluated on, and variance
  explained is computed here against a per-sequence mean.
- **Resid 3 is the outlier that matters.** −17 points, against its neighbours
  resid 2 (−1.4) and resid 4 (−4.7). Attn 4 at −9.7 is second.
- **Two L0 discrepancies worth a look**: MLP 0 (theirs 5, ours 130) and MLP 5
  (theirs 102, ours 164). Note our embed site measures L0 = 5, the same
  anomalous value their table reports for MLP 0 — possibly a coincidence,
  possibly an index-mapping question in the `ctigges` release.

### Root cause of the resid-3 gap — TransformerLens weight processing

**Resolved without needing the original dictionaries.** Same SAE, same 32
sequences, only the model's weight processing differs:

| Site | `from_pretrained` (fold_ln + center_writing_weights + center_unembed) | `from_pretrained_no_processing` | Marks |
|---|---|---|---|
| `blocks.3.hook_resid_post` | 0.790 | **0.942** | 96% |
| `blocks.2.hook_resid_post` | 0.946 | 0.952 | 96% |
| `blocks.0.hook_resid_post` | 0.908 | 0.915 | 92% |
| `blocks.4.hook_attn_out` | 0.733 | 0.734 | 83% |

The −17 point gap becomes −1.8, in line with every other site.

**Why.** `center_writing_weights` subtracts the mean-over-`d_model` component
from every matrix that writes to the residual stream. That component is
invisible to the *model* (the next LayerNorm removes it) but perfectly visible
to an *SAE*, which was trained on the uncentered activation. Sites whose mean
component is large — resid 3 here — are mis-reconstructed by exactly that
amount. Marks ran nnsight on the raw HF model, so unprocessed weights are also
the faithful choice for reproduction.

**Confirmed by sae_lens' own metadata** (`sae.cfg.metadata`), which is
authoritative and matches the measurement:

| Release | `model_from_pretrained_kwargs` | `prepend_bos` |
|---|---|---|
| `gpt2-small-res-jb` | `{'center_writing_weights': True}` | `True` |
| `pythia-70m-deduped-{res,att,mlp}-sm` | `None` | **`False`** |

**The two suites need opposite treatment.** Loading gpt2-small *without*
processing is catastrophic — `blocks.11.hook_resid_post` scores VE **−241**
(L0 877) — because those SAEs were trained on TransformerLens-processed
activations. So this is a per-suite property, not a global switch.

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

### Still unexplained: `blocks.4.hook_attn_out`

0.733 against their 83%, and **unmoved** by weight processing (0.7329 →
0.7338). `blocks.5.hook_attn_out` is also low (0.819 vs 89%). This is the one
question the original `dictionary_learning` `.pt` dictionaries would still
answer: whether the `ctigges` `_processed` conversion is lossy for the later
attention SAEs, or whether our `hook_attn_out` placement differs from their
`layer.attention` output.

---

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
