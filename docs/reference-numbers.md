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

**Not yet measured.** Run
`python -m scripts.check_sae_reconstruction --suite pythia-marks`; it prepends
its results table to this file.

---

## Known divergences to carry into the reproduction report

1. **Metric sign.** The paper (§2) defines `m = log P(patch_answer) −
   log P(clean_answer)`, which is what `sva_logit_diff` implements. Their
   notebook's `metric_fn` is the negation (`clean − patch`). Both faithfulness
   (a ratio in which the sign cancels) and `|IE| > T` selection are invariant to
   this, so it changes no reported quantity — but every individual IE printed
   from their notebook has the opposite sign to ours.
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
