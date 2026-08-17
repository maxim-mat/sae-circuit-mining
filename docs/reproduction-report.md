# Reproduction report — Marks et al. RC circuit

**Status: REPRODUCED.** The 100-example run matches the one published
(threshold, node-count, faithfulness) triple to within 2 nodes and 0.03
faithfulness. See `docs/ha-vs-marks.md` for the phase-2 comparison built on it.

## Result — 2026-08-17, n=100, T_N=0.1, T_E=0.01, aggregation none

| Quantity | Marks (target) | Ours | Δ |
|---|---|---|---|
| Faithfulness | **0.21** | **0.2388** | +0.029 |
| Node count | **86** | **84** | −2 |
| Edge count | not published per-circuit | 323 | — |
| Completeness | not published per-circuit | −0.0221 (ideal 0) | — |
| F(C) | — | +0.3806 | — |
| F(M) | — | +1.6189 | — |
| F(∅) | — | −0.0078 | — |

Source: wandb `maxim0/SAE`, group `marks-repro-rc`, run `marks-threshold-rc-n100`
(`oobc2bhj`). Target: paper §3.3 ("a circuit with 86 nodes and faithfulness
0.21") plus the Fig. 7 caption pinning that circuit to T_N=0.1 / T_E=0.01 — the
only place a threshold, a node count and a faithfulness appear together.
**Fig. 3 is not a check on this**: it is indexed by node count with the threshold
swept away, and its bold line averages the four structures.

Reproduced independently on a second machine from the same seed: 1,345,653 node
rows against the remote run's 1,345,680 (0.002% apart, float nondeterminism at
the activation mask boundary), same mean total effect to 5 decimals.

## Smoke run — 2026-08-17, 5 examples, **after** the loader fix

```
data.num_examples=5  attribution.ig_steps=2  evaluation.num_examples=5
```

| Stage | Result |
|---|---|
| node_attribution | 68,770 rows, 5 examples, 0 skipped |
| selection | 75 of 29,153 nodes |
| edge_attribution | 338,403 edges over 31 specs |
| circuit | **75 nodes, 317 edges** |
| evaluation | **faithfulness 0.2110** (F_C 0.626, F_M 1.526, F_∅ 0.385) |

Against Marks' **86 nodes / 0.21**. All three F terms are now positive, as the
flipped metric orientation predicts.

**This is not the reproduction.** 5 examples at 2 IG steps against their 100 at
10, and 5 eval examples against their 40. Landing on 0.211 at this sample size
is as much luck as signal; the 100-example run below is what settles it. What it
does establish is that the corrected loading moved the number in the right
direction — 0.179 → 0.211 — while node count moved 83 → 75, closer to their 86.

## Superseded smoke run — before the loader fix

```
data.num_examples=5  attribution.ig_steps=2  evaluation.num_examples=5
```

| Stage | Result | Time |
|---|---|---|
| node_attribution | 96,574 rows, 5 examples, 0 skipped, mean total effect **+3.92** | 39 s |
| selection | 83 of 35,161 nodes selected | 0.2 s |
| edge_attribution | 397,869 edges over **31 specs** | 19 s |
| circuit | **83 nodes, 371 edges** | 0.2 s |
| evaluation | **faithfulness 0.179** (F_C −1.003, F_M −1.936, F_∅ −0.799) | 13 s |

31 specs is the expected template size: 6 blocks x 5 edges
(MR/AR/RM/RA/RR, AM suppressed by `parallel_attn`) + the final resid→y.

A second identical invocation replayed all five stages from their manifests in
under a second (`was_cached: True` throughout), so the caching contract holds
end to end.

**Do not read 0.179 as a reproduction of 0.21.** It is 5 examples with 2 IG
steps against their 100 and 10, evaluated on 5 test examples against their 40.
The agreement in magnitude is encouraging, not evidence.

**Superseded sign convention.** This run predates the metric flip. At the time
`sva_logit_diff` was `logit(patch) − logit(clean)`, which is *negative* when the
model behaves correctly, hence F_M = −1.94 sitting *below* F_∅ = −0.80.
`sva_logit_diff` now returns `clean − patch` (Marks' notebook / Wang et al.),
so a rerun reports the same faithfulness **0.179** with all three terms
positive: F_M = +1.94, F_∅ = +0.80, F_C = +1.00, and `total_effect` negative
rather than positive. The ratio is unchanged because the global sign cancels.

**Targets** come from `docs/reference-numbers.md` (arXiv:2403.19647v3 §3.3 and
Fig. 7).

---

## How to run it

### On a fresh machine

Neither config runs straight from a clone: the datasets are gitignored and both
configs enable wandb. Three setup steps, then the two runs.

```bash
# 0a. Dependencies. Install torch FIRST from the wheel index matching your
#     driver, then the rest. Verified on torch 2.9.1+cu130, RTX 4090.
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 0b. Datasets (gitignored, never committed). All four SVA structures.
python -m scripts.fetch_data

# 0c. Verify the SAEs reconstruct before trusting anything downstream.
#     Downloads pythia-70m-deduped + 19 SAEs on first run. Expect PASS.
python -m scripts.check_sae_reconstruction --suite pythia-marks
```

Then the two arms:

```bash
# ARM A -- Marks: |mean IE| > 0.1
python main.py -c configs/pythia_marks_rc.yaml     evaluation.num_examples=40 'logging.backend=[]'

# ARM B -- hypothesis-testing selection, identical in every other respect
python main.py -c configs/pythia_ha_rc.yaml     evaluation.num_examples=40 'logging.backend=[]'
```

Run A first: the configs differ only inside `selection`, so B reuses A's
node-attribution hive from the cache and costs one selection pass plus its own
edges rather than a second attribution run.

The overrides are all defaults you may not want:

| Override | Why |
|---|---|
| `'logging.backend=[]'` | both configs ship `backend: [wandb]`; drop it to log to wandb, but `wandb login` first or the run blocks on a credential prompt. `logging.offline=true` is the middle option. Quote it -- brackets are shell glob characters. |
| `evaluation.num_examples=40` | matches their notebook; the schema default is 100. |
| `suite.device=cpu` | only if there is no GPU. Pythia-70m will run, slowly. |
| `data.template_field=case` | records the prompt->template map. Set it *before* the expensive run: it is in the node-attribution cache key, so adding it later forces a recompute. |
| `run.force=true` | recompute a stage whose parameters are unchanged. |

Flags must precede the dotlist overrides (`main.py -c cfg --stage selection a=1`);
argparse cannot interleave them with a `nargs="*"` positional.

### Smoke test first

```bash
python -m pytest tests/ -q -m "not slow"              # ~3 s, 90 tests
python -m pytest tests/test_end_to_end.py -v -m slow  # ~2 min, 5 examples
```

---

## Known divergences

Carried from `docs/reference-numbers.md` §"Known divergences" — metric sign
(inert for both faithfulness and `|IE|` selection), IG interpolation direction
(paper prose vs. their code; ours follows the code), attn/mlp hook placement,
BOS convention, and edge-position summation. Plus the gaps recorded in
`docs/spec.md` §2: G7 (no `_pe_exact` ground truth) remains open.

## Diagnostics, retained for the remaining structures

The reproduction landed, and two of the originally-ranked suspects were resolved
in the process:

1. ~~Processed-SAE mismatch~~ — **eliminated.** Marks' original
   `dictionary_learning` .pt scores identically to the `ctigges` conversion
   (0.7338 / L0 136.1 at attn 4, four decimals). See `docs/reference-numbers.md`.
2. ~~Model loading~~ — **was a real defect, fixed.** TransformerLens weight
   processing cost `blocks.3.hook_resid_post` 15 points of variance explained;
   the model is now reconstituted from the SAEs' own metadata.
3. **Hook placement** — still unverified. Their `attn` is `layer.attention`'s
   output and `mlp` is `layer.mlp`'s; ours are `hook_attn_out` / `hook_mlp_out`.
4. **IG step grid** — paper prose and their code disagree on interpolation
   direction; ours follows the code (docs/spec.md §2).
5. **The evaluated segment** — `evaluation.start_layer` defaults to
   `ceil(n_blocks/3)` = 2, reproducing their "latter 2/3" rule. Check first if a
   faithfulness number looks far off: it moves numerator and denominator both.
