# Reproduction report — Marks et al. RC circuit

**Status: smoke-tested, full run NOT YET DONE.** The pipeline runs end to end on
a 5-example smoke configuration (see below), but the 100-example reproduction has
not been executed. The results table below is still empty on purpose rather than
filled with smoke-test numbers, which are not comparable to Marks'.

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

```bash
# 0. Verify the processed SAEs reconstruct (plan Task 12 Q1). Stop if this fails.
C:/Users/Maxim/anaconda3/python.exe -m scripts.check_sae_reconstruction --suite pythia-marks

# 1. Smoke test on 5 examples.
C:/Users/Maxim/anaconda3/python.exe -m pytest tests/test_end_to_end.py -v -m slow

# 2. The reproduction run: 100 examples, IG 10 steps, agg none, T_N 0.1, T_E 0.01.
#    Matches scripts/get_circuit.sh EleutherAI/pythia-70m-deduped rc_train 0.1 0.01 none
C:/Users/Maxim/anaconda3/python.exe main.py -c configs/pythia_marks_rc.yaml \
    evaluation.num_examples=40
```

`evaluation.num_examples=40` matches their notebook; the schema default is 100.

---

## Results

| Quantity | Marks (target) | Ours (measured) | Δ |
|---|---|---|---|
| Faithfulness | **0.21** | | |
| Node count | **86** | | |
| Edge count | not published per-circuit | | — |
| F(C) | — | | — |
| F(M) | — | | — |
| F(∅) | — | | — |

**Acceptance:** faithfulness within 0.1 of 0.21, with the edge set built from
`edges.template=marks` so edge topology is comparable at all.

### Per-site node counts

| ordinal | kind | block | selected nodes |
|---|---|---|---|
| | | | |

---

## Known divergences

Carried from `docs/reference-numbers.md` §"Known divergences" — metric sign
(inert for both faithfulness and `|IE|` selection), IG interpolation direction
(paper prose vs. their code; ours follows the code), attn/mlp hook placement,
BOS convention, and edge-position summation. Plus the gaps recorded in
`docs/spec.md` §2: G7 (no `_pe_exact` ground truth) remains open.

## If the number misses

Ranked diagnostic candidates, most likely first:

1. **Processed-SAE mismatch.** `ctigges/pythia-70m-deduped__*_processed` are
   folded/normalised conversions of their `dictionary_learning` dictionaries.
   Step 0 above is exactly this check; compare against the Tables 4–6 rows in
   `docs/reference-numbers.md` site by site, not just against the 0.80 floor.
2. **Hook placement.** Their `attn` is `layer.attention`'s output and `mlp` is
   `layer.mlp`'s; ours are `hook_attn_out` / `hook_mlp_out`.
3. **Stop-gradient coverage in the edge template.** Compare
   `marks_template(build_suite("pythia-marks"))` against `circuit.py`'s
   per-layer emission order edge by edge.
4. **IG step grid or batching** — see divergence 2.
5. **The evaluated segment.** `evaluation.start_layer` defaults to
   `ceil(n_blocks/3)` = 2, reproducing their "latter 2/3" rule. A faithfulness
   number that looks far too high or low is worth re-checking here first, since
   it moves the denominator as well as the numerator.
