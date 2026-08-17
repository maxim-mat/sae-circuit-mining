# Reproduction report — Marks et al. RC circuit

**Status: NOT YET RUN.** The pipeline is implemented and its unit tests pass
(77 tests, `pytest -m "not slow"`), but no attribution run has been executed:
that needs model weights, the 19 SAE downloads and GPU time. This file is the
template the run fills in; every "measured" cell is empty on purpose rather than
estimated.

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
C:/Users/Maxim/anaconda3/python.exe -m src.main -c configs/pythia_marks_rc.yaml \
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
