# Phase 2 — hypothesis-testing selection vs. Marks' threshold

**Question** (spec §1.2, R6): does the calibrated / hypothesis-testing selector in
`src/calibrated_feature_selection.py` buy anything measurable over Marks' plain
`|mean IE| > T_N`, on identical inputs?

**Answer so far: yes, roughly 2x the faithfulness at matched node count** — but
that conclusion only emerged after a bug was fixed, and the first run said the
opposite. Both are recorded below, because the failure mode is instructive.

Setup for everything here: `pythia-marks`, `rc_train.json`, n=100, IG 10 steps,
`aggregation=none`, `T_E=0.01`, mean ablation, evaluated on `rc_test.json`,
`start_layer` resolved to 2. Both arms share one node-attribution hive through
the manifest cache, so only selection onward differs.

---

## 1. First run — the apparent collapse

wandb `maxim0/SAE`, group `marks-repro-rc`.

| arm | run | nodes | edges | faithfulness |
|---|---|---|---|---|
| threshold `T_N=0.1` | `oobc2bhj` | 84 | 323 | 0.2388 |
| calibrated | `6s37amd4` | **338** | 680 | **0.2203** |

Read naively: HA selects 4x the nodes for *lower* faithfulness. That reading is
wrong, and the giveaway is that it is close to impossible — faithfulness rises
monotonically as a circuit approaches the full model, so more nodes and less
faithfulness means the two node sets are not nested.

The reproduction arm itself is sound: **84 nodes / 0.2388 against Marks'
published 86 nodes / 0.21** at the same thresholds (paper §3.3 + Fig. 7).

## 2. The bug — the tested unit was not the unit a node is

Under `aggregation=none` a node is `(site, feature, position)`. The calibrated
selector was working at `(site, feature)`. Two coupled halves:

1. **Evidence pooling.** `recurrence_test` did
   `passes.groupby(feature_cols).agg(n_recurring=(trial_col, "nunique"))` with
   `feature_cols = ["site_ordinal", "feature_idx"]`. A feature passing the screen
   at position 5 in one example and position 1 in another counted as recurring
   twice — but those are different nodes. Position-scattered features got
   inflated recurrence and therefore too-small binomial p-values.
2. **Verdict broadcast.** `run_selection` merged that feature-level verdict onto
   the position-resolved aggregate on `FEATURE_COLS`, marking each selected
   feature selected at *every* position — 6 on this data, where the threshold
   selector uses only 3.

Neither `p0` nor `n_trials` was affected; the null was never wrong. What changed
is the unit of evidence and the unit of verdict.

| | alphabet_tested | bh_significant | selected | circuit nodes |
|---|---|---|---|---|
| before | 465 | 289 | 112 | **338** |
| after | 536 | 335 | 125 | **125** |

The alphabet grew only 15%, so most features pass the screen at a single
position and the pooling distortion was modest. The **broadcast** was the 3x
multiplier that produced the apparent collapse.

Fix: `calibrated_feature_cols(cfg)` returns
`["site_ordinal","feature_idx","token_idx"]` when `aggregation=none`, and is used
for both the recurrence groupby and the join. Pinned by
`tests/test_calibrated_wiring.py::test_tested_unit_matches_what_a_node_is`.

## 3. Corrected result — threshold sweep at matched node count

`edges.enabled=false` (faithfulness depends only on the node set),
`evaluation.num_examples=100`. `F_M = +1.6189`, `F_empty = -0.0078` throughout,
so the metric spans 1.627.

| selector | T_N | nodes | faithfulness | completeness | F_C | F(M\C) |
|---|---|---|---|---|---|---|
| threshold | 0.10 | 84 | 0.2388 | −0.0221 | 0.3806 | −0.0438 |
| threshold | 0.08 | 109 | 0.2955 | −0.0278 | 0.4728 | −0.0531 |
| **calibrated** | — | **125** | **0.6258** | −0.0911 | 1.0101 | −0.1559 |
| threshold | 0.06 | 135 | 0.3563 | −0.0368 | 0.5717 | −0.0677 |
| threshold | 0.04 | 187 | 0.4884 | −0.0515 | 0.7866 | −0.0916 |

Interpolating the threshold arm to 125 nodes gives ~0.33, against the calibrated
arm's **0.626 — about 1.9x**. The calibrated circuit at 125 nodes also beats the
threshold circuit at 187 nodes (0.488).

## 4. Interpretation, and what it does NOT show

**Faithfulness: a real result.** ~2x at matched node count, on the arm whose
baseline independently reproduces the published number.

**Completeness: does not separate the two.** Both papers set the ideal at 0 —
Marks' Fig. 3 caption ("the ideal completeness is 0") and Wang et al. 2023
(arXiv:2211.00593), whose incompleteness score `|F(C\K) − F(M\K)|` is an
absolute difference and so has no sign to overshoot. Our values (−0.02 to −0.09
on a scale where the naive baseline is ~1) are all ≈0: **both circuits are
complete.**

Negative completeness means `F(M\C) < F(∅)` — the complement performs *worse*
than full mean-ablation. It is not "more damage" in a way that favours the
circuit. The known mechanism is counterproductive components left unopposed:
Wang et al.'s **Negative Name Mover Heads** "reliably write in the opposite
direction of the correct answer" (knocking them out *raises* logit difference,
3.13 → 4.07 in their minimality table), and Marks' own circuits mark red nodes
that "cause the model to consistently predict incorrect answers". Ablating the
positive pathway leaves those dominant. So the sign reports the positive/negative
balance of the *complement*, not the quality of C.

A plausible task-specific version here: with the subject-number pathway ablated,
the model falls back on the **distractor** noun's number, which is wrong in
exactly the `singular_plural` and `plural_singular` cases — about half the data.
Testable now that `case` is in the `examples.parquet` sidecar: completeness
should be ≈0 on matched cases and clearly negative on mismatched ones.

**Minimality: worth knowing about, but not a gap in this comparison.** Marks do
not use it — §3.2 evaluates exactly "interpretability, faithfulness, and
completeness". It is Wang et al.'s third criterion (§4.2): every node v ∈ C
should have some K with `|F(C \ (K∪{v})) − F(C \ K)|` large.

Their argument for needing it is that faithfulness and completeness can both be
satisfied by a padded circuit, and that holds for Marks' pair too — adding
irrelevant nodes to C removes them from M\C without moving F(M\C), so
completeness is padding-blind as well. But that bites when circuits are compared
at face value. The sweep above compares at **matched node count**, which prices
padding in directly: a padded 125-node circuit would lose to the threshold at 125
nodes, and it does not.

What minimality would add is a different question — whether each node is
individually necessary, i.e. whether this is a *mechanism* rather than a
well-performing set. That matters for interpreting the circuit, not for deciding
which selector is better. Their exact criterion needs an exponential subset
search; a per-node leave-one-out is cheap on position-resolved nodes if the
question becomes interesting.

## 5. Claims made during this work that did not survive

- **"BH has no discriminative power."** Asserted from an n=8 run where
  `bh_significant == alphabet_tested`. At n=100 BH rejects 38% (289 of 465). The
  saturation was a small-sample artifact: with 8 trials and `p0 ≈ 1e-4`, anything
  recurring twice is significant. No defect in the BH step.
- **"HA fails colossally."** An artifact of the broadcast bug above.
- **"HA is worse on completeness."** True as raw distance from 0, but presented
  as a deficiency when both circuits are effectively at the ideal.
- **"Minimality is the missing leg of this comparison."** Imported from Wang et
  al. into a Marks reproduction, where it does not belong: Marks evaluate
  interpretability, faithfulness and completeness only. And comparing at matched
  node count already prices in the padding that minimality guards against.

## 6. Next

1. Per-`case` completeness breakdown (tests the distractor-fallback story).
2. Minimality (per-node leave-one-out) — optional, and only if the claim
   becomes about the circuit as a mechanism rather than about the selector.
3. Repeat the sweep on `simple`, `nounpp`, `within_rc` — one claim from one
   dataset is not a result. Node hive is ~3 min per structure.
4. Grid over `screen_q x fdr_q x min_recurrence x screen_n0`, now that all four
   are config-exposed; each point is ~2 s of selection plus ~12 s of evaluation
   against the cached hive.
