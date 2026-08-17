# Phase 2 — hypothesis-testing selection vs. Marks' threshold

**Question** (spec §1.2, R6): does the calibrated / hypothesis-testing selector in
`src/calibrated_feature_selection.py` buy anything measurable over Marks' plain
`|mean IE| > T_N`, on identical inputs?

**Answer, as of 2026-08-17: no measurable advantage on this task.** At matched
node count the threshold selector is slightly *more* faithful, the calibrated
selector's nodes are individually *less* necessary, and completeness does not
separate them.

This is the opposite of what an earlier version of this document concluded. Four
implementation bugs stood between the first run and these numbers, and each one
moved the result; §5 records them, because that sequence is the main
methodological lesson here.

Setup: `pythia-marks`, `rc_train.json`, n=100, IG 10 steps, `aggregation=none`,
mean ablation from the patch distribution, evaluated on `rc_test.json` with
n=100, `edges.enabled=false` (faithfulness depends only on the node set). Both
arms share one node-attribution hive through the manifest cache, so only
selection onward differs. `F(M) = 1.6189`.

---

## 1. Result

| selector | T_N | nodes | faithfulness (latter ⅔) | completeness | faithfulness (all blocks) | minimality median | % above 1% floor |
|---|---|---|---|---|---|---|---|
| threshold | 0.10 | 84 | 0.2011 | −0.0032 | 0.0650 | **0.00173** | **36%** |
| threshold | 0.08 | 109 | 0.2416 | −0.0039 | 0.1063 | | |
| **calibrated** | — | **125** | **0.2147** | 0.0045 | 0.0178 | **0.00116** | **22%** |
| threshold | 0.06 | 135 | 0.2836 | 0.0028 | 0.1648 | | |
| threshold | 0.04 | 187 | 0.4079 | −0.0080 | 0.2651 | | |

**Faithfulness.** Interpolating the threshold arm to 125 nodes gives ~0.267
against the calibrated arm's 0.215. The calibrated point sits below the
threshold curve — between its 109- and 135-node points while carrying 125 nodes.

**Minimality.** Per-node leave-one-out, `|F(C\{v}) − F(C)|`. The calibrated
selector is worse on both statistics: median 0.00116 vs 0.00173, and 22% of its
nodes clear Wang et al.'s 1%-of-F(M) floor against the threshold's 36%.

That is coherent with the faithfulness result rather than a second surprise.
Recurrence selects features that fire *consistently*; consistent-but-small
effects are exactly the ones whose individual removal changes little. The
threshold selects on magnitude, which is closer to what leave-one-out measures.

**Completeness** is ≈0 for every row (ideal 0). It does not distinguish the
selectors, and its sign carries no ranking — see §4.

**The reproduction arm still lands.** 84 nodes / 0.2011 against Marks' published
86 / 0.21, so the baseline is sound and the comparison machinery runs correctly.
It is the phase-2 conclusion that moved, not the phase-1 one.

## 2. Both readings of the ⅓ rule

Marks' paper and code disagree, so every evaluation reports both:

- **§3.2**: *"we ignore the first 1/3 of our circuit, and only evaluate the
  latter 2/3"* → `start_layer = ceil(n_blocks/3) = 2`.
- **`ablation.py`**: `--start_layer` defaults to `-1`, filtered as
  `>= start_layer`, which keeps every block → `start_layer = 0`. Their
  faithfulness script also passes `include_embed=False`, which our `block >= 0`
  filter reproduces since embed carries `block = -1`.

The choice is not cosmetic: 0.2011 vs 0.0650 for the same circuit. **Their
published 0.21 corresponds to the paper's reading**, so their CLI default is not
what produced it — anyone running `ablation.py` as shipped gets ~0.065 and
concludes the reproduction failed. Reported as `faithfulness_latter_two_thirds`
and `faithfulness_all_blocks`.

## 3. What was verified against their code

Fetched `saprmarks/feature-circuits@main` `ablation.py` and
`experiments/faithfulness.ipynb` and compared line by line:

| | theirs | ours |
|---|---|---|
| faithfulness | `(fc - fempty) / (fm - fempty)` | identical |
| completeness | `(fccomp - fempty) / (fm - fempty)`, `fccomp` via `complement=True` → `~submod_nodes` | identical, via `invert=True` |
| ablation reference | traced on the **patch** inputs | fixed to match (was clean) |
| node mask | `[seq, d_sae]`, position-resolved | fixed to match (was feature-only) |
| error handling | `resc` per position, `keep`/`remove` overrides | identical |

**Minimality has no reference implementation in either repo.** Marks do not
measure it (§3.2 evaluates interpretability, faithfulness, completeness), and
`redwoodresearch/Easy-Transformer`'s top-level code contains no minimality,
completeness or faithfulness routine. Ours interprets Wang et al. §4.2 with
`K = ∅`, unchecked against running code — the softest number in the table.

## 4. Reading completeness

Both papers set the ideal at 0 — Marks' Fig. 3 caption, and Wang et al., whose
incompleteness score `|F(C\K) − F(M\K)|` is an absolute difference with no sign
to overshoot. Our values (−0.008 to +0.007) are all ≈0: **both circuits are
complete**, and the metric does not rank them.

Negative completeness means `F(M\C) < F(∅)` — the complement performs worse than
full ablation. It is not "more damage" in a circuit's favour. The known
mechanism is counterproductive components left unopposed: Wang et al.'s Negative
Name Mover Heads "reliably write in the opposite direction of the correct
answer" (removing them *raises* logit difference, 3.13 → 4.07), and Marks mark
red nodes that "cause the model to consistently predict incorrect answers". The
sign reports the complement's positive/negative balance, not the quality of C.

## 5. The bugs, in the order they were found

Each changed the answer. The first three are the same mistake — `token_idx`
dropped at a module boundary — and the fourth silently corrupts data.

1. **The calibrated selector tested the wrong unit.** Under `aggregation=none` a
   node is `(site, feature, position)`, but recurrence ran over
   `(site, feature)` and the verdict was broadcast to every position. 125 real
   selections became 338 nodes. *Produced the original "HA fails colossally".*
2. **Evaluation's `keep_mask` dropped `token_idx`.** A feature selected at one
   position was kept at all six: 84 selected nodes were scored as 486 kept, 125
   as 696. *Produced the "HA wins 2×".*
3. **Ablation reference came from the clean distribution.** `ablation.py` traces
   the *patch* inputs to build `patch_states`. Clean activations leave ablated
   slots carrying the information the intervention is meant to remove, and shift
   `F(∅)`, which anchors both ratios.
4. **`resolve_stage_output` appended on recompute.** Hive writers add files
   rather than replacing them, so `run.force=true` — or resuming after a crash —
   stacked a second copy of every row, then rewrote the manifest with the new
   run's counts, making it invisible. One n=100 hive reached 5,790,378 rows
   against a manifest claiming 1,345,653.

All four have regression tests. A fifth hazard is open: **the cache keys on
config, not code**, so changing what a config value *means* — or fixing the
computation behind it — silently serves stale outputs. That has bitten four
times; a `git_commit`-mismatch warning on cache hits would have caught each one.

## 6. Claims made during this work that did not survive

- **"BH has no discriminative power."** From an n=8 run where
  `bh_significant == alphabet_tested`. At n=100 BH rejects 38% (289 of 465);
  the saturation was a small-sample artifact. No defect in the BH step.
- **"HA fails colossally."** Bug 1.
- **"HA is worse on completeness."** True as distance from 0, but presented as a
  deficiency when both circuits are at the ideal.
- **"Minimality is the missing leg of this comparison."** Imported from Wang into
  a Marks reproduction; matched-node-count comparison already prices in the
  padding it guards against.
- **"HA wins ~2× on faithfulness at matched node count."** Bugs 2 and 3. The
  corrected numbers reverse it.

## 7. Next

1. **Repeat on `simple`, `nounpp`, `within_rc`** before treating "no advantage"
   as a result. One dataset is not a finding, and this one has already produced
   two opposite answers. Node hive is ~3 min per structure.
2. **Grid over `screen_q × fdr_q × min_recurrence × screen_n0`** — all four are
   config-exposed now. The calibrated arm has never been tuned; it is being
   compared at its defaults against a threshold swept over five values. That is
   not a fair fight, and it is the single most likely reason the current answer
   is wrong. Each grid point costs ~2 s of selection plus ~30 s of evaluation on
   the cached hive.
3. **Per-`case` completeness breakdown** — tests whether the complement falls
   back on the distractor noun's number (predicted ≈0 on matched cases, negative
   on mismatched).
4. **Minimality with non-empty K** if the leave-one-out result is to be leaned
   on; `K = ∅` structurally penalises redundant nodes.
