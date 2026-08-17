# Why the two selectors disagree — diagnosis and proposed tests

**Parked 2026-08-17 to run the sweep. Return here afterwards.**

This is the "why" behind `docs/ha-vs-marks.md`, which only measures outcomes. All
of it comes from `scripts/explore_selectors.py` (exploratory, read-only, no GPU,
seconds to run) against the n=100 hive, and its per-node output
`docs/selector-difference-sets.csv` — one row per node carrying both selectors'
verdicts plus every calibrated statistic, so further questions are queries rather
than experiments.

```bash
python -m scripts.explore_selectors --node-dir outputs_n100/node_attribution/<hash>
```

---

## 1. The reverse experiment: how does Marks' selection score statistically?

Take the nodes the threshold selects and read off the statistics the calibrated
arm would have judged them by. Nothing is recomputed — `recurrence_test` already
produces all of it.

Of the threshold's **84** picks: **78** reach the recurrence test, **74 are
BH-significant**, 57 are also selected by HA, median recurrence 0.59, median sign
consistency 1.00.

**So the threshold is not selecting statistical noise.** The 27 nodes it keeps
alone are rejected by HA on *recurrence*, not on significance.

## 2. What characterises the difference sets

Medians over the three sets (n = 57 both, 27 marks-only, 68 ha-only):

| set | mean \|IE\| | recurrence | \|IE\| when it fires | Σ\|IE\| mass | median site |
|---|---|---|---|---|---|
| both | 0.239 | 0.72 | 0.353 | 19.1 | 12 |
| **marks_only** | 0.135 | **0.07** | **1.763** | 12.4 | **2** |
| **ha_only** | **0.039** | 0.17 | 0.103 | **2.2** | **9** |

`marks_only` nodes are **rare but enormous**; `ha_only` are **frequent but tiny**
(mean |IE| well below the 0.1 threshold, which is exactly why Marks reject them).

They also live in different parts of the model: `marks_only` concentrates at
sites 0 and 2 (embed, MLP-0), `ha_only` at sites 6/7/9 (resid-1 … resid-2). That
fits Marks' §3.2 remark that early components "are typically involved in
processing specific tokens" — a token detector fires hugely on the few examples
containing its trigger and not at all otherwise, which *is* the rare-but-huge
profile.

## 3. The core finding: the test statistic decides the disagreement

| statistic | marks_only | ha_only | favours |
|---|---|---|---|
| occurrence count — *what the binomial tests* | 7 | **17** | ha_only |
| **Σ\|IE\| mass** | **12.4** | 2.2 | marks_only (5.6×) |
| **conditional median \|IE\|** | **1.76** | 0.10 | marks_only (17×) |

Switching the statistic from a pass *count* to |IE| *mass* reorders exactly the
nodes where the selectors disagree, in the direction that the faithfulness and
minimality measurements already favour.

The defect stated precisely: **each node's per-example IE is zero-inflated — a
Bernoulli "did it clear the noise floor" times a magnitude "by how much" — and
the binomial tests only the Bernoulli part.** The two parts are genuinely
distinct information, not redundant:

- the median node is present in only **8%** of examples (75th pct 24%);
- only **536** of 69,211 nodes ever clear the screen (0.56% of rows);
- Spearman(recurrence rate, conditional median |IE|) = **+0.39**.

## 4. BLOCKER: the scored-node asymmetry

The ⅓ rule ablates only blocks ≥ 2, so a large share of each circuit is never
touched by the metric:

| arm | nodes reported | inside the scored region | never ablated |
|---|---|---|---|
| threshold | 84 | **45 (54%)** | 39 |
| calibrated | 125 | **86 (69%)** | 39 |

**22 of the threshold's 27 exclusive nodes fall outside the scored region**,
because they are those early-layer token detectors. So "faithfulness at 84 vs 125
nodes" is really "at 45 vs 86" — a 1.9× ratio, not 1.5×, and asymmetric by
construction.

Any new test tuned against the current faithfulness number is being tuned
against a biased objective. **Fix first:** record `n_nodes_scored` in the
evaluation manifest and re-plot the curves against it.

## 5. Candidate tests, ranked

**(1) Sign-flip permutation test on the mean IE.** Statistic: `mean_i IE_{v,i}`,
exactly what Marks threshold. Null: under no systematic effect the per-example
signs are exchangeable — draw `ε_i ∈ {±1}`, recompute, build the null
empirically, BH across nodes. Distribution-free (the tails demand it),
magnitude-aware by construction, vectorises to milliseconds over 536 candidates.
Already floated in `gaussianity_report`'s own docstring.

*Why this one first:* it does not compete with Marks' criterion, it **calibrates**
it. The claim becomes "their threshold is an uncontrolled point estimate; here is
the tested version", and its disagreements are exactly the nodes whose high mean
rests on one outlier example — a failure mode a threshold cannot see. ~40 lines,
no GPU, and it drops in as a fourth `selection.method` so it enters the sweep as
its own arm.

**(2) Conditional-magnitude test against the truncated null tail.** Given a
feature cleared the screen, is its |IE| larger than a null feature that also
cleared it? The screen threshold is the `1 − q/n_g` quantile of the per-group
fitted lognormal, so conditional on passing the null is that lognormal truncated
above the threshold — closed form from parameters `mad_screen` already fits. The
most literal version of "contributes significantly to the mass", and it composes
with Stage 1 rather than replacing it.

**(3) Explicit two-part test.** Binomial for the Bernoulli part, (2) for the
magnitude part; require both. Do **not** combine the p-values with Fisher — the
parts correlate at +0.39, so independence fails.

## 6. The entropy idea — a diagnostic, not a selector

Normalising within a group, `q_v = |IE_v| / Σ_u |IE_u|`, a feature's entropy
contribution `−q log q` is **non-monotonic**: it peaks at `q = 1/e ≈ 0.37` and
falls to zero as `q → 1`. A feature carrying *all* the causal mass would score
like one carrying none, so it penalises the dominance you want to reward. Unusable
as a ranking statistic.

Worth keeping as a **descriptive** quantity, stated as a ratio: activation
entropy (or L0) counts features that *fire*; IE-entropy counts features that
*matter*. Their ratio is "of the features that fire, how few carry the causal
mass" — directly relevant to the Petri-net phase, since a site with low
IE-entropy is a bottleneck. A figure, not a criterion.

## 7. Order of work when returning

1. `n_nodes_scored` in the manifest; re-plot the existing n=100 points against it.
   Everything below is tuned against that axis.
2. Implement test (1) as `selection.method=signflip`; add to the sweep's `curves`
   block so it gets a curve, not a point.
3. Read whatever the sweep says about `min_median_abs` — it is the crude version
   of the same fix (a magnitude floor bolted onto a count-based test), so if it
   helps materially, (1) and (2) should help more.
4. Only then consider (3).
