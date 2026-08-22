# The Stage-1 screen's design space — scale, estimator, scope, and Stage 2

**Run 2026-08-22.** Companion to `docs/ha-vs-marks.md` (which compares two fixed
selectors) and `docs/selector-diagnosis.md` (which explains why they disagree).
This one maps the *space* the calibrated selector lives in, and asks what Marks'
single constant threshold looks like from inside it.

Everything here is on the cached n=100 hive
(`outputs_n100/node_attribution/77c6178e`), `pythia-marks` / `rc_train.json`,
IG 10 steps, `aggregation=none`. Faithfulness is evaluated on `rc_test.json`
with 40 examples (Marks' protocol), mean ablation, edges disabled — node sets
are the only thing that varies.

**Two faithfulness columns are reported throughout**, because Marks' paper and
code disagree about what to score: `23_mean` is §3.2's "ignore the first 1/3 of
the circuit" (`start_layer = ceil(n_blocks/3) = 2`), `all_mean` is
`ablation.py`'s default of scoring every block. A ranking that holds under only
one of them is not a ranking.

## What was added to the pipeline

Four config knobs on `selection`, all defaulting to the previous behaviour:

| knob | values | meaning |
|---|---|---|
| `screen_scale` | `log` (default), `linear` | fit and threshold on log abs-IE, or on raw abs-IE |
| `screen_method` | `mad` (default), `z` | robust median/MAD per (prompt, layer), or plain mean/sd per (prompt, layer, **token**) |
| `screen_scope` | `group` (default), `global` | one null per group, or ONE null for the whole run |
| `recurrence` | `true` (default), `false` | Stage 2 on, or keep every feature the screen ever passed |

`screen_group_cols(method, scope)` is the single place that decides what a group
is, and Stage 2's null pass rate `p0 = q/n_g` follows it — per-token groups are
~10x smaller, so a `p0` computed on the coarse grouping would call noise
significant. `mad_screen` now also refuses a `screen_q` so small that
`1 - q/n_g` rounds to 1.0 in float64, which silently made every threshold `+inf`
and selected nothing.

## 1. What Marks' constant threshold means inside each group

`scripts/explore_threshold_zscores.py` -> `figures/marks-threshold-zscores.png`,
`figures/marks-threshold-pvalues.png`, `docs/marks-threshold-zscores.csv`.

Treat `T = 0.1` as a test statistic evaluated against each group's own fitted
null, and read off the z and the implied per-group false-positive rate.

| null | grain | z median | z q10-q90 | z range | implied alpha q10-q90 |
|---|---|---|---|---|---|
| linear | prompt x layer | 3.89 | 0.87 – 22.7 | -0.18 … 68 | 0.19 … 8e-114 |
| linear | prompt x layer x position | 21.1 | 1.21 – 143 | -0.18 … 2019 | 0.11 … ~0 |
| **log** | **prompt x layer** | **3.01** | **2.22 – 4.14** | 0.52 … 5.56 | **1.3e-2 … 1.7e-5** |
| log | prompt x layer x position | 3.67 | 2.08 – 6.46 | 0.52 … 19.7 | 1.9e-2 … 5e-11 |

- **On the linear scale a constant threshold is incoherent.** The same 0.1 lands
  anywhere from below a group's own noise mean (z < 0) to 68 sigma out — more
  than 100 orders of magnitude of implied alpha. 46% of groups get alpha > 1e-3
  while 44% get alpha < 1e-6.
- **On the log scale it is nearly a constant-alpha test**: z = 3 +- 1, alpha
  spanning ~3 orders instead of ~100.
- **The global log z-test sits at z = 3.06 against a per-group median of 3.005.**
  A single global log threshold is approximately a calibrated per-group test at
  alpha ~ 1e-3 in every group.

That is the mechanism behind every result below: one constant is coherent on the
log scale because the per-group nulls are homogeneous there, and is not on the
linear scale.

Caveat: the group nulls are fitted on per-row abs-IE while Marks' 0.1 cuts the
example-mean, so this measures the constant's coherence across groups, not his
node rule exactly.

## 2. The 16-cell grid

`scripts/explore_screen_variants.py` -> `docs/screen-variants.csv` (node counts,
abs-IE statistics, and the margin against Marks per cell).

| variant | nodes | `23_mean` | `all_mean` |
|---|---|---|---|
| linear/mad/global/norec | 16704 | 1.0159 | 0.9165 |
| linear/mad/group/norec | 20250 | 0.9978 | 0.7616 |
| log/mad/group/norec | 536 | 0.5303 | 0.3446 |
| linear/z/group/norec | 2297 | 0.4673 | 0.2688 |
| log/z/group/norec | 812 | 0.4044 | 0.2189 |
| **linear/z/global/norec** | **208** | **0.2669** | **0.2444** |
| linear/mad/group/rec | 1318 | 0.2491 | 0.1201 |
| linear/mad/global/rec | 1024 | 0.2338 | 0.1279 |
| *Marks threshold 0.1* | *84* | *0.2225* | *0.0618* |
| log/mad/group/rec (the shipped default) | 125 | 0.1941 | 0.0278 |
| log/z/group/rec | 100 | 0.1368 | 0.0027 |
| linear/z/group/rec | 128 | 0.0592 | 0.0213 |
| linear/z/global/rec | 57 | 0.0497 | 0.0266 |
| log/z/global/norec | 60 | -0.0077 | -0.0077 |
| log/mad/global/norec | 50 | -0.0204 | -0.0204 |
| log/mad/global/rec | 5 | -0.0336 | -0.0336 |
| log/z/global/rec | 6 | -0.0336 | -0.0336 |

**Faithfulness ranks almost perfectly by node count**, so this ordering is not a
ranking of selectors. Section 4 is the comparison that counts.

Other readings:

- `linear/z/global` is the only variant whose single floor (0.2603 per row)
  lands near Marks' 0.1 node cut. Global *log* screens are brutally strict
  (floor 1.58; 238 of 1.35M rows survive) and give 5-60 node circuits with
  negative faithfulness and completeness 0.67-0.80.
- `linear/mad` at any scope is broken: raw-scale MAD puts the floor at ~0.004,
  20x too low, and `screen_q` cannot lift it (section 3).
- Turning Stage 2 off makes every arm a superset that contains nearly all of
  Marks' nodes (it misses 6 with log/mad/group, 2 with the linear arms). **The
  screen almost never misses a Marks node; the recurrence test is what drops
  them.**

## 3. Sizing knobs, and which ones actually work

- **`screen_q` is a weak knob on the linear scale.** For `linear/mad`, q from 1
  to 1e-12 moves the circuit only 1318 -> 704 nodes; below ~1e-13 the c
  arithmetic overflows and the screen keeps nothing. The floor is
  `med + c*MAD` on raw abs-IE, and c only ranges ~3-8, so it cannot reach Marks'
  scale from a raw-scale MAD of ~1e-3.
- **The Stage-2 magnitude floor `min_median_abs` does size it.** 0.06 -> 126
  nodes for `linear/mad`, matched to the default arm's 125.
- Under a **global** screen the binomial degenerates: `p0 = 7.4e-7`, so all 208
  alphabet nodes are BH-significant and the selection is done entirely by the
  10%-of-prompts recurrence floor (208 -> 65) and the sign check (65 -> 57). The
  28 Marks nodes it drops fail on recurrence (median 7 of 100 prompts), not on
  significance; only 3 of his 84 never clear the screen at all.
- The recurrence test's unit is `(site_ordinal, feature_idx, token_idx)` and its
  trials are the 100 prompts. BH is one family over the whole alphabet, not one
  test per layer.

## 4. Cutoff sweep, Jaccard, and the size-matched curve

`scripts/explore_global_cutoffs.py` -> `figures/global-cutoff-sweep.png`,
`docs/global-cutoff-sweep.csv`. The global linear z-screen *is* `abs-IE > cutoff`
on rows, so the sweep is parameterised by the cutoff and `screen_q` is inverted
from it (verified against `mad_screen`; above cutoff ~0.43 the inversion
underflows float64 and only the direct form works).

| cutoff | screen_q | nodes | equiv. node threshold | shared with Marks' 84 | J vs Marks 84 | J vs size-matched |
|---|---|---|---|---|---|---|
| 1.0 | 3.7e-72 | 89 | 0.0965 | 43 | 0.331 | 0.338 |
| 0.55 | 8.6e-19 | 133 | 0.0621 | 65 | 0.428 | 0.565 |
| **0.45** | 4.0e-11 | 149 | 0.0536 | 71 | **0.438** | 0.594 |
| 0.2603 | 1.0 | 208 | 0.0350 | 81 | 0.384 | 0.651 |
| 0.20 | 152 | 259 | 0.0217 | **84** | 0.324 | **0.666** |
| 0.040 | 3.0e5 | 1243 | 0.0021 | 84 | 0.068 | 0.569 |
| 0.030 | 3.7e5 | 1742 | 0.0014 | 84 | 0.048 | 0.537 |
| 0.020 | 4.5e5 | 3056 | 0.0007 | 84 | 0.027 | 0.501 |
| 0.015 | 5.0e5 | 4549 | 0.0005 | 84 | 0.018 | 0.504 |

Agreement peaks **above** the q=1 operating point, at cutoff 0.45 -> J = 0.438.
From cutoff 0.20 downward the set already *contains* all 84 nodes, so loosening
only dilutes. **Containment is not agreement**: even at the peak the two
selectors share well under half their union, and against a size-matched Marks
threshold the ceiling is 0.67.

### The comparison that is not confounded by size

`docs/faithfulness-vs-size.csv`, `figures/faithfulness-vs-size.png` — 11 Marks
points (58-4301 nodes) against 8 global-z points (162-4549).

| nodes | Marks `23_mean` / `all_mean` | global-z `23_mean` / `all_mean` |
|---|---|---|
| ~161 | **0.3813 / 0.2159** | 0.2298 / 0.1480 |
| 208 | **0.3819 / 0.2755** | 0.2669 / 0.2444 |
| ~1250 | **0.7396 / 0.7956** | 0.6903 / 0.7115 |
| ~1460 | **0.7611 / 0.7716** | 0.7094 / 0.7487 |
| ~1740 | **0.8105 / 0.8005** | 0.7670 / 0.7561 |
| ~2200 | **0.8554 / 0.8538** | 0.8063 / 0.7265 |
| ~3100 | **0.9136 / 0.8991** | 0.8558 / 0.7440 |
| ~4400 | **0.9349 / 0.8563** | 0.8646 / 0.7881 |

**Marks' constant node threshold is above the screen at every size, on both
protocols, with no crossover.** Both curves are monotone in circuit size, which
is why every unmatched comparison in this line of work read as a win for
whichever arm happened to select more nodes.

Earlier size-matched checks point the same way: `linear/mad` sized to 126 nodes
with `min_median_abs=0.06` scores 0.0915 / 0.0516 against `log/mad` at 107 nodes
scoring 0.1810 / 0.0251 — note this pair is the one place the two protocols
disagree, so the log-scale advantage *there* holds only under §3.2's rule.

## 5. Conclusions

1. **The log scale is not cosmetic.** It is the scale on which one constant
   threshold is a constant-alpha test, and it is what makes Marks' single number
   defensible. Linear-scale screens are either incoherent across groups (`mad`)
   or need a floor 20x higher to compensate (`z`).
2. **Per-group calibration is not earning its keep on this task**, and going
   global does not rescue the calibrated arm either — it changes which nodes are
   selected far more than how good they are.
3. **The recurrence test is what loses the Marks nodes**, and under a global
   screen it degenerates into "appeared in >= 10 of 100 prompts with a
   consistent sign". Its marginal trades are bad in every variant: the nodes it
   drops average abs-IE 0.14-0.18, the ones it adds 0.025-0.047.
4. **Inverting the null is not the fix.** The selected set beats the rejected set
   on mean abs-IE by 7-25x in every variant.
5. **Mean abs-IE of a node set does not predict its faithfulness.**
   `log/mad/global/norec` has the second-highest mean abs-IE in the grid (0.176
   over 50 nodes) and scores -0.0204. Three independent instances of this
   appeared in one day; use ablation, never magnitude summaries, as the arbiter.
6. **Report faithfulness only at matched node count.** Every arm's curve is
   monotone in size, so an unmatched number says nothing about the selector.

## Reproducing

```bash
PY=C:/Users/Maxim/anaconda3/python.exe
$PY scripts/explore_threshold_zscores.py        # section 1, both figures
$PY scripts/explore_screen_variants.py          # section 2, the 16-cell grid
$PY scripts/explore_global_cutoffs.py           # section 4, sweep + figure
$PY scripts/plot_faithfulness_vs_size.py        # section 4, matched curve

# one faithfulness cell (~40 s on a 4090; the node hive is cached)
$PY main.py -c configs/pythia_ha_rc.yaml \
    data.num_examples=100 evaluation.num_examples=40 evaluation.minimality=false \
    edges.enabled=false run.output_root=outputs_n100 logging.backend=[] \
    selection.screen_scale=linear selection.screen_method=z \
    selection.screen_scope=global selection.recurrence=false
```

All selection variants share one node-attribution hive: its cache key is
`(suite, data, attribution, seed)`, so no GPU attribution is repeated.
