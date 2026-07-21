# Preliminary plan: incorporating the SAE error term into the selection statistics

Status: **research / undecided.** Parked until the error-vs-inference literature
is reviewed. This file records the directions found so a later session can pick
up without re-deriving them.

## What the error term actually is (framing that constrains the options)

The stored `feature_idx == -1` rows are the SAE **reconstruction residual**
`x - decode(encode(x))`, with an indirect effect computed the same way as any
node (`grad_resid · Δresidual`). Key properties that rule some methods in/out:

- It is **approximation / misspecification error** (the part of the causal
  effect the feature basis fails to carry), *not* stochastic measurement noise.
  It is **deterministic given the prompt** — no sampling distribution.
- It is **per-site** `(layer, token)`, **shared across all features** at that
  site — there is no per-feature standard error.
- The residual-stream IE is known **exactly** (`Σ features + error = total`).
  The uncertainty is in the **attribution to the basis**, not in a measurement.

Consequence: the textbook "measurement-error models" family
(errors-in-variables, Deming/TLS, SIMEX) is a **mismatch** — it corrects a noisy
proxy of a latent truth with known variance, which is not our situation. Same
for inverse-variance meta-analysis (needs a per-feature SE we do not have).

## Direction A (recommended) — error share as an IHW/HAMT covariate

Already on the roadmap as "variance-as-covariate." The error node makes the
covariate concrete and mechanistic instead of using layer index:

- Covariate per event = **site-normalized error share**
  `|IE_error| / (Σ|IE_feature| + |IE_error|)` at that `(prompt, layer, token)`.
- Feed it to covariate-modulated FDR: **IHW** (Ignatiadis & Huber) or **HAMT**
  (Gang, Sun & Banerjee) at the Stage-2 (recurrence) test in
  `calibrated_feature_selection`.
- Effect: features at sites the SAE reconstructs poorly (high error share, the
  decomposition is untrustworthy there) get down-weighted / a wider null.
- Honest caveat: this is **selection-on-reliability** — it biases the alphabet
  toward well-reconstructed regions. Usually desired; must be stated.

## Direction B (simpler, cruder) — site-level completeness gate

Hard-threshold version of A: drop `(prompt, layer[, token])` sites from the
recurrence trials where the error node carries most of the residual-stream IE
("don't test where the instrument can't see"). Pairs naturally with the
faithfulness story since error share *is* a local completeness measure.

## Direction C (likely reject) — errors-in-variables / meta-analysis

Only viable if a per-feature variance can be manufactured (e.g. propagating
`grad` uncertainty). The error node alone does not provide it (per-site, not
per-feature). Document why it was rejected rather than silently dropping it.

## Cross-cutting caveat (applies to A/B/C)

The reconstruction error measures **basis incompleteness**, not the
**linearization error of the IE estimate** (attrib/IG vs. exact patching). If
the goal is "how wrong is this IE number," that is the attrib-vs-`_pe_exact`
gap, estimable only by running exact patching on a subsample — which doubles as
an attribution-method validation. The error node will not give it.

## Next step when resumed

Skim IHW (Ignatiadis & Huber 2016/2021) and HAMT (Gang–Sun–Banerjee) for the
covariate assumptions (independence of covariate and p-value under the null),
check they hold for error share, then wire the covariate into Stage 2 of
`calibrated_feature_selection`. Everything upstream is already in place: the raw
parquet retains the `feature_idx == -1` rows for exactly this.
