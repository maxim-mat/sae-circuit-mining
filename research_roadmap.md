# Process Mining of Transformer Circuits — Research Roadmap

Consolidated from the full design conversation. Ordered by dependency:
data regeneration unblocks Part 1, which unblocks Part 2, which unblocks
evaluation. Framing/extensions run in parallel once results exist.

---

## 0. Data regeneration (prerequisite — blocks clean results everywhere)

- **float16 → float32 attribution storage.** The fp16 subnormal floor
  (~24% of nonzero values on current data) corrupts the null tail that
  calibrates the Part-1 screen. Highest-priority fix; removes a whole class
  of reviewer objections. zstd keeps the size cost modest.
- **Fix counterfactual min-length truncation** that misaligns clean vs. CF
  token positions and injects structured junk into the null tail.
- **Verify BOS handling** in `no_concat` mode (GPT-2 tokenizer + prepend_bos).
- **Add a `template_id` column** so recurrence blocking works (IOI prompts
  sharing a template are correlated trials).
- **Optional: cache per-(prompt,layer) feature gradients** so stop-gradient
  edge IEs become computable post-hoc without a third model pass.

## 1. Part 1 — calibrated feature selection (remaining work)

- Re-validate on regenerated data: `gaussianity_report` tail-calibration
  (tail2 ≈ 1.0) before trusting the q arithmetic; confirm group-size
  histogram modes are layers, not prompt-length bands.
- Enable `block_col` recurrence blocking once `template_id` exists; fall back
  to beta-binomial if the negative control still selects > 0 features.
- **Mediator problem:** features with ~0 marginal node-IE that matter only
  through a downstream edge are currently dropped before edges are even
  considered. Either (a) scope the claim to individually-relevant features,
  or (b) add a looser "edge-candidate" node tier feeding the edge stage, with
  an edge-level recurrence test that can promote a node back in.
- **Borrowable statistical upgrades** (genomics / neuroimaging large-scale
  inference): IHW / HAMT variance-as-covariate (the null scale correlates
  with signal by layer — the regime where naive standardization loses
  validity); HMM-FDR for dependence among co-firing features;
  surrogate-variable decorrelation as a robustness check; Westfall–Young
  max-T permutation (≈ the sign-flip null already built in) and
  Benjamini–Yekutieli as the assumption-light FDR backstop.

## 2. Part 2 — ordering / DFG / discovery (remaining work)

- **Edge density is the current bottleneck** (see initial results): the DFG
  is near-complete bipartite between adjacent layers, which produces the
  silent-transition soup in the net. Fixes, in order:
  1. Tighten the edge gate: `top_k_parents` → 1–2, raise `rel_threshold`
     until inter-layer DFG density visibly drops. (Highest leverage.)
  2. Raise `n_extensions` (3 → 15–30) so the miner sees enough interleavings
     to certify parallel cuts instead of inserting silent skips.
  3. Raise IM-f `noise_threshold` (0.2 → 0.3–0.4) to prune low-frequency
     variants spawning τ-routes.
  - Keep in mind: some silent transitions are *correct* (AND-split
    split/join gateways). Target the "unexplained skip" τ's, not the
    parallel-branch τ's; faithfulness (§3) is the real arbiter.
- **Virtual weights → stop-gradient edge IEs** for the paper version. Virtual
  weights see only the direct residual path — blind to attention/QK routing,
  where IOI's name-mover mechanism lives. Then **diff the two nets**: the
  disagreement is plausibly the attention-mediated structure — a result.
- Decide `ie_weighting="count"` vs `"normalized_ie"` empirically (if the net
  is unchanged, `count` is the simpler story).
- Monitor `mean_concurrent_per_case` as an over-parallelization alarm.

## 3. Evaluation (determines whether the paper lands)

- **Track (i) — Conformance (novel metric).** Partial-order-aware
  fitness/precision: does a held-out prompt's partial order replay in the
  mined net? A genuinely new notion of circuit generalization. Negative
  controls (shuffled-attribution log; different-task log) must score low, or
  the metric is unfalsifiable.
- **Track (ii) — Faithfulness (headline).** Mean-ablate everything outside
  the discovered net; measure metric recovery vs. an SFC subgraph at matched
  node budget. Independently ablate each discovered AND-branch to confirm
  concurrency is real (interaction effects ≈ 0). This is what mech-interp
  reviewers judge, and what tells you which silent transitions were real.

## 4. Extensions / stretch

- **Cross-CF comparison** across the 8 counterfactual types: shared backbone
  vs. CF-specific subcircuits (mine one net per CF, diff them).
- **Second task** (greater-than, Hanna et al.), then **second model**.
- **CLT / crosscoder pivot:** CLT-Forge + Circuit-Tracer give per-layer-causal
  features (dissolving the clustering problem) and native edge attributions.
  Strong v2 once the SAE pipeline is proven. Note fmxcoder result: acausal
  crosscoders fail to learn cross-layer features — use the *causal* CLT
  variant.

## 5. Framing / positioning

- **Three-legged related work:** (1) large-scale-inference statistics from
  genomics/neuroimaging (Efron empirical null, BH/BY, HMM-FDR, IHW/HAMT); (2)
  process-discovery-on-scientific-data precedent in bioinformatics (van der
  Aalst "When Process Mining Meets Bioinformatics"); (3) Petri-nets for
  signal-flow decomposition in systems biology (t-invariants / MCT-sets as
  the AND-split analogue).
- **Core novelty:** the triangulation onto transformer circuits + the
  inverted epistemics — causality is *measured* (edge IEs), order is derived,
  concurrency is the residue (vs. classical PM inferring causality from
  observed order).
- Frame Petri nets as a **behavioral abstraction with executable conformance**,
  not a replacement mechanistic ontology (pre-empts the "continuous dataflow
  ≠ token control flow" critique).
- **Selection procedure** makes the *event log* defensible; **circuit claims**
  still stand or fall on faithfulness ablations. Keep those roles distinct.
- **Venues:** ICPM / BlackboxNLP / interpretability workshops for early
  feedback; ICLR / NeurIPS main track once faithfulness lands.

## Key references (by leg)

- **Process mining:** van der Aalst, *Process Mining: Data Science in Action*
  (2016); van der Aalst, Weijters & Măruster, "Workflow Mining…", IEEE TKDE
  2004 (DF-as-causality-proxy); Leemans et al. (Inductive Miner / IMf);
  Lu, Fahland & van der Aalst (partial-order event data / conformance);
  van Zelst et al. (event abstraction survey); Carmona et al.,
  *Conformance Checking* (2018).
- **Mech interp:** Marks et al. (Sparse Feature Circuits); Wang et al. 2022
  (IOI); Hanna et al. (greater-than); Conmy et al. (ACDC); Syed et al. /
  Hanna (EAP-IG); Kramár et al. (AtP*); Ameisen/Lindsey et al. (Circuit
  Tracing methods; Biology of an LLM); Hanna et al. (Circuit-Tracer library);
  CLT-Forge; Lindsey et al. (crosscoders); Kamath et al. (attention tracing).
- **Statistics:** Efron (empirical null / large-scale inference); Benjamini–
  Hochberg; Benjamini–Yekutieli; Sun & Cai (HMM-FDR); Ignatiadis et al.
  (IHW); Gang & Banerjee (HAMT); Westfall–Young (max-T permutation).
- **Cross-domain:** van der Aalst et al. ("When Process Mining Meets
  Bioinformatics"); Heiner & Koch (Petri nets for biological pathways;
  t-invariants, MCT-sets).
