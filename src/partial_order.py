"""
Part 2 — partial-order construction and Petri-net discovery.

Consumes the event log written by Part 1 (calibrated_feature_selection.py):
    event_logs/eventlog/counterfactual=<cf>/  with columns
      prompt_idx, layer, token_idx, token_id, feature_idx, attribution,
      abs_attr, sign
plus an EDGE-ATTRIBUTION table (computed by a separate model pass — see
compute_edge_attributions docstring) with columns
      prompt_idx, src_layer, src_feat, dst_layer, dst_feat, edge_ie
and produces a directly-follows graph (DFG) for the inductive miner (IM-f).

Design (per case = per prompt; token positions collapsed):

  1. Events.  Node identity is (layer, feature_idx) — a feature index is only
     unique within its layer. A feature firing at several token positions in
     one prompt is ONE prompt-level event. Same-layer nodes read the same
     residual state (gpt2-small-res-jb hooks the residual stream once/layer),
     so no edge can ever run between them: within-layer incomparability is
     architectural, and all ordering lives on cross-layer pairs.

  2. Edges.  Cross-layer candidate pairs (src_layer < dst_layer) get an edge
     IE (stop-gradient chain rule). Optional second calibrated screen on
     |edge_ie|, reusing Part 1's MAD machinery with groups = (case,
     layer-pair) — toggleable, OFF by default.

  3. Order.  Screened edges form a down-layer DAG. Its transitive reduction
     (Hasse diagram) is the "directly-causes" relation = the honest analogue
     of directly-follows. Two events are concurrent in a case iff no directed
     path connects them (includes all co-occurring same-layer pairs).

  4. DFG.   Hasse edge a->b            => directly-follows a>b, weight ~|edge IE|
            concurrent co-occurring {a,b} => BOTH a>b and b>a (interleaving
                                            semantics; IM parallel cut needs
                                            bidirectional edges, absent edges
                                            read as exclusivity/XOR).
            Frequencies aggregate over cases; IM-f prunes by (IE-weighted)
            frequency.

Inverted-epistemics note for the writeup: classical process mining INFERS
causality from observed order (van der Aalst, Weijters & Maruster, "Workflow
Mining: Discovering Process Models from Event Logs", IEEE TKDE 2004; van der
Aalst, "Process Mining", 2016). Here causality is MEASURED (edge IEs) and
order is derived; concurrency is the residue. The edge IEs are an explicit
concurrency oracle replacing the interleaving heuristic.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

EVENT_KEY = ["prompt_idx", "layer", "feature_idx"]
EDGE_COLS = ["prompt_idx", "src_layer", "src_feat", "dst_layer", "dst_feat", "edge_ie"]


# ===========================================================================
# Event preparation — collapse Part 1's per-token log to per-prompt events
# ===========================================================================

def prompt_events(event_log: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-(token) event log to per-(prompt, layer, feature)
    events. Aggregates the multiple token-position rows of a feature within a
    prompt: event magnitude = summed |attribution|, event sign = sign of the
    signed sum (net direction of the feature's contribution in that prompt)."""
    g = event_log.groupby(EVENT_KEY, sort=False)
    ev = g.agg(
        abs_attr=("abs_attr", "sum"),
        signed=("attribution", "sum"),
        n_positions=("token_idx", "nunique"),
    ).reset_index()
    ev["sign"] = np.sign(ev["signed"]).astype(np.int8)
    ev["node"] = list(zip(ev["layer"], ev["feature_idx"]))
    return ev


def activity_label(layer: int, feat: int) -> str:
    """Human-readable, miner-friendly activity id for a node."""
    return f"L{int(layer):02d}.F{int(feat)}"


# ===========================================================================
# Candidate edges — the cross-layer pairs whose edge IE the model pass needs
# ===========================================================================

def candidate_edges(event_log: pd.DataFrame) -> pd.DataFrame:
    """Per prompt, all ordered cross-layer pairs (src_layer < dst_layer) of
    selected features. This is exactly the set compute_edge_attributions must
    score — same-layer pairs are excluded (architecturally incomparable)."""
    ev = prompt_events(event_log)
    out = []
    for pid, g in ev.groupby("prompt_idx", sort=False):
        recs = sorted(g[["layer", "feature_idx"]].itertuples(index=False),
                      key=lambda r: r.layer)
        for a, b in itertools.combinations(recs, 2):
            if a.layer < b.layer:               # strict: no same-layer edges
                out.append((pid, a.layer, a.feature_idx, b.layer, b.feature_idx))
    return pd.DataFrame(out, columns=EDGE_COLS[:-1])


# ===========================================================================
# Edge IE — post-hoc virtual weights (no model pass; needs SAE weight matrices)
# ===========================================================================
#
# The only edge estimate computable WITHOUT re-running the model is the
# virtual weight: feature a writes W_dec[a] into the residual stream, feature
# b reads it via its encoder row, so the linear a->b coupling is
#       vw(a->b) = W_dec[src_layer][a] . W_enc[dst_layer][:, b].
# It captures the direct residual path only (NOT attention-mediated cross-
# position routing) — a documented approximation of the stop-gradient IE.
# We make it per-prompt by routing through a's stored node IE:
#       edge_ie(a->b, prompt) = vw(a->b) * node_ie(a, prompt).
#
# Density warning: vw is nonzero for nearly every pair. Emitting all pairs
# makes each case's DAG near-complete, so transitive reduction collapses the
# partial order to the trivial layer order. Hence top_k_parents density
# control is ON by default for the post-hoc path.

def load_sae_directions(release: str, device: str = "cpu") -> dict:
    """Load per-layer encoder/decoder directions for the residual-stream SAEs,
    mirroring main.py's release handling. Returns
        {layer: {"W_enc": (d_model, d_sae), "W_dec": (d_sae, d_model)}}
    as numpy float32 arrays. Requires sae_lens (run in the user's env)."""
    from sae_lens import SAE
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    saes_map = get_pretrained_saes_directory()[release].saes_map
    dirs = {}
    for sid in saes_map:
        layer = int(sid.split(".")[1]) if "." in sid else 0
        sae = SAE.from_pretrained(release=release, sae_id=sid, device=device)[0]
        W_enc = sae.W_enc.detach().cpu().float().numpy()          # (d_model, d_sae)
        W_dec = sae.W_dec.detach().cpu().float().numpy()          # (d_sae, d_model)
        if W_dec.shape[0] != W_enc.shape[1]:                      # guard orientation
            W_dec = W_dec.T
        dirs[layer] = {"W_enc": W_enc, "W_dec": W_dec}
    return dirs


def compute_edge_attributions(
    event_log: pd.DataFrame,
    sae_dirs: dict,
    weighting: str = "upstream",
    top_k_parents: int | None = 3,
) -> pd.DataFrame:
    """Post-hoc virtual-weight edge IEs for all cross-layer candidate pairs.

    weighting:
      "upstream" — edge_ie = vw(a->b) * node_ie(a)   (default; per-prompt,
                   routes a's measured causal delta through the linear
                   coupling to b)
      "virtual"  — edge_ie = vw(a->b)                (structural coupling only)
    top_k_parents: if set, keep only each downstream feature's k strongest
      incoming edges per prompt (by |edge_ie|) — density control that keeps
      the DAG sparse enough for a non-trivial partial order. None = keep all.

    Returns a dataframe with columns EDGE_COLS. Torch-free (numpy only).
    """
    ev = prompt_events(event_log)
    ev = ev.assign(node_ie=ev["signed"].to_numpy())
    out = []

    for pid, g in ev.groupby("prompt_idx", sort=False):
        by_layer = {int(l): sub for l, sub in g.groupby("layer", sort=False)}
        layers = sorted(by_layer)
        for i, ls in enumerate(layers):
            src = by_layer[ls]
            Wd = sae_dirs[ls]["W_dec"]                       # (d_sae, d_model)
            s_feats = src["feature_idx"].to_numpy()
            s_ie = src["node_ie"].to_numpy()
            Wd_s = Wd[s_feats]                               # (n_src, d_model)
            for ld in layers[i + 1:]:
                dst = by_layer[ld]
                We = sae_dirs[ld]["W_enc"]                   # (d_model, d_sae)
                d_feats = dst["feature_idx"].to_numpy()
                We_d = We[:, d_feats]                        # (d_model, n_dst)
                vw = Wd_s @ We_d                             # (n_src, n_dst)
                if weighting == "virtual":
                    eie = vw
                else:                                        # upstream routing
                    eie = vw * s_ie[:, None]

                if top_k_parents is not None and eie.shape[0] > top_k_parents:
                    # per downstream column, keep k strongest-|eie| parents
                    keep = np.zeros_like(eie, dtype=bool)
                    idx = np.argpartition(-np.abs(eie), top_k_parents - 1, axis=0)[:top_k_parents]
                    np.put_along_axis(keep, idx, True, axis=0)
                else:
                    keep = np.ones_like(eie, dtype=bool)

                sr, dc = np.nonzero(keep)
                for r, c in zip(sr, dc):
                    out.append((pid, ls, int(s_feats[r]), ld,
                                int(d_feats[c]), float(eie[r, c])))

    return pd.DataFrame(out, columns=EDGE_COLS)


# ===========================================================================
# Optional second screen on edge IEs — reuse Part 1's MAD machinery
# ===========================================================================

def screen_edges(
    edges: pd.DataFrame,
    enabled: bool = False,
    rel_threshold: float = 0.1,
) -> pd.DataFrame:
    """Add a 'kept_edge' column. If disabled, keep every candidate edge.

    IMPORTANT — why this is NOT the node MAD screen. Part 1's node screen is a
    NOISE-FLOOR test: it works because each (prompt, layer) group is ~99%
    task-irrelevant features, so median/MAD estimate the noise bulk. The edge
    candidate set is the opposite regime: candidate_edges() only pairs
    features that ALREADY survived node selection, so most candidate edges are
    between genuinely relevant features and the |edge_ie| population is
    SIGNAL-dominated, not noise-dominated. A robust noise floor there has no
    noise bulk to latch onto (contamination far exceeds MAD's 50% breakdown),
    so it either keeps everything or, calibrated tightly, erases the real
    chains. (This was verified empirically; see the design notes.)

    The appropriate edge gate is therefore RELATIVE MAGNITUDE, not a noise
    test: within each layer-pair, drop edges whose |edge_ie| is below
    rel_threshold * median(|edge_ie|) for that layer-pair. This removes only
    the weakest mediated/leak edges (e.g. a direct a->c that is really a->b->c)
    while preserving the causal backbone, and it lets transitive reduction do
    the rest. OFF by default; the unscreened DAG + transitive reduction is
    already a strong baseline (redundant edges are removed structurally).
    """
    edges = edges.copy()
    if not enabled or len(edges) == 0:
        edges["kept_edge"] = True
        return edges

    edges["_pair"] = (edges["src_layer"].astype(int) * 1000
                      + edges["dst_layer"].astype(int))
    absie = edges["edge_ie"].abs()
    pair_med = edges.assign(_a=absie).groupby("_pair")["_a"].transform("median")
    edges["kept_edge"] = absie >= rel_threshold * pair_med
    return edges.drop(columns="_pair")


# ===========================================================================
# Per-case causal DAG and its partial-order relations
# ===========================================================================

def case_dag(ev_case: pd.DataFrame, edges_case: pd.DataFrame) -> nx.DiGraph:
    """Build one prompt's causal DAG. Nodes = (layer, feature_idx) events;
    edges = kept cross-layer edges among present nodes (down-layer only)."""
    g = nx.DiGraph()
    for r in ev_case.itertuples(index=False):
        g.add_node((int(r.layer), int(r.feature_idx)), layer=int(r.layer),
                   sign=int(r.sign), abs_attr=float(r.abs_attr))
    for r in edges_case.itertuples(index=False):
        if not getattr(r, "kept_edge", True):
            continue
        s, d = (int(r.src_layer), int(r.src_feat)), (int(r.dst_layer), int(r.dst_feat))
        if g.has_node(s) and g.has_node(d) and r.src_layer < r.dst_layer:
            g.add_edge(s, d, weight=abs(float(r.edge_ie)))
    return g


def case_relations(g: nx.DiGraph):
    """Return (hasse_edges, concurrent_pairs) for one case.
    hasse: transitive reduction = directly-causes.
    concurrent: co-occurring node pairs with no directed path either way
    (includes all same-layer pairs)."""
    hasse = set(nx.transitive_reduction(g).edges()) if g.number_of_edges() else set()
    reach = {n: nx.descendants(g, n) for n in g.nodes()}
    concurrent = set()
    for a, b in itertools.combinations(g.nodes(), 2):
        if b not in reach[a] and a not in reach[b]:
            concurrent.add(frozenset((a, b)))
    return hasse, concurrent


# ===========================================================================
# DFG construction
# ===========================================================================

def build_dfg(
    event_log: pd.DataFrame,
    edges: pd.DataFrame,
    ie_weighting: str = "normalized_ie",
    screen_edges_enabled: bool = False,
    rel_threshold: float = 0.1,
):
    """Aggregate all cases into a DFG for the inductive miner.

    ie_weighting:
      "normalized_ie" — a directly-causes occurrence contributes
                        |edge_ie| / median(|edge_ie|)  (mean ~1, so IE
                        strength modulates frequency while staying on the same
                        scale as concurrency's unit contributions);
      "count"         — every relation occurrence contributes 1 (standard DFG,
                        IE ignored — useful as a structural baseline).
    Concurrent co-occurring pairs always contribute 1.0 in BOTH directions.
    screen_edges_enabled toggles the relative-magnitude edge gate (see
    screen_edges); transitive reduction removes redundant edges regardless.

    Returns (dfg, starts, ends, stats).
    """
    ev = prompt_events(event_log)
    edges = screen_edges(edges, enabled=screen_edges_enabled,
                         rel_threshold=rel_threshold)

    kept = edges.loc[edges["kept_edge"], "edge_ie"] if len(edges) else pd.Series(dtype=float)
    med_ie = float(np.median(np.abs(kept))) if len(kept) else 1.0
    med_ie = med_ie if med_ie > 0 else 1.0

    dfg = defaultdict(float)
    starts, ends = defaultdict(float), defaultdict(float)
    n_cases = n_hasse = n_conc = 0

    ev_by_case = dict(tuple(ev.groupby("prompt_idx", sort=False)))
    ed_by_case = dict(tuple(edges.groupby("prompt_idx", sort=False))) if len(edges) else {}
    empty_edges = edges.iloc[0:0]

    for pid, ev_case in ev_by_case.items():
        ed_case = ed_by_case.get(pid, empty_edges)
        g = case_dag(ev_case, ed_case)
        if g.number_of_nodes() == 0:
            continue
        n_cases += 1
        hasse, concurrent = case_relations(g)
        n_hasse += len(hasse); n_conc += len(concurrent)

        for a, b in hasse:
            w = 1.0 if ie_weighting == "count" else g[a][b].get("weight", med_ie) / med_ie
            dfg[(activity_label(*a), activity_label(*b))] += w

        for pair in concurrent:
            a, b = tuple(pair)
            la, lb = activity_label(*a), activity_label(*b)
            dfg[(la, lb)] += 1.0
            dfg[(lb, la)] += 1.0

        for n in g.nodes():
            if g.in_degree(n) == 0:
                starts[activity_label(*n)] += 1.0
            if g.out_degree(n) == 0:
                ends[activity_label(*n)] += 1.0

    stats = {
        "cases": n_cases,
        "activities": len({a for e in dfg for a in e}),
        "dfg_edges": len(dfg),
        "mean_hasse_per_case": n_hasse / max(n_cases, 1),
        "mean_concurrent_per_case": n_conc / max(n_cases, 1),
        "edges_kept": int(edges["kept_edge"].sum()) if len(edges) else 0,
        "median_edge_ie": med_ie,
    }
    return dict(dfg), dict(starts), dict(ends), stats


# ===========================================================================
# Inductive miner (IM-f) on the DFG
# ===========================================================================

def discover_net(
    event_log: pd.DataFrame,
    edges: pd.DataFrame,
    noise_threshold: float = 0.2,
    n_extensions: int = 3,
    seed: int = 0,
    **build_kwargs,
):
    """Discover a Petri net with the Inductive Miner infrequent (IM-f).

    The internal DFG-miner API (inductive_miner.Variants.IMf / IMInstance)
    changed across pm4py versions and is brittle. The stable, public route is
    pm4py.discover_petri_net_inductive(log, noise_threshold=...), which takes
    an event LOG. Our DFG is tiny, so we materialize a small event log by
    sampling linear extensions of each case's partial order (concurrent
    activities appear in randomized order across traces, which is exactly how
    IM's parallel cut infers AND), then mine that log. This keeps concurrency
    semantics intact while using only the supported API.

    Returns (net, initial_marking, final_marking, dfg_tuple, stats).
    """
    import pm4py

    dfg_dict, starts, ends, stats = build_dfg(event_log, edges, **build_kwargs)
    log_df = case_traces(event_log, edges, n_extensions=n_extensions,
                         seed=seed, **_build_only(build_kwargs))

    net, im, fm = pm4py.discover_petri_net_inductive(
        log_df, noise_threshold=noise_threshold,
        activity_key="concept:name", case_id_key="case:concept:name",
        timestamp_key="time:timestamp",
    )
    return net, im, fm, (dfg_dict, starts, ends), stats


def _build_only(build_kwargs: dict) -> dict:
    """Pass through only the kwargs case_traces understands."""
    keep = {"screen_edges_enabled", "rel_threshold"}
    return {k: v for k, v in build_kwargs.items() if k in keep}


def case_traces(
    event_log: pd.DataFrame,
    edges: pd.DataFrame,
    n_extensions: int = 3,
    seed: int = 0,
    screen_edges_enabled: bool = False,
    rel_threshold: float = 0.1,
) -> pd.DataFrame:
    """Materialize an event log by sampling linear extensions of each case's
    causal partial order. Columns follow pm4py conventions:
    case:concept:name, concept:name (activity), time:timestamp.

    Each case contributes n_extensions traces; concurrent activities are
    randomly ordered per trace (subject to the causal DAG), so the miner sees
    both interleavings and infers parallelism. Timestamps are synthetic
    integer positions (topological order), only used to fix trace order."""
    import datetime as _dt

    rng = np.random.default_rng(seed)
    ev = prompt_events(event_log)
    edges = screen_edges(edges, enabled=screen_edges_enabled, rel_threshold=rel_threshold)
    ev_by_case = dict(tuple(ev.groupby("prompt_idx", sort=False)))
    ed_by_case = dict(tuple(edges.groupby("prompt_idx", sort=False))) if len(edges) else {}
    empty = edges.iloc[0:0]

    rows = []
    base = _dt.datetime(2020, 1, 1)
    for pid, ev_case in ev_by_case.items():
        g = case_dag(ev_case, ed_by_case.get(pid, empty))
        if g.number_of_nodes() == 0:
            continue
        for k in range(n_extensions):
            order = _random_topo_order(g, rng)
            cid = f"{pid}_{k}"
            for t, node in enumerate(order):
                rows.append({
                    "case:concept:name": cid,
                    "concept:name": activity_label(*node),
                    "time:timestamp": base + _dt.timedelta(seconds=t),
                })
    return pd.DataFrame(rows)


def _random_topo_order(g, rng):
    """A uniformly-random-ish topological order: repeatedly pick a random
    node whose predecessors are all already placed."""
    import networkx as nx
    indeg = dict(g.in_degree())
    ready = [n for n, d in indeg.items() if d == 0]
    order = []
    succ = {n: list(g.successors(n)) for n in g.nodes()}
    while ready:
        i = rng.integers(len(ready))
        n = ready.pop(int(i))
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return order


def export_dfg(dfg_dict, starts, ends, path: str):
    """Persist the DFG so any miner/version can consume it downstream."""
    pd.DataFrame([{"src": a, "dst": b, "weight": w}
                  for (a, b), w in dfg_dict.items()]
                 ).to_parquet(path + ".edges.parquet", index=False)
    acts = {x for e in dfg_dict for x in e}
    pd.DataFrame([{"activity": a, "start": starts.get(a, 0.0),
                   "end": ends.get(a, 0.0)} for a in acts]
                 ).to_parquet(path + ".nodes.parquet", index=False)


# ===========================================================================
# Loading Part 1 output + CLI
# ===========================================================================

def load_event_log(path: str) -> pd.DataFrame:
    """Load a Part-1 event log: the hive partition dir
    event_logs/eventlog/counterfactual=<cf>/ (or a single parquet file)."""
    df = pd.read_parquet(path)
    needed = {"prompt_idx", "layer", "token_idx", "feature_idx", "attribution"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"event log missing columns: {missing}")
    if "abs_attr" not in df:
        df["abs_attr"] = df["attribution"].abs()
    return df


def run(
    event_log_path: str,
    release: str,
    out_dir: str,
    weighting: str = "upstream",
    top_k_parents: int | None = 3,
    screen_edges_enabled: bool = True,
    rel_threshold: float = 0.1,
    ie_weighting: str = "normalized_ie",
    noise_threshold: float = 0.2,
    device: str = "cpu",
):
    """Full Part-2 run on a Part-1 event log: post-hoc virtual-weight edges ->
    DFG -> IM-f. Writes the DFG and (if pm4py present) a PNML net."""
    import os

    os.makedirs(out_dir, exist_ok=True)

    print(f"loading event log {event_log_path} ...")
    event_log = load_event_log(event_log_path)
    # Part 2 mines one counterfactual at a time. The counterfactual is a hive
    # partition column when the eventlog root is read; pointing at a single
    # counterfactual=<cf>/ dir hides it, so fall back to the dir name.
    if "counterfactual" in event_log:
        cfs = event_log["counterfactual"].astype(str).unique()
        if len(cfs) > 1:
            raise ValueError(f"event log mixes counterfactuals {list(cfs)}; "
                             f"pass one counterfactual=<cf>/ partition")
        cf = cfs[0]
    else:
        cf = os.path.basename(event_log_path.rstrip("/\\")).split("=")[-1]
    ev = prompt_events(event_log)
    print(f"[{cf}] {len(event_log):,} log rows -> {len(ev):,} prompt-events "
          f"over {ev['prompt_idx'].nunique()} prompts, "
          f"{ev.groupby('layer').ngroups} layers")

    print(f"[{cf}] loading SAE directions ({release}) ...")
    sae_dirs = load_sae_directions(release, device=device)

    print(f"[{cf}] computing post-hoc virtual-weight edges "
          f"(weighting={weighting}, top_k_parents={top_k_parents}) ...")
    edges = compute_edge_attributions(
        event_log, sae_dirs, weighting=weighting, top_k_parents=top_k_parents)
    print(f"[{cf}] {len(edges):,} candidate edges")

    dfg, starts, ends, stats = build_dfg(
        event_log, edges,
        ie_weighting=ie_weighting,
        screen_edges_enabled=screen_edges_enabled,
        rel_threshold=rel_threshold,
    )
    print(f"[{cf}] DFG stats:", {k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in stats.items()})

    dfg_path = os.path.join(out_dir, f"dfg_{cf}")
    export_dfg(dfg, starts, ends, dfg_path)
    edges.to_parquet(os.path.join(out_dir, f"edges_{cf}.parquet"), index=False)
    print(f"[{cf}] wrote {dfg_path}.{{edges,nodes}}.parquet and edges_{cf}.parquet")

    try:
        net, im, fm, _, _ = discover_net(
            event_log, edges,
            noise_threshold=noise_threshold,
            ie_weighting=ie_weighting,
            screen_edges_enabled=screen_edges_enabled,
            rel_threshold=rel_threshold,
        )
        from pm4py.write import write_pnml
        pnml_path = os.path.join(out_dir, f"net_{cf}.pnml")
        write_pnml(net, im, fm, pnml_path)
        print(f"[{cf}] discovered net: {len(net.places)} places, "
              f"{len(net.transitions)} transitions -> {pnml_path}")
    except Exception as e:
        print(f"[{cf}] miner/export skipped ({type(e).__name__}: {e}); "
              f"DFG saved for external mining")


def _selftest():
    """Synthetic ground truth respecting virtual-weight physics: each feature
    has ONE residual write direction, so we build a consistent chain
    A -> M -> D where the decoder of each stage aligns with the encoder of the
    next. Verifies the virtual-weight edge path, top-k density control, and the
    graph/DFG logic without SAEs. (Independent a->b, a->c couplings are NOT
    expressible via virtual weights — a feature writes one direction — so the
    test uses a single strong downstream target per feature.)"""
    rng = np.random.default_rng(0)
    d_model, d_sae = 32, 64
    A, M, D = (0, 10), (3, 11), (6, 13)   # a clean causal chain A -> M -> D
    sae_dirs = {l: {"W_enc": rng.normal(0, 0.1, (d_model, d_sae)),
                    "W_dec": rng.normal(0, 0.1, (d_sae, d_model))} for l in (0, 3, 6)}

    def couple(src, dst, strength=6.0):
        (ls, fs), (ld, fd) = src, dst
        v = rng.normal(0, 1, d_model); v /= np.linalg.norm(v)
        sae_dirs[ls]["W_dec"][fs] = strength * v      # a's single write direction
        sae_dirs[ld]["W_enc"][:, fd] = v              # b reads exactly that
    couple(A, M); couple(M, D)

    ev_rows = []
    for case in range(80):
        for (l, f) in [A, M, D]:
            for tok in range(int(rng.integers(1, 4))):
                v = rng.lognormal(-4, 0.5)
                ev_rows.append((case, l, tok, 50256, f, v, v, 1))
    event_log = pd.DataFrame(ev_rows, columns=[
        "prompt_idx", "layer", "token_idx", "token_id", "feature_idx",
        "attribution", "abs_attr", "sign"])

    edges = compute_edge_attributions(event_log, sae_dirs, top_k_parents=3)
    laA, laM, laD = activity_label(*A), activity_label(*M), activity_label(*D)
    # inspect magnitudes: planted A>M and M>D should dominate the noise A>D
    mag = (edges.assign(p=edges.src_layer.astype(str) + ">" + edges.dst_layer.astype(str))
           .groupby("p")["edge_ie"].apply(lambda s: s.abs().mean()))
    dfg, starts, ends, stats = build_dfg(event_log, edges, screen_edges_enabled=True)
    print("mean |edge_ie| by layer-pair:", {k: round(v, 4) for k, v in mag.items()})
    print("stats:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    print("chain A>M>D recovered:",
          (laA, laM) in dfg and (laM, laD) in dfg
          and (laA, laD) not in dfg)   # A>D redundant, removed by transitive reduction
    print("no reverse edges:",
          (laM, laA) not in dfg and (laD, laM) not in dfg)
    print("starts:", {k: round(v, 1) for k, v in starts.items()},
          "ends:", {k: round(v, 1) for k, v in ends.items()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Part 2: post-hoc virtual-weight edges + Petri-net discovery "
                    "from a Part-1 event log.")
    ap.add_argument("--event_log",
                    help="Part-1 event log: event_logs/eventlog/counterfactual=<cf>/")
    ap.add_argument("--release", default="gpt2-small-res-jb",
                    help="SAE release (for W_enc/W_dec virtual weights)")
    ap.add_argument("--out_dir", default="nets")
    ap.add_argument("--weighting", choices=["upstream", "virtual"], default="upstream")
    ap.add_argument("--top_k_parents", type=int, default=3,
                    help="keep each downstream feature's k strongest parents "
                         "(density control; 0 = keep all)")
    ap.add_argument("--no_edge_screen", action="store_true",
                    help="disable the relative-magnitude edge gate")
    ap.add_argument("--rel_threshold", type=float, default=0.1)
    ap.add_argument("--ie_weighting", choices=["normalized_ie", "count"],
                    default="normalized_ie")
    ap.add_argument("--noise_threshold", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or not args.event_log:
        _selftest()
    else:
        run(args.event_log, args.release, args.out_dir,
            weighting=args.weighting,
            top_k_parents=(None if args.top_k_parents == 0 else args.top_k_parents),
            screen_edges_enabled=not args.no_edge_screen,
            rel_threshold=args.rel_threshold,
            ie_weighting=args.ie_weighting,
            noise_threshold=args.noise_threshold,
            device=args.device)
