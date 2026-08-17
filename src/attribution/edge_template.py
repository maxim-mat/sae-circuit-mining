"""Which (upstream, downstream, stop-gradient) triples to score.

Marks et al. (arXiv:2403.19647) do not score all site pairs. circuit.py walks
layers in reverse and emits a fixed local template, applying stop-gradients to
intermediate submodules so every edge is a DIRECT effect -- the influence of u
on d that is not mediated by anything else in the layer. Long-range influence is
represented by composing edges along the residual chain, never by a long-range
edge. Reproducing their edge counts requires reproducing this topology exactly.

`total_effect_template` preserves this repo's earlier all-pairs / consecutive
behaviour, which computes TOTAL effects (no stop-gradients, gradient flows
through the whole frozen network). That matches the process-mining
"eventually-follows" relation and is transitive and non-exclusive: direct and
total edges must never be summed or mixed in one graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.suites import Site, Suite

# Sentinel destination for the metric node. circuit.py:
#   edges[f"resid_{len(resids)-1}"]["y"] = effects[resids[-1]]
Y_SITE = Site(ordinal=-1, block=-1, kind="y", release="", sae_id="", hook="")


@dataclass(frozen=True)
class EdgeSpec:
    src: Site
    dst: Site
    stop: tuple[Site, ...] = ()


def _by_kind(suite: Suite):
    embed = next((s for s in suite.sites if s.kind == "embed"), None)
    resids = sorted((s for s in suite.sites if s.kind == "resid"),
                    key=lambda s: s.ordinal)
    idx = {(s.kind, s.block): s for s in suite.sites}
    return embed, resids, idx


def marks_template(suite: Suite) -> list[EdgeSpec]:
    """The per-layer template from circuit.py, plus the final resid -> y edge.

    `prev` walks the residual chain by ordinal rather than by block index:
    gpt2-res carries two resid sites for block 11 (resid_pre and resid_post),
    so block number does not identify a residual site uniquely, while the
    chain order always does.
    """
    embed, resids, idx = _by_kind(suite)
    if not resids:
        raise ValueError(f"Suite {suite.name!r} has no resid sites to build a chain from")
    specs: list[EdgeSpec] = []

    prev = embed
    for resid in resids:
        b = resid.block
        attn, mlp = idx.get(("attn", b)), idx.get(("mlp", b))
        if mlp is not None:
            specs.append(EdgeSpec(mlp, resid))                        # MR
        if attn is not None:
            specs.append(EdgeSpec(attn, resid, tuple(x for x in (mlp,) if x)))  # AR
            if not suite.parallel_attn and mlp is not None:
                specs.append(EdgeSpec(attn, mlp))                     # AM

        if prev is not None:
            if mlp is not None:
                specs.append(EdgeSpec(prev, mlp, tuple(x for x in (attn,) if x)))  # RM
            if attn is not None:
                specs.append(EdgeSpec(prev, attn))                    # RA
            specs.append(EdgeSpec(prev, resid,
                                  tuple(x for x in (attn, mlp) if x)))  # RR
        prev = resid

    specs.append(EdgeSpec(resids[-1], Y_SITE))
    return specs


def total_effect_template(suite: Suite, pairs: str) -> list[EdgeSpec]:
    """All (pairs='all') or adjacent (pairs='consecutive') site pairs, no
    stop-gradients. TOTAL-effect edges; not comparable to Marks' edge counts."""
    if pairs not in ("all", "consecutive"):
        raise ValueError(f"Unknown pairs {pairs!r}; expected 'all' or 'consecutive'")
    sites = sorted(suite.sites, key=lambda s: s.ordinal)
    if pairs == "consecutive":
        return [EdgeSpec(a, b) for a, b in zip(sites, sites[1:])]
    return [EdgeSpec(a, b) for i, a in enumerate(sites) for b in sites[i + 1:]]


def build_template(suite: Suite, cfg_edges) -> list[EdgeSpec]:
    if cfg_edges.template == "marks":
        return marks_template(suite)
    if cfg_edges.template == "total_effect":
        return total_effect_template(suite, cfg_edges.pairs)
    raise ValueError(
        f"Unknown edge template {cfg_edges.template!r}; "
        f"expected 'marks' or 'total_effect'"
    )
