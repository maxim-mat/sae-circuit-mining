"""Model + SAE-dictionary suites, and the site table that orders them.

A "site" is one hook point with an SAE attached. Sites carry an explicit
ordinal rather than an integer parsed out of the SAE id, because typed
dictionaries (embed / attn / mlp / resid) cannot be ordered by block number
alone:

  * resid_post of block L is the same tensor as resid_pre of block L+1, so
    parsing L collides two disjoint feature spaces (the reason the previous
    sae_layer_ordinal existed at all);
  * attn and mlp are PARALLEL in Pythia, so no total order over integers
    describes them honestly. `Suite.parallel_attn` records this so downstream
    partial-order work can recover the concurrency; the consecutive ordinals
    assigned here are a linearisation for the edge code's src < dst test.
    Edges across a falsely-ordered pair score ~0.

The pythia-marks suite is the SAE set from Marks et al. 2024
("Sparse Feature Circuits", arXiv:2403.19647), served through the sae_lens
catalog as ctigges/pythia-70m-deduped__{res,att,mlp}-sm_processed rather than
the dictionary_learning .pt files their repo downloads.

Self-check: python -m pytest tests/test_suites.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass(frozen=True)
class Site:
    """One SAE-instrumented hook point."""
    ordinal: int      # strict position in execution order; the total order edges use
    block: int        # transformer block index; -1 for embed
    kind: str         # "embed" | "attn" | "mlp" | "resid"
    release: str      # sae_lens release
    sae_id: str       # sae_lens id within the release
    hook: str         # TransformerLens hook name


@dataclass(frozen=True)
class Suite:
    name: str
    model_name: str
    parallel_attn: bool
    sites: tuple[Site, ...]


def _gpt2_res_sites() -> tuple[Site, ...]:
    rel = "gpt2-small-res-jb"
    sites = [
        Site(ordinal=i, block=i, kind="resid", release=rel,
             sae_id=f"blocks.{i}.hook_resid_pre", hook=f"blocks.{i}.hook_resid_pre")
        for i in range(12)
    ]
    sites.append(Site(ordinal=12, block=11, kind="resid", release=rel,
                      sae_id="blocks.11.hook_resid_post",
                      hook="blocks.11.hook_resid_post"))
    return tuple(sites)


def _pythia_marks_sites() -> tuple[Site, ...]:
    res, att, mlp = (
        "pythia-70m-deduped-res-sm",
        "pythia-70m-deduped-att-sm",
        "pythia-70m-deduped-mlp-sm",
    )
    sites = [Site(ordinal=0, block=-1, kind="embed", release=res,
                  sae_id="blocks.0.hook_resid_pre", hook="blocks.0.hook_resid_pre")]
    for b in range(6):
        sites += [
            Site(ordinal=3 * b + 1, block=b, kind="attn", release=att,
                 sae_id=f"blocks.{b}.hook_attn_out", hook=f"blocks.{b}.hook_attn_out"),
            Site(ordinal=3 * b + 2, block=b, kind="mlp", release=mlp,
                 sae_id=f"blocks.{b}.hook_mlp_out", hook=f"blocks.{b}.hook_mlp_out"),
            Site(ordinal=3 * b + 3, block=b, kind="resid", release=res,
                 sae_id=f"blocks.{b}.hook_resid_post", hook=f"blocks.{b}.hook_resid_post"),
        ]
    return tuple(sites)


SUITES: dict[str, Suite] = {
    "gpt2-res": Suite("gpt2-res", "gpt2-small", False, _gpt2_res_sites()),
    "pythia-marks": Suite("pythia-marks", "pythia-70m-deduped", True,
                          _pythia_marks_sites()),
}


def build_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"Unknown suite {name!r}; known: {sorted(SUITES)}")
    return SUITES[name]


def load_model_and_saes(suite: Suite, device: str, dtype: str = "float32"):
    """Load the model and every SAE in the suite. Returns (model, {sae_id: SAE})."""
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch_dtype = _DTYPES[dtype]
    logger.info("Loading model %s on %s", suite.model_name, device)
    model = HookedTransformer.from_pretrained(
        suite.model_name, device=device, dtype=torch_dtype
    )

    saes = {}
    for site in suite.sites:
        logger.info("Loading SAE %s/%s", site.release, site.sae_id)
        try:
            saes[site.sae_id] = SAE.from_pretrained(
                release=site.release, sae_id=site.sae_id, device=device
            )[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAE {site.release}/{site.sae_id} for suite "
                f"{suite.name!r}. Check the sae_lens catalog entry."
            ) from e
    return model, saes
