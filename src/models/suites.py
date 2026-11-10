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
from dataclasses import dataclass, field

import torch

from src.models.transcoders import load_transcoder

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
    hook: str         # TransformerLens hook name; where the dictionary's output lives
    # Transcoders read a different hook than they predict. None = read at `hook`.
    input_hook: str | None = None
    # "sae_lens", or "safetensors": release is a HF repo or local directory and
    # sae_id a file inside it (see src/models/transcoders.py).
    loader: str = "sae_lens"

    @property
    def read_hook(self) -> str:
        return self.input_hook or self.hook


@dataclass(frozen=True)
class Suite:
    name: str
    model_name: str
    parallel_attn: bool
    sites: tuple[Site, ...]
    # Model weight processing, for dictionaries that carry no sae_lens metadata.
    model_kwargs: dict = field(default_factory=dict, hash=False)


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


def _gpt2_transcoder_sites() -> tuple[Site, ...]:
    """Dunefsky et al.'s MLP transcoders (arXiv:2406.11944), in the safetensors
    repackaging of pchlenski/gpt2-transcoders, whose files name the input hook."""
    return tuple(
        Site(ordinal=b, block=b, kind="mlp", release="jacobdunefsky/gpt2small-transcoders",
             sae_id=f"gpt2-small-dun-chl-mlp-tc{b}/sae.safetensors",
             hook=f"blocks.{b}.hook_mlp_out", input_hook=f"blocks.{b}.ln2.hook_normalized",
             loader="safetensors")
        for b in range(12)
    )


SUITES: dict[str, Suite] = {
    "gpt2-res": Suite("gpt2-res", "gpt2-small", False, _gpt2_res_sites()),
    "pythia-marks": Suite("pythia-marks", "pythia-70m-deduped", True,
                          _pythia_marks_sites()),
    # ponytail: center_writing_weights is assumed from TransformerLens' defaults at
    # training time; confirm with a reconstruction check before trusting results.
    "gpt2-transcoders": Suite("gpt2-transcoders", "gpt2-small", False,
                              _gpt2_transcoder_sites(),
                              model_kwargs={"center_writing_weights": True}),
}


def build_suite(name: str) -> Suite:
    if name not in SUITES:
        raise KeyError(f"Unknown suite {name!r}; known: {sorted(SUITES)}")
    return SUITES[name]


def _sae_metadata(sae):
    """sae_lens 6.x keeps provenance on cfg.metadata, not on cfg itself."""
    return getattr(getattr(sae, "cfg", None), "metadata", None)


def prepend_bos_for(saes) -> bool:
    """The BOS convention the SAEs were trained under.

    Read from cfg.metadata rather than cfg: a `getattr(sae.cfg, "prepend_bos",
    True)` silently returns its default under sae_lens 6.x, which had us feeding
    the pythia-marks SAEs a BOS token they declare they do not want
    (`prepend_bos = False`, matching Marks' bare-HF-tokenizer loader).
    """
    md = _sae_metadata(next(iter(saes.values())))
    value = getattr(md, "prepend_bos", None) if md is not None else None
    return True if value is None else bool(value)


def model_load_kwargs(saes) -> dict:
    """Weight-processing kwargs the SAEs were trained against.

    An SAE is trained on one specific version of a model's activations, so the
    model must be reconstituted the same way. TransformerLens' default
    processing is NOT neutral here: `center_writing_weights` removes the
    mean-over-d_model component of everything written to the residual stream,
    which the model cannot see (LayerNorm removes it anyway) but the SAE very
    much can. Loading pythia-70m with full processing costs
    blocks.3.hook_resid_post 15 points of variance explained (0.94 -> 0.79);
    loading gpt2-small WITHOUT it is catastrophic (VE -241 at resid_post_11).

    sae_lens records the right answer per release, so take it from there rather
    than hardcoding a per-suite flag that can drift from the catalog.
    """
    md = _sae_metadata(next(iter(saes.values())))
    kwargs = getattr(md, "model_from_pretrained_kwargs", None) if md is not None else None
    return dict(kwargs or {})


def load_model_and_saes(suite: Suite, device: str, dtype: str = "float32",
                        allow_transcoders: bool = False):
    """Load every SAE in the suite, then the model they were trained against.

    Returns (model, {sae_id: SAE}). SAEs come first because their metadata is
    what determines how the model must be loaded.

    allow_transcoders is opt-in because a caller that encodes at site.hook gets
    plausible, wrong numbers from a dictionary that reads site.read_hook.
    """
    if not allow_transcoders and any(s.read_hook != s.hook for s in suite.sites):
        raise ValueError(f"Suite {suite.name!r} has transcoder sites, which only node "
                         "attribution supports so far.")
    unknown = {s.loader for s in suite.sites} - {"sae_lens", "safetensors"}
    if unknown:
        raise ValueError(f"Unknown site loader(s) {sorted(unknown)} in suite {suite.name!r}")

    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch_dtype = _DTYPES[dtype]

    saes = {}
    for site in suite.sites:
        logger.info("Loading SAE %s/%s", site.release, site.sae_id)
        try:
            if site.loader == "safetensors":
                saes[site.sae_id] = load_transcoder(site.release, site.sae_id, device, torch_dtype)
                continue
            saes[site.sae_id] = SAE.from_pretrained(
                release=site.release, sae_id=site.sae_id, device=device
            )[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAE {site.release}/{site.sae_id} for suite "
                f"{suite.name!r}. Check the sae_lens catalog entry."
            ) from e

    kwargs = model_load_kwargs(saes) or dict(suite.model_kwargs)
    logger.info("Loading model %s on %s (no_processing + %s)",
                suite.model_name, device, kwargs or "{}")
    model = HookedTransformer.from_pretrained_no_processing(
        suite.model_name, device=device, dtype=torch_dtype, **kwargs
    )
    if any(s.read_hook.endswith("hook_mlp_in") for s in suite.sites):
        model.set_use_hook_mlp_in(True)     # off by default in TransformerLens
    return model, saes
