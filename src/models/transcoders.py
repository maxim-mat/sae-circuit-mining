"""Transcoders loaded straight from safetensors, for checkpoints sae_lens lacks.

A transcoder (Dunefsky et al., "Transcoders Find Interpretable LLM Feature
Circuits", arXiv:2406.11944) reads one hook and predicts another. encode() and
decode() match the sae_lens surface the attribution code uses, so a Site's
input_hook is the only thing that distinguishes one from an SAE.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_REQUIRED = ("W_enc", "W_dec", "b_enc", "b_dec")


class Transcoder:
    """relu(x @ W_enc + b_enc) -> f @ W_dec + b_dec, JumpReLU given a `threshold`."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        missing = [k for k in _REQUIRED if k not in tensors]
        if missing:
            raise ValueError(f"Transcoder checkpoint lacks {missing}; has {sorted(tensors)}")
        if "W_skip" in tensors:
            logger.warning("Ignoring W_skip: the skip term's effect lands in the error node.")
        self.b_enc, self.b_dec = tensors["b_enc"], tensors["b_dec"]
        d_sae = len(self.b_enc)
        # Conventions differ on orientation; the feature count settles it.
        W_enc, W_dec = tensors["W_enc"], tensors["W_dec"]
        self.W_enc = W_enc if W_enc.shape[1] == d_sae else W_enc.T
        self.W_dec = W_dec if W_dec.shape[0] == d_sae else W_dec.T
        self.threshold = tensors.get("threshold")
        self.cfg = SimpleNamespace(d_sae=d_sae, metadata=None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        f = torch.relu(pre)
        return f if self.threshold is None else f * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


def load_transcoder(source: str, filename: str, device: str, dtype: torch.dtype) -> Transcoder:
    """Load `filename` from a local directory or, failing that, a Hugging Face repo."""
    from safetensors.torch import load_file

    path = Path(source) / filename
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=source, filename=filename)
    return Transcoder({k: v.to(dtype=dtype) for k, v in load_file(path, device=device).items()})
