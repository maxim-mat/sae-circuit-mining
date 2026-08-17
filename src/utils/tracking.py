"""wandb experiment tracking.

Deliberately not Lightning: this pipeline has no training loop, optimizer, or
DataLoader, so a LightningModule/Trainer would wrap a plain iteration and
WandbLogger without a Trainer adds nothing over wandb.log.
"""
from __future__ import annotations

import logging

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
_run = None


def init_tracking(cfg):
    """Start a wandb run if configured. No-op when logging.backend is empty."""
    global _run
    if "wandb" not in cfg.logging.backend:
        return None
    import wandb
    _run = wandb.init(
        project=cfg.logging.project, entity=cfg.logging.entity,
        name=cfg.logging.run_name, group=cfg.logging.group,
        mode="offline" if cfg.logging.offline else "online",
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
    )
    return _run


def log_stage(stage: str, metrics: dict) -> None:
    logger.info("[%s] %s", stage, metrics)
    if _run is not None:
        _run.log({f"{stage}/{k}": v for k, v in metrics.items()})


def finish_tracking() -> None:
    global _run
    if _run is not None:
        _run.finish()
        _run = None
