"""Config loading and CLI parsing."""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.config import Config


def load_config(config_path: Path, overrides: list[str]) -> Config:
    """Merge schema defaults <- YAML file <- CLI dotlist overrides.

    Raises FileNotFoundError if the YAML is missing, and omegaconf errors on
    keys absent from the schema (structured configs are strict by default).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(schema, file_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="SAE circuit discovery pipeline")
    parser.add_argument("--config", "-c", required=True, type=Path,
                        help="Path to YAML configuration file")
    parser.add_argument("--stage", default=None,
                        help="Run a single stage instead of the full pipeline")
    parser.add_argument("overrides", nargs="*",
                        help="Dotlist overrides, e.g. selection.node_threshold=0.2")
    return parser.parse_args()
