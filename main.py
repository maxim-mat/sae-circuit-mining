"""Entry point for the SAE circuit-discovery pipeline.

Usage:
    python -m src.main -c configs/pythia_marks_rc.yaml
    python -m src.main -c configs/pythia_marks_rc.yaml selection.node_threshold=0.2
    python -m src.main -c configs/pythia_marks_rc.yaml --stage node_attribution
"""
import logging
import sys

from omegaconf import OmegaConf

from src.utils.initialization import load_config, parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sae_circuit_mining")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    from src.pipeline import run_pipeline   # imported late: heavy deps
    run_pipeline(cfg, only_stage=args.stage)


if __name__ == "__main__":
    main()
