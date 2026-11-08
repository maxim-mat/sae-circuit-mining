"""IG node attribution on Pythia for mib-bench/ioi and Marks' SVA datasets.

Writes outputs/<dataset>/attributions/nodes/<counterfactual>/. Finished runs are
skipped, so re-running resumes; a failing run does not stop the others. Uses every
prompt unless data.num_examples is set, which takes a sample fixed by `seed`.

    python -m scripts.fetch_data                            # Marks stimuli, once
    python -m scripts.run_node_attributions
    python -m scripts.run_node_attributions --only rc data.num_examples=1000

Other MIB tasks are not listed because mib_logit_diff only handles IOI's
two-way string choices.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_node_attributions")

BASE = Path("configs/pythia_marks_rc.yaml")

IOI_COUNTERFACTUALS = [
    "abc_counterfactual",
    "random_names_counterfactual",
    "s1_io_flip_counterfactual",
    "s2_io_flip_counterfactual",
    "random_names_s1_ioi_flip_counterfactual",
    "random_names_s2_ioi_flip_counterfactual",
    "s1_ioi_flip_s2_ioi_flip_counterfactual",
    "random_names_s1_ioi_flip_s2_ioi_flip_counterfactual",
]
MARKS_STRUCTURES = ["simple", "nounpp", "rc", "within_rc"]


def plan() -> list[list[str]]:
    """Config overrides for each run, on top of BASE."""
    # Every prompt of every length; length filtering is left to later stages.
    common = ["attribution.method=ig", "logging.backend=[]",
              "data.num_examples=null", "data.use_min_length_only=false"]
    ioi = [common + ["data.path=mib-bench/ioi", "data.loader=mib",
                     "data.template_field=template",
                     f"data.counterfactual={cf}", "attribution.metric=mib_logit_diff"]
           for cf in IOI_COUNTERFACTUALS]
    marks = [common + [f"data.path=data/marks/{s}_train.json", "data.loader=sva",
                       "data.template_field=case", "attribution.metric=sva_logit_diff"]
             for s in MARKS_STRUCTURES]
    return ioi + marks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None,
                    help="Run only datasets/counterfactuals whose output path contains this.")
    ap.add_argument("overrides", nargs="*",
                    help="Dotlist config overrides, e.g. data.num_examples=1000")
    args = ap.parse_args()

    from src.attribution.nodes import run_node_attribution
    from src.utils.initialization import load_config
    from src.utils.manifest import resolve_stage_output, stage_dir, stage_params

    failed = []
    for overrides in plan():
        cfg = load_config(BASE, overrides + args.overrides)
        out = stage_dir(cfg.run.output_root, "node_attribution",
                        stage_params(cfg, "node_attribution"))
        if args.only and args.only not in out.as_posix():
            continue
        logger.info("=== %s", out)
        try:
            _, cached = resolve_stage_output(
                cfg, "node_attribution",
                lambda d, c=cfg: run_node_attribution(c, d))
            logger.info("%s: %s", out, "cached" if cached else "computed")
        except Exception as e:                      # one bad run must not end the rest
            logger.error("%s failed: %s: %s", out, type(e).__name__, e)
            failed.append(out)

    logger.info("Done; %d failed%s", len(failed), f": {failed}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
