"""Position-resolved edge attribution for every finished node run, one store per
filtering method: outputs/<dataset>/attributions/edges/<counterfactual>/<method>/.

Each node run is read back with the exact config in its manifest, so edges use
the same examples, metric and integrated-gradients steps. A method only decides
which node instances get edges computed INTO them (src/selection/destinations.py);
all upstream nodes are kept and nothing is aggregated or thresholded. Finished
methods are skipped, so re-running resumes; a failing run does not stop the others.

    python -m scripts.run_edge_attributions                       # default methods
    python -m scripts.run_edge_attributions --only ioi --methods case marks+case
    python -m scripts.run_edge_attributions --methods marks --force   # recompute

Methods are defined in METHODS below; change a definition there, not on the
command line, so its parameters land in the store's manifest.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.attribution.positional_edges import run_positional_edges

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_edge_attributions")

T_N = 0.1               # Marks' node threshold (their SVA circuits used 0.1-0.2)

METHODS = {
    "marks": {"rule": "marks", "threshold": T_N, "aggregation": "sum"},
    "case": {"rule": "case", "threshold": T_N},
    "marks+case": {"rule": "marks+case", "threshold": T_N, "aggregation": "sum"},
    "recurrence": {"rule": "recurrence", "threshold": T_N, "aggregation": "sum", "min_frac": 0.1},
    "mass": {"rule": "mass", "aggregation": "sum", "mass": 0.8},
    "ttest_bh": {"rule": "ttest_bh", "aggregation": "sum", "fdr": 0.05},
}
DEFAULT_METHODS = ["marks", "case", "marks+case"]


def node_runs(output_root, only: str | None = None) -> list[Path]:
    """Finished node-attribution runs under output_root, optionally filtered by a
    substring of their path."""
    runs = sorted(p.parent for p in Path(output_root).glob("*/attributions/nodes/*/manifest.json"))
    return [r for r in runs if only is None or only in r.as_posix()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--only", default=None,
                    help="Run only node runs whose path contains this.")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(METHODS))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite stores that hold a finished run with other parameters.")
    ap.add_argument("--device", default=None, help="Override the node run's suite.device.")
    args = ap.parse_args()

    methods = {name: METHODS[name] for name in args.methods}
    runs = node_runs(args.output_root, args.only)
    logger.info("%d finished node runs, methods %s", len(runs), list(methods))

    failed = []
    for node_dir in runs:
        logger.info("=== %s", node_dir)
        try:
            run_positional_edges(node_dir, methods, force=args.force, device=args.device)
        except Exception:                       # one bad run must not end the rest
            logger.exception("%s failed", node_dir)
            failed.append(node_dir)

    logger.info("Done; %d failed%s", len(failed), f": {failed}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
