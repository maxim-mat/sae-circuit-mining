"""Walk the phase-2 grid, one wandb run per cell.

Structure, and why it is not a full cross product:

  Phase 1  node attribution, once per n. This is the only expensive stage
           (~0.9 s/example) and every cell at that n reuses its hive through the
           manifest cache. Doing it up front means a crash later never re-runs
           attribution.

  Phase 2  cells. Three blocks, because the full cross product of everything we
           want to vary is ~2000 cells and most of it is uninformative:

    curves    n x aggregation x {threshold(5 thresholds), calibrated(defaults),
              screen_magnitude(matched k), random(matched k)}
              -- the faithfulness-vs-nodes curves, with the random floor and the
              screen/test decomposition. This is the block that answers phase 2.
    ha_grid   the calibrated arm's own knobs, at one n only. The comparison so
              far pits a swept threshold against an untuned calibrated arm,
              which is not a fair fight; this is the fix.
    errors    handle_errors in {remove, resid_only} for the headline arms only
              -- Marks' features_wo_errs and features_wo_some_errs (Fig. 3).

Each cell becomes its own wandb run (`logging.run_name` encodes the cell), so
the sweep reads as a tidy table in the UI. Cells whose evaluation is already
cached are skipped without creating a duplicate run, so the sweep is resumable.

    python -m scripts.sweep --dry-run                 # print the plan
    python -m scripts.sweep --blocks curves
    python -m scripts.sweep --n 100 1000 --blocks curves ha_grid errors
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sweep")

BASE = Path("configs/pythia_marks_rc.yaml")

# Marks sweep thresholds over logspace(-4, 0, 15); five points spanning the
# region that produces circuits of a comparable size to theirs.
THRESHOLDS = [0.02, 0.04, 0.07, 0.12, 0.20]
# Node budgets for the baselines, chosen to bracket what the other arms produce.
K_VALUES = [50, 84, 125, 187, 300]
AGGREGATIONS = ["none", "sum"]
# Marks evaluate on 40 test examples for 100 train.
TEST_FRACTION = 0.4

HA_GRID = {
    "selection.screen_q": [0.5, 1.0, 2.0],
    "selection.fdr_q": [0.01, 0.05, 0.20],
    "selection.min_recurrence": [0.05, 0.10, 0.25],
    "selection.min_median_abs": ["null", 0.01, 0.05],
    "selection.screen_n0": [10.0, 50.0],
    "selection.screen_pool_col": ["site_ordinal", "kind"],
}


def cells(block: str, n: int, ha_grid_n: int) -> list[tuple[str, dict]]:
    """(run_name, overrides) for every cell of one block at one n."""
    out: list[tuple[str, dict]] = []

    if block == "curves":
        for agg in AGGREGATIONS:
            for t in THRESHOLDS:
                out.append((f"thr{t}-{agg}-n{n}", {
                    "selection.method": "threshold",
                    "selection.node_threshold": t,
                    "selection.aggregation": agg}))
            out.append((f"cal-default-{agg}-n{n}", {
                "selection.method": "calibrated",
                "selection.aggregation": agg}))
            for k in K_VALUES:
                out.append((f"rand{k}-{agg}-n{n}", {
                    "selection.method": "random", "selection.k": k,
                    "selection.aggregation": agg}))
                out.append((f"screenmag{k}-{agg}-n{n}", {
                    "selection.method": "screen_magnitude", "selection.k": k,
                    "selection.aggregation": agg}))

    elif block == "ha_grid":
        if n != ha_grid_n:
            return out                      # tune at one n only
        keys = list(HA_GRID)
        for combo in itertools.product(*(HA_GRID[k] for k in keys)):
            over = {"selection.method": "calibrated",
                    "selection.aggregation": "none"}
            over.update(dict(zip(keys, combo)))
            tag = "-".join(str(v).replace(".", "") for v in combo)
            out.append((f"ha-{tag}-n{n}", over))

    elif block == "errors":
        for he in ["remove", "resid_only"]:
            out.append((f"thr0.1-{he}-n{n}", {
                "selection.method": "threshold",
                "selection.node_threshold": 0.1,
                "evaluation.handle_errors": he}))
            out.append((f"cal-{he}-n{n}", {
                "selection.method": "calibrated",
                "evaluation.handle_errors": he}))

    else:
        raise ValueError(f"Unknown block {block!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="*", default=[100, 1000, 10000])
    ap.add_argument("--blocks", nargs="*", default=["curves", "errors"],
                    choices=["curves", "ha_grid", "errors"])
    ap.add_argument("--ha-grid-n", type=int, default=1000,
                    help="Which n the calibrated grid is tuned at.")
    ap.add_argument("--output-root", default="outputs_sweep")
    # Minimality costs one forward per circuit node. At n=10k (eval 4000) a
    # 300-node cell is ~15 min on its own, so it is opt-in and capped by n.
    ap.add_argument("--minimality-for", nargs="*", default=[],
                    help="Blocks that also pay for per-node leave-one-out.")
    ap.add_argument("--minimality-max-n", type=int, default=1000,
                    help="Never run minimality above this n, whatever --minimality-for says.")
    # run_with_ablations batches the WHOLE eval set into one forward, so each SAE
    # hook holds several [eval_n, seq, d_sae] tensors at once -- 3.1GB apiece at
    # eval_n=4000, times four live tensors, against 24GB of VRAM. Cap it: 1000
    # eval examples is already 25x what Marks use.
    ap.add_argument("--max-eval-n", type=int, default=1000,
                    help="Upper bound on evaluation.num_examples regardless of n.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N cells (smoke-testing the sweep).")
    ap.add_argument("--wandb", action="store_true", help="Log each cell to wandb.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results", type=Path, default=Path("docs/sweep-results.csv"))
    args = ap.parse_args()

    from src.utils.initialization import load_config
    from src.utils.manifest import find_cached, stage_params

    plan = [(n, block, name, over)
            for n in args.n
            for block in args.blocks
            for name, over in cells(block, n, args.ha_grid_n)]
    if args.limit:
        plan = plan[:args.limit]
    logger.info("Plan: %d cells over n=%s, blocks=%s", len(plan), args.n, args.blocks)
    if args.dry_run:
        for n, block, name, over in plan:
            print(f"  [{block:8s}] {name:44s} {over}")
        # Attribution dominates: ~0.9 s/example, everything else reuses the hive.
        print(f"\n{len(plan)} cells; node attribution "
              f"~{sum(args.n) * 0.9 / 60:.0f} min total for n={args.n}")
        return

    # ---- Phase 1: the expensive shared stage, once per n -------------------
    for n in args.n:
        cfg = load_config(BASE, [
            f"data.num_examples={n}",
            "edges.enabled=false",
            f"run.output_root={args.output_root}",
            "logging.backend=[]",
        ])
        from src.attribution.nodes import run_node_attribution
        from src.utils.manifest import resolve_stage_output
        _, was_cached = resolve_stage_output(
            cfg, "node_attribution", lambda d: run_node_attribution(cfg, d))
        logger.info("n=%d node attribution %s", n,
                    "cached" if was_cached else "computed")

    # ---- Phase 2: one wandb run per cell ----------------------------------
    from src.pipeline import run_pipeline

    done = skipped = failed = 0
    for n, block, name, over in plan:
        overrides = [
            f"data.num_examples={n}",
            f"evaluation.num_examples="
            f"{min(args.max_eval_n, max(1, int(n * TEST_FRACTION)))}",
            "edges.enabled=false",
            f"run.output_root={args.output_root}",
            f"evaluation.minimality="
            f"{str(block in args.minimality_for and n <= args.minimality_max_n).lower()}",
            f"logging.backend=[{'wandb' if args.wandb else ''}]",
            "logging.experiment=phase2-sweep",
            f"logging.run_name={name}",
            f"logging.tags=[{block},n{n}]",
        ] + [f"{k}={v}" for k, v in over.items()]

        cfg = load_config(BASE, overrides)
        if find_cached(cfg.run.output_root, "evaluation",
                       stage_params(cfg, "evaluation")) is not None:
            skipped += 1
            continue

        logger.info("[%d/%d] %s", done + skipped + failed + 1, len(plan), name)
        try:
            run_pipeline(cfg)
            done += 1
        except Exception as e:                      # one bad cell must not end the sweep
            logger.error("cell %s failed: %s: %s", name, type(e).__name__, e)
            failed += 1

    logger.info("Cells: %d run, %d already cached, %d failed", done, skipped, failed)
    collect(args.output_root, args.results)


def collect(output_root: str, dest: Path) -> None:
    """Every evaluation manifest under output_root as one tidy table."""
    import pandas as pd

    rows = []
    for f in Path(output_root).glob("evaluation/*/manifest.json"):
        m = json.loads(f.read_text())
        p, s = m["params"], m["params"]["selection"]
        rows.append({
            "n": p["data"]["num_examples"], "eval_n": m.get("n_examples"),
            "aggregation": s["aggregation"], "selector": s["method"],
            "node_threshold": s["node_threshold"], "k": s.get("k"),
            "screen_q": s.get("screen_q"), "fdr_q": s.get("fdr_q"),
            "min_recurrence": s.get("min_recurrence"),
            "min_median_abs": s.get("min_median_abs"),
            "screen_n0": s.get("screen_n0"), "pool": s.get("screen_pool_col"),
            "handle_errors": p["evaluation"]["handle_errors"],
            "nodes": m.get("n_nodes"),
            **{k: m[k] for k in m if k.startswith(("faithfulness", "completeness",
                                                   "minimality"))},
            "git_commit": m.get("git_commit", "")[:8],
        })
    if not rows:
        logger.warning("No evaluation manifests under %s", output_root)
        return
    df = pd.DataFrame(rows).sort_values(["n", "aggregation", "selector", "nodes"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Wrote %d rows to %s", len(df), dest)


if __name__ == "__main__":
    main()
