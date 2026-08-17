#!/usr/bin/env bash
# Phase-2 sweep, all four stages, for unattended execution:
#
#     nohup bash scripts/run_sweeps.sh > sweeps.out 2>&1 &
#     tail -f logs/sweep-*/stage1-curves-n100.log
#
# Each stage writes its own log under logs/sweep-<timestamp>/ and each sweep cell
# becomes its own wandb run. Stages run in sequence but a failing stage does NOT
# abort the rest -- a bad cell in stage 2 should not cost you stages 3 and 4.
#
# Resumable: every stage skips cells whose evaluation is already cached, so
# re-running this script after an interruption picks up where it stopped and
# creates no duplicate wandb runs.
set -uo pipefail

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-outputs_sweep}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="logs/sweep-${STAMP}"
mkdir -p "$LOGDIR"

# Minimality (per-node leave-one-out) costs one forward pass per circuit node.
# Enabled for every block, and --minimality-max-n keeps it off the n=10000 stage
# where a 300-node cell would cost ~15 min on its own.
MIN_BLOCKS="curves errors ha_grid"
MIN_MAX_N=1000

# Evaluation batches its whole example set into one forward, so eval size is
# bounded by VRAM rather than by statistics. 0.4*n would ask for 4000 at
# n=10000, which does not fit 24GB; 1000 is still 25x Marks' 40.
MAX_EVAL_N="${MAX_EVAL_N:-1000}"

run_stage () {
  local name="$1"; shift
  local log="${LOGDIR}/${name}.log"
  echo "=== [$(date +%H:%M:%S)] START ${name}" | tee -a "${LOGDIR}/summary.log"
  local t0=$SECONDS
  if "$PY" -m scripts.sweep "$@" >"$log" 2>&1; then
    local status="OK"
  else
    local status="FAILED (see ${log})"
  fi
  local mins=$(( (SECONDS - t0) / 60 ))
  echo "=== [$(date +%H:%M:%S)] ${status} ${name} after ${mins} min" \
    | tee -a "${LOGDIR}/summary.log"
}

echo "Sweep starting; logs in ${LOGDIR}" | tee "${LOGDIR}/summary.log"
"$PY" -m scripts.sweep --dry-run --n 100 1000 10000 --blocks curves ha_grid errors \
  --ha-grid-n 100 > "${LOGDIR}/plan.txt" 2>&1
echo "Plan written to ${LOGDIR}/plan.txt" | tee -a "${LOGDIR}/summary.log"

# ---- Stage 1: the curves at n=100, plus the error variants. ~45 min ---------
run_stage "stage1-curves-n100" \
  --n 100 --blocks curves errors --wandb \
  --minimality-for $MIN_BLOCKS --minimality-max-n $MIN_MAX_N \n  --max-eval-n $MAX_EVAL_N \
  --output-root "$OUT_ROOT"

# ---- Stage 2: the calibrated arm's own knobs, tuned at n=100. ~4-12 h -------
# 324 cells. Minimality roughly triples this stage; drop it from MIN_BLOCKS if
# the wall time matters more than per-node necessity for the grid.
run_stage "stage2-ha_grid-n100" \
  --n 100 --blocks ha_grid --ha-grid-n 100 --wandb \
  --minimality-for $MIN_BLOCKS --minimality-max-n $MIN_MAX_N \n  --max-eval-n $MAX_EVAL_N \
  --output-root "$OUT_ROOT"

# ---- Stage 3: scale the curves to n=1000. ~2.5 h ---------------------------
run_stage "stage3-curves-n1000" \
  --n 1000 --blocks curves errors --wandb \
  --minimality-for $MIN_BLOCKS --minimality-max-n $MIN_MAX_N \n  --max-eval-n $MAX_EVAL_N \
  --output-root "$OUT_ROOT"

# ---- Stage 4: n=10000, no minimality (see MIN_MAX_N). ~19 h ----------------
run_stage "stage4-curves-n10000" \
  --n 10000 --blocks curves errors --wandb \
  --minimality-for $MIN_BLOCKS --minimality-max-n $MIN_MAX_N \n  --max-eval-n $MAX_EVAL_N \
  --output-root "$OUT_ROOT"

echo "=== [$(date +%H:%M:%S)] ALL STAGES DONE" | tee -a "${LOGDIR}/summary.log"
"$PY" - <<'EOF' | tee -a "${LOGDIR}/summary.log"
from pathlib import Path
import pandas as pd
f = Path("docs/sweep-results.csv")
if f.is_file():
    df = pd.read_csv(f)
    print(f"\n{len(df)} cells in {f}")
    cols = [c for c in ("n", "aggregation", "selector", "nodes",
                        "faithfulness_23_mean", "completeness_23_mean",
                        "minimality_frac_nontrivial") if c in df.columns]
    print(df[cols].to_string(index=False, max_rows=40))
else:
    print(f"\n{f} missing -- no cell completed?")
EOF
