#!/usr/bin/env bash
# Both discovery scripts in series, for unattended execution:
#
#     nohup bash scripts/run_discovery.sh > discovery.out 2>&1 &
#     tail -f logs/discovery-*/summary.log
#
# Configure each run by editing the variables at the top of
# scripts/simple_discovery.py and scripts/partial_order_discovery.py. Each
# writes outputs/nets/<EXPERIMENT_NAME>/ and refuses to overwrite an existing
# run, so change EXPERIMENT_NAME before re-running.
#
# The two are independent: a failure in one does NOT skip the other. The exit
# status is non-zero if either failed.
set -uo pipefail

cd "$(dirname "$0")/.."              # repo root, for `python -m scripts.*`
PY="${PY:-python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="logs/discovery-${STAMP}"
mkdir -p "$LOGDIR"
FAILED=0

run_step () {
  local name="$1"
  local log="${LOGDIR}/${name}.log"
  echo "=== [$(date +%H:%M:%S)] START ${name}" | tee -a "${LOGDIR}/summary.log"
  local t0=$SECONDS
  if "$PY" -m "scripts.${name}" >"$log" 2>&1; then
    local status="OK"
  else
    local status="FAILED (see ${log})"
    FAILED=1
  fi
  local mins=$(( (SECONDS - t0) / 60 ))
  echo "=== [$(date +%H:%M:%S)] ${status} ${name} after ${mins} min" \
    | tee -a "${LOGDIR}/summary.log"
}

echo "Discovery starting with $("$PY" --version 2>&1); logs in ${LOGDIR}" \
  | tee "${LOGDIR}/summary.log"
run_step simple_discovery
run_step partial_order_discovery
echo "=== [$(date +%H:%M:%S)] DONE" | tee -a "${LOGDIR}/summary.log"
exit $FAILED
