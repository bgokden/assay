#!/usr/bin/env bash
# Third overnight queue: per-item predictions and conformal thresholds for the models that
# were trained before predictions were saved. Starts once the second queue (unit "night2")
# has finished.
#
#   systemd-run --user --unit night3 -p WorkingDirectory=$PWD -p CPUAffinity=0-5,7-23 scripts/night_conformal.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/night
LOG=runs/night/queue.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

retry() {  # logfile, cmd...
  local logfile="$1"; shift
  for attempt in 1 2 3; do
    if "$@" >> "$logfile" 2>&1; then return 0; fi
    log "retry ($attempt): $*"; sleep 10
  done
  return 1
}

predictions() {  # run, data, batch, maxstate
  local run="$1" data="$2" batch="$3" maxstate="$4"
  if [ -f "runs/$run/conformal.json" ]; then log "skip $run (done)"; return 0; fi
  log "start predictions $run"
  retry "runs/$run/conformal.log" uv run python -X faulthandler -m assay.calibrate --model "runs/$run" --data "$data/calibration.jsonl" --batch-size "$batch" --max-state-tokens "$maxstate" || return 1
  retry "runs/$run/conformal.log" uv run python -X faulthandler -m assay.evaluate --model "runs/$run" --data "$data/holdout.jsonl" --out "runs/$run/eval-holdout.json" --batch-size "$batch" --max-state-tokens "$maxstate" --temperature 1.0 || return 1
  retry "runs/$run/conformal.log" uv run python -X faulthandler -m assay.evaluate --model "runs/$run" --data data/suites/kev-transfer-v4-dev.jsonl --out "runs/$run/eval-transfer-v4.json" --batch-size "$batch" --max-state-tokens "$maxstate" --temperature 1.0 || return 1
  uv run python -m assay.conformal --model "runs/$run" > "runs/$run/conformal.txt" 2>&1 && log "done $run"
}

while systemctl --user is-active --quiet night2; do sleep 60; done
log "third queue starting"
predictions assay-1.7b-v2 data/v2 16 2048 || true
predictions assay-4b-v4 data/v4 16 2048 || true
predictions assay-27b data/v2 4 1024 || true
log "third queue finished"
