#!/usr/bin/env bash
# Second overnight queue: decoder runs with the content-scored option term, started once the
# first queue (unit "night") has finished.
#
#   systemd-run --user --unit night2 -p WorkingDirectory=$PWD scripts/night_decoder_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/night
LOG=runs/night/queue.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

stage() {  # name, cmd...
  local name="$1"; shift
  mkdir -p "runs/$name"
  if [ -f "runs/$name/eval-transfer-v4-scaled.json" ]; then log "skip $name (done)"; return 0; fi
  for attempt in 1 2 3; do
    log "start $name attempt $attempt: $*"
    if "$@" >> "runs/$name/run.log" 2>&1; then log "done $name"; return 0; fi
    log "failed $name (exit $?)"
    sleep 30
  done
  return 1
}

while systemctl --user is-active --quiet night; do sleep 60; done
log "second queue starting"
stage assay-1.7b-v2-content scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/assay-1.7b-v2-content data/v2 --content-term || true
stage assay-4b-v4-content scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b-v4-content data/v4 --content-term || true
log "second queue finished"
