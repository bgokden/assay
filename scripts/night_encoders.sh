#!/usr/bin/env bash
# Overnight queue: compiled-function and cross-encoder tiers, then second seeds for the
# decoders. Each stage retries on failure (sporadic native crashes on this GPU); decoder
# stages resume from their checkpoints.
#
#   systemd-run --user --unit night -p WorkingDirectory=$PWD scripts/night_encoders.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/night
LOG=runs/night/queue.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

stage() {  # name, cmd...
  local name="$1"; shift
  mkdir -p "runs/$name"
  if [ -f "runs/$name/eval-transfer-v4-scaled.json" ] || [ -f "runs/$name/eval-transfer-v4-zeroshot.json" ]; then log "skip $name (done)"; return 0; fi
  for attempt in 1 2 3; do
    log "start $name attempt $attempt: $*"
    if "$@" >> "runs/$name/run.log" 2>&1; then log "done $name"; return 0; fi
    log "failed $name (exit $?)"
    sleep 30
  done
  return 1
}

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

stage compiled-gte-base-zeroshot $COMPILED --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --out runs/compiled-gte-base-zeroshot --zero-shot-only || true
stage compiled-gte-base $COMPILED --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/compiled-gte-base || true
stage cross-gte-base $COMPILED --arch cross --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/cross-gte-base || true
stage compiled-modernbert-large $COMPILED --encoder answerdotai/ModernBERT-large --data data/v4 --extra $GENERIC --out runs/compiled-modernbert-large --batch-size 16 --pair-budget 128 || true

stage assay-4b-v4-s1 scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b-v4-s1 data/v4 --seed 1 || true
stage assay-1.7b-v2-s1 scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/assay-1.7b-v2-s1 data/v2 --seed 1 || true
stage assay-1.7b-v2-hard-s1 scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/assay-1.7b-v2-hard-s1 data/v2 --seed 1 --hard-targets || true
log "queue finished"
