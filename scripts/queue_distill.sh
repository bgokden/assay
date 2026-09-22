#!/usr/bin/env bash
# Label the distillation corpus with the 4B teacher once the backbone queue has released the
# GPU: 400k states x 3 rubric questions, packed per state, resumable.
#
#   systemd-run --user --unit distill -p WorkingDirectory=$PWD scripts/queue_distill.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

while systemctl --user is-active --quiet backbone; do sleep 60; done
log "distillation labelling starting"
for attempt in 1 2 3 4 5; do
  if uv run python -X faulthandler -m assay.data.distill --teacher runs/assay-4b-v4 --train data/v4/train.jsonl --out data/distill \
      --corpus data/distill/corpus.jsonl --corpus-limit 400000 --questions-per-text 3 --batch-size 16 --max-state-tokens 1024 \
      >> runs/night/distill.log 2>&1; then log "done labelling"; break; fi
  log "labelling failed (attempt $attempt), resuming"; sleep 30
done
log "distillation queue finished"
