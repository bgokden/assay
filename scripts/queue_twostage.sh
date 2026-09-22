#!/usr/bin/env bash
# Two-stage distillation for the encoder tier: the model already pretrained on 1.2M
# teacher-labelled rubric items (runs/compiled-late-distilled) is fine-tuned on the task
# mix, which is the distribution the evaluations measure. Mixing the two sets in one pass
# lost 4 points on unseen tasks, most of it on pairwise tasks the rubric bank does not cover.
#
#   systemd-run --user --unit twostage -p WorkingDirectory=$PWD scripts/queue_twostage.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

log "two-stage queue starting"
stage compiled-twostage $COMPILED --init runs/compiled-late-distilled --data data/v4 --extra $GENERIC \
  --out runs/compiled-twostage --epochs 3 --lr 2e-5 --batch-size 32 || true
log "two-stage queue finished"
