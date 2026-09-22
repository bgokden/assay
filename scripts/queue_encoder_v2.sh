#!/usr/bin/env bash
# Encoder tier, next round on gte-modernbert-base (the 0.6B backbone swap was dropped):
# the joint multi-option reader, and the same reader on the distilled corpus when it exists.
#
#   systemd-run --user --unit encoder2 -p WorkingDirectory=$PWD scripts/queue_encoder_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

ENCODER=Alibaba-NLP/gte-modernbert-base
GENERIC=data/distill/generic.jsonl
CORPUS=data/distill/corpus_labels.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

log "encoder v2 queue starting"
stage joint-gte-base $COMPILED --arch joint --encoder $ENCODER --data data/v4 --extra $GENERIC \
  --out runs/joint-gte-base --epochs 2 --lr 5e-5 --slots 8 --reader-layers 3 --batch-size 16 || true

# the distilled corpus is labelled in parallel; use whatever exists when this stage starts
# the data lever is tested on the stronger architecture first: on the standard data the
# joint reader (holdout 0.576, 2 epochs) is behind the simpler reader (0.606, 3 epochs)
if [ -s "$CORPUS" ]; then
  stage compiled-late-distilled $COMPILED --encoder $ENCODER --data data/v4 --extra $GENERIC $CORPUS \
    --out runs/compiled-late-distilled --epochs 1 --lr 5e-5 --slots 8 --late-interaction || true
  stage joint-gte-base-distilled $COMPILED --arch joint --encoder $ENCODER --data data/v4 --extra $GENERIC $CORPUS \
    --out runs/joint-gte-base-distilled --epochs 1 --lr 5e-5 --slots 8 --reader-layers 3 --batch-size 16 || true
fi
log "encoder v2 queue finished"
