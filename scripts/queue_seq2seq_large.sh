#!/usr/bin/env bash
# Scale the encoder-decoder tier: t5gemma-l-l-ul2-it is 24 layers each side at hidden 1024
# against the base model's 12 at 768, in the arrangement that trains best.
#
# NOT RUN (2026-09-23): the smaller T5Gemma-1 b-b-ul2-it reached 0.521 on unseen tasks against
# 0.615 for the newer T5Gemma-2 270m-270m, so this lineage is the wrong one to scale. Kept for
# the record; see docs/roadmap.md step 8.
#
#   systemd-run --user --unit seq2seqlarge -p WorkingDirectory=$PWD scripts/queue_seq2seq_large.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

M=google/t5gemma-l-l-ul2-it
T="uv run python -X faulthandler -m assay.train_compiled --arch seq2seq --encoder $M --data data/v4"
COMMON="--lora 16 --lr 1e-4 --batch-size 8 --pair-budget 32 --max-state-tokens 512"

while systemctl --user is-active --quiet seq2seqit; do sleep 60; done
log "large seq2seq queue starting"
stage seq2seqlarge-enc $T $COMMON --encoder-question --epochs 2 --extra data/distill/generic.jsonl --out runs/seq2seqlarge-enc || true
log "large seq2seq queue finished"
