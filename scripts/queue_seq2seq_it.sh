#!/usr/bin/env bash
# The instruction-tuned encoder-decoder: T5Gemma 1 (Gemma 2 lineage) has -it variants where
# T5Gemma 2 does not, which is the property that made Flan-T5 worth considering at all.
# Zero-shot in both arrangements, then training in the encoder arrangement only: with
# T5Gemma 2 at 270m, state and question both in the encoder reached holdout 0.615 against
# 0.475 for the question in the decoder, so the encode-once trick is not worth its cost.
#
#   systemd-run --user --unit seq2seqit -p WorkingDirectory=$PWD scripts/queue_seq2seq_it.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

M=google/t5gemma-b-b-ul2-it
T="uv run python -X faulthandler -m assay.train_compiled --arch seq2seq --encoder $M --data data/v4"
COMMON="--lora 16 --lr 1e-4 --batch-size 16 --pair-budget 64 --max-state-tokens 512"

while systemctl --user is-active --quiet seq2seq; do sleep 60; done
log "instruction-tuned seq2seq queue starting"
stage seq2seqit-enc-zeroshot $T $COMMON --encoder-question --epochs 0 --out runs/seq2seqit-enc-zeroshot || true
stage seq2seqit-dec-zeroshot $T $COMMON --epochs 0 --out runs/seq2seqit-dec-zeroshot || true
stage seq2seqit-enc $T $COMMON --encoder-question --epochs 2 --extra data/distill/generic.jsonl --out runs/seq2seqit-enc || true
log "instruction-tuned seq2seq queue finished"
