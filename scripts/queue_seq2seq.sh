#!/usr/bin/env bash
# Encoder-decoder tier on T5Gemma 2 (Gemma licence; experimental). Four runs separate the two
# questions: does an encoder-decoder readout beat the encoder tier's learned head (0.606 on
# unseen tasks, cross-encoder ceiling 0.619), and what does encode-once cost?
#
#   --encoder-question: state and question both in the encoder (how the model was pretrained)
#   default:            state in the encoder, question in the decoder (one state, many questions)
#
#   systemd-run --user --unit seq2seq -p WorkingDirectory=$PWD scripts/queue_seq2seq.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

M=google/t5gemma-2-270m-270m
T="uv run python -X faulthandler -m assay.train_compiled --arch seq2seq --encoder $M --data data/v4"
COMMON="--lora 16 --lr 1e-4 --batch-size 16 --pair-budget 64 --max-state-tokens 512"

# zero-shot readouts first: cheap, and they bound what training has to beat
stage seq2seq-enc-zeroshot $T $COMMON --encoder-question --epochs 0 --out runs/seq2seq-enc-zeroshot || true
stage seq2seq-dec-zeroshot $T $COMMON --epochs 0 --out runs/seq2seq-dec-zeroshot || true

stage seq2seq-enc $T $COMMON --encoder-question --epochs 2 --extra data/distill/generic.jsonl --out runs/seq2seq-enc || true
stage seq2seq-dec $T $COMMON --epochs 2 --extra data/distill/generic.jsonl --out runs/seq2seq-dec || true
log "seq2seq queue finished"
