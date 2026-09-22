#!/usr/bin/env bash
# Encoder-tier backbone experiments: the gate (a 0.6B decoder's own readout, untrained and
# LoRA-trained) and the compiled tier on the same decoder's features at two depths.
#
#   systemd-run --user --unit backbone -p WorkingDirectory=$PWD scripts/queue_backbone.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"
FEATURES="--encoder Qwen/Qwen3-0.6B-Base --pooling last --lora 16 --lr 1e-4 --slots 8 --late-interaction --epochs 2 --batch-size 16 --pair-budget 128"

gate_eval() {  # name, model, data dir
  local name="$1" model="$2" data="$3"
  mkdir -p "runs/$name"
  if [ -f "runs/$name/eval-transfer-v4.json" ]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  uv run python -m assay.evaluate --model "$model" --data "$data/holdout.jsonl" --out "runs/$name/eval-holdout.json" --batch-size 16 > "runs/$name/eval-holdout.txt" 2>&1
  uv run python -m assay.evaluate --model "$model" --data data/suites/kev-transfer-v4-dev.jsonl --out "runs/$name/eval-transfer-v4.json" --batch-size 16 > "runs/$name/eval-transfer-v4.txt" 2>&1
  log "done $name: holdout $(grep -o 'acc=[0-9.]*' runs/$name/eval-holdout.txt | head -1) transfer $(grep -o 'acc=[0-9.]*' runs/$name/eval-transfer-v4.txt | head -1)"
}

gate_eval base-0.6b base:Qwen/Qwen3-0.6B-Base data/v4
stage assay-0.6b-v4 scripts/run_experiment.sh Qwen/Qwen3-0.6B-Base runs/assay-0.6b-v4 data/v4 || true
stage compiled-qwen0.6b-l16 $COMPILED $FEATURES --encoder-layers 16 --data data/v4 --extra $GENERIC --out runs/compiled-qwen0.6b-l16 || true
stage compiled-qwen0.6b-l28 $COMPILED $FEATURES --data data/v4 --extra $GENERIC --out runs/compiled-qwen0.6b-l28 || true
log "backbone queue finished"
