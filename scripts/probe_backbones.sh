#!/usr/bin/env bash
# Zero-shot ceiling of untrained backbones with the Assay readout, on the public transfer suite
# (and the holdout split where cheap). Big checkpoints are removed from the HF cache afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."
SUITE=data/suites/kev-transfer-v4-dev.jsonl
HOLD=data/v2/holdout.jsonl

probe() {  # name, hf id, extra evaluate args...
  local name="$1"; local id="$2"; shift 2
  mkdir -p "runs/base-$name"
  echo "=== $name transfer $(date +%T)"
  uv run python -m assay.evaluate --model "base:$id" --data "$SUITE" --out "runs/base-$name/eval-transfer-v4.json" "$@" > "runs/base-$name/eval-transfer-v4.txt" 2>&1
  grep '^overall' "runs/base-$name/eval-transfer-v4.txt" || tail -5 "runs/base-$name/eval-transfer-v4.txt"
}

probe 8b Qwen/Qwen3-8B-Base
echo "=== 8b holdout $(date +%T)"
uv run python -m assay.evaluate --model base:Qwen/Qwen3-8B-Base --data "$HOLD" --out runs/base-8b/eval-holdout.json > runs/base-8b/eval-holdout.txt 2>&1
grep '^overall' runs/base-8b/eval-holdout.txt

probe 14b Qwen/Qwen3-14B-Base --quant 8bit --batch-size 8
echo "=== 14b holdout $(date +%T)"
uv run python -m assay.evaluate --model base:Qwen/Qwen3-14B-Base --quant 8bit --batch-size 8 --data "$HOLD" --out runs/base-14b/eval-holdout.json > runs/base-14b/eval-holdout.txt 2>&1
grep '^overall' runs/base-14b/eval-holdout.txt
uv run hf cache rm model/Qwen/Qwen3-14B-Base --yes

probe 30b-a3b Qwen/Qwen3-30B-A3B-Base --gpu-memory 26GiB --cpu-memory 52GiB --batch-size 8
uv run hf cache rm model/Qwen/Qwen3-30B-A3B-Base --yes
echo "=== done $(date +%T)"
