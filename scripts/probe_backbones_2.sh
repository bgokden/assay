#!/usr/bin/env bash
# Second probe round: the hybrid Qwen3.5/3.8 series with the Assay readout (one question per
# sequence). Waits for the first probe unit to release the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
SUITE=data/suites/kev-transfer-v4-dev.jsonl
HOLD=data/v2/holdout.jsonl
until ! systemctl --user is-active --quiet assay-probe; do sleep 30; done

probe() {  # name, hf id, extra evaluate args...
  local name="$1"; local id="$2"; shift 2
  mkdir -p "runs/base-$name"
  echo "=== $name transfer $(date +%T)"
  uv run python -m assay.evaluate --model "base:$id" --data "$SUITE" --out "runs/base-$name/eval-transfer-v4.json" "$@" > "runs/base-$name/eval-transfer-v4.txt" 2>&1
  grep '^overall' "runs/base-$name/eval-transfer-v4.txt" || tail -3 "runs/base-$name/eval-transfer-v4.txt"
  echo "=== $name holdout $(date +%T)"
  uv run python -m assay.evaluate --model "base:$id" --data "$HOLD" --out "runs/base-$name/eval-holdout.json" "$@" > "runs/base-$name/eval-holdout.txt" 2>&1
  grep '^overall' "runs/base-$name/eval-holdout.txt" || tail -3 "runs/base-$name/eval-holdout.txt"
}

probe 3.5-4b Qwen/Qwen3.5-4B-Base
probe 3.5-9b Qwen/Qwen3.5-9B-Base
probe 3.8-27b Qwen/Qwen3.8-27B --quant 4bit --batch-size 8
uv run hf cache rm model/Qwen/Qwen3.8-27B --yes
echo "=== done $(date +%T)"
