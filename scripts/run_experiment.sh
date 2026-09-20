#!/usr/bin/env bash
# Train, fit temperature, and evaluate on dev (seen tasks), holdout (unseen tasks) and the
# public kev transfer-v4 suite. Results land in <out>/eval-*.json, full reports in <out>/eval-*.txt.
#
#   scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/assay-1.7b data/v1 [extra train args]
set -euo pipefail

BASE="$1"; OUT="$2"; DATA="$3"; shift 3
cd "$(dirname "$0")/.."

evaluate() {  # name, data file, extra args...
  local name="$1"; local data="$2"; shift 2
  uv run python -m assay.evaluate --model "$OUT" --data "$data" --out "$OUT/eval-$name.json" "$@" \
    > "$OUT/eval-$name.txt" 2>&1
  echo "$name: $(grep '^overall' "$OUT/eval-$name.txt")"
}

uv run python -m assay.train --base "$BASE" --data "$DATA" --out "$OUT" "$@"
uv run python -m assay.calibrate --model "$OUT" --data "$DATA/calibration.jsonl"

evaluate dev "$DATA/dev.jsonl" --temperature 1.0
evaluate dev-scaled "$DATA/dev.jsonl"
evaluate holdout "$DATA/holdout.jsonl" --temperature 1.0
evaluate holdout-scaled "$DATA/holdout.jsonl"
evaluate transfer-v4 data/suites/kev-transfer-v4-dev.jsonl --temperature 1.0
evaluate transfer-v4-scaled data/suites/kev-transfer-v4-dev.jsonl
