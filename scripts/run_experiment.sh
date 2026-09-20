#!/usr/bin/env bash
# Train, fit temperature, and evaluate on dev (seen tasks), holdout (unseen tasks) and the
# public kev transfer-v4 suite. Results land in <out>/eval-*.json.
#
#   scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/sezgi-1.7b data/v1 [extra train args]
set -euo pipefail

BASE="$1"; OUT="$2"; DATA="$3"; shift 3
cd "$(dirname "$0")/.."

uv run python -m sezgi.train --base "$BASE" --data "$DATA" --out "$OUT" "$@"
uv run python -m sezgi.calibrate --model "$OUT" --data "$DATA/calibration.jsonl"

for suite in dev holdout; do
  uv run python -m sezgi.evaluate --model "$OUT" --data "$DATA/$suite.jsonl" \
    --out "$OUT/eval-$suite.json" --temperature 1.0 | head -1 | sed "s/^/$suite raw: /"
  uv run python -m sezgi.evaluate --model "$OUT" --data "$DATA/$suite.jsonl" \
    --out "$OUT/eval-$suite-scaled.json" | head -1 | sed "s/^/$suite scaled: /"
done
uv run python -m sezgi.evaluate --model "$OUT" --data data/suites/kev-transfer-v4-dev.jsonl \
  --out "$OUT/eval-transfer-v4.json" --temperature 1.0 | head -1 | sed "s/^/transfer raw: /"
uv run python -m sezgi.evaluate --model "$OUT" --data data/suites/kev-transfer-v4-dev.jsonl \
  --out "$OUT/eval-transfer-v4-scaled.json" | head -1 | sed "s/^/transfer scaled: /"
