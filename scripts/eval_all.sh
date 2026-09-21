#!/usr/bin/env bash
# Run the standard evaluations for a saved model, retrying each one on a crash.
#   scripts/eval_all.sh runs/assay-27b data/v2 4
set -uo pipefail
OUT="$1"; DATA="$2"; BATCH="${3:-16}"; MAXSTATE="${4:-2048}"
cd "$(dirname "$0")/.."
run_eval() {  # name, data file, extra args...
  local name="$1"; local data="$2"; shift 2
  for attempt in 1 2 3; do
    uv run python -X faulthandler -m assay.evaluate --model "$OUT" --data "$data" --out "$OUT/eval-$name.json" --batch-size "$BATCH" --max-state-tokens "$MAXSTATE" "$@" > "$OUT/eval-$name.txt" 2>&1
    if grep -q '^overall' "$OUT/eval-$name.txt"; then break; fi
    echo "retry $name ($attempt)"; sleep 10
  done
  echo "$name: $(grep '^overall' "$OUT/eval-$name.txt" || echo FAILED)"
}
run_eval dev "$DATA/dev.jsonl" --temperature 1.0
run_eval dev-scaled "$DATA/dev.jsonl"
run_eval holdout "$DATA/holdout.jsonl" --temperature 1.0
run_eval holdout-scaled "$DATA/holdout.jsonl"
run_eval transfer-v4 data/suites/kev-transfer-v4-dev.jsonl --temperature 1.0
run_eval transfer-v4-scaled data/suites/kev-transfer-v4-dev.jsonl
echo "=== done"
