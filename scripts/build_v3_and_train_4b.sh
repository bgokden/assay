#!/usr/bin/env bash
# Roadmap step 2 + 5a: distillation set from the 27B, data/v3, retrain the 4B on it.
set -euo pipefail
cd "$(dirname "$0")/.."
for attempt in 1 2 3 4 5 6; do
  if uv run python -m assay.data.distill --teacher runs/assay-27b --train data/v2/train.jsonl --out data/distill \
    --generic 40000 --policy-hard 5000 --dates 6000 --dev 300 --batch-size 8; then break; fi; sleep 10
done
mkdir -p data/v3
cat data/v2/train.jsonl data/distill/generic.jsonl data/distill/policy_hard.jsonl data/distill/dates.jsonl > data/v3/train.jsonl
cat data/v2/dev.jsonl data/distill/policy_hard_dev.jsonl data/distill/dates_dev.jsonl > data/v3/dev.jsonl
cp data/v2/calibration.jsonl data/v3/calibration.jsonl
cp data/v2/holdout.jsonl data/v3/holdout.jsonl
wc -l data/v3/*.jsonl
mkdir -p runs/assay-4b-v3
for attempt in 1 2 3; do
  if scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b-v3 data/v3 --log-every 100 --checkpoint-every 500 >> runs/assay-4b-v3/run.log 2>&1; then break; fi
  sleep 15
done
echo "=== done"
