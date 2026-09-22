#!/usr/bin/env bash
# Sixth overnight queue: the late-interaction term on the compiled and conditioned tiers.
# Starts after unit "night5".
#
#   systemd-run --user --unit night6 -p WorkingDirectory=$PWD scripts/night_late.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

while systemctl --user is-active --quiet night5; do sleep 60; done
log "sixth queue starting"
stage compiled-late-gte-base $COMPILED --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/compiled-late-gte-base --epochs 3 --lr 5e-5 --slots 8 --late-interaction || true
stage conditioned-late-gte-base $COMPILED --arch conditioned --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/conditioned-late-gte-base --epochs 2 --lr 5e-5 --slots 8 --late-interaction || true
log "sixth queue finished"
