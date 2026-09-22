#!/usr/bin/env bash
# Fifth overnight queue: the conditioned encoder tier (instruction + state in one pass,
# compiled options). Starts after unit "night4".
#
#   systemd-run --user --unit night5 -p WorkingDirectory=$PWD scripts/night_conditioned.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

while systemctl --user is-active --quiet night4; do sleep 60; done
log "fifth queue starting"
stage conditioned-gte-base $COMPILED --arch conditioned --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/conditioned-gte-base --epochs 2 --lr 5e-5 --slots 8 || true
log "fifth queue finished"
