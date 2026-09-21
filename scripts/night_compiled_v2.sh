#!/usr/bin/env bash
# Fourth overnight queue: the compiled tier with more optimisation (it underfits in one epoch
# at lr 2e-5: train loss 0.86 against 0.4-0.6 for the decoders). Starts after unit "night3".
#
#   systemd-run --user --unit night4 -p WorkingDirectory=$PWD -p CPUAffinity=0-5,7-23 scripts/night_compiled_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

GENERIC=data/distill/generic.jsonl
COMPILED="uv run python -X faulthandler -m assay.train_compiled"

while systemctl --user is-active --quiet night3; do sleep 60; done
log "fourth queue starting"
stage compiled-gte-base-3ep $COMPILED --encoder Alibaba-NLP/gte-modernbert-base --data data/v4 --extra $GENERIC --out runs/compiled-gte-base-3ep --epochs 3 --lr 5e-5 --slots 8 || true
log "fourth queue finished"
