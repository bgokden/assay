#!/usr/bin/env bash
# Second overnight queue: decoder runs with the content-scored option term, started once the
# first queue (unit "night") has finished.
#
#   systemd-run --user --unit night2 -p WorkingDirectory=$PWD -p CPUAffinity=0-5,7-23 scripts/night_decoder_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/stage.sh

while systemctl --user is-active --quiet night; do sleep 60; done
log "second queue starting"
stage assay-1.7b-v2-content scripts/run_experiment.sh Qwen/Qwen3-1.7B-Base runs/assay-1.7b-v2-content data/v2 --content-term || true
stage assay-4b-v4-content scripts/run_experiment.sh Qwen/Qwen3-4B-Base runs/assay-4b-v4-content data/v4 --content-term || true
log "second queue finished"
