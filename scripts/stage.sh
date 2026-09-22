#!/usr/bin/env bash
# Shared stage runner for the overnight queues. A stage retries on failure and is killed and
# retried when its log stops growing (a hung CUDA process shows no error). Decoder and
# compiled trainers resume from their checkpoints, so a retry continues rather than restarts.
#
#   source scripts/stage.sh; stage <name> <cmd...>
mkdir -p runs/night
LOG=runs/night/queue.log
STALL_SECONDS="${STALL_SECONDS:-900}"
GPU_LOCK="${GPU_LOCK:-/tmp/assay-gpu.lock}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# One GPU job at a time across queues: a stage holds an exclusive flock for its whole run,
# so a queue that reaches its next stage waits instead of crashing into a running job.
with_gpu() { flock "$GPU_LOCK" "$@"; }

run_watched() {  # logfile, cmd... ; returns the command's exit code, 124 when killed as stalled
  local logfile="$1"; shift
  touch "$GPU_LOCK"
  setsid flock "$GPU_LOCK" "$@" >> "$logfile" 2>&1 &  # own process group; holds the GPU lock
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local age=$(( $(date +%s) - $(stat -c %Y "$logfile") ))
    if [ "$age" -gt "$STALL_SECONDS" ]; then
      log "stalled for ${age}s, killing process group $pid"
      kill -TERM -- "-$pid" 2>/dev/null; sleep 15
      kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

stage() {  # name, cmd...
  local name="$1"; shift
  mkdir -p "runs/$name"
  if [ -f "runs/$name/eval-transfer-v4-scaled.json" ] || [ -f "runs/$name/eval-transfer-v4-zeroshot.json" ]; then
    log "skip $name (done)"; return 0
  fi
  for attempt in 1 2 3 4 5; do
    log "start $name attempt $attempt: $*"
    touch "runs/$name/run.log"
    if run_watched "runs/$name/run.log" "$@"; then log "done $name"; return 0; fi
    log "failed $name (exit $?)"
    sleep 30
  done
  return 1
}
