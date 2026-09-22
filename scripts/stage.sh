#!/usr/bin/env bash
# Shared stage runner for the overnight queues. A stage retries on failure and is killed and
# retried when its log stops growing (a hung CUDA process shows no error). Decoder and
# compiled trainers resume from their checkpoints, so a retry continues rather than restarts.
#
#   source scripts/stage.sh; stage <name> <cmd...>
mkdir -p runs/night
LOG=runs/night/queue.log
STALL_SECONDS="${STALL_SECONDS:-900}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
GPU_LOCK="${GPU_LOCK:-/tmp/assay-gpu.lock}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# A run that cannot write its checkpoints is worse than a run that never starts: it burns
# the GPU for hours and stops at the first save. Wait for space, then give up on the stage.
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }
wait_for_disk() {  # name
  local waited=0
  while [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; do
    if [ "$waited" -ge 3600 ]; then
      log "skip $1: only $(free_gb) GiB free, needs ${MIN_FREE_GB}"
      return 1
    fi
    [ "$waited" -eq 0 ] && log "waiting for disk before $1: $(free_gb) GiB free"
    sleep 300; waited=$((waited + 300))
  done
  return 0
}

# One GPU job at a time across queues: a stage holds an exclusive flock for its whole run,
# so a queue that reaches its next stage waits instead of crashing into a running job.
with_gpu() { flock "$GPU_LOCK" "$@"; }

run_watched() {  # logfile, cmd... ; returns the command's exit code, 124 when killed as stalled
  local logfile="$1"; shift
  touch "$GPU_LOCK"
  local launched=$(stat -c %Y "$logfile")
  # own process group (so a stall kill reaches the whole tree) holding the GPU lock; the log
  # is touched the moment the lock is acquired, which is when the stall clock starts
  setsid flock "$GPU_LOCK" bash -c 'touch "$1"; shift; exec "$@"' _ "$logfile" "$@" >> "$logfile" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local mtime=$(stat -c %Y "$logfile")
    if [ "$mtime" -le "$launched" ]; then
      continue  # still queued behind another GPU job, not stalled
    fi
    local age=$(( $(date +%s) - mtime ))
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
  wait_for_disk "$name" || return 1
  for attempt in 1 2 3 4 5; do
    log "start $name attempt $attempt: $*"
    touch "runs/$name/run.log"
    local code=0
    run_watched "runs/$name/run.log" "$@" || code=$?
    if [ "$code" -eq 0 ]; then log "done $name"; return 0; fi
    log "failed $name (exit $code)"   # an if with no else returns 0, so keep the code itself
    sleep 30
  done
  return 1
}
