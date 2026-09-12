#!/usr/bin/env bash
# Wait for the active baseline/TeacherAvg queue to finish, then start ANT-FS-GR.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
UPSTREAM_LOCK="$TAGFEX_LOCK_DIR/tagfex_baseline_teacheravg.lock"
UPSTREAM_LOG="$SCRIPT_DIR/logs/auto_experiments/baseline_teacheravg_orchestrator.log"
UPSTREAM_QUEUE="$SCRIPT_DIR/experiments/baseline_teacheravg/queues/queue_baseline_teacheravg.txt"
CHAIN_LOG="$SCRIPT_DIR/logs/auto_experiments/ant_fs_gr_chain.log"
CHAIN_LOCK="$TAGFEX_LOCK_DIR/tagfex_ant_fs_gr_chain.lock"

mkdir -p "$(dirname "$CHAIN_LOG")"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$CHAIN_LOG"
}

# Prevent duplicate waiters from being created when the enqueue command is
# issued more than once.
exec 7>"$CHAIN_LOCK"
if ! flock -n 7; then
    log "ALREADY QUEUED: another ANT-FS-GR chain owns $CHAIN_LOCK"
    exit 0
fi

# The active baseline orchestrator owns this lock for its complete lifetime.
# Blocking here prevents ANT-FS-GR from taking the GPU between baseline runs.
exec 8>"$UPSTREAM_LOCK"
log "QUEUED ANT-FS-GR behind baseline/TeacherAvg lock"
flock 8

upstream_total="$(grep -cE '^[^#[:space:]]' "$UPSTREAM_QUEUE" 2>/dev/null || true)"
if tail -n 1 "$UPSTREAM_LOG" | grep -q 'ORCHESTRATOR DONE'; then
    log "UPSTREAM DONE: final orchestrator record found"
elif [ "$upstream_total" -gt 0 ] && \
     grep -Fq "DONE [$upstream_total/$upstream_total]" "$UPSTREAM_LOG"; then
    # The final experiment can be complete even if a later best-effort report
    # refresh terminates the shell before it writes ORCHESTRATOR DONE.
    log "UPSTREAM VERIFIED: final experiment DONE [$upstream_total/$upstream_total]"
else
    log "ABORT: upstream lock was released without a completed orchestrator record"
    exit 1
fi

# Do not retain the baseline/TeacherAvg lock throughout the downstream queue.
flock -u 8
exec 8>&-

log "STARTING ANT-FS-GR queue"
exec bash "$SCRIPT_DIR/experiments/ant_fs_gr/run_ant_fs_gr_queue.sh"
