#!/usr/bin/env bash
# Wait for ANT-FS-GR to finish successfully, then complete CUB ANT-IV-GR.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
UPSTREAM_LOCK="$TAGFEX_LOCK_DIR/tagfex_ant_fs_gr.lock"
UPSTREAM_LOG="$SCRIPT_DIR/logs/auto_experiments/ant_fs_gr_orchestrator.log"
UPSTREAM_QUEUE="$SCRIPT_DIR/experiments/ant_fs_gr/queues/queue_ant_fs_gr.txt"
CHAIN_LOG="$SCRIPT_DIR/logs/auto_experiments/cub_iv_gr_chain.log"
CHAIN_LOCK="$TAGFEX_LOCK_DIR/tagfex_cub_iv_gr_chain.lock"

mkdir -p "$(dirname "$CHAIN_LOG")"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$CHAIN_LOG"
}

exec 7>"$CHAIN_LOCK"
if ! flock -n 7; then
    log "ALREADY QUEUED: another CUB ANT-IV-GR chain owns $CHAIN_LOCK"
    exit 0
fi

# Waiting on the orchestrator lock keeps this composition strictly after the
# active ANT-FS-GR queue, rather than competing for a GPU between its runs.
exec 8>"$UPSTREAM_LOCK"
log "QUEUED CUB ANT-IV-GR behind ANT-FS-GR lock"
flock 8

upstream_total="$(grep -cE '^[^#[:space:]]' "$UPSTREAM_QUEUE" 2>/dev/null || true)"
if ! awk -v total="$upstream_total" '
    /ORCHESTRATOR START/ { seen = 1; failed = 0; final_done = 0; success = 0 }
    seen && /(FAIL|MISSING|MALFORMED)/ { failed = 1 }
    seen && index($0, "DONE [" total "/" total "]") { final_done = 1 }
    seen && index($0, "ORCHESTRATOR DONE — all " total " experiments finished successfully") { success = 1 }
    END { exit !(seen && !failed && (success || final_done)) }
' "$UPSTREAM_LOG"; then
    log "ABORT: ANT-FS-GR lock was released without a successful complete queue"
    exit 1
fi

log "UPSTREAM VERIFIED: ANT-FS-GR completed $upstream_total/$upstream_total"
flock -u 8
exec 8>&-

log "STARTING CUB 20-20 ANT-IV-GR queue"
exec bash "$SCRIPT_DIR/experiments/cub_iv_gr/run_cub_iv_gr_queue.sh"
