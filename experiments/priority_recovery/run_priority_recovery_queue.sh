#!/usr/bin/env bash
# Run the missing TIN-20-20 seed before resuming the interrupted RefDetach queue.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/priority_recovery/queues/queue_priority_recovery.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_priority_recovery.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/priority_recovery_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/priority_recovery_console" \
GPUS=1 \
THRESHOLD=100 \
MEMORY_THRESHOLD=10 \
ALLOWED_GPU_IDS="0" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
