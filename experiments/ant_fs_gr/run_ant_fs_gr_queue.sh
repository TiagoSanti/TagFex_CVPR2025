#!/usr/bin/env bash
# Run the pure ANT-FS-GR factorial-completion queue.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/ant_fs_gr/queues/queue_ant_fs_gr.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_fs_gr.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/ant_fs_gr_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/ant_fs_gr_console" \
GPUS="${GPUS:-1}" \
THRESHOLD="${THRESHOLD:-100}" \
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-10}" \
ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-0 1}" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
