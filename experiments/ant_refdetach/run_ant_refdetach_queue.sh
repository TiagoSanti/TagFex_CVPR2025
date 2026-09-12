#!/usr/bin/env bash
# Three-seed ANT reference-detach ablation on the four primary protocols.
# Uses a memory gate so it waits safely for either local GPU while other queues run.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

# Record dataset structure without rehashing every image for each of the 12 runs.
export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_refdetach/queues/queue_ant_refdetach.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_refdetach.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/ant_refdetach_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/ant_refdetach_console" \
GPUS=1 \
THRESHOLD=100 \
MEMORY_THRESHOLD=10 \
ALLOWED_GPU_IDS="1" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
