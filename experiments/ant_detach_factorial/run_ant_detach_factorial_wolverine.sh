#!/usr/bin/env bash
# Missing detach cells allocated to Wolverine GPU 1 (18 runs).
# Policy: docs/EXPERIMENT_QUEUE_GUIDELINES.md

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_detach_factorial/queues/queue_ant_detach_factorial_wolverine.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_detach_factorial_wolverine.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/ant_detach_factorial_wolverine_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/ant_detach_factorial_wolverine_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-10000}" \
ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-1}" \
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
