#!/usr/bin/env bash
# Missing detach cells allocated to this host (18 runs).
# Policy: docs/EXPERIMENT_QUEUE_GUIDELINES.md

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
LOCAL_LOGS_ROOT="$(tagfex_profile_path configs/hosts/xavier/ant_detach_factorial/storage.yaml)" || exit 2

export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_detach_factorial/queues/queue_ant_detach_factorial_local.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_detach_factorial_local.lock" \
ORCH_LOG="$LOCAL_LOGS_ROOT/auto_experiments/ant_detach_factorial_local_orchestrator.log" \
CONSOLE_DIR="$LOCAL_LOGS_ROOT/auto_experiments/ant_detach_factorial_local_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-10000}" \
ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-0 1}" \
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
