#!/usr/bin/env bash
# Seeds 1996/1997 for the baseline and four main detached ANT methods.
# This shard is balanced by historical GPU-hours against both Wolverine GPUs.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
LOCAL_LOGS_ROOT="$(tagfex_profile_path configs/hosts/xavier/ant_central_extra_seeds/storage.yaml)" || exit 2
export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_central_extra_seeds/queues/queue_ant_central_extra_seeds_local.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_central_extra_seeds_local.lock" \
ORCH_LOG="$LOCAL_LOGS_ROOT/auto_experiments/ant_central_extra_seeds_local_orchestrator.log" \
CONSOLE_DIR="$LOCAL_LOGS_ROOT/auto_experiments/ant_central_extra_seeds_local_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-11000}" \
ALLOWED_GPU_IDS=0 \
STOP_ON_FAILURE=1 \
UPDATE_REPORT=0 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
