#!/usr/bin/env bash
# CUB-200 100+20, five seeds: baseline and canonical detached ANT methods.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
LOCAL_LOGS_ROOT="$(tagfex_profile_path configs/hosts/xavier/cub100_canonical_5seeds/storage.yaml)" || exit 2
export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

"${TAGFEX_PYTHON:-$SCRIPT_DIR/.venv/bin/python}" \
    "$SCRIPT_DIR/validation/validate_cub100_canonical_5seeds.py"

QUEUE="$SCRIPT_DIR/experiments/cub100_canonical_5seeds/queues/queue_cub100_canonical_5seeds_local.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_cub100_canonical_5seeds_local.lock" \
ORCH_LOG="$LOCAL_LOGS_ROOT/auto_experiments/cub100_canonical_5seeds_local_orchestrator.log" \
CONSOLE_DIR="$LOCAL_LOGS_ROOT/auto_experiments/cub100_canonical_5seeds_local_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-11000}" \
ALLOWED_GPU_IDS=0 \
STOP_ON_FAILURE=1 \
UPDATE_REPORT=0 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
