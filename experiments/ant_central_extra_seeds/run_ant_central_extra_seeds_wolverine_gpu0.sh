#!/usr/bin/env bash
# Seeds 1996/1997, Wolverine GPU 0 shard. It may safely wait behind CUB detach.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
CAMPAIGN_ROOT="$(tagfex_profile_path configs/hosts/wolverine/ant_central_extra_seeds/storage.yaml --parent)" || exit 2
export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_central_extra_seeds/queues/queue_ant_central_extra_seeds_wolverine_gpu0.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_central_extra_seeds_wolverine_gpu0.lock" \
ORCH_LOG="$CAMPAIGN_ROOT/orchestrator/ant_central_extra_seeds_wolverine_gpu0.log" \
CONSOLE_DIR="$CAMPAIGN_ROOT/orchestrator/gpu0_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-11000}" \
ALLOWED_GPU_IDS=0 \
STOP_ON_FAILURE=1 \
UPDATE_REPORT=0 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
