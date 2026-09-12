#!/usr/bin/env bash
# CUB-200 20-20 detach factorial on Wolverine GPU 0.
# The llama-server must be stopped before this queue starts. Requiring 11 GiB
# free prevents an accidental relaunch while another GPU-0 workload is active.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
CAMPAIGN_ROOT="$(tagfex_profile_path configs/hosts/wolverine/ant_detach_cub20/storage.yaml --parent)" || exit 2

export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_detach_cub20/queues/queue_ant_detach_cub20_wolverine_gpu0.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_detach_cub20_wolverine_gpu0.lock" \
ORCH_LOG="$CAMPAIGN_ROOT/orchestrator/ant_detach_cub20_wolverine_gpu0.log" \
CONSOLE_DIR="$CAMPAIGN_ROOT/orchestrator/console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-11000}" \
ALLOWED_GPU_IDS=0 \
STOP_ON_FAILURE=1 \
UPDATE_REPORT=0 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
