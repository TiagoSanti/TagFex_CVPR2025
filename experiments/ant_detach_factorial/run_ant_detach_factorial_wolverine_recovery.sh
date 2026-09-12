#!/usr/bin/env bash
# Execute the failed seed-1995 IV-GR cells after the active GPU-1 queue.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
UPSTREAM_LOCK="$TAGFEX_LOCK_DIR/tagfex_ant_detach_factorial_wolverine.lock"
CAMPAIGN_ROOT="$(tagfex_profile_path configs/hosts/wolverine/ant_detach_factorial/storage.yaml --parent)" || exit 2
mkdir -p "$CAMPAIGN_ROOT/orchestrator"

# Do not edit the queue currently held open by the upstream orchestrator.
# Waiting on its lock provides an unambiguous and race-free queue tail.
while ! flock -n "$UPSTREAM_LOCK" -c true; do
    sleep 30
done

export TAGFEX_DATASET_HASH_MODE="${TAGFEX_DATASET_HASH_MODE:-structure}"

QUEUE="$SCRIPT_DIR/experiments/ant_detach_factorial/queues/queue_ant_detach_factorial_wolverine_recovery.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ant_detach_factorial_wolverine_recovery.lock" \
ORCH_LOG="$CAMPAIGN_ROOT/orchestrator/ant_detach_factorial_wolverine_recovery.log" \
CONSOLE_DIR="$CAMPAIGN_ROOT/orchestrator/recovery_console" \
GPUS=1 \
THRESHOLD="${THRESHOLD:-100}" \
MIN_FREE_MB="${MIN_FREE_MB:-10000}" \
ALLOWED_GPU_IDS=1 \
STOP_ON_FAILURE=1 \
UPDATE_REPORT=0 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
