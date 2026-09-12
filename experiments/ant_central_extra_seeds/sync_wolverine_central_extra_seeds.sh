#!/usr/bin/env bash
# Collect only completed seeds-1996/1997 runs from both Wolverine GPU shards.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
LOCAL_LOGS_ROOT="$(tagfex_profile_path configs/hosts/xavier/ant_central_extra_seeds/storage.yaml)" || exit 2
CAMPAIGN_ROOT="$(tagfex_profile_path configs/hosts/wolverine/ant_central_extra_seeds/storage.yaml --parent)" || exit 2

REMOTE_ROOT="$CAMPAIGN_ROOT" \
REMOTE_ORCHESTRATOR="$CAMPAIGN_ROOT/orchestrator" \
REMOTE_UNITS="${REMOTE_UNITS:-ant-central-extra-seeds-wolverine-gpu0-20260906.service ant-central-extra-seeds-wolverine-gpu1-20260906.service}" \
LOCAL_RESULTS="$LOCAL_LOGS_ROOT" \
LOCAL_ORCHESTRATOR="$LOCAL_LOGS_ROOT/auto_experiments/wolverine_central_extra_seeds_20260906" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_wolverine_central_extra_seeds_sync.lock" \
COMPLETED_ONLY=1 \
RUN_MODE="${RUN_MODE:-watch}" \
exec bash "$SCRIPT_DIR/scripts/sync/sync_wolverine_refdetach_results.sh"
