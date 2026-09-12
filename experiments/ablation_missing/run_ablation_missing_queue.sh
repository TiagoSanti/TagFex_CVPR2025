#!/usr/bin/env bash
# run_ablation_missing_queue.sh — Runs all missing multi-seed ablation experiments.
#
# Delegates to run_avgk_queue.sh with configs/queue_ablation_missing.txt.
# Experiments already completed (console log contains avg_nme1) are skipped.
#
# Usage:
#   screen -dmS ablation_missing bash run_ablation_missing_queue.sh
#   screen -r ablation_missing
#
# Or directly:
#   bash run_ablation_missing_queue.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/ablation_missing/queues/queue_ablation_missing.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_ablation_missing.lock" \
exec bash "$SCRIPT_DIR/scripts/execution/run_avgk_queue.sh"
