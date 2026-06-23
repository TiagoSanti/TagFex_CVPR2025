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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUEUE="$SCRIPT_DIR/configs/queue_ablation_missing.txt" \
LOCKFILE="/tmp/tagfex_ablation_missing.lock" \
exec bash "$SCRIPT_DIR/run_avgk_queue.sh"
