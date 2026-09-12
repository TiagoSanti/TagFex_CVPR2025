#!/usr/bin/env bash
# run_cub200_queue.sh — CUB-200 benchmark queue (20-20 and 100-20 scenarios).
#
# Runs seeds 1993, 1994, and 1995 for all 14 promising variants from section 0
# across both CUB-200 scenarios (20-20 and 100-20).
# Total: 14 variants × 2 scenarios × 3 seeds = 84 runs.
#
# Usage:
#   screen -dmS cub200 bash run_cub200_queue.sh
#   screen -r cub200
#
#   tail -f logs/auto_experiments/cub200_orchestrator.log
#   tail -f logs/auto_experiments/cub200_console/<exp>.log
#
# Prerequisites:
#   python setup_cub200.py  # download and verify the CUB-200-2011 dataset

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/cub200/queues/queue_cub200.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_cub200.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/cub200_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/cub200_console" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
