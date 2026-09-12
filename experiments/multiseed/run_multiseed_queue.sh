#!/usr/bin/env bash
# run_multiseed_queue.sh — Multi-seed runs for competitive variants.
#
# Runs seeds 1994 and 1995 for:
#   β=0.5 aSymFull avgK5 | β=0.5 aLocal avgK3 | β=0.5 aLocal avgK5
# across all 4 scenarios (CIFAR-100 10-10, 50-10 | TIN 100-20, 20-20).
# Total: 3 variants × 4 scenarios × 2 seeds = 24 runs.
#
# Usage:
#   screen -dmS multiseed bash run_multiseed_queue.sh
#   screen -r multiseed
#
#   tail -f logs/auto_experiments/multiseed_orchestrator.log
#   tail -f logs/auto_experiments/multiseed_console/<exp>.log

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/multiseed/queues/queue_multiseed.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_multiseed.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/multiseed_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/multiseed_console" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
