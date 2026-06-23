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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUEUE="$SCRIPT_DIR/configs/queue_multiseed.txt" \
LOCKFILE="/tmp/tagfex_multiseed.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/multiseed_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/multiseed_console" \
exec bash "$SCRIPT_DIR/run_sbs_queue.sh"
