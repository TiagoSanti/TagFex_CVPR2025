#!/usr/bin/env bash
# Run the Baseline InfoNCE + TeacherAvg-K ablation queue.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"

QUEUE="$SCRIPT_DIR/experiments/baseline_teacheravg/queues/queue_baseline_teacheravg.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_baseline_teacheravg.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/baseline_teacheravg_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/baseline_teacheravg_console" \
GPUS="${GPUS:-1}" \
THRESHOLD="${THRESHOLD:-100}" \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
