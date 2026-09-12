#!/usr/bin/env bash
# Run the ImageNet-100 50-10 pure ANT 2x2 ablation.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
DATASET_ROOT="$HOME/data/datasets/imagenet100"

# Fail before consuming the queue when the licensed ImageNet subset is absent.
if [ ! -d "$DATASET_ROOT/train" ] || [ ! -d "$DATASET_ROOT/val" ]; then
    echo "[$(date -Iseconds)] BLOCKED: ImageNet-100 not found at $DATASET_ROOT (expected train/ and val/)." >&2
    exit 2
fi

QUEUE="$SCRIPT_DIR/experiments/imagenet100/queues/queue_imagenet100_50-10_ant_main.txt" \
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_imagenet100_50-10_ant.lock" \
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/imagenet100_50-10_ant_orchestrator.log" \
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/imagenet100_50-10_ant_console" \
GPUS=1 \
THRESHOLD=100 \
MEMORY_THRESHOLD=10 \
MIN_FREE_MB=11000 \
ALLOWED_GPU_IDS="0" \
STOP_ON_FAILURE=1 \
exec bash "$SCRIPT_DIR/scripts/execution/run_sbs_queue.sh"
