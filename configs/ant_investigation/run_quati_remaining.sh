#!/usr/bin/env bash
# ANT Investigation — remaining experiments (topk32 + active_only)
# Run after the first 4 completed successfully.

set -uo pipefail

SEED=1995
GPU=0
REPO=/home/tiago/TagFex_CVPR2025
VENV=$REPO/.venv
OVERLAY="configs/ant_investigation/machine_overlays/quati.yaml"

cd "$REPO"
source "$VENV/bin/activate"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

run_exp() {
    local label=$1
    local cfg=$2
    log "=== $label starting ==="
    log "  Config: $cfg"
    CUDA_VISIBLE_DEVICES=$GPU python main.py train \
        --exp-configs "$cfg" "$OVERLAY" \
        --seed $SEED --device cuda --force-no-debug
    log "=== $label done (exit $?) ==="
}

log "=========================================="
log "ANT Investigation — remaining experiments"
log "=========================================="

run_exp "5/6 stage2_topk32" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_topk32.yaml"

run_exp "6/6 stage2_activeonly" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_activeonly.yaml"

log "=========================================="
log "DONE"
log "=========================================="
