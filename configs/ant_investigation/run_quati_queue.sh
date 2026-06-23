#!/usr/bin/env bash
# ANT Flattening Investigation — full queue for quaTII (RTX 4090, 24 GB)
#
# Runs all 6 experiments sequentially on GPU 0. Logs go directly to /mnt/raid.
#
# Usage:
#   Launch in a detached screen session (recommended):
#       screen -dmS ant_queue bash configs/ant_investigation/run_quati_queue.sh
#       screen -r ant_queue      # attach to watch progress
#
#   Or run directly:
#       bash configs/ant_investigation/run_quati_queue.sh
#
# Log location: /mnt/raid/home/tiago/logs/ant_investigation/

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
log "ANT Investigation Queue — quaTII RTX 4090"
log "=========================================="

run_exp "1/6 stage1_diag (logsumexp + full diagnostics)" \
    "configs/ant_investigation/stage1_cifar100_10-10_aSymFull_nLocal_diag.yaml"

run_exp "2/6 stage2_logsumexp (baseline formulation)" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_logsumexp.yaml"

run_exp "3/6 stage2_expm1" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_expm1.yaml"

run_exp "4/6 stage2_softplus" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_softplus.yaml"

run_exp "5/6 stage2_topk32" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_topk32.yaml"

run_exp "6/6 stage2_activeonly" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_activeonly.yaml"

log "=========================================="
log "ALL EXPERIMENTS COMPLETE"
log "Logs at: /mnt/raid/home/tiago/logs/ant_investigation/"
log "=========================================="
