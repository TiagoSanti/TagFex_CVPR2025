#!/usr/bin/env bash
# ANT Flattening Investigation — full queued pipeline for wolverine (2× RTX 3080 Ti)
#
# Runs all 3 stage-pairs sequentially. Each stage-pair runs 2 experiments in
# parallel (one per GPU). After every stage-pair completes, logs are pushed to
# quaTII via rsync over SSH.
#
# Usage:
#   Run directly (e.g. inside an existing screen session):
#       bash configs/ant_investigation/run_wolverine_parallel.sh
#
#   Launch detached in a new screen session (recommended):
#       screen -dmS ant_queue bash configs/ant_investigation/run_wolverine_parallel.sh
#       screen -r ant_queue       # attach to watch progress
#
# Log locations:
#   wolverine (local): ~/TagFex_CVPR2025/logs/ant_investigation/
#   quaTII   (synced): /mnt/raid/home/tiago/logs/ant_investigation/wolverine/

set -uo pipefail

SEED=1995
REPO=/home/tiago/TagFex_CVPR2025
VENV=$REPO/.venv
OVERLAY="configs/ant_investigation/machine_overlays/wolverine.yaml"
LOCAL_LOG_DIR="$REPO/logs/ant_investigation"
QUATI_USER_HOST="tiago@10.87.10.217"
QUATI_PORT=2222
QUATI_LOG_DEST="/mnt/raid/home/tiago/logs/ant_investigation/wolverine/"

cd "$REPO"
source "$VENV/bin/activate"
mkdir -p "$LOCAL_LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(ts)] $*"; }

# Run two experiments in parallel (GPU0 + GPU1) and wait for both to finish.
# Usage: run_pair <label> <cfg_gpu0> <cfg_gpu1>
run_pair() {
    local label=$1
    local cfg0=$2
    local cfg1=$3

    log "=== $label — starting ==="
    log "  GPU0: $cfg0"
    log "  GPU1: $cfg1"

    CUDA_VISIBLE_DEVICES=0 python main.py train \
        --exp-configs "$cfg0" "$OVERLAY" \
        --seed $SEED --device cuda --force-no-debug &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 python main.py train \
        --exp-configs "$cfg1" "$OVERLAY" \
        --seed $SEED --device cuda --force-no-debug &
    local pid1=$!

    local exit0=0 exit1=0
    wait $pid0 || exit0=$?
    wait $pid1 || exit1=$?

    log "=== $label — done (GPU0 exit=$exit0, GPU1 exit=$exit1) ==="
    [[ $exit0 -ne 0 ]] && log "WARNING: GPU0 experiment exited with code $exit0"
    [[ $exit1 -ne 0 ]] && log "WARNING: GPU1 experiment exited with code $exit1"
}

# Push logs from wolverine to quaTII.
sync_logs() {
    log "--- Syncing logs to quaTII ---"
    rsync -az -e "ssh -p $QUATI_PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes" \
        "$LOCAL_LOG_DIR/" \
        "${QUATI_USER_HOST}:${QUATI_LOG_DEST}" \
        && log "--- Sync OK ---" \
        || log "--- Sync FAILED (non-fatal, continuing) ---"
}

log "=========================================="
log "ANT Investigation Queue starting on $(hostname)"
log "=========================================="

# Stage 1: diagnostic baseline + logsumexp (both need to run to establish ground truth)
run_pair "Stage 1/3" \
    "configs/ant_investigation/stage1_cifar100_10-10_aSymFull_nLocal_diag.yaml" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_logsumexp.yaml"
sync_logs

# Stage 2: expm1 and softplus formulations
run_pair "Stage 2/3" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_expm1.yaml" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_softplus.yaml"
sync_logs

# Stage 3: topk32 and active_only formulations
run_pair "Stage 3/3" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_topk32.yaml" \
    "configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_activeonly.yaml"
sync_logs

log "=========================================="
log "ALL STAGES COMPLETE"
log "=========================================="
