#!/usr/bin/env bash
# ANT Flattening Investigation — experiment runner
#
# Stage 1: Single diagnostic run to identify the source of ANT loss flattening.
#           Run FIRST, analyse logs before proceeding to Stage 2.
# Stage 2: Compare all loss formulations under identical conditions.
#
# Usage:
#   ./configs/ant_investigation/run_investigation.sh [stage1|stage2|all]
#
# Logs are written to /mnt/raid/home/tiago/logs/ant_investigation/
# The per-experiment log suffix encodes hyperparameters + seed for traceability.
#
# After completion, parse diagnostics from exp_debug0.log:
#   grep "ANT flattening" <log_dir>/exp_debug0.log | head -200
#   grep "ANT distance stats" <log_dir>/exp_debug0.log | head -200

set -e
STAGE=${1:-all}
SEED=1995
GPU=${GPU:-0}
VENV=/home/tiago/TagFex_CVPR2025/.venv

activate_venv() {
    if [[ -f "$VENV/bin/activate" ]]; then
        source "$VENV/bin/activate"
    fi
}

run_exp() {
    local config=$1
    echo "========================================"
    echo "Running: $config  (seed=$SEED, gpu=$GPU)"
    echo "========================================"
    CUDA_VISIBLE_DEVICES=$GPU python main.py train \
        --exp-configs "$config" \
        --seed $SEED \
        --device cuda
}

activate_venv
cd "$(dirname "$0")/../.."

# Stage 1 — diagnostic run (current logsumexp, all new metrics logged)
if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
    run_exp configs/ant_investigation/stage1_cifar100_10-10_aSymFull_nLocal_diag.yaml
fi

# Stage 2 — loss formulation comparison (same seed, all formulations)
if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    run_exp configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_logsumexp.yaml
    run_exp configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_expm1.yaml
    run_exp configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_softplus.yaml
    run_exp configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_topk32.yaml
    run_exp configs/ant_investigation/stage2_cifar100_10-10_aSymFull_nLocal_activeonly.yaml
fi

echo "All requested stages complete."
