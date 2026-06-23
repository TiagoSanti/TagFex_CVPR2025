#!/bin/bash
# run_exp_queue_restart.sh — Restart queue after crash at [7/21]
#
# Queue status when this script was created (2026-05-04):
#   [1-3/21] TinyIN-20-20 s1995  (all 3 variants)          ✓ DONE
#   [4/21]   TinyIN-100-20 β=0 aLocal nLocal s1993          ✓ DONE (avg_acc≈59.70%)
#   [5/21]   TinyIN-100-20 β=0.5 aLocal nLocal s1993        ✓ DONE but BUGGY (avg_acc≈11.75%)
#   [6/21]   TinyIN-100-20 β=0.5 aSymFull nLocal s1993      ✓ DONE but BUGGY (avg_acc≈11.75%)
#   [7/21]   TinyIN-100-20 β=0 aLocal nLocal s1994          ✗ CRASHED at T6E37
#   [8-21/21] remaining                                      ✗ NOT STARTED
#
# Bug found: trans_cls_loss explodes to 5858 at T2E1 for β=0.5 on 100-20
#   Root cause: runaway weight growth in TSAttention.weight_v + trans_classifier
#               due to positive feedback loop (merged_feature grows unboundedly)
#   Fix: grad_clip_norm: 5.0 added to both β=0.5 100-20 config files
#        + torch.nn.utils.clip_grad_norm_() in tagfex.py backward step
#
# This script runs:
#   • Re-runs of buggy s1993 β=0.5 experiments (with grad_clip fix applied)
#   • Restart of crashed s1994 β=0 experiment
#   • Remaining [8-21] from original queue
#
# Usage:
#   screen -dmS exp_queue bash run_exp_queue_restart.sh
#   screen -r exp_queue
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Virtual environment ──────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── GPU / launcher settings ──────────────────────────────────────────────────
GPUS=1
THRESHOLD=100.0      # % GPU memory free threshold to consider GPU available
INTERVAL=30          # seconds between GPU polling attempts
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# ── Progress log ─────────────────────────────────────────────────────────────
LOG_DIR="/mnt/raid/home/tiago/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress.log"
CONSOLE_LOG="$LOG_DIR/queue_console_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# ── Terminal colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_progress() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $1" | tee -a "$PROGRESS_LOG"
}

EXP_COUNTER=0
EXP_TOTAL=17

queue_experiment() {
    local config_file=$1
    local description=$2
    local seed=${3:-1993}

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$EXP_TOTAL]"

    log_progress ">> $pos START: $description  [seed=$seed]"
    echo -e "${YELLOW}>> $pos${NC} $description  ${CYAN}[seed=$seed]${NC}"
    echo -e "   Config  : $config_file"
    echo -e "   Waiting for free GPU (mem < ${THRESHOLD}%)...\n"

    local train_cmd="python3 main.py train --exp-configs $config_file --seed $seed"

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus "$GPUS" \
        --threshold "$THRESHOLD" \
        --interval "$INTERVAL" \
        --no-screen

    if [ $? -eq 0 ]; then
        log_progress "[OK] $pos DONE: $description  [seed=$seed]"
        echo -e "${GREEN}[OK] $pos Done (${EXP_COUNTER}/${EXP_TOTAL})${NC}\n"
    else
        log_progress "[ERR] $pos FAILED: $description  [seed=$seed]"
        echo -e "${RED}[ERR] $pos Experiment failed — aborting queue${NC}\n"
        exit 1
    fi

    sleep 5
}

# ═════════════════════════════════════════════════════════════════════════════
main() {
echo -e "${GREEN}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Restart Queue — 17 experiments  ${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Total       : ${EXP_TOTAL} experiments${NC}"
echo -e "${BLUE} Re-runs     : TinyIN-100-20 β=0.5 s1993 × 2 (grad_clip fix applied)${NC}"
echo -e "${BLUE} Restart     : TinyIN-100-20 β=0 s1994 (crashed at T6E37)${NC}"
echo -e "${BLUE} Continuing  : TinyIN-100-20 β=0.5 s1994+s1995 + β=0 s1995${NC}"
echo -e "${BLUE} True baseline (β=0 nGlobal):${NC}"
echo -e "   • CIFAR-100 10-10     s1994, s1995"
echo -e "   • CIFAR-100 50-10     s1994, s1995"
echo -e "   • TinyIN-20-20        s1994, s1995"
echo -e "   • TinyIN-100-20       s1993, s1994, s1995"
echo -e ""

log_progress ">> Restart queue started  (total: $EXP_TOTAL)"

# ─────────────────────────────────────────────────────────────────────────────
# RE-RUNS: TinyIN-100-20  β=0.5  seed 1993  (grad_clip fix, was buggy)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── TinyIN-100-20  β=0.5  seed 1993 RE-RUNS (with grad_clip fix) ──${NC}"

queue_experiment \
    "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-100-20] β=0.5 aLocal nLocal [RE-RUN]" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antSymmetricFull_debug_resnet18.yaml" \
    "[TinyIN-100-20] β=0.5 aSymFull nLocal [RE-RUN]" 1993

# ─────────────────────────────────────────────────────────────────────────────
# RESTART: TinyIN-100-20  β=0  seed 1994  (was [7/21], crashed at T6E37)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── TinyIN-100-20  β=0  seed 1994 (restart, was [7/21]) ──────────${NC}"

queue_experiment \
    "configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-100-20] β=0 aLocal nLocal [RESTART]" 1994

# ─────────────────────────────────────────────────────────────────────────────
# NEW: TinyIN-100-20  β=0.5  seeds 1994, 1995  (was [8,9,11,12/21])
#      β=0  seed 1995  (was [10/21])
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── TinyIN-100-20  β=0.5  seeds 1994 + 1995  (new, with fix) ────${NC}"

for seed in 1994 1995; do
    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0.5 aLocal nLocal" $seed

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antSymmetricFull_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0.5 aSymFull nLocal" $seed
done

echo -e "\n${CYAN}── TinyIN-100-20  β=0  seed 1995  ───────────────────────────────${NC}"

queue_experiment \
    "configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-100-20] β=0 aLocal nLocal" 1995

# ─────────────────────────────────────────────────────────────────────────────
# TRUE BASELINE: β=0 nGlobal  (was [13-21/21])
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── True Baseline β=0 nGlobal  (seeds 1994, 1995) ────────────────${NC}"

for seed in 1994 1995; do
    queue_experiment \
        "configs/all_in_one/cifar100_10-10_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
        "[C100-10-10] β=0 nGlobal" $seed

    queue_experiment \
        "configs/all_in_one/cifar100_50-10_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
        "[C100-50-10] β=0 nGlobal" $seed

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
        "[TinyIN-20-20] β=0 nGlobal" $seed
done

echo -e "\n${CYAN}── True Baseline β=0 nGlobal  TinyIN-100-20  (seeds 1993–1995) ──${NC}"

for seed in 1993 1994 1995; do
    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0 nGlobal" $seed
done

# ─────────────────────────────────────────────────────────────────────────────
log_progress "[OK] Restart queue finished!  ($EXP_TOTAL/$EXP_TOTAL)"
echo -e "\n${GREEN}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] All ${EXP_TOTAL} experiments completed!${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════════════${NC}\n"
echo -e "  Progress log : ${PROGRESS_LOG}"
echo -e "  Console log  : ${CONSOLE_LOG}"
echo -e "  Results      : /mnt/raid/home/tiago/logs/"
}

main 2>&1 | tee -a "$CONSOLE_LOG"
