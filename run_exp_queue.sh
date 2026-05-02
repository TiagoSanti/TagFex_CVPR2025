#!/bin/bash
# run_exp_queue.sh — Continuation + true baseline queue
#
# Purpose:
#   Continues the interrupted run at [TinyIN-20-20] β=0 aLocal nLocal s1995,
#   completes the remaining original experiments (TinyIN-20-20 s1995 tail +
#   TinyIN-100-20 all seeds), and adds the true baseline (β=0 nGlobal) for
#   all 4 datasets across seeds 1994, 1995 (seed 1993 already done for the
#   first 3 datasets) and all 3 seeds for TinyIN-100-20 (new dataset).
#
# Completed (removed from queue):
#   CIFAR-100 10-10   × seeds 1994, 1995  (3 variants × 2 = 6)   ✓ done
#   CIFAR-100 50-10   × seeds 1994, 1995  (3 variants × 2 = 6)   ✓ done
#   TinyIN-20-20      × seed  1994        (3 variants × 1 = 3)   ✓ done
#   TinyIN-20-20 β=0.5 aSymFull nLocal s1994                      ✓ done
#
# Remaining / new experiments (21 total):
#   1  — TinyIN-20-20  β=0 aLocal nLocal s1995          (interrupted, restart)
#   2–3 — TinyIN-20-20 β=0.5 variants s1995
#   4–12 — TinyIN-100-20  3 variants × seeds 1993,1994,1995
#   13–14 — C100-10-10   β=0 nGlobal s1994, s1995       (true baseline)
#   15–16 — C100-50-10   β=0 nGlobal s1994, s1995       (true baseline)
#   17–18 — TinyIN-20-20 β=0 nGlobal s1994, s1995       (true baseline)
#   19–21 — TinyIN-100-20 β=0 nGlobal s1993, s1994, s1995 (true baseline)
#
# True baseline: β=0, ant_max_global=true, infonce_max_global=true
# (seed 1993 nGlobal results already exist for C100-10-10, C100-50-10, TinyIN-20-20)
#
# Debug logs are kept as raw files (no zipping after each run).
# All logs go to /mnt/raid/home/tiago/logs/ as set in each config's log_dir.
#
# Usage:
#   screen -dmS exp_queue bash run_exp_queue.sh
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
EXP_TOTAL=21

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
echo -e "${GREEN}  Queue — 21 experiments (continuation + true baseline)  ${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Total       : ${EXP_TOTAL} experiments${NC}"
echo -e "${BLUE} Restarting  : [TinyIN-20-20] β=0 aLocal nLocal s1995 (interrupted)${NC}"
echo -e "${BLUE} Continuing  : TinyIN-20-20 s1995 tail + TinyIN-100-20 (3 seeds)${NC}"
echo -e "${BLUE} True baseline (β=0 nGlobal, seeds 1994+1995):${NC}"
echo -e "   • CIFAR-100 10-10     →  2 runs  (s1993 already done)"
echo -e "   • CIFAR-100 50-10     →  2 runs  (s1993 already done)"
echo -e "   • Tiny-ImageNet 20-20 →  2 runs  (s1993 already done)"
echo -e "   • Tiny-ImageNet 100-20 → 3 runs  (all seeds new)"
echo -e ""

log_progress ">> Queue started  (total: $EXP_TOTAL)"

# ─────────────────────────────────────────────────────────────────────────────
# Tiny-ImageNet 20-20  ·  seed 1995  (interrupted + tail)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── Tiny-ImageNet 20-20  (seed 1995 — restart + tail) ────────────${NC}"

# [16/27 original] — was interrupted, restarting
queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-20-20] β=0 aLocal nLocal" 1995

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-20-20] β=0.5 aLocal nLocal" 1995

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antSymmetricFull_debug_resnet18.yaml" \
    "[TinyIN-20-20] β=0.5 aSymFull nLocal" 1995

# ─────────────────────────────────────────────────────────────────────────────
# Tiny-ImageNet 100-20  ·  seeds 1993, 1994, 1995  (ALL NEW)
# 100 base classes + 5 incremental sessions of 20 = 6 tasks total
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}── Tiny-ImageNet 100-20  (seeds 1993, 1994, 1995 — ALL NEW) ─────${NC}"

for seed in 1993 1994 1995; do
    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0 aLocal nLocal" $seed

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0.5 aLocal nLocal" $seed

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0.5_nceA1_antM0.5_antSymmetricFull_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0.5 aSymFull nLocal" $seed
done

# ─────────────────────────────────────────────────────────────────────────────
# TRUE BASELINE: β=0  nGlobal  (ant_max_global=true, infonce_max_global=true)
# seed 1993 already done for C100-10-10, C100-50-10, TinyIN-20-20
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

echo -e "\n${CYAN}── True Baseline β=0 nGlobal  TinyIN-100-20  (seeds 1993, 1994, 1995) ──${NC}"

for seed in 1993 1994 1995; do
    queue_experiment \
        "configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
        "[TinyIN-100-20] β=0 nGlobal" $seed
done

# ─────────────────────────────────────────────────────────────────────────────
log_progress "[OK] Queue finished!  ($EXP_TOTAL/$EXP_TOTAL)"
echo -e "\n${GREEN}═════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] All ${EXP_TOTAL} experiments completed!${NC}"
echo -e "${GREEN}═════════════════════════════════════════════════════════════════${NC}\n"
echo -e "  Progress log : ${PROGRESS_LOG}"
echo -e "  Console log  : ${CONSOLE_LOG}"
echo -e "  Results log  : /mnt/raid/home/tiago/logs/results_report.md"
echo -e "  (re-run generate_report.py to refresh the report)"
} # end main

echo "Console log: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
