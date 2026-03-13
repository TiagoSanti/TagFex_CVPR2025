#!/bin/bash
# Fila de experimentos de debug — wolverine (2× RTX 3080 Ti 12 GB)
#
# Objetivo: rodar os configs com debug_similarity=true para gerar logs reais de
# matrizes de similaridade, utilizados no Similarity Viewer.
#
# GPUs disponíveis:
#   [0] RTX 3080 Ti 12 GB  — pode estar parcialmente ocupada (root ~3 GB)
#   [1] RTX 3080 Ti 12 GB  — idle por padrão → usar aqui preferencialmente
#
# Experimentos:
#   1. CIFAR-100 10-10 Baseline Local  (ant_beta=0, debug)   seed 1993   GPU 1
#   2. CIFAR-100 10-10 ANT β=0.5 m=0.5 Local (debug)        seed 1993   GPU 1
#
# Uso:
#   screen -dmS debug_wolverine bash run_debug_wolverine.sh
#   screen -r debug_wolverine
#
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Parâmetros de GPU (wolverine — RTX 3080 Ti 12 GB) ────────────────────────
# CIFAR-100 batch_size=128 + ResNet18 + proj head → ~4-5 GB VRAM
# Pedimos 7 GB livres para segurança confortável dentro dos 12 GB
INTERVAL=30
THRESHOLD=100.0
MIN_FREE_MB=7000

# GPU alvo: 1 (idle). Muda para "0 1" para usar ambas em experimentos paralelos
# rodados em terminais/screens separados.
TARGET_GPU=1

# ── Logs ──────────────────────────────────────────────────────────────────────
LOG_DIR="$LOGS_DIR/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress_debug.log"
mkdir -p "$LOG_DIR"

# Cores
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$PROGRESS_LOG"
}

is_done() {
    [ -d "$LOGS_DIR/$1" ]
}

# queue_experiment <config> <desc> <seed> <done_dir_or_empty> <gpu_id>
queue_experiment() {
    local config=$1 desc=$2 seed=$3 done_dir=${4:-} gpu=$5

    if [ -n "$done_dir" ] && is_done "$done_dir"; then
        log_progress "[SKIP] Já concluído: ${desc} [seed=${seed}]"
        echo -e "  ${BLUE}[SKIP]${NC} Já concluído: ${desc} [seed=${seed}]"
        return 0
    fi

    log_progress ">> Iniciando: ${desc} [seed=${seed}] GPU=${gpu}"
    echo -e "${YELLOW}>> Iniciando:${NC} ${desc} [seed=${seed}]"
    echo -e "   GPU alvo: ${gpu} | Aguardando >= ${MIN_FREE_MB} MB livres...\n"

    python3 "$AUTO_LAUNCHER" \
        --command "python3 main.py train --exp-configs $config --seed $seed" \
        --gpus 1 \
        --threshold "$THRESHOLD" \
        --min-free-mb "$MIN_FREE_MB" \
        --interval "$INTERVAL" \
        --no-screen \
        --allowed-gpu-ids "$gpu"

    if [ $? -eq 0 ]; then
        log_progress "[OK] Concluído: ${desc} [seed=${seed}]"
        echo -e "${GREEN}[OK] Concluído!${NC} ${desc} [seed=${seed}]\n"
    else
        log_progress "[ERRO] Falhou: ${desc} [seed=${seed}]"
        echo -e "${RED}[ERRO] Falhou!${NC} ${desc} [seed=${seed}]\n"
        return 1
    fi

    sleep 5
}

# ═════════════════════════════════════════════════════════════════════════════
# Cabeçalho
# ═════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Debug Queue — wolverine (2× RTX 3080 Ti 12 GB)     ║${NC}"
echo -e "${CYAN}║  GPU alvo: ${TARGET_GPU}  |  Min free: ${MIN_FREE_MB} MB          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"
log_progress "=== Iniciando debug queue (wolverine, GPU ${TARGET_GPU}) ==="

# ─────────────────────────────────────────────────────────────────────────────
# 1. CIFAR-100 10-10 Baseline Local (debug) — seed 1993
#    done_dir: debug_exp_cifar100_10-10_antB0_nceA1_antLocal_s1993
# ─────────────────────────────────────────────────────────────────────────────
queue_experiment \
    "configs/all_in_one/cifar100_10-10_baseline_local_debug_resnet18.yaml" \
    "CIFAR-100 10-10 Baseline Local [debug]" \
    1993 \
    "debug_exp_cifar100_10-10_antB0_nceA1_antLocal_s1993" \
    "$TARGET_GPU"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CIFAR-100 10-10 ANT β=0.5 m=0.5 Local (debug) — seed 1993
#    done_dir: debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993
# ─────────────────────────────────────────────────────────────────────────────
queue_experiment \
    "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_debug_resnet18.yaml" \
    "CIFAR-100 10-10 ANT β=0.5 m=0.5 Local [debug]" \
    1993 \
    "debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993" \
    "$TARGET_GPU"

# ═════════════════════════════════════════════════════════════════════════════
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Debug queue concluída! ✅                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
log_progress "=== Debug queue concluída ==="
