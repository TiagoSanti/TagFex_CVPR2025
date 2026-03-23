#!/bin/bash
# Fila de experimentos de debug — wolverine (2× RTX 3080 Ti 12 GB)
#
# Objetivo: rodar os configs com debug_similarity=true para gerar logs reais de
# matrizes de similaridade, utilizados no Similarity Viewer.
#
# GPUs disponíveis:
#   [0] RTX 3080 Ti 12 GB  — pode estar parcialmente ocupada (root ~3 GB)
#   [1] RTX 3080 Ti 12 GB  — idle por padrão
#
# Experimentos (em paralelo, um por GPU):
#   GPU 0 — CIFAR-100 10-10 ANT β=0.5 m=0.5 Local (debug)   seed 1993
#   GPU 1 — CIFAR-100 10-10 Baseline Global       (debug)   seed 1993
#
# Uso:
#   screen -dmS debug_wolverine bash run_debug_wolverine.sh
#   screen -r debug_wolverine
#
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Parâmetros de GPU (wolverine — RTX 3080 Ti 12 GB) ────────────────────────
# CIFAR-100 batch_size=128 + ResNet18 + proj head → ~4-5 GB VRAM
# GPU 0 tem ~3 GB ocupados (root); pedimos 7 GB livres → seguro nos 12 GB
INTERVAL=30
THRESHOLD=100.0
MIN_FREE_MB=7000

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
# Lane A — GPU 1 (idle)    Baseline Local
# Lane B — GPU 0 (parcial) ANT β=0.5 m=0.5 Local
# As duas lanes correm em paralelo (&) e o script aguarda ambas.
# ═════════════════════════════════════════════════════════════════════════════

lane_baseline() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane A — GPU 1 — Baseline Global [debug]           ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane A] Iniciando — Baseline Global (GPU 1)"

    queue_experiment \
        "configs/all_in_one/cifar100_10-10_baseline_global_debug_resnet18.yaml" \
        "CIFAR-100 10-10 Baseline Global [debug]" \
        1993 \
        "debug_exp_cifar100_10-10_antB0_nceA1_antGlobal_s1993" \
        1

    log_progress "[OK] [Lane A] Baseline Global concluída"
    echo -e "${GREEN}[OK] [Lane A] Baseline Global concluída!${NC}\n"
}

lane_ant() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane B — GPU 0 — ANT β=0.5 m=0.5 Local [debug]    ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane B] Iniciando — ANT β=0.5 (GPU 0)"

    queue_experiment \
        "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_debug_resnet18.yaml" \
        "CIFAR-100 10-10 ANT β=0.5 m=0.5 Local [debug]" \
        1993 \
        "debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993" \
        0


    log_progress "[OK] [Lane B] ANT β=0.5 concluída"
    echo -e "${GREEN}[OK] [Lane B] ANT β=0.5 concluída!${NC}\n"
}

# ─────────────────────────────────────────────────────────────────────────────
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Debug Queue — wolverine (2× RTX 3080 Ti 12 GB)     ║${NC}"
echo -e "${CYAN}║  Lane A → GPU 1 | Lane B → GPU 0  (paralelas)       ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"
log_progress "=== Iniciando debug queue (wolverine, GPUs 0+1 paralelo) ==="

lane_baseline &
LANE_A_PID=$!
lane_ant &
LANE_B_PID=$!

echo -e "${CYAN}Lanes A e B em paralelo (PID A=${LANE_A_PID}, B=${LANE_B_PID})${NC}\n"
wait $LANE_A_PID; A_STATUS=$?
wait $LANE_B_PID; B_STATUS=$?

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
if [ $A_STATUS -eq 0 ] && [ $B_STATUS -eq 0 ]; then
    echo -e "${GREEN}[OK] wolverine — ambas as lanes concluídas! ✅${NC}"
    log_progress "[FIM] debug queue concluída (A=${A_STATUS}, B=${B_STATUS})"
else
    echo -e "${RED}[ERRO] uma ou mais lanes falharam (A=${A_STATUS}, B=${B_STATUS})${NC}"
    log_progress "[ERRO] debug queue com falha (A=${A_STATUS}, B=${B_STATUS})"
    exit 1
fi
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
