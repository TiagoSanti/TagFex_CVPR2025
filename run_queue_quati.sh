#!/bin/bash
# Fila consolidada de experimentos — Quati (RTX 4090 · 24 GB · 1 GPU)
#
# Experimentos pendentes (22 mar 2026):
#   CIFAR-100 50-10 Baseline Local  — seeds 1994 1995 1996 1997
#   CIFAR-100 50-10 ANT β=0.5 m=0.5 Local — seeds 1993 1994 1995 1996 1997
#   Total: 9 experimentos sequenciais
#
# Uso (lança em screen para persistir após desconexão SSH):
#   screen -dmS quati_queue ./run_queue_quati.sh
#   screen -r quati_queue    # para anexar e acompanhar
#
# Para monitorar sem anexar:
#   tail -f logs/auto_experiments/quati_queue_console_*.log
#   tail -f logs/auto_experiments/quati_queue_progress.log
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Ambiente virtual ──
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Parâmetros de GPU (RTX 4090 · 24 GB · 1 GPU) ──
# VRAM idle ~43 MiB; threshold 10% ≈ 2.4 GB de margem.
GPUS=1
MEMORY_THRESHOLD="--memory-threshold 10.0"
THRESHOLD=100.0
INTERVAL=30
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# ── Logs ──
LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/quati_queue_progress.log"
CONSOLE_LOG="$LOG_DIR/quati_queue_console_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# ── Cores ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_progress() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $1" | tee -a "$PROGRESS_LOG"
}

EXP_COUNTER=0
EXP_TOTAL=9

queue_experiment() {
    local config_file=$1
    local description=$2
    local seed=${3:-1993}

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$EXP_TOTAL]"

    log_progress ">> $pos Iniciando: $description  [seed=$seed]"
    echo -e "${YELLOW}>> $pos Iniciando:${NC} $description  [seed=$seed]"
    echo -e "   Config: $config_file"
    echo -e "   Aguardando GPU disponível (mem < 10%)...\n"

    local train_cmd="python3 main.py train --exp-configs $config_file --seed $seed"

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus "$GPUS" \
        --threshold "$THRESHOLD" \
        $MEMORY_THRESHOLD \
        --interval "$INTERVAL" \
        --no-screen

    if [ $? -eq 0 ]; then
        log_progress "[OK] $pos Concluído: $description  [seed=$seed]"
        echo -e "${GREEN}[OK] $pos Concluído! (${EXP_COUNTER}/${EXP_TOTAL} feitos)${NC}\n"
    else
        log_progress "[ERRO] $pos ERRO: $description  [seed=$seed]"
        echo -e "${RED}[ERRO] $pos Erro no experimento — abortando fila${NC}\n"
        exit 1
    fi

    sleep 5
}

# ═══════════════════════════════════════════════════════════
main() {
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    Quati Queue — RTX 4090 24GB · CIFAR-100 50-10${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Total: ${EXP_TOTAL} experimentos${NC}"
echo -e "${BLUE} Ordem de execução:${NC}"
echo -e "   1-4 : CIFAR-100 50-10 Baseline Local  — seeds 1994 1995 1996 1997"
echo -e "   5-9 : CIFAR-100 50-10 ANT β=0.5 m=0.5 Local — seeds 1993 1994 1995 1996 1997"
echo -e ""

log_progress ">> Iniciando quati_queue  (total: $EXP_TOTAL)"

# ── CIFAR-100 50-10 Baseline Local — seeds 1994–1997 ──
echo -e "${YELLOW}═══ CIFAR-100 50-10 Baseline Local ═══${NC}\n"

for seed in 1994 1995 1996 1997; do
    queue_experiment \
        "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
        "CIFAR-100 50-10 Baseline Local" \
        $seed
done

# ── CIFAR-100 50-10 ANT β=0.5 m=0.5 Local — seeds 1993–1997 ──
echo -e "${YELLOW}═══ CIFAR-100 50-10 ANT β=0.5 m=0.5 Local ═══${NC}\n"

for seed in 1993 1994 1995 1996 1997; do
    queue_experiment \
        "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local" \
        $seed
done

# ═══════════════════════════════════════════════════════════
log_progress "[OK] quati_queue concluída! ($EXP_TOTAL/$EXP_TOTAL)"
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] Todos os ${EXP_TOTAL} experimentos concluídos!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e " Log de progresso : ${PROGRESS_LOG}"
echo -e " Log completo     : ${CONSOLE_LOG}"
} # fim main

echo " Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
