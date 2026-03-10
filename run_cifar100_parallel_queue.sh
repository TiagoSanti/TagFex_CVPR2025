#!/bin/bash
# Fila paralela de experimentos CIFAR-100 — roda em paralelo com run_experiments_queue.sh
#
# Usa --min-free-mb 8000 em vez de um threshold percentual, de modo que a GPU
# seja considerada disponível se houver pelo menos 8 GB livres — mesmo que outro
# experimento (Tiny ImageNet ~6.5 GB) já esteja rodando.
#
# Uso recomendado (em sessão screen separada):
#   screen -dmS cifar100_queue ./run_cifar100_parallel_queue.sh
#   screen -r cifar100_queue
#
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# Ativar ambiente virtual se existir
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Parâmetros de GPU (quati: RTX 4090 24GB) ──
GPUS=1
INTERVAL=30
# Exige >= 8 GB livres antes de iniciar cada experimento
# CIFAR-100 ResNet-18 usa ~5 GB de pico → 8 GB garante margem segura
# Tiny ImageNet usa ~6.5 GB → 24.5 - 6.5 = 18 GB livres → OK para CIFAR-100
MIN_FREE_MB=8000
# threshold de utilização desabilitado (modo memória livre absoluta)
THRESHOLD=100.0

# ── Logs ──
LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/cifar100_queue_progress.log"
mkdir -p "$LOG_DIR"

# Cores
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

# ── Contadores ──
EXP_COUNTER=0
# CIFAR-100 10-10: baseline 4 seeds + ANT 4 seeds = 8
# CIFAR-100 50-10: baseline 4 seeds + ANT 5 seeds = 9
# Total: 17
EXP_TOTAL=17

queue_experiment() {
    local config_file=$1
    local description=$2
    local seed=${3:-1993}

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$EXP_TOTAL]"

    log_progress ">> $pos Iniciando: $description  [seed=$seed]"
    echo -e "${YELLOW}>> $pos Iniciando:${NC} $description  [seed=$seed]"
    echo -e "   Config: $config_file"
    echo -e "   Aguardando >= ${MIN_FREE_MB} MB livres na GPU...\n"

    local train_cmd="python3 main.py train --exp-configs $config_file --seed $seed"

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus "$GPUS" \
        --threshold "$THRESHOLD" \
        --min-free-mb "$MIN_FREE_MB" \
        --interval "$INTERVAL" \
        --no-screen

    if [ $? -eq 0 ]; then
        log_progress "[OK] $pos Concluído: $description  [seed=$seed]"
        echo -e "${GREEN}[OK] $pos Concluído! (${EXP_COUNTER}/${EXP_TOTAL} feitos)${NC}\n"
    else
        log_progress "[ERRO] $pos ERRO: $description  [seed=$seed]"
        echo -e "${RED}[ERRO] $pos Erro no experimento${NC}\n"
        return 1
    fi

    sleep 5
}

queue_experiment_remaining_seeds() {
    local config_file=$1
    local description=$2
    for seed in 1994 1995 1996 1997; do
        queue_experiment "$config_file" "$description" "$seed"
    done
}

queue_experiment_5seeds() {
    local config_file=$1
    local description=$2
    for seed in 1993 1994 1995 1996 1997; do
        queue_experiment "$config_file" "$description" "$seed"
    done
}

# ═══════════════════════════════════════════════════════════
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    CIFAR-100 Parallel Queue (quati · paralelo com TinyImageNet)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Total: ${EXP_TOTAL} experimentos${NC}"
echo -e "${BLUE} Requisito: >= ${MIN_FREE_MB} MB livres na GPU antes de cada run${NC}\n"
echo -e "${BLUE}Ordem de execução:${NC}"
echo -e "   1-4  : CIFAR-100 10-10 Baseline Local  — seeds 1994, 1995, 1996, 1997"
echo -e "   5-8  : CIFAR-100 10-10 ANT β=0.5 m=0.5 — seeds 1994, 1995, 1996, 1997"
echo -e "   9-12 : CIFAR-100 50-10 Baseline Local  — seeds 1994, 1995, 1996, 1997"
echo -e "   13-17: CIFAR-100 50-10 ANT β=0.5 m=0.5 — seeds 1993, 1994, 1995, 1996, 1997"
echo -e ""
log_progress ">> Iniciando fila CIFAR-100 paralela  (total: $EXP_TOTAL)"

# ── CIFAR-100 10-10 ── seed 1993 já executado → apenas 1994-1997
echo -e "${YELLOW}═══ CIFAR-100 10-10 (seeds 1994-1997) ═══${NC}\n"

queue_experiment_remaining_seeds \
    "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml" \
    "CIFAR-100 10-10 Baseline Local"

queue_experiment_remaining_seeds \
    "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
    "CIFAR-100 10-10 ANT β=0.5 m=0.5 Local"

# ── CIFAR-100 50-10 ── baseline seed 1993 já executado; ANT best nunca executado
echo -e "${YELLOW}═══ CIFAR-100 50-10 ═══${NC}\n"

queue_experiment_remaining_seeds \
    "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
    "CIFAR-100 50-10 Baseline Local"

# ANT 50-10: seed 1993 nunca foi executado → 5 seeds completos
queue_experiment_5seeds \
    "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
    "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

# ═══════════════════════════════════════════════════════════
log_progress "[OK] Fila CIFAR-100 concluída! ($EXP_TOTAL/$EXP_TOTAL)"
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] Todos os ${EXP_TOTAL} experimentos CIFAR-100 concluídos!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e " Log de progresso: ${PROGRESS_LOG}"
