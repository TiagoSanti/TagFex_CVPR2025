#!/bin/bash
# Fila de experimentos debug — 3 datasets × 6 combinações de âncora = 18 experimentos
#
# Datasets:
#   - CIFAR-100 10-10
#   - CIFAR-100 50-10
#   - Tiny ImageNet 20-20
#
# Variantes (por dataset):
#   1. Baseline: ANT off, InfoNCE global
#   2. Baseline: ANT off, InfoNCE local
#   3. ANT β=0.5 m=0.5: âncora global para ambos
#   4. ANT β=0.5 m=0.5: ANT local + InfoNCE global
#   5. ANT β=0.5 m=0.5: ANT global + InfoNCE local
#   6. ANT β=0.5 m=0.5: âncora local para ambos
#
# Após cada experimento, os logs de debug (exp_debug0.log, similarity_debug.log,
# similarity_heatmaps/) são compactados em debug_logs.zip e apagados.
#
# Uso:
#   screen -dmS debug_queue ./run_queue_debug_anchors.sh
#   screen -r debug_queue
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Ambiente virtual ──
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Parâmetros de GPU ──
GPUS=1
MEMORY_THRESHOLD="--memory-threshold 10.0"
THRESHOLD=100.0
INTERVAL=30
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# ── Logs ──
LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/debug_anchors_progress.log"
CONSOLE_LOG="$LOG_DIR/debug_anchors_console_$(date +%Y%m%d_%H%M%S).log"
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
EXP_TOTAL=18

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
        compress_debug_logs
    else
        log_progress "[ERRO] $pos ERRO: $description  [seed=$seed]"
        echo -e "${RED}[ERRO] $pos Erro no experimento — abortando fila${NC}\n"
        exit 1
    fi

    sleep 5
}

# Compacta os logs de debug de todos os diretórios debug_exp_* que ainda não foram
# compactados, apagando os ficheiros originais em caso de sucesso.
compress_debug_logs() {
    local log_base="$SCRIPT_DIR/logs"
    for exp_dir in "$log_base"/debug_exp_*/; do
        [ -d "$exp_dir" ] || continue
        local zip_file="${exp_dir}debug_logs.zip"
        [ -f "$zip_file" ] && continue  # já compactado

        local files=()
        [ -f "${exp_dir}exp_debug0.log"      ] && files+=("exp_debug0.log")
        [ -f "${exp_dir}similarity_debug.log" ] && files+=("similarity_debug.log")
        [ -d "${exp_dir}similarity_heatmaps" ] && files+=("similarity_heatmaps")

        [ ${#files[@]} -eq 0 ] && continue  # nada para compactar

        echo -e "  ${BLUE}[zip]${NC} Compactando logs de debug: $(basename "$exp_dir")"
        if (cd "$exp_dir" && zip -r "debug_logs.zip" "${files[@]}" -q); then
            rm -f  "${exp_dir}exp_debug0.log"
            rm -f  "${exp_dir}similarity_debug.log"
            rm -rf "${exp_dir}similarity_heatmaps"
            echo -e "  ${GREEN}[zip OK]${NC} $(basename "$exp_dir")/debug_logs.zip"
            log_progress "[zip] Logs de debug compactados: $(basename "$exp_dir")"
        else
            echo -e "  ${RED}[zip ERRO]${NC} Falha ao compactar $(basename "$exp_dir") — ficheiros mantidos"
            log_progress "[zip ERRO] Falha ao compactar: $(basename "$exp_dir")"
        fi
    done
}

# ═══════════════════════════════════════════════════════════
main() {
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Debug Queue — 3 datasets × 6 âncoras = 18 experimentos  ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Total: ${EXP_TOTAL} experimentos (seed=1993, debug_similarity=true)${NC}"
echo -e "${BLUE} Datasets: CIFAR-100 10-10 · CIFAR-100 50-10 · Tiny ImageNet 20-20${NC}"
echo -e "${BLUE} Variantes por dataset:${NC}"
echo -e "   1 : Baseline ANT off + InfoNCE global"
echo -e "   2 : Baseline ANT off + InfoNCE local"
echo -e "   3 : ANT β=0.5 m=0.5 — global/global"
echo -e "   4 : ANT β=0.5 m=0.5 — ANT local / InfoNCE global"
echo -e "   5 : ANT β=0.5 m=0.5 — ANT global / InfoNCE local"
echo -e "   6 : ANT β=0.5 m=0.5 — local/local"
echo -e ""

log_progress ">> Iniciando debug_anchors_queue  (total: $EXP_TOTAL)"

# ═══════════════════════════════════════════════════════════
# CIFAR-100 10-10
# ═══════════════════════════════════════════════════════════
echo -e "\n${BLUE}── CIFAR-100 10-10 ──────────────────────────────────────────${NC}"

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[C100-10-10] Baseline ANT off + InfoNCE global" 1993

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[C100-10-10] Baseline ANT off + InfoNCE local" 1993

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[C100-10-10] ANT β=0.5 m=0.5 — antGlobal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceGlobal_debug_resnet18.yaml" \
    "[C100-10-10] ANT β=0.5 m=0.5 — antLocal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antGlobal_nceLocal_debug_resnet18.yaml" \
    "[C100-10-10] ANT β=0.5 m=0.5 — antGlobal/nceLocal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
    "[C100-10-10] ANT β=0.5 m=0.5 — antLocal/nceLocal" 1993

# ═══════════════════════════════════════════════════════════
# CIFAR-100 50-10
# ═══════════════════════════════════════════════════════════
echo -e "\n${BLUE}── CIFAR-100 50-10 ──────────────────────────────────────────${NC}"

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[C100-50-10] Baseline ANT off + InfoNCE global" 1993

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[C100-50-10] Baseline ANT off + InfoNCE local" 1993

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0.5_nceA1_antM0.5_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[C100-50-10] ANT β=0.5 m=0.5 — antGlobal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0.5_nceA1_antM0.5_antLocal_nceGlobal_debug_resnet18.yaml" \
    "[C100-50-10] ANT β=0.5 m=0.5 — antLocal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0.5_nceA1_antM0.5_antGlobal_nceLocal_debug_resnet18.yaml" \
    "[C100-50-10] ANT β=0.5 m=0.5 — antGlobal/nceLocal" 1993

queue_experiment \
    "configs/all_in_one/cifar100_50-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
    "[C100-50-10] ANT β=0.5 m=0.5 — antLocal/nceLocal" 1993

# ═══════════════════════════════════════════════════════════
# Tiny ImageNet 20-20
# ═══════════════════════════════════════════════════════════
echo -e "\n${BLUE}── Tiny ImageNet 20-20 ──────────────────────────────────────${NC}"

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[TinyIN-20-20] Baseline ANT off + InfoNCE global" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0_nceA1_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-20-20] Baseline ANT off + InfoNCE local" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antGlobal_nceGlobal_debug_resnet18.yaml" \
    "[TinyIN-20-20] ANT β=0.5 m=0.5 — antGlobal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antLocal_nceGlobal_debug_resnet18.yaml" \
    "[TinyIN-20-20] ANT β=0.5 m=0.5 — antLocal/nceGlobal" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antGlobal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-20-20] ANT β=0.5 m=0.5 — antGlobal/nceLocal" 1993

queue_experiment \
    "configs/all_in_one/tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antLocal_nceLocal_debug_resnet18.yaml" \
    "[TinyIN-20-20] ANT β=0.5 m=0.5 — antLocal/nceLocal" 1993

# ═══════════════════════════════════════════════════════════
log_progress "[OK] debug_anchors_queue concluída! ($EXP_TOTAL/$EXP_TOTAL)"
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}[OK] Todos os ${EXP_TOTAL} experimentos concluídos!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e " Log de progresso : ${PROGRESS_LOG}"
echo -e " Log completo     : ${CONSOLE_LOG}"
} # fim main

echo " Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
