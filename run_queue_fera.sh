#!/bin/bash
# Fila de experimentos ImageNet-100 — fera (2× RTX A6000 49GB)
#
# Lane A — ImageNet-100 10-10   seeds 1993/1995/1996  GPU 0  >= 25 GB livres
# Lane B — ImageNet-100 50-10   seeds 1993/1995/1996  GPU 1  >= 25 GB livres
#
# Experimentos por lane/seed:
#   1. Baseline           (baseline_global: TagFex padrão, sem âncora local)
#   2. Baseline + local   (baseline_local:  TagFex com âncora local, β=0)
#   3. ANT β=0.5 m=0.5    (ant_beta0.5_margin0.5_local: melhores parâmetros)
#
# Skip automático: experimentos com diretório done_exp_* já existente são pulados.
# GPU pinning: Lane A usa GPU 0, Lane B usa GPU 1 (via --allowed-gpu-ids).
#
# Uso:
#   screen -dmS tagfex_fera ./run_queue_fera.sh
#
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Seeds ────────────────────────────────────────────────────────────────────
# Trio {1993, 1995, 1996}: soma L2 pairwise = 5.96 (máxima incluindo s1993)
SEEDS_ALL=(1993 1995 1996)
SEEDS_DEFERRED=(1994 1997)

# ── Parâmetros de GPU (fera — RTX A6000 49GB) ────────────────────────────────
INTERVAL=30
THRESHOLD=100.0
MIN_FREE_MB_A=25000   # ImageNet-100 ~15 GB → 25 GB margem GPU 0
MIN_FREE_MB_B=25000   # ImageNet-100 ~15 GB → 25 GB margem GPU 1

# ── Logs ────────────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_fera_progress.log"
CONSOLE_LOG="$LOG_DIR/queue_fera_console_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Cores
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$PROGRESS_LOG"
}

# Verifica se experimento já foi concluído (diretório done_exp_* existe)
is_done() {
    local done_dir="$1"
    [ -d "$LOGS_DIR/$done_dir" ]
}

# queue_experiment <config> <description> <seed> <min_free_mb> <lane> [done_dir] [gpu_ids...]
# gpu_ids: IDs de GPU a pinar (ex: 0 ou 1); omitir = todas as GPUs
queue_experiment() {
    local config=$1 desc=$2 seed=$3 min_free=$4 lane=$5 done_dir=${6:-}
    shift 6
    local gpu_ids=("$@")

    if [ -n "$done_dir" ] && is_done "$done_dir"; then
        log_progress "[SKIP] [Lane ${lane}] Já concluído: ${desc} [seed=${seed}]"
        echo -e "  ${BLUE}[SKIP]${NC} Já concluído: ${desc} [seed=${seed}]"
        return 0
    fi

    log_progress ">> [Lane ${lane}] Iniciando: ${desc} [seed=${seed}]"
    echo -e "${YELLOW}>> [Lane ${lane}] Iniciando:${NC} ${desc} [seed=${seed}]"
    echo -e "   Aguardando >= ${min_free} MB livres na GPU...\n"

    local gpu_flag=()
    if [ ${#gpu_ids[@]} -gt 0 ]; then
        gpu_flag=(--allowed-gpu-ids "${gpu_ids[@]}")
    fi

    python3 "$AUTO_LAUNCHER" \
        --command "python3 main.py train --exp-configs $config --seed $seed" \
        --gpus 1 \
        --threshold "$THRESHOLD" \
        --min-free-mb "$min_free" \
        --interval "$INTERVAL" \
        --no-screen \
        "${gpu_flag[@]}"

    if [ $? -eq 0 ]; then
        log_progress "[OK] [Lane ${lane}] Concluído: ${desc} [seed=${seed}]"
        echo -e "${GREEN}[OK] [Lane ${lane}] Concluído!${NC} ${desc} [seed=${seed}]\n"
    else
        log_progress "[ERRO] [Lane ${lane}] ERRO: ${desc} [seed=${seed}]"
        echo -e "${RED}[ERRO] [Lane ${lane}] Erro!${NC} ${desc} [seed=${seed}]\n"
        return 1
    fi

    sleep 5
}

# ═══════════════════════════════════════════════════════════════════════════
# Lane A — ImageNet-100 10-10  ·  seeds 1993/1995/1996  ·  GPU 0
# ═══════════════════════════════════════════════════════════════════════════
lane_imagenet100_10() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane A — ImageNet-100 10-10 (seeds ${SEEDS_ALL[*]})  GPU 0  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane A] Iniciando — ImageNet-100 10-10 (GPU 0)"

    for seed in "${SEEDS_ALL[@]}"; do
        echo -e "${YELLOW}  ── Seed ${seed} (ImageNet-100 10-10) ──────────────────────${NC}\n"

        # 1. Baseline (sem âncora local)
        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_baseline_global_resnet18.yaml" \
            "ImageNet-100 10-10 Baseline" "$seed" "$MIN_FREE_MB_A" "A" \
            "done_exp_imagenet100_10-10_antB0_nceA1_antGlobal_s${seed}" \
            0

        # 2. Baseline com âncora local (β=0)
        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml" \
            "ImageNet-100 10-10 Baseline Local" "$seed" "$MIN_FREE_MB_A" "A" \
            "done_exp_imagenet100_10-10_antB0_nceA1_antLocal_s${seed}" \
            0

        # 3. ANT melhores parâmetros (β=0.5, m=0.5, local)
        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "ImageNet-100 10-10 ANT β=0.5 m=0.5 Local" "$seed" "$MIN_FREE_MB_A" "A" \
            "done_exp_imagenet100_10-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}" \
            0
    done

    log_progress "[OK] [Lane A] ImageNet-100 10-10 — concluída"
    echo -e "${GREEN}[OK] Lane A — ImageNet-100 10-10 concluída!${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# Lane B — ImageNet-100 50-10  ·  seeds 1993/1995/1996  ·  GPU 1
# ═══════════════════════════════════════════════════════════════════════════
lane_imagenet100_50() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane B — ImageNet-100 50-10 (seeds ${SEEDS_ALL[*]})  GPU 1  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane B] Iniciando — ImageNet-100 50-10 (GPU 1)"

    for seed in "${SEEDS_ALL[@]}"; do
        echo -e "${YELLOW}  ── Seed ${seed} (ImageNet-100 50-10) ──────────────────────${NC}\n"

        # 1. Baseline (sem âncora local)
        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_baseline_global_resnet18.yaml" \
            "ImageNet-100 50-10 Baseline" "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_imagenet100_50-10_antB0_nceA1_antGlobal_s${seed}" \
            1

        # 2. Baseline com âncora local (β=0)
        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_baseline_local_resnet18.yaml" \
            "ImageNet-100 50-10 Baseline Local" "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_imagenet100_50-10_antB0_nceA1_antLocal_s${seed}" \
            1

        # 3. ANT melhores parâmetros (β=0.5, m=0.5, local)
        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "ImageNet-100 50-10 ANT β=0.5 m=0.5 Local" "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_imagenet100_50-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}" \
            1
    done

    log_progress "[OK] [Lane B] ImageNet-100 50-10 — concluída"
    echo -e "${GREEN}[OK] Lane B — ImageNet-100 50-10 concluída!${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
main() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}    TagFex Queue — fera (2× RTX A6000 49GB)${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  ${BLUE}Lane A:${NC} ImageNet-100 10-10  seeds ${SEEDS_ALL[*]}  GPU 0  >= ${MIN_FREE_MB_A} MB"
    echo -e "  ${BLUE}Lane B:${NC} ImageNet-100 50-10  seeds ${SEEDS_ALL[*]}  GPU 1  >= ${MIN_FREE_MB_B} MB"
    echo -e "  ${BLUE}Seeds diferidas:${NC} ${SEEDS_DEFERRED[*]}"
    echo -e "  Experimentos com diretório done_exp_* são automaticamente pulados.\n"
    log_progress ">> Fila iniciada (fera)"

    lane_imagenet100_10 &
    LANE_A_PID=$!
    lane_imagenet100_50 &
    LANE_B_PID=$!

    echo -e "${CYAN}Lanes A e B em paralelo (PID A=${LANE_A_PID}, B=${LANE_B_PID})${NC}\n"
    wait $LANE_A_PID; A_STATUS=$?
    wait $LANE_B_PID; B_STATUS=$?

    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
    if [ $A_STATUS -eq 0 ] && [ $B_STATUS -eq 0 ]; then
        echo -e "${GREEN}[OK] fera — todas as lanes concluídas!${NC}"
        log_progress "[FIM] fera concluída (A=${A_STATUS}, B=${B_STATUS})"
    else
        echo -e "${RED}[AVISO] Erros: Lane A=${A_STATUS}  Lane B=${B_STATUS}${NC}"
        log_progress "[FIM] fera com erros (A=${A_STATUS}, B=${B_STATUS})"
    fi
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  Log de progresso : ${PROGRESS_LOG}"
    echo -e "  Log completo     : ${CONSOLE_LOG}"
}

# Executar e capturar toda saída em arquivo de log
echo " Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
