#!/bin/bash
# Fila unificada — Lane A (Tiny ImageNet) e Lane B (CIFAR-100) em paralelo
#
# Ambas as lanes rodam simultaneamente na mesma GPU (RTX 4090 24GB):
#   Lane A — Tiny ImageNet (~6.5 GB VRAM)  · aguarda >= 16 GB livres
#   Lane B — CIFAR-100    (~5.0 GB VRAM)   · aguarda >=  8 GB livres
#   Pico combinado: ~11.5 GB de 24 GB
#
# Skip automático: experimentos com diretório done_exp_* já existente são pulados.
#
# Uso:
#   screen -dmS tagfex_queue ./run_queue.sh
#   screen -r tagfex_queue
#
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Seeds selecionadas por análise de diversidade L2 ────────────────────────
# Trio {1993, 1995, 1996}: soma L2 pairwise = 5.96 (máxima incluindo s1993)
# Diferidas: 1994 (próxima a 1993, L2=2.02) e 1997 (próxima a 1996, L2=1.26)
SEEDS_ACTIVE=(1995 1996)          # seeds adicionais além de 1993
SEEDS_ALL=(1993 1995 1996)        # todas as seeds ativas
SEEDS_DEFERRED=(1994 1997)        # diferidas para validação futura

# ── Parâmetros de GPU (quati — RTX 4090 24GB) ───────────────────────────────
INTERVAL=30
THRESHOLD=100.0       # desabilitado (só verifica memória livre)
MIN_FREE_MB_A=16000   # Lane A: Tiny ImageNet ~6.5 GB → exige 16 GB livres
MIN_FREE_MB_B=8000    # Lane B: CIFAR-100     ~5.0 GB → exige  8 GB livres

# ── Logs ────────────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress.log"
CONSOLE_LOG="$LOG_DIR/queue_console_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Cores
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$PROGRESS_LOG"
}

# Verifica se experimento já foi concluído (diretório done_exp_* existe)
# Aceita o nome exato do diretório ou uma string de busca (glob parcial)
is_done() {
    local done_dir="$1"
    [ -d "$LOGS_DIR/$done_dir" ]
}

# queue_experiment <config> <description> <seed> <min_free_mb> <lane> [done_dir]
# Se done_dir for passado e o diretório existir, o experimento é pulado.
queue_experiment() {
    local config=$1 desc=$2 seed=$3 min_free=$4 lane=$5 done_dir=${6:-}

    if [ -n "$done_dir" ] && is_done "$done_dir"; then
        log_progress "[SKIP] [Lane ${lane}] Já concluído: ${desc} [seed=${seed}]"
        echo -e "  ${BLUE}[SKIP]${NC} Já concluído: ${desc} [seed=${seed}]"
        return 0
    fi

    log_progress ">> [Lane ${lane}] Iniciando: ${desc} [seed=${seed}]"
    echo -e "${YELLOW}>> [Lane ${lane}] Iniciando:${NC} ${desc} [seed=${seed}]"
    echo -e "   Aguardando >= ${min_free} MB livres na GPU...\n"

    python3 "$AUTO_LAUNCHER" \
        --command "python3 main.py train --exp-configs $config --seed $seed" \
        --gpus 1 \
        --threshold "$THRESHOLD" \
        --min-free-mb "$min_free" \
        --interval "$INTERVAL" \
        --no-screen

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
# Lane A — Tiny ImageNet 20-20 (seed 1993, sequencial)
# VRAM: ~6.5 GB por exp  |  threshold: >= 16 GB livres
#
# Histórico:
#   [✅] Baseline Local  → done_exp_tiny_imagenet_20-20_antB0_nceA1_antLocal_s1993
#   [✅] Baseline Global → done_exp_tiny_imagenet_20-20_antB0_nceA1_antGlobal_s1993
#   [🔄] ANT Local       → exp_tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antLocal_s1993
#   [ ] ANT Global       → pendente
# ═══════════════════════════════════════════════════════════════════════════
lane_tiny_imagenet() {
    echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane A — Tiny ImageNet 20-20 (seed 1993)   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane A] Iniciando — Tiny ImageNet 20-20"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_baseline_local_resnet18.yaml" \
        "Tiny ImageNet Baseline Local" \
        1993 "$MIN_FREE_MB_A" "A" \
        "done_exp_tiny_imagenet_20-20_antB0_nceA1_antLocal_s1993"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_baseline_global_resnet18.yaml" \
        "Tiny ImageNet Baseline Global" \
        1993 "$MIN_FREE_MB_A" "A" \
        "done_exp_tiny_imagenet_20-20_antB0_nceA1_antGlobal_s1993"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "Tiny ImageNet ANT β=0.5 m=0.5 Local" \
        1993 "$MIN_FREE_MB_A" "A" \
        "done_exp_tiny_imagenet_20-20_antB0.5_nceA1_antM0.5_antLocal_s1993"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_ant_beta0.5_margin0.5_global_resnet18.yaml" \
        "Tiny ImageNet ANT β=0.5 m=0.5 Global" \
        1993 "$MIN_FREE_MB_A" "A"
    # sem done_dir → sempre roda (ainda não executado)

    log_progress "[OK] [Lane A] Tiny ImageNet — concluída"
    echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  [Lane A] Tiny ImageNet 20-20 concluída! ✅  ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# Lane B — CIFAR-100 (seeds 1993/1995/1996, sequencial por seed)
# VRAM: ~5.0 GB por exp  |  threshold: >= 8 GB livres (corre junto com Lane A)
#
# Histórico por config e seed:
#   10-10 Baseline:     s1993 ✅ s1995 ✅ s1996 ✅ (todos done)
#   10-10 ANT β=0.5:    s1993 ✅ s1995 ✅ s1996 ✅ (todos done)
#   50-10 Baseline:     s1993 ✅ s1995 ⏳ s1996 ⏳
#   50-10 ANT β=0.5:    s1993 ⏳ s1995 ⏳ s1996 ⏳
#
# Nota: s1993 do 10-10 Baseline está em done_exp_cifar100_10-10_baseline_tagfex_original_s1993
# ═══════════════════════════════════════════════════════════════════════════
lane_cifar100() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane B — CIFAR-100 (seeds 1993 · 1995 · 1996)        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane B] Iniciando — CIFAR-100"

    for seed in "${SEEDS_ALL[@]}"; do
        echo -e "${YELLOW}  ── Seed ${seed} ──────────────────────────────────────${NC}\n"

        # 10-10 Baseline Local
        # s1993 usa nome especial (experimento original do TagFex)
        if [ "$seed" -eq 1993 ]; then
            queue_experiment \
                "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml" \
                "CIFAR-100 10-10 Baseline Local" \
                "$seed" "$MIN_FREE_MB_B" "B" \
                "done_exp_cifar100_10-10_baseline_tagfex_original_s1993"
        else
            queue_experiment \
                "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml" \
                "CIFAR-100 10-10 Baseline Local" \
                "$seed" "$MIN_FREE_MB_B" "B" \
                "done_exp_cifar100_10-10_antB0_nceA1_antLocal_s${seed}"
        fi

        # 10-10 ANT β=0.5 m=0.5 Local
        queue_experiment \
            "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "CIFAR-100 10-10 ANT β=0.5 m=0.5 Local" \
            "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}"

        # 50-10 Baseline Local
        queue_experiment \
            "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
            "CIFAR-100 50-10 Baseline Local" \
            "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_cifar100_50-10_antB0_nceA1_antLocal_s${seed}"

        # 50-10 ANT β=0.5 m=0.5 Local
        queue_experiment \
            "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local" \
            "$seed" "$MIN_FREE_MB_B" "B" \
            "done_exp_cifar100_50-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}"

    done

    log_progress "[OK] [Lane B] CIFAR-100 — concluída"
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  [Lane B] CIFAR-100 concluída! ✅                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main — lançar ambas as lanes em paralelo
# ═══════════════════════════════════════════════════════════════════════════
main() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}    TagFex Unified Queue — Lane A ∥ Lane B${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  ${BLUE}GPU:${NC}    RTX 4090 24 GB (quati)"
    echo -e "  ${BLUE}Lane A:${NC} Tiny ImageNet 20-20 · seed 1993 · >= ${MIN_FREE_MB_A} MB livres"
    echo -e "  ${BLUE}Lane B:${NC} CIFAR-100 · seeds ${SEEDS_ALL[*]} · >= ${MIN_FREE_MB_B} MB livres"
    echo -e "  ${BLUE}Seeds diferidas:${NC} ${SEEDS_DEFERRED[*]}\n"
    echo -e "  Experimentos com diretório done_exp_* são automaticamente pulados.\n"

    log_progress ">> Fila unificada iniciada (Lane A ∥ Lane B)"

    # Lançar lanes em paralelo como subshells
    lane_tiny_imagenet &
    LANE_A_PID=$!

    lane_cifar100 &
    LANE_B_PID=$!

    echo -e "${CYAN}Ambas as lanes lançadas em paralelo (PID A=${LANE_A_PID}, B=${LANE_B_PID})${NC}\n"

    # Aguardar ambas
    wait $LANE_A_PID
    LANE_A_STATUS=$?
    wait $LANE_B_PID
    LANE_B_STATUS=$?

    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
    if [ $LANE_A_STATUS -eq 0 ] && [ $LANE_B_STATUS -eq 0 ]; then
        echo -e "${GREEN}[OK] Fila unificada concluída! Todas as lanes finalizadas.${NC}"
        log_progress "[FIM] Fila unificada concluída com sucesso"
    else
        echo -e "${RED}[AVISO] Uma ou mais lanes terminaram com erro.${NC}"
        echo -e "  Lane A: $([ $LANE_A_STATUS -eq 0 ] && echo OK || echo ERRO)"
        echo -e "  Lane B: $([ $LANE_B_STATUS -eq 0 ] && echo OK || echo ERRO)"
        log_progress "[FIM] Fila concluída com erros (A=${LANE_A_STATUS}, B=${LANE_B_STATUS})"
    fi
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  Log de progresso : ${PROGRESS_LOG}"
    echo -e "  Log completo     : ${CONSOLE_LOG}"
}

# Executar e capturar toda saída em arquivo de log
echo " Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
