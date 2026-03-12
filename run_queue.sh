#!/bin/bash
# Fila unificada — distribui experimentos entre quati e fera automaticamente
#
# Arquitetura:
#   quati (RTX 4090 24GB)
#     Lane A — Tiny ImageNet 20-20   seed 1993             >= 16 GB livres
#     Lane B — CIFAR-100             seeds 1993/1995/1996  >=  8 GB livres
#
#   fera (2x RTX A6000 49GB)
#     Lane C — ImageNet-100 10-10    seeds 1993/1995/1996  GPU 0  >= 25 GB
#     Lane D — ImageNet-100 50-10    seeds 1993/1995/1996  GPU 1  >= 25 GB
#
# Ao rodar em quati com LAUNCH_FERA=true (padrão), faz SSH em fera e lança
# o mesmo script lá; fera auto-detecta o hostname e roda apenas as lanes C/D.
# Skip automático: experimentos com diretório done_exp_* já existente são pulados.
#
# Uso (quati):
#   screen -dmS tagfex_queue ./run_queue.sh
# Uso (fera manual):
#   screen -dmS tagfex_fera ./run_queue.sh
#
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOGS_DIR="$SCRIPT_DIR/logs"

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Configuração de máquinas ────────────────────────────────────────────────
MACHINE="auto"   # "quati" | "fera" | "auto"

# SSH: quati lança fera automaticamente ao iniciar
LAUNCH_FERA=true
FERA_HOST="tiago@10.87.10.209"

# ── Seeds ────────────────────────────────────────────────────────────────────
# Trio {1993, 1995, 1996}: soma L2 pairwise = 5.96 (máxima incluindo s1993)
# Diferidas: 1994 (próxima a 1993, L2=2.02) e 1997 (próxima a 1996, L2=1.26)
SEEDS_ACTIVE=(1995 1996)          # seeds adicionais além de 1993
SEEDS_ALL=(1993 1995 1996)        # todas as seeds ativas
SEEDS_DEFERRED=(1994 1997)        # diferidas para validação futura

# ── Auto-detecção de máquina por hostname ────────────────────────────────────
if [ "$MACHINE" = "auto" ]; then
    HOSTNAME_LOWER=$(hostname | tr '[:upper:]' '[:lower:]')
    if [[ "$HOSTNAME_LOWER" == *"qua"* ]]; then
        MACHINE="quati"
    elif [[ "$HOSTNAME_LOWER" == *"fera"* ]]; then
        MACHINE="fera"
    else
        echo "[ERRO] Hostname '$(hostname)' não reconhecido. Defina MACHINE='quati' ou 'fera'."
        exit 1
    fi
fi

# ── Thresholds de GPU por máquina ────────────────────────────────────────────
INTERVAL=30
THRESHOLD=100.0

case "$MACHINE" in
    quati)
        MACHINE_LABEL="quati (RTX 4090 24GB)"
        MIN_FREE_MB_A=16000   # Tiny ImageNet ~6.5 GB → 16 GB margem
        MIN_FREE_MB_B=8000    # CIFAR-100    ~5.0 GB → 8 GB (corre junto A)
        ;;
    fera)
        MACHINE_LABEL="fera (2× RTX A6000 49GB)"
        MIN_FREE_MB_C=25000   # ImageNet-100 ~15 GB → 25 GB mínimo GPU 0
        MIN_FREE_MB_D=25000   # ImageNet-100 ~15 GB → 25 GB mínimo GPU 1
        ;;
esac

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
is_done() {
    local done_dir="$1"
    [ -d "$LOGS_DIR/$done_dir" ]
}

# queue_experiment <config> <description> <seed> <min_free_mb> <lane> [done_dir] [gpu_ids...]
# gpu_ids: IDs das GPUs permitidas (ex: "0" ou "1"); omitir = todas as GPUs
queue_experiment() {
    local config=$1 desc=$2 seed=$3 min_free=$4 lane=$5 done_dir=${6:-}
    shift 6
    local gpu_ids=("$@")   # IDs de GPU passados como argumentos restantes

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
# Lane C — ImageNet-100 10-10  ·  seeds 1993/1995/1996  ·  fera GPU 0
# ═══════════════════════════════════════════════════════════════════════════
lane_imagenet100_10() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane C — ImageNet-100 10-10 (seeds ${SEEDS_ALL[*]})  GPU 0  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane C] Iniciando — ImageNet-100 10-10 (GPU 0)"

    for seed in "${SEEDS_ALL[@]}"; do
        echo -e "${YELLOW}  ── Seed ${seed} (ImageNet-100 10-10) ─────────────────────${NC}\n"

        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml" \
            "ImageNet-100 10-10 Baseline Local" "$seed" "$MIN_FREE_MB_C" "C" \
            "done_exp_imagenet100_10-10_antB0_nceA1_antLocal_s${seed}" \
            0

        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_baseline_global_resnet18.yaml" \
            "ImageNet-100 10-10 Baseline Global" "$seed" "$MIN_FREE_MB_C" "C" \
            "done_exp_imagenet100_10-10_antB0_nceA1_antGlobal_s${seed}" \
            0

        queue_experiment \
            "configs/all_in_one/imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "ImageNet-100 10-10 ANT β=0.5 m=0.5 Local" "$seed" "$MIN_FREE_MB_C" "C" \
            "done_exp_imagenet100_10-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}" \
            0
    done

    log_progress "[OK] [Lane C] ImageNet-100 10-10 — concluída"
    echo -e "${GREEN}[OK] Lane C — ImageNet-100 10-10 concluída!${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# Lane D — ImageNet-100 50-10  ·  seeds 1993/1995/1996  ·  fera GPU 1
# ═══════════════════════════════════════════════════════════════════════════
lane_imagenet100_50() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  Lane D — ImageNet-100 50-10 (seeds ${SEEDS_ALL[*]})  GPU 1  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}\n"
    log_progress ">> [Lane D] Iniciando — ImageNet-100 50-10 (GPU 1)"

    for seed in "${SEEDS_ALL[@]}"; do
        echo -e "${YELLOW}  ── Seed ${seed} (ImageNet-100 50-10) ─────────────────────${NC}\n"

        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_baseline_local_resnet18.yaml" \
            "ImageNet-100 50-10 Baseline Local" "$seed" "$MIN_FREE_MB_D" "D" \
            "done_exp_imagenet100_50-10_antB0_nceA1_antLocal_s${seed}" \
            1

        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_baseline_global_resnet18.yaml" \
            "ImageNet-100 50-10 Baseline Global" "$seed" "$MIN_FREE_MB_D" "D" \
            "done_exp_imagenet100_50-10_antB0_nceA1_antGlobal_s${seed}" \
            1

        queue_experiment \
            "configs/all_in_one/imagenet100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            "ImageNet-100 50-10 ANT β=0.5 m=0.5 Local" "$seed" "$MIN_FREE_MB_D" "D" \
            "done_exp_imagenet100_50-10_antB0.5_nceA1_antM0.5_antLocal_s${seed}" \
            1
    done

    log_progress "[OK] [Lane D] ImageNet-100 50-10 — concluída"
    echo -e "${GREEN}[OK] Lane D — ImageNet-100 50-10 concluída!${NC}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
# SSH: quati lança fera em background antes de começar as lanes locais
# ═══════════════════════════════════════════════════════════════════════════
launch_fera_remote() {
    echo -e "${CYAN}[Remote] Tentando lançar Lanes C/D em fera (${FERA_HOST})...${NC}"
    local remote_cmd="cd ~/TagFex_CVPR2025 && git pull --quiet && screen -dmS tagfex_fera ./run_queue.sh && echo 'fera_launched'"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$FERA_HOST" "$remote_cmd" 2>/dev/null; then
        log_progress ">> [Remote] Lanes C/D lançadas em ${FERA_HOST}"
        echo -e "${GREEN}[Remote] Fila fera lançada! (screen: tagfex_fera em ${FERA_HOST})${NC}\n"
    else
        log_progress "[AVISO] [Remote] Falha ao conectar em ${FERA_HOST} — apenas quati ativo"
        echo -e "${YELLOW}[AVISO] Não foi possível lançar fera (sem SSH key ou host indisponível).${NC}"
        echo -e "${YELLOW}        Continuando apenas com quati.${NC}\n"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
main() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}    TagFex Unified Queue  ·  ${MACHINE_LABEL}${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  ${BLUE}Seeds ativas:${NC}   ${SEEDS_ALL[*]}"
    echo -e "  ${BLUE}Seeds diferidas:${NC} ${SEEDS_DEFERRED[*]}"
    echo -e "  Experimentos com diretório done_exp_* são automaticamente pulados.\n"
    log_progress ">> Fila iniciada em ${MACHINE_LABEL}"

    if [ "$MACHINE" = "quati" ]; then
        echo -e "  ${BLUE}Lanes:${NC}"
        echo -e "    A  Tiny ImageNet 20-20  seed 1993             >= ${MIN_FREE_MB_A} MB"
        echo -e "    B  CIFAR-100            seeds ${SEEDS_ALL[*]}  >= ${MIN_FREE_MB_B} MB"
        if [ "$LAUNCH_FERA" = "true" ]; then
            echo -e "    C  ImageNet-100 10-10  seeds ${SEEDS_ALL[*]}  GPU 0  [fera]"
            echo -e "    D  ImageNet-100 50-10  seeds ${SEEDS_ALL[*]}  GPU 1  [fera]\n"
            launch_fera_remote
        else
            echo ""
        fi

        lane_tiny_imagenet &
        LANE_A_PID=$!
        lane_cifar100 &
        LANE_B_PID=$!

        echo -e "${CYAN}Lanes A e B em paralelo (PID A=${LANE_A_PID}, B=${LANE_B_PID})${NC}\n"
        wait $LANE_A_PID; A_STATUS=$?
        wait $LANE_B_PID; B_STATUS=$?

        echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
        if [ $A_STATUS -eq 0 ] && [ $B_STATUS -eq 0 ]; then
            echo -e "${GREEN}[OK] quati — todas as lanes concluídas!${NC}"
            log_progress "[FIM] quati concluída (A=${A_STATUS}, B=${B_STATUS})"
        else
            echo -e "${RED}[AVISO] Erros: Lane A=${A_STATUS}  Lane B=${B_STATUS}${NC}"
            log_progress "[FIM] quati com erros (A=${A_STATUS}, B=${B_STATUS})"
        fi

    elif [ "$MACHINE" = "fera" ]; then
        echo -e "  ${BLUE}Lanes:${NC}"
        echo -e "    C  ImageNet-100 10-10  seeds ${SEEDS_ALL[*]}  GPU 0  >= ${MIN_FREE_MB_C} MB"
        echo -e "    D  ImageNet-100 50-10  seeds ${SEEDS_ALL[*]}  GPU 1  >= ${MIN_FREE_MB_D} MB\n"

        lane_imagenet100_10 &
        LANE_C_PID=$!
        lane_imagenet100_50 &
        LANE_D_PID=$!

        echo -e "${CYAN}Lanes C e D em paralelo (PID C=${LANE_C_PID}, D=${LANE_D_PID})${NC}\n"
        wait $LANE_C_PID; C_STATUS=$?
        wait $LANE_D_PID; D_STATUS=$?

        echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
        if [ $C_STATUS -eq 0 ] && [ $D_STATUS -eq 0 ]; then
            echo -e "${GREEN}[OK] fera — todas as lanes concluídas!${NC}"
            log_progress "[FIM] fera concluída (C=${C_STATUS}, D=${D_STATUS})"
        else
            echo -e "${RED}[AVISO] Erros: Lane C=${C_STATUS}  Lane D=${D_STATUS}${NC}"
            log_progress "[FIM] fera com erros (C=${C_STATUS}, D=${D_STATUS})"
        fi
    fi

    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
    echo -e "  Log de progresso : ${PROGRESS_LOG}"
    echo -e "  Log completo     : ${CONSOLE_LOG}"
}

# Executar e capturar toda saída em arquivo de log
echo " Máquina: ${MACHINE_LABEL}"
echo " Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
