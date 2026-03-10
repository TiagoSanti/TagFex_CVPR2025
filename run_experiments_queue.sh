#!/bin/bash
# Script para enfileirar múltiplos experimentos
# Cada experimento é disparado automaticamente quando uma GPU fica disponível
#
# RECOMENDADO: Execute este script em uma sessão screen para continuar
# monitorando mesmo após desconectar do SSH:
#
#   ./start_queue_monitor.sh
#
# Ou manualmente:
#   screen -dmS tagfex_queue ./run_experiments_queue.sh
#   screen -r tagfex_queue  # para anexar
#
# ═══════════════════════════════════════════════════════════
# SELEÇÃO DE MÁQUINA
# ═══════════════════════════════════════════════════════════
# Defina MACHINE abaixo ou deixe em "auto" para detecção automática
# pelo hostname. Valores válidos: "quati" | "fera" | "auto"
#
#   quati — RTX 4090 24GB, single GPU  (hostname: quaTII)
#   fera  — 2× GPU ~49GB cada, multi-GPU (hostname: fera*)
#
MACHINE="auto"

# Diretório base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# Ativar ambiente virtual se existir
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ───────────────────────────────────────────────────────────
# Perfis de máquina
# Cada perfil define:
#   GPUS               — nº de GPUs por experimento
#   MEMORY_THRESHOLD   — flag --memory-threshold (% de VRAM ocupada)
#   INTERVAL           — intervalo entre checagens de disponibilidade (s)
# ───────────────────────────────────────────────────────────
configure_machine() {
    local machine=$1

    case "$machine" in
        quati)
            # RTX 4090 · 24 GB · 1 GPU
            # VRAM idle ~43 MiB (~0.2%); threshold 10% ≈ 2.4 GB de folga
            GPUS=1
            MEMORY_THRESHOLD="--memory-threshold 10.0"
            INTERVAL=30
            MACHINE_LABEL="quati (RTX 4090 24GB · 1 GPU)"
            ;;
        fera)
            # 2× GPU ~49 GB cada · multi-GPU via torchrun
            # VRAM idle ~200 MiB (~0.4%); threshold 5% ≈ 2.4 GB de folga por GPU
            GPUS=2
            MEMORY_THRESHOLD="--memory-threshold 5.0"
            INTERVAL=30
            MACHINE_LABEL="fera (2× ~49GB · 2 GPUs)"
            ;;
        *)
            echo -e "${RED}❌ Perfil de máquina desconhecido: '$machine'. Use 'quati' ou 'fera'.${NC}"
            exit 1
            ;;
    esac
}

# Auto-detecção por hostname
if [ "$MACHINE" = "auto" ]; then
    HOSTNAME_LOWER=$(hostname | tr '[:upper:]' '[:lower:]')
    if [[ "$HOSTNAME_LOWER" == *"qua"* ]]; then
        MACHINE="quati"
    elif [[ "$HOSTNAME_LOWER" == *"fera"* ]]; then
        MACHINE="fera"
    else
        echo -e "\033[0;31m❌ Hostname '$(hostname)' não reconhecido. Defina MACHINE='quati' ou 'fera' no topo do script.\033[0m"
        exit 1
    fi
fi

configure_machine "$MACHINE"

# Configurações globais
THRESHOLD=100.0      # Desabilitado (só verifica memória)
LOG_DIR="./logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress.log"
CONSOLE_LOG="$LOG_DIR/queue_console_$(date +%Y%m%d_%H%M%S).log"

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"

# Função para log com timestamp
log_progress() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$PROGRESS_LOG"
}

# Função principal que será logada
main() {

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    TagFex Auto Experiment Queue${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE}🖥️  Máquina:${NC} $MACHINE_LABEL"
echo -e "${BLUE}⚙️  GPUs/exp:${NC} $GPUS   ${BLUE}Threshold:${NC} ${MEMORY_THRESHOLD#--memory-threshold }% ocupada\n"

log_progress "🚀 Iniciando fila de experimentos"
log_progress "🖥️  Máquina: $MACHINE_LABEL  (GPUs/exp: $GPUS)"
log_progress "📁 Logs salvos em: $LOG_DIR"
log_progress "📊 Log de progresso: $PROGRESS_LOG"

# Contador global de posição na fila
EXP_COUNTER=0
EXP_TOTAL=0  # Definido em cada bloco de máquina abaixo

# Função para enfileirar experimento
# Arguments: 1=config_file  2=description  3=seed (opcional, padrão=1993)
# Usa variável global GPUS para escolher entre single- ou multi-gpu (torchrun)
queue_experiment() {
    local config_file=$1
    local description=$2
    local seed=${3:-1993}

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$EXP_TOTAL]"

    log_progress "🔄 $pos Iniciando: $description  [seed=$seed]"
    log_progress "   Config: $config_file"
    log_progress "   GPUs necessárias: $GPUS"

    echo -e "${YELLOW}🔄 $pos Iniciando:${NC} $description  [seed=$seed]"
    echo -e "   Config: $config_file"
    echo -e "   GPUs necessárias: $GPUS"
    echo -e "   Aguardando GPU(s) disponíveis...\n"

    # montar comando de treino; suporta multi-gpu com torchrun
    if [ "$GPUS" -gt 1 ]; then
        train_cmd="torchrun --nproc_per_node=$GPUS python3 main.py train --exp-configs $config_file --seed $seed"
    else
        train_cmd="python3 main.py train --exp-configs $config_file --seed $seed"
    fi

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus $GPUS \
        --threshold $THRESHOLD \
        $MEMORY_THRESHOLD \
        --interval $INTERVAL \
        --no-screen

    if [ $? -eq 0 ]; then
        log_progress "✅ $pos Concluído: $description  [seed=$seed]"
        echo -e "${GREEN}✅ $pos Concluído com sucesso! (${EXP_COUNTER}/${EXP_TOTAL} feitos)${NC}\n"
    else
        log_progress "❌ $pos ERRO: $description  [seed=$seed]"
        echo -e "${RED}❌ $pos Erro no experimento${NC}\n"
        return 1
    fi
    
    # Pequeno delay entre disparos
    sleep 5
}

# Roda 5 seeds (1993-1997) para uma config — resultados do paper
# Arguments: 1=config_file  2=description_base
queue_experiment_5seeds() {
    local config_file=$1
    local description=$2
    for seed in 1993 1994 1995 1996 1997; do
        queue_experiment "$config_file" "$description" "$seed"
    done
}

# Roda apenas seeds 1994-1997 (quando seed 1993 já foi executado)
# Arguments: 1=config_file  2=description_base
queue_experiment_remaining_seeds() {
    local config_file=$1
    local description=$2
    for seed in 1994 1995 1996 1997; do
        queue_experiment "$config_file" "$description" "$seed"
    done
}

# ═══════════════════════════════════════════════════════════
# Filas de experimentos — separadas por máquina
# ═══════════════════════════════════════════════════════════

if [ "$MACHINE" = "quati" ]; then

    # ─────────────────────────────────────────────────────
    # quati · RTX 4090 24GB · single GPU
    # Datasets que cabem em 24GB: Tiny ImageNet, CIFAR-100
    # ─────────────────────────────────────────────────────
    # 4 Tiny ImageNet (seed 1993)
    # + 4×2 CIFAR-100 10-10 (seeds 1994-1997) = 8
    # + 4   CIFAR-100 50-10 baseline (seeds 1994-1997) = 4
    # + 5   CIFAR-100 50-10 ANT (seeds 1993-1997) = 5
    # Total: 4 + 8 + 4 + 5 = 21
    EXP_TOTAL=21
    log_progress "📋 Total de experimentos nesta fila: $EXP_TOTAL"
    echo -e "${BLUE}📋 Total de experimentos: ${EXP_TOTAL}${NC}\n"
    echo -e "${BLUE}Ordem de execução:${NC}"
    echo -e "   1-4  : Tiny ImageNet 20-20 (baseline local/global + ANT local/global) — seed 1993"
    echo -e "   5-8  : CIFAR-100 10-10 Baseline Local — seeds 1994, 1995, 1996, 1997"
    echo -e "   9-12 : CIFAR-100 10-10 ANT β=0.5 m=0.5 Local — seeds 1994, 1995, 1996, 1997"
    echo -e "   13-16: CIFAR-100 50-10 Baseline Local — seeds 1994, 1995, 1996, 1997"
    echo -e "   17-21: CIFAR-100 50-10 ANT β=0.5 m=0.5 Local — seeds 1993, 1994, 1995, 1996, 1997"
    echo -e ""

    # ── Tiny ImageNet 20-20 ── seed 1993 (primeira execução)
    echo -e "${YELLOW}═══ Tiny ImageNet 20-20 — seed 1993 ═══${NC}\n"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_baseline_local_resnet18.yaml" \
        "Tiny ImageNet 20-20 Baseline (Local Anchor)"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_baseline_global_resnet18.yaml" \
        "Tiny ImageNet 20-20 Baseline (Global Anchor)"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "Tiny ImageNet 20-20 ANT (β=0.5, margin=0.5, Local)"

    queue_experiment \
        "configs/all_in_one/tiny_imagenet_20-20_ant_beta0.5_margin0.5_global_resnet18.yaml" \
        "Tiny ImageNet 20-20 ANT (β=0.5, margin=0.5, Global)"

    # ── CIFAR-100 10-10 — paper reproducibility (5 seeds) ──
    # Seed 1993 já executado → apenas remaining seeds 1994-1997
    echo -e "${YELLOW}═══ CIFAR-100 10-10 — Paper Reproducibility (seeds 1994-1997) ═══${NC}\n"

    queue_experiment_remaining_seeds \
        "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml" \
        "CIFAR-100 10-10 Baseline (Local)"

    queue_experiment_remaining_seeds \
        "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "CIFAR-100 10-10 ANT β=0.5 m=0.5 Local"

    # ── CIFAR-100 50-10 — paper reproducibility (5 seeds) ──
    # Baseline local seed 1993 já executado; ANT best seed 1993 NÃO foi executado ainda
    echo -e "${YELLOW}═══ CIFAR-100 50-10 — Paper Reproducibility ═══${NC}\n"

    queue_experiment_remaining_seeds \
        "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
        "CIFAR-100 50-10 Baseline (Local)"

    # 50-10 ANT best (β=0.5, m=0.5) — seed 1993 ainda não executado → 5 seeds completos
    queue_experiment_5seeds \
        "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

elif [ "$MACHINE" = "fera" ]; then

    # ─────────────────────────────────────────────────────
    # fera · 2× ~49GB · torchrun 2 GPUs
    # NOTA: fera atualmente indisponível. Manter para quando retornar.
    # 2 CIFAR-100 global anchor (seed 1993) + 6 ImageNet-100 (seed 1993) = 8
    # ─────────────────────────────────────────────────────
    EXP_TOTAL=8
    log_progress "📋 Total de experimentos nesta fila: $EXP_TOTAL"
    echo -e "${BLUE}📋 Total de experimentos: ${EXP_TOTAL}${NC}\n"

    # ── CIFAR-100 — seed 1993 (exploratory / global anchor) ──
    echo -e "${YELLOW}═══ CIFAR-100 Experiments (fera) ═══${NC}\n"

    queue_experiment \
        "configs/all_in_one/cifar100_10-10_baseline_global_resnet18.yaml" \
        "CIFAR-100 10-10 Baseline (Global Anchor)"

    queue_experiment \
        "configs/all_in_one/cifar100_50-10_baseline_global_resnet18.yaml" \
        "CIFAR-100 50-10 Baseline (Global Anchor)"

    # ── ImageNet-100 — seed 1993 (primeira execução) ──
    echo -e "${YELLOW}═══ ImageNet-100 Experiments (fera) ═══${NC}\n"

    queue_experiment \
        "configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml" \
        "ImageNet-100 10-10 Baseline (Local Anchor)"

    queue_experiment \
        "configs/all_in_one/imagenet100_10-10_baseline_global_resnet18.yaml" \
        "ImageNet-100 10-10 Baseline (Global Anchor)"

    queue_experiment \
        "configs/all_in_one/imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "ImageNet-100 10-10 ANT (β=0.5, margin=0.5, Local)"

    queue_experiment \
        "configs/all_in_one/imagenet100_50-10_baseline_local_resnet18.yaml" \
        "ImageNet-100 50-10 Baseline (Local Anchor)"

    queue_experiment \
        "configs/all_in_one/imagenet100_50-10_baseline_global_resnet18.yaml" \
        "ImageNet-100 50-10 Baseline (Global Anchor)"

    queue_experiment \
        "configs/all_in_one/imagenet100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        "ImageNet-100 50-10 ANT (β=0.5, margin=0.5, Local)"

fi

log_progress "✅ Todos os experimentos enfileirados!"
log_progress "📺 $(screen -ls | grep -c Detached) sessões screen ativas"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Todos os experimentos enfileirados!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"

echo "📊 Para monitorar os experimentos:"
echo "   - GPUs: gpustat --watch"
echo "   - Progresso: tail -f $PROGRESS_LOG"
echo "   - Logs: tail -f $LOG_DIR/*.log"
echo "   - Sessões: screen -ls"
echo "   - Processos: ps aux | grep 'main.py train'"
echo ""

log_progress "🏁 Script de enfileiramento concluído"

} # Fim da função main

# Executar main e capturar toda saída em arquivo de log
echo "📝 Log completo será salvo em: $CONSOLE_LOG"
main 2>&1 | tee -a "$CONSOLE_LOG"
