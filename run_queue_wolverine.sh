#!/bin/bash
# Fila de experimentos para Wolverine (2× RTX 3080 Ti · 12 GB/GPU · 2 GPUs)
#
# Experimentos pendentes (CIFAR-100 50-10, 8 total):
#   GPU 0 — seeds 1994 e 1996:
#     · CIFAR-100 50-10 Baseline Local s1994
#     · CIFAR-100 50-10 ANT β=0.5 m=0.5 Local s1994
#     · CIFAR-100 50-10 Baseline Local s1996
#     · CIFAR-100 50-10 ANT β=0.5 m=0.5 Local s1996
#   GPU 1 — seeds 1995 e 1997:
#     · CIFAR-100 50-10 Baseline Local s1995
#     · CIFAR-100 50-10 ANT β=0.5 m=0.5 Local s1995
#     · CIFAR-100 50-10 Baseline Local s1997
#     · CIFAR-100 50-10 ANT β=0.5 m=0.5 Local s1997
#
# Uso — executar uma vez; o script cria dois screen sessions automaticamente:
#   ./run_queue_wolverine.sh
#
# Para monitorar:
#   screen -ls                            # listar sessões
#   screen -r wolverine_gpu0              # attach GPU 0
#   screen -r wolverine_gpu1              # attach GPU 1
#   tail -f logs/auto_experiments/wolverine_gpu0_progress.log
#   tail -f logs/auto_experiments/wolverine_gpu1_progress.log
#
# Para desanexar de dentro do screen: Ctrl+A depois D
#
# ── Variável interna — NÃO definir manualmente ──
# GPU_ID é definida internamente quando o script se auto-invoca dentro do screen.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# ════════════════════════════════════════════════════════════
# MODO WORKER: executado dentro de um screen session
# Roda a fila de uma GPU específica de forma sequencial.
# ════════════════════════════════════════════════════════════
if [ -n "$GPU_ID" ]; then
    # Ativar ambiente virtual
    if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    PROGRESS_LOG="$LOG_DIR/wolverine_gpu${GPU_ID}_progress.log"

    log_w() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS_LOG"
    }

    run_exp() {
        local config=$1
        local seed=$2
        local desc=$3
        log_w ">> Iniciando: $desc  [seed=$seed]  (CUDA_VISIBLE_DEVICES=$GPU_ID)"
        echo -e "${YELLOW}>> Iniciando:${NC} $desc  [seed=$seed]  (GPU $GPU_ID)"
        python3 main.py train --exp-configs "$config" --seed "$seed"
        if [ $? -eq 0 ]; then
            log_w "[OK] Concluído: $desc  [seed=$seed]"
            echo -e "${GREEN}[OK] Concluído: $desc${NC}\n"
        else
            log_w "[ERRO] $desc  [seed=$seed]"
            echo -e "${RED}[ERRO] $desc — abortando GPU $GPU_ID${NC}\n"
            exit 1
        fi
        sleep 5
    }

    # Ajuste de memória: CIFAR-100 50-10 usa ~5-6 GB de pico com ResNet-18.
    # GPU 3080 Ti 12 GB tem margem confortável por experimento.

    if [ "$GPU_ID" = "0" ]; then
        echo -e "${GREEN}═══ Wolverine GPU 0 — seeds 1994 e 1996 ═══${NC}\n"
        log_w ">> Iniciando GPU 0 queue (seeds 1994, 1996)"

        run_exp \
            "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
            1994 \
            "CIFAR-100 50-10 Baseline Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            1994 \
            "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
            1996 \
            "CIFAR-100 50-10 Baseline Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            1996 \
            "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

        log_w "[OK] GPU 0 queue completa (4/4)"
        echo -e "${GREEN}[OK] GPU 0: todos os 4 experimentos concluídos!${NC}"

    elif [ "$GPU_ID" = "1" ]; then
        echo -e "${GREEN}═══ Wolverine GPU 1 — seeds 1995 e 1997 ═══${NC}\n"
        log_w ">> Iniciando GPU 1 queue (seeds 1995, 1997)"

        run_exp \
            "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
            1995 \
            "CIFAR-100 50-10 Baseline Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            1995 \
            "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_baseline_local_resnet18.yaml" \
            1997 \
            "CIFAR-100 50-10 Baseline Local"

        run_exp \
            "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
            1997 \
            "CIFAR-100 50-10 ANT β=0.5 m=0.5 Local"

        log_w "[OK] GPU 1 queue completa (4/4)"
        echo -e "${GREEN}[OK] GPU 1: todos os 4 experimentos concluídos!${NC}"
    else
        echo -e "${RED}[ERRO] GPU_ID inválido: $GPU_ID${NC}"
        exit 1
    fi

    exit 0
fi

# ════════════════════════════════════════════════════════════
# MODO DISPATCHER: cria os dois screen sessions e sai
# ════════════════════════════════════════════════════════════

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    Wolverine Queue — 2× RTX 3080 Ti 12GB · Dispatcher${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "${BLUE} Lançando 2 screen sessions (uma por GPU):${NC}"
echo -e "     wolverine_gpu0 → seeds 1994 e 1996 (CIFAR-100 50-10 Baseline + ANT)"
echo -e "     wolverine_gpu1 → seeds 1995 e 1997 (CIFAR-100 50-10 Baseline + ANT)"
echo ""

# Detectar ambiente virtual para passar para os workers
VENV_ACTIVATE=""
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_ACTIVATE="source $VIRTUAL_ENV/bin/activate && "
fi

# Lançar GPU 0
screen -dmS wolverine_gpu0 bash -c \
    "${VENV_ACTIVATE}cd '$SCRIPT_DIR' && GPU_ID=0 bash '$0' 2>&1 | tee '$LOG_DIR/wolverine_gpu0_console_$(date +%Y%m%d_%H%M%S).log'"
echo -e "${GREEN}[OK] Screen 'wolverine_gpu0' lançado${NC}"

# Lançar GPU 1
screen -dmS wolverine_gpu1 bash -c \
    "${VENV_ACTIVATE}cd '$SCRIPT_DIR' && GPU_ID=1 bash '$0' 2>&1 | tee '$LOG_DIR/wolverine_gpu1_console_$(date +%Y%m%d_%H%M%S).log'"
echo -e "${GREEN}[OK] Screen 'wolverine_gpu1' lançado${NC}"

echo ""
echo -e "${BLUE}Para acompanhar:${NC}"
echo "   screen -ls"
echo "   screen -r wolverine_gpu0"
echo "   screen -r wolverine_gpu1"
echo "   tail -f $LOG_DIR/wolverine_gpu0_progress.log"
echo "   tail -f $LOG_DIR/wolverine_gpu1_progress.log"
echo ""
echo -e "${BLUE}Para desanexar de dentro do screen:${NC}  Ctrl+A  depois  D"
