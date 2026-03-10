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

# Diretório base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"

# Ativar ambiente virtual se existir
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo -e "${GREEN}✅ Ambiente virtual .venv ativado${NC}\n"
fi

# Configurações globais
THRESHOLD=100.0      # Desabilitado (só verifica memória)
MEMORY_THRESHOLD="--memory-threshold 5.0"  # GPU com < 5% memória é considerada livre
INTERVAL=30          # Checar a cada 30 segundos
LOG_DIR="./logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress.log"
CONSOLE_LOG="$LOG_DIR/queue_console_$(date +%Y%m%d_%H%M%S).log"

# Número de GPUs por experimento (alterar se quiser multi-gpu)
# Como os experimentos de ImageNet-100 são grandes, definimos 2 GPUs.
# O launcher (`auto_run_on_free_gpu.py`) aguardará pelo menos esse número livre
# antes de disparar – portanto as duas devem estar abaixo do threshold.
GPUS=2    # set to >1 to run with torchrun (distributed); mudar caso necessario

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
NC='\033[0m' # No Color

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    TagFex Auto Experiment Queue${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"

log_progress "🚀 Iniciando fila de experimentos"
log_progress "📁 Logs salvos em: $LOG_DIR"
log_progress "📊 Log de progresso: $PROGRESS_LOG"

# Função para enfileirar experimento
# Arguments: 1=config_file  2=description
# Usa variável global GPUS para escolher entre single- ou multi-gpu (torchrun)
queue_experiment() {
    local config_file=$1
    local description=$2
    
    log_progress "📋 Enfileirando: $description"
    log_progress "   Config: $config_file"
    log_progress "   GPUs necessárias: $GPUS"
    
    echo -e "${YELLOW}📋 Enfileirando:${NC} $description"
    echo -e "   Config: $config_file"
    echo -e "   GPUs necessárias: $GPUS"
    echo -e "   Aguardando GPU(s) disponíveis...\n"
    
    # montar comando de treino; suporta multi-gpu com torchrun
    if [ "$GPUS" -gt 1 ]; then
        train_cmd="torchrun --nproc_per_node=$GPUS python3 main.py train --exp-configs $config_file"
    else
        train_cmd="python3 main.py train --exp-configs $config_file"
    fi

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus $GPUS \
        --threshold $THRESHOLD \
        $MEMORY_THRESHOLD \
        --interval $INTERVAL \
        --log-dir $LOG_DIR \
        --no-wait
    
    if [ $? -eq 0 ]; then
        log_progress "✅ Experimento disparado: $description"
        # Extrair nome da sessão screen do output (última sessão criada)
        local screen_session=$(screen -ls | grep -oE '[0-9]+\.[a-zA-Z0-9_-]+' | tail -1)
        if [ -n "$screen_session" ]; then
            log_progress "   📺 Sessão screen: $screen_session"
        fi
        echo -e "${GREEN}✅ Experimento disparado com sucesso!${NC}\n"
    else
        log_progress "❌ ERRO ao disparar experimento: $description"
        echo -e "${RED}❌ Erro ao disparar experimento${NC}\n"
        return 1
    fi
    
    # Pequeno delay entre disparos
    sleep 5
}

# ═══════════════════════════════════════════════════════════
# Queue de experimentos ImageNet-100
# ═══════════════════════════════════════════════════════════

echo -e "${YELLOW}═══ ImageNet-100 Experiments ═══${NC}\n"

# ImageNet-100 50-10 Baseline Local
queue_experiment \
    "configs/all_in_one/imagenet100_50-10_baseline_local_resnet18.yaml" \
    "ImageNet-100 50-10 Baseline (Local Anchor)"

# ImageNet-100 10-10 ANT (Best Configuration)
queue_experiment \
    "configs/all_in_one/imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
    "ImageNet-100 10-10 ANT (β=0.5, margin=0.5, Local)"

# ImageNet-100 50-10 ANT (Best Configuration)
queue_experiment \
    "configs/all_in_one/imagenet100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
    "ImageNet-100 50-10 ANT (β=0.5, margin=0.5, Local)"

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
