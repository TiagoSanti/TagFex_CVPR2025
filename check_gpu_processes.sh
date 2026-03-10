#!/bin/bash
# Script para verificar processos em GPUs e identificar processos idle

echo "════════════════════════════════════════════════════════════════"
echo "🔍 GPU Process Checker - Identifica processos idle ocupando memória"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verificar nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi não encontrado${NC}"
    exit 1
fi

# Status geral
echo -e "${BLUE}═══ Status Geral das GPUs ═══${NC}"
gpustat 2>/dev/null || nvidia-smi
echo ""

# Analisar cada GPU
echo -e "${BLUE}═══ Análise Detalhada por GPU ═══${NC}"
echo ""

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

for (( gpu=0; gpu<$GPU_COUNT; gpu++ )); do
    echo -e "${YELLOW}━━━ GPU $gpu ━━━${NC}"
    
    # Obter utilização e memória
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i $gpu)
    UTIL=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
    MEM_USED=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    MEM_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | xargs)
    MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($MEM_USED/$MEM_TOTAL)*100}")
    
    echo "  Utilização: ${UTIL}%"
    echo "  Memória: ${MEM_USED}MB / ${MEM_TOTAL}MB (${MEM_PERCENT}%)"
    
    # Detectar se está idle (baixa utilização mas alta memória)
    if [ "$UTIL" -lt 5 ] && [ $(echo "$MEM_PERCENT > 50" | bc -l) -eq 1 ]; then
        echo -e "  ${RED}⚠️  POSSÍVEL PROCESSO IDLE (util<5% mas mem>${MEM_PERCENT}%)${NC}"
    elif [ "$UTIL" -lt 2 ]; then
        echo -e "  ${GREEN}✅ GPU disponível${NC}"
    else
        echo -e "  ${YELLOW}🔴 GPU ocupada${NC}"
    fi
    
    # Processos na GPU
    echo ""
    echo "  Processos:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i $gpu 2>/dev/null | while IFS=',' read -r pid pname mem; do
        pid=$(echo $pid | xargs)
        pname=$(echo $pname | xargs)
        mem=$(echo $mem | xargs)
        
        if [ -n "$pid" ]; then
            # Verificar se processo existe
            if ps -p $pid > /dev/null 2>&1; then
                USER=$(ps -o user= -p $pid)
                CMD=$(ps -o cmd= -p $pid | cut -c1-60)
                echo "    PID $pid ($USER): $mem - $CMD"
            else
                echo -e "    ${RED}PID $pid: $mem - [processo morto, memória não liberada]${NC}"
            fi
        fi
    done
    
    # Verificar /dev/nvidia*
    FUSER_OUTPUT=$(fuser /dev/nvidia${gpu} 2>/dev/null)
    if [ -n "$FUSER_OUTPUT" ]; then
        echo ""
        echo "  PIDs usando /dev/nvidia${gpu}: $FUSER_OUTPUT"
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo -e "${BLUE}💡 Dicas:${NC}"
echo "  • Processos com util<5% e mem alta = idle (podem ser mortos)"
echo "  • Use 'kill <PID>' para matar processo específico"
echo "  • Use 'fuser -k /dev/nvidia<N>' para limpar GPU (cuidado!)"
echo "  • Use '--memory-threshold' no auto_run_on_free_gpu.py"
echo "════════════════════════════════════════════════════════════════"
