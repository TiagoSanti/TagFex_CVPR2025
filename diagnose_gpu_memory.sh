#!/bin/bash
# Diagnóstico avançado de memória GPU
# Identifica processos mortos, idle e sugere ações

echo "════════════════════════════════════════════════════════════════"
echo "🔬 GPU Memory Diagnostic Tool"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Verificar dependências
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi não encontrado${NC}"
    exit 1
fi

# Função para verificar se processo está ativo
check_process_status() {
    local pid=$1
    
    if ! ps -p $pid > /dev/null 2>&1; then
        echo -e "${RED}💀 MORTO${NC}"
        return 1
    fi
    
    # Verificar CPU usage (se < 0.1% por muito tempo, provavelmente está idle)
    local cpu_usage=$(ps -p $pid -o %cpu= 2>/dev/null | xargs)
    
    if [ -z "$cpu_usage" ]; then
        echo -e "${RED}💀 MORTO${NC}"
        return 1
    fi
    
    # Verificar estado do processo
    local state=$(ps -p $pid -o state= 2>/dev/null | xargs)
    
    case $state in
        R|D)
            echo -e "${GREEN}✅ ATIVO${NC}"
            return 0
            ;;
        S)
            if (( $(echo "$cpu_usage < 0.1" | bc -l) )); then
                echo -e "${YELLOW}😴 IDLE/SLEEP${NC}"
                return 2
            else
                echo -e "${GREEN}✅ ATIVO${NC}"
                return 0
            fi
            ;;
        T|Z)
            echo -e "${RED}⚠️  PARADO/ZOMBIE${NC}"
            return 3
            ;;
        *)
            echo -e "${YELLOW}❓ DESCONHECIDO${NC}"
            return 4
            ;;
    esac
}

# Função para verificar há quanto tempo processo está rodando
get_process_runtime() {
    local pid=$1
    ps -p $pid -o etime= 2>/dev/null | xargs
}

# Status geral
echo -e "${CYAN}═══ Resumo das GPUs ═══${NC}"
echo ""
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | while IFS=',' read -r idx name util mem_used mem_total temp; do
    idx=$(echo $idx | xargs)
    name=$(echo $name | xargs)
    util=$(echo $util | xargs | sed 's/ %//')
    mem_used=$(echo $mem_used | xargs | sed 's/ MiB//')
    mem_total=$(echo $mem_total | xargs | sed 's/ MiB//')
    temp=$(echo $temp | xargs)
    
    mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
    
    echo -e "${BLUE}GPU $idx${NC} ($name)"
    echo "  🔥 Temperatura: $temp"
    echo "  ⚡ Utilização: $util%"
    echo "  💾 Memória: ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
    
    # Alerta se memória alta mas utilização baixa
    if [ "$util" -lt 5 ] && (( $(echo "$mem_percent > 50" | bc -l) )); then
        echo -e "  ${RED}⚠️  SUSPEITO: Alta memória (${mem_percent}%) mas baixa utilização (${util}%)${NC}"
        echo -e "     ${YELLOW}→ Possível processo morto ou idle${NC}"
    fi
    echo ""
done

# Análise detalhada por GPU
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
TOTAL_ZOMBIES=0
TOTAL_IDLE=0
PIDS_TO_KILL=()

echo -e "${CYAN}═══ Análise Detalhada de Processos ═══${NC}"
echo ""

for (( gpu=0; gpu<$GPU_COUNT; gpu++ )); do
    echo -e "${MAGENTA}┌─────────────────────────────────────────────────────┐${NC}"
    echo -e "${MAGENTA}│  GPU $gpu - Análise de Processos                      │${NC}"
    echo -e "${MAGENTA}└─────────────────────────────────────────────────────┘${NC}"
    echo ""
    
    # Pegar processos computacionais
    HAS_PROCESSES=false
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i $gpu 2>/dev/null | while IFS=',' read -r pid pname mem; do
        HAS_PROCESSES=true
        pid=$(echo $pid | xargs)
        pname=$(echo $pname | xargs)
        mem=$(echo $mem | xargs)
        
        if [ -z "$pid" ]; then
            continue
        fi
        
        echo -e "${YELLOW}  📊 PID $pid${NC} | $mem | $pname"
        
        # Verificar se processo existe
        if ps -p $pid > /dev/null 2>&1; then
            USER=$(ps -o user= -p $pid 2>/dev/null)
            CMD=$(ps -o cmd= -p $pid 2>/dev/null | head -c 80)
            RUNTIME=$(get_process_runtime $pid)
            CPU=$(ps -p $pid -o %cpu= 2>/dev/null | xargs)
            STATUS_CODE=0
            STATUS=$(check_process_status $pid)
            STATUS_CODE=$?
            
            echo "     👤 Usuário: $USER"
            echo "     ⏱️  Tempo ativo: $RUNTIME"
            echo "     🖥️  CPU usage: ${CPU}%"
            echo "     📝 Comando: $CMD"
            echo "     🔍 Status: $STATUS"
            
            # Ação recomendada baseada no status
            case $STATUS_CODE in
                1|3)
                    echo -e "     ${RED}💡 AÇÃO: Processo morto/zombie - PODE MATAR SEGURAMENTE${NC}"
                    echo -e "     ${RED}   → kill -9 $pid  (ou fuser -k /dev/nvidia$gpu)${NC}"
                    TOTAL_ZOMBIES=$((TOTAL_ZOMBIES + 1))
                    PIDS_TO_KILL+=($pid)
                    ;;
                2)
                    echo -e "     ${YELLOW}💡 AÇÃO: Processo idle - VERIFICAR antes de matar${NC}"
                    echo -e "     ${YELLOW}   → Pode estar em debug, aguardando input, ou legitimamente pausado${NC}"
                    echo -e "     ${YELLOW}   → Se certeza que pode matar: kill $pid${NC}"
                    TOTAL_IDLE=$((TOTAL_IDLE + 1))
                    ;;
                0)
                    echo -e "     ${GREEN}💡 AÇÃO: Processo ativo - NÃO MATAR${NC}"
                    ;;
            esac
        else
            echo -e "     ${RED}💀 Processo não encontrado no sistema${NC}"
            echo -e "     ${RED}💡 AÇÃO: Memória órfã - será liberada ao reiniciar driver NVIDIA${NC}"
            echo -e "     ${RED}   → sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia${NC}"
            echo -e "     ${RED}   → sudo modprobe nvidia${NC}"
            echo -e "     ${RED}   OU mais simples: Reiniciar a máquina${NC}"
            TOTAL_ZOMBIES=$((TOTAL_ZOMBIES + 1))
        fi
        
        echo ""
    done
    
    if [ "$HAS_PROCESSES" = false ]; then
        # Verificar processos via fuser (detecta processos sem compute capability)
        FUSER_PIDS=$(fuser /dev/nvidia${gpu} 2>/dev/null)
        if [ -n "$FUSER_PIDS" ]; then
            echo -e "${YELLOW}  ⚠️  Processos usando /dev/nvidia${gpu} (via fuser):${NC}"
            for fpid in $FUSER_PIDS; do
                if ps -p $fpid > /dev/null 2>&1; then
                    CMD=$(ps -o cmd= -p $fpid | head -c 60)
                    USER=$(ps -o user= -p $fpid)
                    echo "     PID $fpid ($USER): $CMD"
                fi
            done
            echo ""
        else
            echo -e "${GREEN}  ✅ Nenhum processo detectado${NC}"
            echo ""
        fi
    fi
done

# Resumo e recomendações
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📋 Resumo e Recomendações${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $TOTAL_ZOMBIES -gt 0 ]; then
    echo -e "${RED}⚠️  Encontrados $TOTAL_ZOMBIES processo(s) morto(s)/zombie(s)${NC}"
    echo ""
    echo -e "${YELLOW}Opções para limpar:${NC}"
    echo ""
    
    if [ ${#PIDS_TO_KILL[@]} -gt 0 ]; then
        echo -e "${RED}1. Matar processos específicos (RECOMENDADO):${NC}"
        for pid in "${PIDS_TO_KILL[@]}"; do
            echo "   kill -9 $pid"
        done
        echo ""
        echo -e "   ${CYAN}Ou todos de uma vez:${NC}"
        echo "   kill -9 ${PIDS_TO_KILL[*]}"
        echo ""
    fi
    
    echo -e "${RED}2. Limpar GPU específica (CUIDADO - mata TODOS processos):${NC}"
    echo "   sudo fuser -k /dev/nvidia0  # Para GPU 0"
    echo "   sudo fuser -k /dev/nvidia1  # Para GPU 1"
    echo ""
    
    echo -e "${RED}3. Reiniciar driver NVIDIA (ÚLTIMA OPÇÃO):${NC}"
    echo "   sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia"
    echo "   sudo modprobe nvidia"
    echo ""
    
    echo -e "${RED}4. Reiniciar máquina (MAIS SIMPLES):${NC}"
    echo "   sudo reboot"
    echo ""
fi

if [ $TOTAL_IDLE -gt 0 ]; then
    echo -e "${YELLOW}ℹ️  Encontrados $TOTAL_IDLE processo(s) idle/suspenso(s)${NC}"
    echo "  → Verificar se são legítimos (debug, pausa intencional)"
    echo "  → Se não, matar manualmente com 'kill <PID>'"
    echo ""
fi

if [ $TOTAL_ZOMBIES -eq 0 ] && [ $TOTAL_IDLE -eq 0 ]; then
    echo -e "${GREEN}✅ Todos os processos em GPUs estão ativos e legítimos${NC}"
    echo ""
fi

echo -e "${CYAN}💡 Para usar GPUs com processos idle no auto-launcher:${NC}"
echo "   python3 auto_run_on_free_gpu.py \\"
echo "       --command \"...\" \\"
echo "       --threshold 1.0 \\"
echo "       --memory-threshold 20.0  # Só usa GPU com <20% memória"
echo ""

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
