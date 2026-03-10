#!/bin/bash
# Wrapper para iniciar o monitoramento de GPUs em sessão screen
# Permite desconectar do SSH sem interromper o monitoramento

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_SCRIPT="$SCRIPT_DIR/run_experiments_queue.sh"
MONITOR_SESSION="tagfex_gpu_monitor"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    TagFex GPU Monitor - Screen Launcher${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Verificar se screen está instalado
if ! command -v screen &> /dev/null; then
    echo -e "${RED}❌ 'screen' não encontrado. Instalando...${NC}\n"
    sudo apt-get update && sudo apt-get install -y screen
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Erro ao instalar screen. Abortando.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ screen instalado com sucesso!${NC}\n"
fi

# Verificar se já existe uma sessão com o mesmo nome
if screen -list | grep -q "$MONITOR_SESSION"; then
    echo -e "${YELLOW}⚠️  Sessão '$MONITOR_SESSION' já existe!${NC}\n"
    echo "Opções:"
    echo "  1) Anexar à sessão existente"
    echo "  2) Matar sessão existente e criar nova"
    echo "  3) Cancelar"
    echo ""
    read -p "Escolha (1-3): " choice
    
    case $choice in
        1)
            echo -e "\n${GREEN}📺 Anexando à sessão existente...${NC}"
            screen -r "$MONITOR_SESSION"
            exit 0
            ;;
        2)
            echo -e "\n${YELLOW}🔪 Matando sessão existente...${NC}"
            screen -X -S "$MONITOR_SESSION" quit
            sleep 1
            ;;
        3)
            echo -e "\n${RED}❌ Cancelado.${NC}"
            exit 0
            ;;
        *)
            echo -e "\n${RED}❌ Opção inválida. Cancelado.${NC}"
            exit 1
            ;;
    esac
fi

echo -e "${GREEN}🚀 Iniciando monitoramento em sessão screen...${NC}"
echo -e "${BLUE}📝 Logs serão salvos em: logs/auto_experiments/${NC}\n"

# Criar sessão screen detached e executar o script de queue
# Ativa o ambiente virtual antes de executar o script
screen -dmS "$MONITOR_SESSION" bash -c "cd $SCRIPT_DIR && [ -f .venv/bin/activate ] && source .venv/bin/activate; $QUEUE_SCRIPT; echo ''; echo 'Fila de experimentos concluída. Pressione ENTER para fechar.'; read"

# Aguardar 1 segundo para garantir que a sessão foi criada
sleep 1

# Verificar se a sessão foi criada com sucesso
if screen -list | grep -q "$MONITOR_SESSION"; then
    echo -e "${GREEN}✅ Sessão criada com sucesso!${NC}\n"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}    Informações da Sessão${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
    echo -e "📺 ${GREEN}Nome da sessão:${NC} $MONITOR_SESSION"
    echo -e "📋 ${GREEN}Script executado:${NC} $QUEUE_SCRIPT"
    echo ""
    echo -e "${YELLOW}Comandos úteis:${NC}"
    echo -e "  📺 Anexar à sessão (ver progresso):  ${GREEN}screen -r $MONITOR_SESSION${NC}"
    echo -e "  🔌 Desanexar (dentro da sessão):    ${GREEN}Ctrl+A, depois D${NC}"
    echo -e "  📋 Listar todas as sessões:          ${GREEN}screen -ls${NC}"
    echo -e "  ❌ Matar a sessão:                   ${GREEN}screen -X -S $MONITOR_SESSION quit${NC}"
    echo ""
    echo -e "${YELLOW}📝 Monitoramento de logs:${NC}"
    echo -e "  📊 Log de progresso:                 ${GREEN}tail -f logs/auto_experiments/queue_progress.log${NC}"
    echo -e "  📄 Log completo do console:          ${GREEN}tail -f logs/auto_experiments/queue_console_*.log${NC}"
    echo -e "  📁 Logs individuais:                 ${GREEN}ls -lh logs/auto_experiments/auto_gpu*.log${NC}"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
    echo -e "${GREEN}💡 Dica:${NC} Você pode desconectar do SSH agora."
    echo -e "    O monitoramento continuará rodando e disparando experimentos!"
    echo ""
    echo -e "${YELLOW}📺 Deseja anexar à sessão agora? (s/N)${NC} "
    read -p "" attach_now
    
    if [[ "$attach_now" =~ ^[Ss]$ ]]; then
        echo ""
        echo -e "${GREEN}Anexando à sessão...${NC}"
        echo -e "${YELLOW}(Para desanexar: Ctrl+A, depois D)${NC}"
        sleep 2
        screen -r "$MONITOR_SESSION"
    else
        echo ""
        echo -e "${GREEN}✅ Monitoramento iniciado em background!${NC}"
        echo -e "   Use '${GREEN}screen -r $MONITOR_SESSION${NC}' para ver o progresso."
        echo ""
    fi
else
    echo -e "${RED}❌ Erro ao criar sessão screen.${NC}"
    exit 1
fi
