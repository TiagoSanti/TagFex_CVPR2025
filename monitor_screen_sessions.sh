#!/bin/bash
# Script para monitorar sessões screen e logar status dos experimentos
# Uso: ./monitor_screen_sessions.sh [intervalo_segundos]

INTERVAL=${1:-60}  # Intervalo padrão: 60 segundos
LOG_DIR="./logs/auto_experiments"
PROGRESS_LOG="$LOG_DIR/queue_progress.log"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Criar diretório de logs se não existir
mkdir -p "$LOG_DIR"

# Função para log com timestamp
log_status() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$PROGRESS_LOG"
}

# Arquivo temporário para rastrear sessões
SESSIONS_FILE="/tmp/tagfex_screen_sessions.txt"

# Inicializar arquivo de sessões se não existir
if [ ! -f "$SESSIONS_FILE" ]; then
    screen -ls | grep -oE '[0-9]+\.[a-zA-Z0-9_-]+' > "$SESSIONS_FILE"
    log_status "🔍 Iniciando monitoramento de sessões screen"
    log_status "📋 $(wc -l < $SESSIONS_FILE) sessões ativas detectadas"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    TagFex Screen Session Monitor${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
echo -e "Intervalo: ${GREEN}${INTERVAL}s${NC}"
echo -e "Log de progresso: ${GREEN}${PROGRESS_LOG}${NC}"
echo -e "\nPressione ${YELLOW}Ctrl+C${NC} para sair\n"

log_status "🚀 Monitor de sessões iniciado (intervalo: ${INTERVAL}s)"

# Loop de monitoramento
while true; do
    # Obter lista atual de sessões
    CURRENT_SESSIONS=$(screen -ls 2>/dev/null | grep -oE '[0-9]+\.[a-zA-Z0-9_-]+' || true)
    
    if [ -f "$SESSIONS_FILE" ]; then
        # Verificar sessões que terminaram
        while IFS= read -r session; do
            if ! echo "$CURRENT_SESSIONS" | grep -q "$session"; then
                log_status "✅ Sessão encerrada: $session"
            fi
        done < "$SESSIONS_FILE"
        
        # Verificar novas sessões
        for session in $CURRENT_SESSIONS; do
            if ! grep -q "$session" "$SESSIONS_FILE"; then
                log_status "🆕 Nova sessão detectada: $session"
            fi
        done
    fi
    
    # Atualizar arquivo de sessões
    echo "$CURRENT_SESSIONS" > "$SESSIONS_FILE"
    
    # Exibir status atual
    NUM_SESSIONS=$(echo "$CURRENT_SESSIONS" | wc -w)
    timestamp=$(date '+%H:%M:%S')
    echo -ne "\r[$timestamp] 📺 Sessões ativas: ${GREEN}$NUM_SESSIONS${NC} | Log: $(wc -l < "$PROGRESS_LOG" 2>/dev/null || echo 0) linhas    "
    
    sleep "$INTERVAL"
done
