#!/bin/bash
# Script de teste para verificar o sistema de monitoramento em screen

echo "🧪 Testando Sistema de Monitoramento em Screen"
echo "════════════════════════════════════════════════════════════"
echo ""

# Verificar se screen está instalado
if ! command -v screen &> /dev/null; then
    echo "❌ 'screen' não encontrado."
    exit 1
fi

echo "✅ screen está instalado"
echo ""

# Criar um script de teste temporário que simula experimentos
cat > /tmp/test_queue_tagfex.sh << 'EOF'
#!/bin/bash
echo "═══════════════════════════════════════════════════════════"
echo "    Teste de Fila de Experimentos"
echo "═══════════════════════════════════════════════════════════"
echo ""

for i in 1 2 3; do
    echo "📋 Experimento $i: Aguardando GPU..."
    sleep 2
    echo "✅ GPU disponível! Disparando experimento $i..."
    sleep 2
    echo "🚀 Experimento $i em execução..."
    sleep 3
    echo "✅ Experimento $i concluído!"
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "✅ Todos os experimentos concluídos!"
echo "═══════════════════════════════════════════════════════════"
EOF

chmod +x /tmp/test_queue_tagfex.sh

# Iniciar sessão screen de teste
MONITOR_SESSION="test_tagfex_monitor"

echo "🚀 Iniciando sessão screen de teste: $MONITOR_SESSION"
echo ""

# Verificar se já existe
if screen -list | grep -q "$MONITOR_SESSION"; then
    echo "⚠️  Matando sessão existente..."
    screen -X -S "$MONITOR_SESSION" quit
    sleep 1
fi

# Criar sessão
screen -dmS "$MONITOR_SESSION" bash -c "/tmp/test_queue_tagfex.sh; echo ''; echo 'Pressione ENTER para fechar.'; read"

sleep 1

if screen -list | grep -q "$MONITOR_SESSION"; then
    echo "✅ Sessão criada com sucesso!"
    echo ""
    echo "📋 Sessões ativas:"
    screen -ls
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "💡 Comandos para testar:"
    echo ""
    echo "  1. Anexar à sessão (ver execução em tempo real):"
    echo "     screen -r $MONITOR_SESSION"
    echo ""
    echo "  2. Desanexar (dentro da sessão):"
    echo "     Ctrl+A, depois D"
    echo ""
    echo "  3. Ver se sessão ainda existe:"
    echo "     screen -ls"
    echo ""
    echo "  4. Matar a sessão:"
    echo "     screen -X -S $MONITOR_SESSION quit"
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "⏱️  A sessão de teste rodará por ~20 segundos."
    echo "   Use 'screen -r $MONITOR_SESSION' para ver!"
    echo ""
else
    echo "❌ Erro ao criar sessão."
    exit 1
fi
