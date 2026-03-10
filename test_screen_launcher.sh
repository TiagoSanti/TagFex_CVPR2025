#!/bin/bash
# Script de teste para verificar funcionalidade de sessões screen

echo "🧪 Testando Auto GPU Launcher com Screen Sessions"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Verificar se screen está instalado
if ! command -v screen &> /dev/null; then
    echo "⚠️  'screen' não encontrado. O script tentará instalar automaticamente."
    echo ""
fi

# Criar comando de teste simples
TEST_COMMAND="echo 'Iniciando teste...' && sleep 5 && echo 'Teste completo!' && nvidia-smi"

echo "📋 Informações do teste:"
echo "   Comando: $TEST_COMMAND"
echo "   Config: configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml"
echo "   Threshold: 100.0 (modo: apenas memória)"
echo "   Memory Threshold: 90.0%"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Executar teste
python3 auto_run_on_free_gpu.py \
    --command "python -c \"$TEST_COMMAND\"" \
    --threshold 100.0 \
    --memory-threshold 90.0 \
    --interval 5

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Teste concluído com sucesso!"
    echo ""
    echo "📋 Verificar sessões screen ativas:"
    echo "   screen -ls"
    echo ""
    echo "📺 Para ver sessões em execução:"
    screen -ls
else
    echo "❌ Teste falhou com código: $EXIT_CODE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "💡 Comandos úteis:"
echo "   • Listar sessões:    screen -ls"
echo "   • Anexar a sessão:   screen -r <nome>"
echo "   • Desanexar:         Ctrl+A, depois D (dentro da sessão)"
echo "   • Matar sessão:      screen -X -S <nome> quit"
echo ""
