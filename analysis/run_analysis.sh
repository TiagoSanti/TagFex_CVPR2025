#!/bin/bash
# Script auxiliar para executar análises TagFex
# Uso: ./run_analysis.sh <script_name> [args...]

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Diretório raiz do projeto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"
SCRIPTS_DIR="$PROJECT_ROOT/analysis/scripts"

# Verificar se ambiente virtual existe
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Erro: Ambiente virtual não encontrado em $VENV_PATH${NC}"
    exit 1
fi

# Ativar ambiente virtual
source "$VENV_PATH/bin/activate"

# Verificar se script foi especificado
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Uso: $0 <script_name> [args...]${NC}"
    echo ""
    echo "Scripts disponíveis:"
    echo "  1. compare_baseline_vs_local      - Compara InfoNCE baseline vs âncora local"
    echo "  2. visualize_local_anchor_theory  - Visualiza teoria da âncora local"
    echo "  3. compare_tagfex_ant             - Compara experimentos TagFex+ANT"
    echo "  4. plot_nme1_curves              - Plota curvas NME1"
    echo "  5. analyze_baseline               - Analisa distâncias baseline"
    echo ""
    echo "Exemplo:"
    echo "  $0 compare_baseline_vs_local"
    exit 1
fi

SCRIPT_NAME="$1"
shift  # Remove primeiro argumento

# Adicionar .py se não tiver
if [[ ! "$SCRIPT_NAME" == *.py ]]; then
    SCRIPT_NAME="${SCRIPT_NAME}.py"
fi

SCRIPT_PATH="$SCRIPTS_DIR/$SCRIPT_NAME"

# Verificar se script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Erro: Script não encontrado: $SCRIPT_PATH${NC}"
    echo ""
    echo "Scripts disponíveis em $SCRIPTS_DIR:"
    ls -1 "$SCRIPTS_DIR"/*.py 2>/dev/null | xargs -n1 basename
    exit 1
fi

# Executar script
echo -e "${GREEN}Executando: $SCRIPT_NAME${NC}"
echo -e "${YELLOW}Diretório: $SCRIPTS_DIR${NC}"
echo ""

cd "$SCRIPTS_DIR" && python "$SCRIPT_NAME" "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Script executado com sucesso!${NC}"
else
    echo ""
    echo -e "${RED}✗ Script falhou com código de saída: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
