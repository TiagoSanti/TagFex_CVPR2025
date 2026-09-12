#!/usr/bin/env bash
# Run on the selected training host before launching the nine-condition matrix.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

cd "$ROOT"
test -x "$PYTHON_BIN"
test -d "$HOME/data/datasets/cifar100/cifar-100-python"

echo "host=$(hostname) root=$ROOT gpu=$GPU"
df -h "$ROOT" /tmp
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

MPLCONFIGDIR="/tmp/tagfex-mpl-preflight-${USER:-user}" \
    "$PYTHON_BIN" -m unittest discover -s tests \
    -p 'test_ant_mechanism_*.py' -v

echo "ANT mechanism preflight passed"
