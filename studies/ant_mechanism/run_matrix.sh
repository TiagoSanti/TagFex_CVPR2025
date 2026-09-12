#!/usr/bin/env bash
# Execute the nine-condition ANT mechanism matrix on one GPU.

set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "short" ]]; then
    echo "usage: GPU=1 bash studies/ant_mechanism/run_matrix.sh [smoke|short]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
GPU="${GPU:-1}"
SEED="${SEED:-1993}"
BASE="configs/all_in_one/cifar100_10-10_antB0_nceA1_antGlobal_nceGlobal_resnet18.yaml"
COMMON="configs/ant_mechanism_study/common.yaml"
SMOKE="configs/ant_mechanism_study/smoke.yaml"
VARIANTS=(
    baseline
    ant_iv_gr
    ant_iv_ar
    ant_fs_gr
    ant_fs_ar
    ant_iv_gr_detach
    ant_iv_ar_detach
    ant_fs_gr_detach
    ant_fs_ar_detach
)

# A recovery launch can select only unfinished conditions without generating
# _v2 directories for conditions already completed by a previous matrix run.
# Example: ANT_STUDY_VARIANTS="ant_iv_ar_detach ant_fs_gr_detach".
if [[ -n "${ANT_STUDY_VARIANTS:-}" ]]; then
    read -r -a VARIANTS <<< "$ANT_STUDY_VARIANTS"
fi

if [[ ${#VARIANTS[@]} -eq 0 ]]; then
    echo "ANT_STUDY_VARIANTS selected no conditions" >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tagfex-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

for variant in "${VARIANTS[@]}"; do
    configs=(
        "$BASE"
        "$COMMON"
        "configs/ant_mechanism_study/${variant}.yaml"
    )
    if [[ "$MODE" == "smoke" ]]; then
        configs+=("$SMOKE")
    fi
    echo "ANT study: mode=$MODE variant=$variant seed=$SEED gpu=$GPU"
    "$PYTHON_BIN" main.py train \
        --exp-configs "${configs[@]}" \
        --seed "$SEED" \
        --device cuda \
        --disable-save-ckpt
done

echo "ANT study matrix completed: mode=$MODE seed=$SEED"
