#!/usr/bin/env bash
# Controlled smoke tests and seed-1993 pilot on Wolverine GPU 1.

set -uo pipefail

REPO=/home/tiago/ANT_validation_20260812/repo
PYTHON=/home/tiago/TagFex_updated/.venv/bin/python
GPU=1
BASE_COMMIT=d72033940ba0eab675e55ea7f65eedaa733aef02
BASE_CONFIG=configs/all_in_one/tiny_imagenet_100-20_antB0_nceA1_antGlobal_nceGlobal_debug_resnet18.yaml
COMMON_CONFIG=validation/tin100_clip/common.yaml
BASELINE_CONFIG=validation/tin100_clip/baseline_clip5.yaml
ANT_CONFIG=validation/tin100_clip/ant_fs_ar_clip5.yaml
SMOKE_CONFIG=validation/tin100_clip/smoke.yaml
PROVENANCE=validation/tin100_clip/provenance.py
RUNTIME=/home/tiago/ANT_validation_20260812/validation_runtime
MODE=${MODE:-smoke}

mkdir -p "$RUNTIME/manifests"
cd "$REPO" || exit 1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

gpu_guard() {
    local gpu_uuid
    gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
    if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | grep -Fxq "$gpu_uuid"; then
        log "ERROR: physical GPU $GPU already has a compute process"
        return 1
    fi
}

expected_output_dir() {
    local phase=$1 condition=$2 seed=$3 root suffix
    if [[ "$phase" == smoke ]]; then
        root=/home/tiago/ANT_validation_20260812/logs/smoke
    else
        root=/home/tiago/ANT_validation_20260812/logs/full
    fi
    if [[ "$condition" == baseline_clip5 ]]; then
        suffix="antB0_nceA1_antSymmetricFull_nceGlobal_s${seed}"
    else
        suffix="antB0.5_nceA1_antM0.5_antSymmetricFull_nceGlobal_s${seed}"
    fi
    printf '%s/tin100_%s\n' "$root" "$suffix"
}

validate_output() {
    local output_dir=$1
    [[ -s "$output_dir/exp_gistlog.log" ]] || {
        log "ERROR: missing performance log in $output_dir"
        return 1
    }
    if grep -Eiq 'Traceback|CUDA out of memory|Loss is NaN|has NaN: True' \
        "$output_dir/exp_stdlog0.log"; then
        log "ERROR: failure marker found in $output_dir/exp_stdlog0.log"
        return 1
    fi
}

run_one() {
    local phase=$1 condition=$2 seed=$3 condition_config output_dir run_id manifest
    local -a configs command

    if [[ "$condition" == baseline_clip5 ]]; then
        condition_config=$BASELINE_CONFIG
    else
        condition_config=$ANT_CONFIG
    fi
    configs=("$BASE_CONFIG" "$COMMON_CONFIG" "$condition_config")
    if [[ "$phase" == smoke ]]; then
        configs+=("$SMOKE_CONFIG")
    fi

    output_dir=$(expected_output_dir "$phase" "$condition" "$seed")
    run_id="tin100_${phase}_${condition}_s${seed}"
    manifest="$RUNTIME/manifests/${run_id}.json"
    if [[ -e "$output_dir" || -e "$manifest" ]]; then
        log "ERROR: refusing ambiguous rerun; output or manifest exists for $run_id"
        return 1
    fi

    command=("$PYTHON" main.py train --exp-configs "${configs[@]}" --seed "$seed" \
        --device cuda --force-no-debug --disable-save-ckpt)

    log "START $run_id"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PROVENANCE" prepare \
        --repo "$REPO" --manifest "$manifest" --run-id "$run_id" \
        --condition "$condition" --phase "$phase" --seed "$seed" \
        --output-dir "$output_dir" --command "CUDA_VISIBLE_DEVICES=$GPU ${command[*]}" \
        --config "${configs[@]}"

    local exit_code=0
    CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" || exit_code=$?
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PROVENANCE" finalize \
        --manifest "$manifest" --exit-code "$exit_code"
    if [[ "$exit_code" -ne 0 ]]; then
        log "FAILED $run_id exit=$exit_code"
        return "$exit_code"
    fi
    validate_output "$output_dir" || return 1
    log "DONE $run_id"
}

log "Validating source snapshot"
"$PYTHON" "$PROVENANCE" snapshot --repo "$REPO" --base-commit "$BASE_COMMIT"
gpu_guard || exit 1

if [[ "$MODE" == smoke || "$MODE" == all ]]; then
    run_one smoke baseline_clip5 1993 || exit 1
    run_one smoke ant_fs_ar_clip5 1993 || exit 1
    touch "$RUNTIME/smoke_complete.marker"
    log "SMOKE COMPLETE"
fi

if [[ "$MODE" == pilot || "$MODE" == all ]]; then
    [[ -e "$RUNTIME/smoke_complete.marker" ]] || {
        log "ERROR: pilot requires a completed smoke phase"
        exit 1
    }
    # Re-check that GPU 1 was not claimed before the full pilot.
    gpu_guard || exit 1
    run_one full baseline_clip5 1993 || exit 1
    run_one full ant_fs_ar_clip5 1993 || exit 1
    touch "$RUNTIME/pilot_complete.marker"
    log "PILOT COMPLETE"
fi

if [[ "$MODE" != smoke && "$MODE" != pilot && "$MODE" != all ]]; then
    log "ERROR: MODE must be smoke, pilot, or all"
    exit 2
fi

log "Seeds 1994/1995 intentionally require a separate decision"
