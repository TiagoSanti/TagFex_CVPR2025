#!/usr/bin/env bash
# Complete the missing Tiny-ImageNet 20--20 ANT-IV-GR seed on Wolverine GPU 1.
# WAIT_PID may point to an existing queue; this run starts only after it exits
# and the target physical GPU has no compute process.

set -uo pipefail

REPO=/home/tiago/ANT_validation_20260812/repo
PYTHON=/home/tiago/TagFex_updated/.venv/bin/python
GPU=1
BASE_COMMIT=d72033940ba0eab675e55ea7f65eedaa733aef02
BASE_CONFIG=validation_runtime/tiny_imagenet_20-20_ant_iv_gr.yaml
RUNTIME_CONFIG=validation_runtime/tin20_fill_runtime.yaml
PROVENANCE=validation/tin100_clip/provenance.py
RUNTIME=/home/tiago/ANT_validation_20260812/validation_runtime
OUTPUT_DIR=/home/tiago/ANT_validation_20260812/logs/full/tin20_antB0.5_nceA1_antM0.5_antGlobal_nceGlobal_s1995
MANIFEST="$RUNTIME/manifests/tin20_full_ant_iv_gr_s1995.json"
RUN_ID=tin20_full_ant_iv_gr_s1995
WAIT_PID=${WAIT_PID:-}

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

gpu_is_busy() {
    local gpu_uuid
    gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU") || return 0
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | grep -Fxq "$gpu_uuid"
}

validate_output() {
    [[ -s "$OUTPUT_DIR/exp_gistlog.log" ]] || {
        log "ERROR: missing performance log in $OUTPUT_DIR"
        return 1
    }
    [[ $(grep -c 'avg_acc1' "$OUTPUT_DIR/exp_gistlog.log") -eq 10 ]] || {
        log "ERROR: expected ten completed tasks in exp_gistlog.log"
        return 1
    }
    if grep -Eiq 'Traceback|CUDA out of memory|Loss is NaN|has NaN: True' \
        "$OUTPUT_DIR/exp_stdlog0.log"; then
        log "ERROR: failure marker found in $OUTPUT_DIR/exp_stdlog0.log"
        return 1
    fi
}

mkdir -p "$RUNTIME/manifests"
cd "$REPO" || exit 1

if [[ -n "$WAIT_PID" ]]; then
    log "QUEUED $RUN_ID behind PID $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
fi

while gpu_is_busy; do
    log "WAIT GPU $GPU is still busy"
    sleep 60
done

if [[ -e "$OUTPUT_DIR" || -e "$MANIFEST" ]]; then
    log "ERROR: refusing ambiguous rerun; output or manifest already exists"
    exit 1
fi

configs=("$BASE_CONFIG" "$RUNTIME_CONFIG")
command=("$PYTHON" main.py train --exp-configs "${configs[@]}" --seed 1995 \
    --device cuda --force-no-debug --disable-save-ckpt)

log "Validating source snapshot"
"$PYTHON" "$PROVENANCE" snapshot --repo "$REPO" --base-commit "$BASE_COMMIT" || exit 1

log "START $RUN_ID"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PROVENANCE" prepare \
    --repo "$REPO" --manifest "$MANIFEST" --run-id "$RUN_ID" \
    --condition ant_iv_gr --phase full --seed 1995 \
    --output-dir "$OUTPUT_DIR" \
    --command "CUDA_VISIBLE_DEVICES=$GPU ${command[*]}" \
    --config "${configs[@]}" || exit 1

exit_code=0
CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" || exit_code=$?
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$PROVENANCE" finalize \
    --manifest "$MANIFEST" --exit-code "$exit_code"
if [[ "$exit_code" -ne 0 ]]; then
    log "FAILED $RUN_ID exit=$exit_code"
    exit "$exit_code"
fi

validate_output || exit 1
log "DONE $RUN_ID"
