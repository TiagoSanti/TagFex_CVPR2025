#!/usr/bin/env bash
# Incrementally mirror Wolverine results to Xavier over its SSH port 2222.
# Remote files are never deleted. COMPLETED_ONLY=1 copies only experiment
# directories whose gist contains every task expected by the protocol.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
REMOTE_HOST="${REMOTE_HOST:-tiago@10.87.10.46}"
# Wolverine's sshd is exposed on 2222; keep this explicit for rsync as well as
# for the systemd status probe.
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_KEY="${REMOTE_KEY:-/home/tiago/.ssh/id_ed25519_wolverine}"
REMOTE_ROOT="${REMOTE_ROOT:-/var/tmp/tiago/ANT_refdetach_20260824}"
REMOTE_ORCHESTRATOR="${REMOTE_ORCHESTRATOR-$REMOTE_ROOT/repo/logs/auto_experiments}"
REMOTE_UNIT="${REMOTE_UNIT:-ant-refdetach-20260824.service}"
REMOTE_UNITS="${REMOTE_UNITS:-$REMOTE_UNIT}"
LOCAL_RESULTS="${LOCAL_RESULTS:-$SCRIPT_DIR/logs}"
LOCAL_ORCHESTRATOR="${LOCAL_ORCHESTRATOR:-$SCRIPT_DIR/logs/auto_experiments/wolverine_refdetach_20260824}"
SYNC_LOG="${SYNC_LOG:-$LOCAL_ORCHESTRATOR/sync.log}"
INTERVAL="${INTERVAL:-300}"
LOCKFILE="${LOCKFILE:-$TAGFEX_LOCK_DIR/tagfex_wolverine_refdetach_sync.lock}"
RUN_MODE="${RUN_MODE:-watch}"
COMPLETED_ONLY="${COMPLETED_ONLY:-0}"

SSH_ARGS=(
    -p "$REMOTE_PORT"
    -i "$REMOTE_KEY"
    -o IdentitiesOnly=yes
    -o BatchMode=yes
    -o ConnectTimeout=15
)
RSYNC_SSH="ssh -p $REMOTE_PORT -i $REMOTE_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"

mkdir -p "$LOCAL_RESULTS" "$LOCAL_ORCHESTRATOR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
    printf '[%s] sync already running; exiting\n' "$(date '+%Y-%m-%d %H:%M:%S')" >&2
    exit 0
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SYNC_LOG"
}

sync_once() {
    local result_rc=0 orchestrator_rc=0

    if [[ "$COMPLETED_ONLY" == "1" ]]; then
        sync_completed_results || result_rc=$?
    else
        rsync -az --partial \
            -e "$RSYNC_SSH" \
            "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
            "$LOCAL_RESULTS/" || result_rc=$?
    fi

    if [[ -n "$REMOTE_ORCHESTRATOR" ]]; then
        rsync -az --partial \
            -e "$RSYNC_SSH" \
            "$REMOTE_HOST:$REMOTE_ORCHESTRATOR/" \
            "$LOCAL_ORCHESTRATOR/" || orchestrator_rc=$?
    fi

    if [[ $result_rc -eq 0 && $orchestrator_rc -eq 0 ]]; then
        log "SYNC OK"
        return 0
    fi

    log "SYNC WARNING results_rc=$result_rc orchestrator_rc=$orchestrator_rc"
    return 1
}

expected_tasks() {
    local name="$1" total init increment
    if [[ "$name" =~ ^exp_cifar100_([0-9]+)-([0-9]+)_ ]]; then
        total=100
        init="${BASH_REMATCH[1]}"
        increment="${BASH_REMATCH[2]}"
    elif [[ "$name" =~ ^exp_(cub200|tiny_imagenet)_([0-9]+)-([0-9]+)_ ]]; then
        total=200
        init="${BASH_REMATCH[2]}"
        increment="${BASH_REMATCH[3]}"
    else
        return 1
    fi

    printf '%d\n' "$((1 + (total - init) / increment))"
}

sync_completed_results() {
    local remote_dir count name expected copied=0 rc=0 inventory
    inventory="$(mktemp)"
    if ! ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" \
        "for d in '$REMOTE_ROOT'/logs/exp_*; do
            [ -d \"\$d\" ] || continue
            f=\"\$d/exp_gistlog.log\"
            [ -f \"\$f\" ] || continue
            n=\$(grep -c avg_nme1 \"\$f\" 2>/dev/null || true)
            printf '%s\\t%s\\n' \"\$n\" \"\$d\"
        done" > "$inventory"; then
        rm -f -- "$inventory"
        log "ERROR could not inventory remote results"
        return 1
    fi

    while IFS=$'\t' read -r count remote_dir; do
        [[ -n "$remote_dir" ]] || continue
        name="${remote_dir##*/}"
        if ! expected="$(expected_tasks "$name")"; then
            log "SKIP unrecognized experiment name: $name"
            continue
        fi
        if (( count < expected )); then
            log "SKIP partial $name tasks=$count/$expected"
            continue
        fi

        log "COPY complete $name tasks=$count/$expected"
        if rsync -az --partial \
            -e "$RSYNC_SSH" \
            "$REMOTE_HOST:$remote_dir/" \
            "$LOCAL_RESULTS/$name/"; then
            copied=$((copied + 1))
        else
            rc=$?
        fi
    done < "$inventory"
    rm -f -- "$inventory"
    log "COMPLETED-ONLY copied=$copied"
    return "$rc"
}

remote_queue_active() {
    local unit
    for unit in $REMOTE_UNITS; do
        if ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" \
            systemctl --user is-active --quiet "$unit"; then
            return 0
        fi
    done
    return 1
}

log "SYNC START remote=$REMOTE_HOST:$REMOTE_ROOT local=$LOCAL_RESULTS"

if [[ "$RUN_MODE" == "once" ]]; then
    sync_once
    log "SYNC DONE (one-shot)"
    exit 0
fi

while true; do
    sync_once || true
    if ! remote_queue_active; then
        log "REMOTE QUEUE INACTIVE; performing final sync"
        sync_once || true
        log "SYNC DONE"
        exit 0
    fi
    sleep "$INTERVAL"
done
