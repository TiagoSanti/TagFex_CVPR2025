#!/usr/bin/env bash
# run_sbs_queue.sh — Generic resumable orchestrator (historical filename).
#
# Queue policy and setup rules: docs/EXPERIMENT_QUEUE_GUIDELINES.md
# It accepts composed YAML overlays and is the canonical orchestrator for new
# queues until it is moved to a generically named module.
#
# Usage:
#   screen -dmS sbs_queue bash run_sbs_queue.sh
#   screen -r sbs_queue
#
#   tail -f logs/auto_experiments/sbs_orchestrator.log
#   tail -f logs/auto_experiments/sbs_console/<exp>.log
#
# Queue file format (configs/queue_sbs.txt):
#   <config_path>[,<override_path>...]|<description>|<seed>
#   Lines starting with # or blank are ignored.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
QUEUE="${QUEUE:-$SCRIPT_DIR/experiments/sbs/queues/queue_sbs.txt}"
GPUS="${GPUS:-1}"
THRESHOLD="${THRESHOLD:-100.0}"
INTERVAL="${INTERVAL:-30}"
MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-}"
MIN_FREE_MB="${MIN_FREE_MB:-}"
ALLOWED_GPU_IDS="${ALLOWED_GPU_IDS:-}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
UPDATE_REPORT="${UPDATE_REPORT:-1}"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOCKFILE="${LOCKFILE:-$TAGFEX_LOCK_DIR/tagfex_sbs.lock}"

ORCH_LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
ORCH_LOG="${ORCH_LOG:-$ORCH_LOG_DIR/sbs_orchestrator.log}"
CONSOLE_DIR="${CONSOLE_DIR:-$ORCH_LOG_DIR/sbs_console}"

[[ -f "$QUEUE" ]] || { printf 'Queue not found: %s\n' "$QUEUE" >&2; exit 2; }

# ── flock: one orchestrator at a time ─────────────────────────────────────────
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date -Iseconds)] ALREADY RUNNING — another orchestrator holds $LOCKFILE; exiting." >&2
    exit 0
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "$ORCH_LOG_DIR" "$CONSOLE_DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m';  CYAN='\033[0;36m';  GRAY='\033[0;37m'; NC='\033[0m'

# ── Logging ───────────────────────────────────────────────────────────────────
log() {
    local ts msg
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    msg="[$ts] $*"
    echo "$msg" | tee -a "$ORCH_LOG"
}

# ── Completion check: shared with training and the read-only auditor ─────────
is_done() {
    python3 "$SCRIPT_DIR/scripts/maintenance/check_queue_entry.py" "$1" "$2"
}

# ── Count total (non-comment, non-blank) entries in queue ─────────────────────
count_total() {
    grep -cE '^[^#[:space:]]' "$QUEUE" 2>/dev/null || echo 0
}

# ── Run one experiment ────────────────────────────────────────────────────────
EXP_COUNTER=0

run_one() {
    local config_spec="$1"
    local description="$2"
    local seed="$3"
    local total="$4"

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$total]"
    local slug
    slug="$(printf '%s' "$description" | tr -cs '[:alnum:]_.-' '_')_s${seed}"
    local console_log="$CONSOLE_DIR/${slug}.log"

    local completion_rc=0
    is_done "$config_spec" "$seed" || completion_rc=$?
    if [[ "$completion_rc" == 0 ]]; then
        log "SKIP $pos $description [seed=$seed] — already complete"
        echo -e "${GRAY}[SKIP] $pos $description [seed=$seed]${NC}"
        return 0
    elif [[ "$completion_rc" != 1 ]]; then
        log "FAIL $pos $description [seed=$seed] — completion check failed rc=$completion_rc"
        return "$completion_rc"
    fi

    log "START $pos $description [seed=$seed]"
    echo -e "\n${YELLOW}>> $pos${NC} $description  ${CYAN}[seed=$seed]${NC}"
    echo -e "   Configs: $config_spec"
    echo -e "   Log    : $console_log"
    echo -e "   Waiting for GPU (util < ${THRESHOLD}%)...\n"

    local train_cmd="python3 main.py train --exp-configs"
    local train_config
    local -a train_configs
    IFS=',' read -r -a train_configs <<< "$config_spec"
    for train_config in "${train_configs[@]}"; do
        printf -v train_cmd '%s %q' "$train_cmd" "$train_config"
    done
    printf -v train_cmd '%s --seed %q' "$train_cmd" "$seed"

    local launcher_args=(
        --command "$train_cmd"
        --gpus "$GPUS"
        --threshold "$THRESHOLD"
        --interval "$INTERVAL"
        --no-screen
    )
    if [ -n "$MEMORY_THRESHOLD" ]; then
        launcher_args+=(--memory-threshold "$MEMORY_THRESHOLD")
    fi
    if [ -n "$MIN_FREE_MB" ]; then
        launcher_args+=(--min-free-mb "$MIN_FREE_MB")
    fi
    if [ -n "$ALLOWED_GPU_IDS" ]; then
        read -r -a allowed_gpu_ids <<< "$ALLOWED_GPU_IDS"
        launcher_args+=(--allowed-gpu-ids "${allowed_gpu_ids[@]}")
    fi

    python3 "$AUTO_LAUNCHER" "${launcher_args[@]}" \
        >> "$console_log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        log "DONE $pos $description [seed=$seed]"
        echo -e "${GREEN}[OK]  $pos $description${NC}\n"
        if [ "$UPDATE_REPORT" = "1" ]; then
            TAGFEX_LOGS_DIR="$(dirname "$ORCH_LOG_DIR")" \
            python3 "$SCRIPT_DIR/generate_html_pdf_report.py" --short \
                >> "$ORCH_LOG_DIR/report_update.log" 2>&1 || true
        fi
    else
        log "FAIL $pos $description [seed=$seed] rc=$rc"
        echo -e "${RED}[ERR] $pos $description — exit $rc (see $console_log)${NC}"
        echo -e "${RED}       Continuing queue...${NC}\n"
    fi

    sleep 3
    return $rc
}

# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

TOTAL="$(count_total)"

echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SBS Ablation Queue  (${TOTAL} experiments)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Queue   : $QUEUE${NC}"
echo -e "${BLUE}  Lock    : $LOCKFILE${NC}"
echo -e "${BLUE}  OrcLog  : $ORCH_LOG${NC}"
echo -e ""

log "ORCHESTRATOR START — queue=$QUEUE  total=$TOTAL  GPU=$GPUS  threshold=$THRESHOLD"

FAIL_COUNT=0

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
        ""|\#*) continue ;;
    esac

    IFS='|' read -r config_spec description seed <<< "$line"

    config_spec="${config_spec// /}"
    seed="${seed// /}"

    if [ -z "${config_spec:-}" ] || [ -z "${description:-}" ] || [ -z "${seed:-}" ]; then
        log "MALFORMED line (skipping): $line"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    absolute_config_spec=""
    IFS=',' read -r -a config_files <<< "$config_spec"
    missing_config=0
    for config_file in "${config_files[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$config_file" ]; then
            log "MISSING config $config_file (skipping)"
            echo -e "${RED}[MISS] Config not found: $config_file${NC}"
            missing_config=1
            break
        fi
        absolute_config_spec+="${absolute_config_spec:+,}$SCRIPT_DIR/$config_file"
    done
    if [ "$missing_config" -eq 1 ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    if ! run_one "$absolute_config_spec" "$description" "$seed" "$TOTAL"; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$STOP_ON_FAILURE" = "1" ]; then
            log "ORCHESTRATOR ABORT — stopping after first failure to protect remaining queue entries"
            break
        fi
    fi

done < "$QUEUE"

echo -e ""
if [ "$FAIL_COUNT" -eq 0 ]; then
    log "ORCHESTRATOR DONE — all $TOTAL experiments finished successfully"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}[OK]  All $TOTAL experiments done.${NC}"
else
    log "ORCHESTRATOR DONE — $FAIL_COUNT/$TOTAL experiments failed"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}[WARN] Done with $FAIL_COUNT/$TOTAL failures. See $ORCH_LOG${NC}"
fi
echo -e "${BLUE}  Results : $SCRIPT_DIR/results_report_short.html${NC}"
echo -e "${BLUE}  OrcLog  : $ORCH_LOG${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

# Report unsuccessful queues to service managers and chained callers.
[[ "$FAIL_COUNT" -eq 0 ]]
