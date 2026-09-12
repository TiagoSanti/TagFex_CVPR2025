#!/usr/bin/env bash
# run_avgk_queue.sh — Resumable orchestrator for Avg-K teacher ablations.
#
# Combines:
#   • our GPU poller  (auto_run_on_free_gpu.py + color/progress UX)
#   • professor's flock + skip-if-done + cron-resume pattern
#
# Key features:
#   • Single-instance guarantee via flock — safe to re-run or run from cron.
#   • Skips already-completed experiments (checks for "avg_nme1" in gistlog).
#   • Reads queue from configs/queue_avgk.txt (config|description|seed).
#   • Refreshes results_report_short.html after every finished run.
#   • On reboot: cron restarts this script; flock prevents double-launch.
#
# Usage:
#   # Run in background (screen):
#   screen -dmS avgk_queue bash run_avgk_queue.sh
#   screen -r avgk_queue
#
#   # Or foreground:
#   bash run_avgk_queue.sh
#
#   # Install cron @reboot auto-resume:
#   bash install_avgk_autoresume.sh
#
# Queue file format (configs/queue_avgk.txt):
#   <config_path>|<description>|<seed>
#   Lines starting with # or blank are ignored.

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/paths.sh"
tagfex_require_operational_checkout
tagfex_validate_lock_dir
SCRIPT_DIR="$TAGFEX_ROOT"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
QUEUE="${QUEUE:-$SCRIPT_DIR/experiments/avgk/queues/queue_avgk.txt}"
GPUS="${GPUS:-1}"
THRESHOLD="${THRESHOLD:-100.0}"     # % GPU util — 100 means "wait for memory only"
INTERVAL="${INTERVAL:-30}"          # seconds between GPU polls
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOCKFILE="${LOCKFILE:-$TAGFEX_LOCK_DIR/tagfex_avgk.lock}"

ORCH_LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
ORCH_LOG="$ORCH_LOG_DIR/avgk_orchestrator.log"
CONSOLE_DIR="$ORCH_LOG_DIR/avgk_console"

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

# ── CUDA allocator: use expandable segments to avoid OOM from fragmentation ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    local config_file="$1"
    local description="$2"
    local seed="$3"
    local total="$4"

    EXP_COUNTER=$((EXP_COUNTER + 1))
    local pos="[$EXP_COUNTER/$total]"
    # sanitize name for filesystem
    local slug
    slug="$(basename "$config_file" .yaml)_s${seed}"
    local console_log="$CONSOLE_DIR/${slug}.log"

    # ── skip if already done ────────────────────────────────────────────────
    local completion_rc=0
    is_done "$config_file" "$seed" || completion_rc=$?
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
    echo -e "   Config : $config_file"
    echo -e "   Log    : $console_log"
    echo -e "   Waiting for GPU (util < ${THRESHOLD}%)...\n"

    local train_cmd="python3 main.py train --exp-configs $config_file --seed $seed"

    python3 "$AUTO_LAUNCHER" \
        --command "$train_cmd" \
        --gpus "$GPUS" \
        --threshold "$THRESHOLD" \
        --interval "$INTERVAL" \
        --no-screen \
        >> "$console_log" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        log "DONE $pos $description [seed=$seed]"
        echo -e "${GREEN}[OK]  $pos $description${NC}\n"
        # Refresh report incrementally after every completed run.
        TAGFEX_LOGS_DIR="$(dirname "$ORCH_LOG_DIR")" \
        python3 "$SCRIPT_DIR/generate_html_pdf_report.py" --short \
            >> "$ORCH_LOG_DIR/report_update.log" 2>&1 || true
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
echo -e "${GREEN}  Avg-K Teacher Ablation Queue  (${TOTAL} experiments)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Queue   : $QUEUE${NC}"
echo -e "${BLUE}  Lock    : $LOCKFILE${NC}"
echo -e "${BLUE}  OrcLog  : $ORCH_LOG${NC}"
echo -e ""

log "ORCHESTRATOR START — queue=$QUEUE  total=$TOTAL  GPU=$GPUS  threshold=$THRESHOLD"

FAIL_COUNT=0

while IFS= read -r line || [ -n "$line" ]; do
    # skip blank lines and comments
    case "$line" in
        ""|\#*) continue ;;
    esac

    IFS='|' read -r config_file description seed <<< "$line"

    # trim whitespace
    config_file="${config_file// /}"
    seed="${seed// /}"

    if [ -z "${config_file:-}" ] || [ -z "${description:-}" ] || [ -z "${seed:-}" ]; then
        log "MALFORMED line (skipping): $line"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    if [ ! -f "$SCRIPT_DIR/$config_file" ]; then
        log "MISSING config $config_file (skipping)"
        echo -e "${RED}[MISS] Config not found: $config_file${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    run_one "$SCRIPT_DIR/$config_file" "$description" "$seed" "$TOTAL" || \
        FAIL_COUNT=$((FAIL_COUNT + 1))

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
