#!/usr/bin/env bash
# run_sbs_queue.sh — Resumable orchestrator for SBS (Speed-Based Sampling) ablations.
#
# Identical architecture to run_avgk_queue.sh but:
#   • Reads queue from configs/queue_sbs.txt
#   • is_done() includes SBS suffix logic (prevents false-positives vs non-SBS runs)
#   • Separate lockfile, orchestrator log, and console-log directory
#
# Usage:
#   screen -dmS sbs_queue bash run_sbs_queue.sh
#   screen -r sbs_queue
#
#   tail -f logs/auto_experiments/sbs_orchestrator.log
#   tail -f logs/auto_experiments/sbs_console/<exp>.log
#
# Queue file format (configs/queue_sbs.txt):
#   <config_path>|<description>|<seed>
#   Lines starting with # or blank are ignored.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────
QUEUE="${QUEUE:-$SCRIPT_DIR/configs/queue_sbs.txt}"
GPUS="${GPUS:-1}"
THRESHOLD="${THRESHOLD:-100.0}"
INTERVAL="${INTERVAL:-30}"
AUTO_LAUNCHER="$SCRIPT_DIR/auto_run_on_free_gpu.py"
LOCKFILE="${LOCKFILE:-/tmp/tagfex_sbs.lock}"

ORCH_LOG_DIR="$SCRIPT_DIR/logs/auto_experiments"
ORCH_LOG="${ORCH_LOG:-$ORCH_LOG_DIR/sbs_orchestrator.log}"
CONSOLE_DIR="${CONSOLE_DIR:-$ORCH_LOG_DIR/sbs_console}"

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

# ── Completion check ──────────────────────────────────────────────────────────
# Replicates _adjust_log_dir_with_loss_params() suffix logic INCLUDING SBS,
# so SBS runs are not falsely considered done by matching a non-SBS directory.
is_done() {
    local config_file="$1"
    local seed="$2"
    python3 - "$config_file" "$seed" <<'PYEOF'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(1)

config_file, seed = sys.argv[1], sys.argv[2]

try:
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
except Exception:
    sys.exit(1)

log_dir = cfg.get("log_dir", "")
if not log_dir:
    sys.exit(1)

log_dir = Path(log_dir)

# ── Replicate _adjust_log_dir_with_loss_params() suffix logic ─────────────────
suffix_parts = []

ant_beta = cfg.get("ant_beta", 0.0)
suffix_parts.append(f"antB{ant_beta:.3f}".rstrip("0").rstrip("."))

nce_alpha = cfg.get("nce_alpha", 1.0)
suffix_parts.append(f"nceA{nce_alpha:.3f}".rstrip("0").rstrip("."))

if ant_beta > 0:
    ant_margin = cfg.get("ant_margin", 0.1)
    suffix_parts.append(f"antM{ant_margin:.3f}".rstrip("0").rstrip("."))

ant_max_global = cfg.get("ant_max_global", True)
infonce_max_global = cfg.get("infonce_max_global", ant_max_global)

ant_symmetric_full = cfg.get("ant_symmetric_full", False)
if ant_symmetric_full:
    suffix_parts.append("antSymmetricFull")
else:
    suffix_parts.append("antGlobal" if ant_max_global else "antLocal")
suffix_parts.append("nceGlobal" if infonce_max_global else "nceLocal")

ant_formulation = cfg.get("ant_formulation", "logsumexp")
if ant_formulation != "logsumexp":
    suffix_parts.append(f"form{ant_formulation}")

avg_last_k = cfg.get("avg_last_k", 0)
if avg_last_k > 0:
    suffix_parts.append(f"avgK{avg_last_k}")

# ── SBS suffix (critical: prevents false-positives vs non-SBS runs) ───────────
sbs_q = cfg.get("sbs_q", 0.0)
sbs_s = cfg.get("sbs_s", 0.0)
if sbs_q > 0 or sbs_s > 0:
    suffix_parts.append(f"sbsQ{sbs_q:.2f}S{sbs_s:.2f}".rstrip("0").rstrip("."))

suffix_parts.append(f"s{seed}")

if suffix_parts:
    suffix = "_" + "_".join(suffix_parts)
    if log_dir.name == "logs" or str(log_dir) in ["./logs", "logs"]:
        dataset_name = cfg.get("dataset_name", "dataset")
        scenario = cfg.get("scenario", "").split()[-1]
        expected_base = f"exp_{dataset_name}_{scenario}{suffix}"
        parent = log_dir
    else:
        expected_base = log_dir.name + suffix
        parent = log_dir.parent
else:
    expected_base = log_dir.name
    parent = log_dir.parent

for gistlog in parent.glob(f"{expected_base}*/exp_gistlog.log"):
    try:
        text = gistlog.read_text(errors="replace")
        if "avg_nme1" in text:
            sys.exit(0)   # done
    except Exception:
        continue

sys.exit(1)  # not done
PYEOF
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
    local slug
    slug="$(basename "$config_file" .yaml)_s${seed}"
    local console_log="$CONSOLE_DIR/${slug}.log"

    if is_done "$config_file" "$seed"; then
        log "SKIP $pos $description [seed=$seed] — already complete"
        echo -e "${GRAY}[SKIP] $pos $description [seed=$seed]${NC}"
        return 0
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
        TAGFEX_LOGS_DIR="$(dirname "$ORCH_LOG_DIR")" \
        python3 "$SCRIPT_DIR/generate_report.py" \
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

    IFS='|' read -r config_file description seed <<< "$line"

    config_file="${config_file// /}"
    seed="${seed// /}"

    if [ -z "${config_file:-}" ] || [ -z "${description:-}" ] || [ -z "${seed:-}" ]; then
        log "MALFORMED line (skipping): $line"
        continue
    fi

    if [ ! -f "$SCRIPT_DIR/$config_file" ]; then
        log "MISSING config $config_file (skipping)"
        echo -e "${RED}[MISS] Config not found: $config_file${NC}"
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
echo -e "${BLUE}  Results : $SCRIPT_DIR/results_report.md${NC}"
echo -e "${BLUE}  OrcLog  : $ORCH_LOG${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
