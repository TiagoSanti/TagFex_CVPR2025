#!/usr/bin/env bash
# install_avgk_autoresume.sh — Install/remove cron @reboot entry for run_avgk_queue.sh.
#
# On every machine restart, cron will re-launch the orchestrator.
# flock in run_avgk_queue.sh guarantees only one instance runs at a time,
# so this is safe to install alongside manual invocations.
#
# Usage:
#   bash install_avgk_autoresume.sh           # install
#   bash install_avgk_autoresume.sh --remove  # remove
#   bash install_avgk_autoresume.sh --status  # check if installed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$REPO_ROOT/run_avgk_queue.sh"
ORCH_LOG="/home/tiago/logs/auto_experiments/avgk_orchestrator.log"
MARKER="# tagfex-avgk-autoresume"

CRON_LINE="@reboot mkdir -p $(dirname $ORCH_LOG) && cd $REPO_ROOT && nohup bash $RUN_SCRIPT >> $ORCH_LOG 2>&1 &  $MARKER"

case "${1:-}" in
  --remove)
    (crontab -l 2>/dev/null | grep -vF "$MARKER") | crontab -
    echo "Removed cron @reboot entry."
    exit 0
    ;;
  --status)
    if crontab -l 2>/dev/null | grep -qF "$MARKER"; then
      echo "INSTALLED. Entry:"
      crontab -l 2>/dev/null | grep -F "$MARKER"
    else
      echo "NOT installed."
    fi
    exit 0
    ;;
esac

# Make run script executable
chmod +x "$RUN_SCRIPT"

# Replace any existing entry with the same marker, then add the new one.
new_crontab="$( (crontab -l 2>/dev/null | grep -vF "$MARKER"; echo "$CRON_LINE") )"
echo "$new_crontab" | crontab -

echo "Installed cron @reboot entry:"
echo "  $CRON_LINE"
echo ""
echo "After a reboot the orchestrator will resume the queue automatically."
echo "Duplicate launches are suppressed by flock ($REPO_ROOT/run_avgk_queue.sh)."
echo ""
echo "Verify : crontab -l"
echo "Remove : bash $0 --remove"
echo "Status : bash $0 --status"
