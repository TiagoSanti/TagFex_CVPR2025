#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: scripts/execution/run_sbs_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/execution/run_sbs_queue.sh" "$@"
