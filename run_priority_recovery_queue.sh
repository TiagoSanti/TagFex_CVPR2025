#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/priority_recovery/run_priority_recovery_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/priority_recovery/run_priority_recovery_queue.sh" "$@"
