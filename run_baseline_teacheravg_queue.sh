#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/baseline_teacheravg/run_baseline_teacheravg_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/baseline_teacheravg/run_baseline_teacheravg_queue.sh" "$@"
