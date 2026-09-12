#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ablation_missing/run_ablation_missing_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ablation_missing/run_ablation_missing_queue.sh" "$@"
