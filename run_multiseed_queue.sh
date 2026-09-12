#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/multiseed/run_multiseed_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/multiseed/run_multiseed_queue.sh" "$@"
