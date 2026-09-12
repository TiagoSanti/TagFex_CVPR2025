#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/cub200/run_cub200_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/cub200/run_cub200_queue.sh" "$@"
