#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/cub_iv_gr/run_cub_iv_gr_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/cub_iv_gr/run_cub_iv_gr_queue.sh" "$@"
