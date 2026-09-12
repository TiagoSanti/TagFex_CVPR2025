#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_detach_cub20/run_ant_detach_cub20_wolverine_gpu0.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_detach_cub20/run_ant_detach_cub20_wolverine_gpu0.sh" "$@"
