#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_central_extra_seeds/run_ant_central_extra_seeds_local.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_central_extra_seeds/run_ant_central_extra_seeds_local.sh" "$@"
