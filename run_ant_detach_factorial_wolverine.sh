#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_detach_factorial/run_ant_detach_factorial_wolverine.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_detach_factorial/run_ant_detach_factorial_wolverine.sh" "$@"
