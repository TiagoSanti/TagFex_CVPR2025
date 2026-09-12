#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_refdetach/run_ant_refdetach_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_refdetach/run_ant_refdetach_queue.sh" "$@"
