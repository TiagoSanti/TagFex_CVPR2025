#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_fs_gr/run_ant_fs_gr_after_current.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_fs_gr/run_ant_fs_gr_after_current.sh" "$@"
