#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/ant_central_extra_seeds/sync_wolverine_central_extra_seeds.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/ant_central_extra_seeds/sync_wolverine_central_extra_seeds.sh" "$@"
