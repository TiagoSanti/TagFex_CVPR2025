#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: scripts/sync/sync_wolverine_refdetach_results.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/sync/sync_wolverine_refdetach_results.sh" "$@"
