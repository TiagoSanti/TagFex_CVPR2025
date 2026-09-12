#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: scripts/execution/trainddp.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/execution/trainddp.sh" "$@"
