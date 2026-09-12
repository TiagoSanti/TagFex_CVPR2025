#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: scripts/maintenance/install_avgk_autoresume.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/scripts/maintenance/install_avgk_autoresume.sh" "$@"
