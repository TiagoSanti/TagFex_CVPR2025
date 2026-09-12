#!/usr/bin/env bash
# Source from a canonical script; no cwd assumptions and no filesystem writes.
TAGFEX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly TAGFEX_ROOT

# Keep existing lock identities until a coordinated host-wide migration.
# A private directory may be selected explicitly for ALL producers and waiters.
export TAGFEX_LOCK_DIR="${TAGFEX_LOCK_DIR:-/tmp}"

tagfex_validate_lock_dir() {
    [[ "$TAGFEX_LOCK_DIR" == /tmp ]] && return 0
    if [[ "$TAGFEX_LOCK_DIR" != /* || ! -d "$TAGFEX_LOCK_DIR" || -L "$TAGFEX_LOCK_DIR" ]] ||
       [[ "$(stat -c '%u:%a' -- "$TAGFEX_LOCK_DIR")" != "$UID:700" ]]; then
        printf '%s\n' 'TAGFEX_LOCK_DIR must be an existing absolute directory owned by this user with mode 0700.' >&2
        exit 78
    fi
}

tagfex_profile_path() {
    local interpreter
    if [[ -n "${TAGFEX_PYTHON:-}" ]]; then
        interpreter="$TAGFEX_PYTHON"
    elif [[ -x "$TAGFEX_ROOT/.venv/bin/python" ]]; then
        interpreter="$TAGFEX_ROOT/.venv/bin/python"
    else
        interpreter=python3
    fi
    "$interpreter" "$TAGFEX_ROOT/scripts/maintenance/profile_path.py" "$@"
}

tagfex_require_operational_checkout() {
    if [[ -f "$TAGFEX_ROOT/.snapshot-isolated" ]]; then
        printf '%s\n' 'Isolated snapshot: operational commands are disabled; use scripts/maintenance/audit_queues.py for read-only inspection.' >&2
        exit 78
    fi
}
