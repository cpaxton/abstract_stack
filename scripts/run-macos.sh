#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Darwin ]]; then
    echo "This launcher requires macOS. See README.md for other platforms." >&2
    exit 1
fi
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing .venv. Run $PROJECT_ROOT/scripts/install-macos.sh first." >&2
    exit 1
fi

cd -- "$PROJECT_ROOT"
export MUJOCO_GL=cgl
case "${1:-}" in
    sim|serve)
        USE_VIEWER=true
        for arg in "$@"; do
            case "$arg" in
                --headless|--help|-h) USE_VIEWER=false ;;
            esac
        done
        if [[ "$USE_VIEWER" == true ]]; then
            # mjpython moves the executable outside uv Python's installation,
            # so @rpath cannot locate libpython without its original lib dir.
            PYTHON_LIBDIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
            if [[ -n "$PYTHON_LIBDIR" ]]; then
                export DYLD_FALLBACK_LIBRARY_PATH="$PYTHON_LIBDIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}:/usr/local/lib:/usr/lib"
            fi
            PYTHON="$PROJECT_ROOT/.venv/bin/mjpython"
        fi
        ;;
esac
exec "$PYTHON" -m abstract_stack.cli "$@"
