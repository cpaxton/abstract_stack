#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: scripts/install-macos.sh

Install the lightweight native macOS simulator and development tools into .venv,
fetch the pinned MABEL assets, and install Git hooks. Emet is excluded.
Requires Git and uv. Defaults to Python 3.11; set PYTHON_VERSION to override.
HELP
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Unexpected arguments. Run scripts/install-macos.sh --help for usage." >&2
    exit 2
fi
if [[ "$(uname -s)" != Darwin ]]; then
    echo "This installer requires macOS. See README.md for other platforms." >&2
    exit 1
fi
for tool in git uv; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing required command: $tool. See README.md for prerequisites." >&2
        exit 1
    fi
done

cd -- "$PROJECT_ROOT"
export GIT_LFS_SKIP_SMUDGE=1
# Match CI's simulator-only environment; do not install Emet's Linux GPU stack.
uv sync --locked --only-group ci --no-install-project --python "${PYTHON_VERSION:-3.11}"
uv pip install --no-deps -e .
.venv/bin/python - <<'PY'
from importlib.metadata import version

import abstract_stack  # noqa: F401
import mujoco

print(f"Abstract Stack {version('abstract-stack')}; MuJoCo {mujoco.__version__}")
PY
.venv/bin/abstract-stack fetch-mabel
.venv/bin/pre-commit install --install-hooks

echo "Native Mac simulator ready. Launch: $PROJECT_ROOT/scripts/run-macos.sh sim"
