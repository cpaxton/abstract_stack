#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: ./install.sh [uv sync options]

Install Abstract Stack, emet's simulation extra, and development tools into .venv.
Requires Git and uv. Defaults to Python 3.11; set PYTHON_VERSION to override.
Additional arguments are passed to uv sync (e.g. --no-dev or --locked).
HELP
    exit 0
fi

for tool in git uv; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing required command: $tool" >&2
        if [[ "$tool" == "uv" ]]; then
            echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
        fi
        exit 1
    fi
done

cd -- "$PROJECT_ROOT"
# Emet tracks documentation media in LFS; these files are not needed at runtime.
# Some upstream media objects are unavailable, so do not smudge them on checkout.
export GIT_LFS_SKIP_SMUDGE=1
uv sync --python "${PYTHON_VERSION:-3.11}" "$@"
uv run --no-sync python - <<'PY'
from importlib.metadata import version

import abstract_stack
import emet
import mujoco

print(f"Abstract Stack {version('abstract-stack')} installed.")
print(f"emet {version('emet')}; MuJoCo {mujoco.__version__}")
PY

echo "Activate the environment: source \"$PROJECT_ROOT/.venv/bin/activate\""
