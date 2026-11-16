#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec uv run --locked --no-sync python -m pytest "$@"
