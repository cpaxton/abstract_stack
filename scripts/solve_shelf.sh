#!/usr/bin/env bash
# Attempt a measured physical stocking rollout; rejected/failed phases exit 2.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -x .venv/bin/abstract-stack ]]; then
    echo "Run ./install.sh first, then abstract-stack fetch-mabel." >&2
    exit 1
fi
exec .venv/bin/abstract-stack solve "$@"
