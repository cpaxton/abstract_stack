#!/usr/bin/env bash
# Baseline -> bounded development search -> paired validation -> report.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
BENCH_ROOT="${1:-outputs/stocking-benchmark}"
if [[ -e "$BENCH_ROOT" ]]; then
  echo "Use a fresh benchmark directory: $BENCH_ROOT" >&2
  exit 1
fi
mkdir -p -- "$BENCH_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
run_campaign() {
  local result=0
  "$PYTHON" -m abstract_stack.cli batch "$@" || result=$?
  # Exit 2 means retained task failures. Still evaluate and report those cases.
  if [[ "$result" != 0 && "$result" != 2 ]]; then
    return "$result"
  fi
}
run_campaign --seeds 0 1 2 3 4 5 6 7 8 9 --objects 2 --legacy-contact-control \
  --no-thumb-preshape \
  --output-dir "$BENCH_ROOT/baseline"
"$PYTHON" -m abstract_stack.cli tune --seeds 100 101 --objects 2 --rounds 1 \
  --output-dir "$BENCH_ROOT/search"
run_campaign --seeds 0 1 2 3 4 5 6 7 8 9 --objects 2 \
  --control-config "$BENCH_ROOT/search/best-controls.json" \
  --output-dir "$BENCH_ROOT/tuned"
"$PYTHON" scripts/report_stocking.py --baseline "$BENCH_ROOT/baseline/campaign.json" \
  --tuned "$BENCH_ROOT/tuned/campaign.json" --search "$BENCH_ROOT/search/search.json" \
  --output "$BENCH_ROOT/report"
