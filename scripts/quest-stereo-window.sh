#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_path="$repo_root/unity/QuestTrackingProbe"
unity_cli="${UNITY_CLI:-$HOME/.unity/bin/unity}"
case "${1:-help}" in
  configure)
    "$unity_cli" --proxy-disable command eval --caller plugin --skill unity-cli --project-path "$project_path" 'QuestStereoBuilder.Configure(); return "configured";' --format json
    ;;
  build)
    "$unity_cli" --proxy-disable command eval --caller plugin --skill unity-cli --project-path "$project_path" --timeout 600 'QuestStereoBuilder.Build(); return "built";' --format json
    ;;
  test)
    "$unity_cli" --proxy-disable command 'test run' --caller plugin --skill unity-cli --project-path "$project_path" --mode editor --filter QuestStereo.Tests --filter-type Assembly --timeout 60 --format json
    ;;
  *)
    echo "Open this Unity project first: $project_path"
    echo "Usage: $0 {configure|build|test}"
    ;;
esac
