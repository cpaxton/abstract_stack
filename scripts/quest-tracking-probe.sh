#!/usr/bin/env bash
# Standalone diagnostics only; never connects to a robot or simulation controller.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_path="$repo_root/unity/QuestTrackingProbe"
package_id=org.abstractstack.questtrackingprobe
activity=com.unity3d.player.UnityPlayerGameActivity
editor_is_open() {
  unity status --format json | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    sys.exit(0 if any((i.get("projectPath") or i.get("project")) == sys.argv[1] for i in (data.get("data") or {}).get("instances", [])) else 1)
except (ValueError, TypeError):
    sys.exit(1)
' "$project_path"
}
case "${1:-}" in
  configure|build)
    mkdir -p "$repo_root/outputs"
    if editor_is_open; then
      if [[ "$1" == configure ]]; then method=Configure; else method=Build; fi
      unity command eval --caller plugin --skill unity-cli --project-path "$project_path" \
        --timeout 900 "ProbeBuilder.$method();" --format json
    elif [[ "$1" == configure ]]; then
      unity run "$project_path" --editor-version 6000.6.9f1 --timeout 600 \
        --log-file "$repo_root/outputs/quest-tracking-configure.log" -- -executeMethod ProbeBuilder.Configure
    else
      unity build "$project_path" --editor-version 6000.6.9f1 --target Android \
        --execute-method ProbeBuilder.Build --timeout 900 \
        --log-file "$repo_root/outputs/quest-tracking-build.log"
    fi
    ;;
  install|start|pull|marker|stop)
    action="$1"
    device_serial="${2:?Pass the intended adb device serial explicitly}"
    adb_bin="${ADB:-adb}"
    destination="/sdcard/Android/data/$package_id/files"
    case "$action" in
      install) "$adb_bin" -s "$device_serial" install -r "$project_path/Builds/QuestTrackingProbe.apk" ;;
      start) "$adb_bin" -s "$device_serial" shell am start -n "$package_id/$activity" ;;
      pull)
        output_dir="${3:-$repo_root/outputs/quest-tracking}"
        mkdir -p "$output_dir"
        "$adb_bin" -s "$device_serial" pull "$destination" "$output_dir/"
        ;;
      marker)
        label="${3:?Pass a short phase label}"
        marker_path="$(mktemp)"
        trap 'rm -f "$marker_path"' EXIT
        printf '%s\n' "$label" > "$marker_path"
        "$adb_bin" -s "$device_serial" push "$marker_path" "$destination/marker.txt"
        ;;
      stop) "$adb_bin" -s "$device_serial" shell am force-stop "$package_id" ;;
    esac
    ;;
  *) printf 'Usage: %s {configure|build|install SERIAL|start SERIAL|marker SERIAL LABEL|pull SERIAL [DIR]|stop SERIAL}\n' "$0" >&2; exit 2 ;;
esac
