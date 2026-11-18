# Quest stereo viewer: device previews

These reduced-resolution documentation previews come from actual Meta VR CLI
compositor screencaps captured on the connected Quest 3 on 2026-11-17. They
contain the app view rather than a camera view of the operator's room. Full
captures and logs remain in ignored `outputs/quest-stereo/`.

| Preview | What it demonstrates |
| --- | --- |
| [Streamed pattern](streamed-pattern.png) | Distinct LEFT/RIGHT textures, matching pair ID7526 and LIVE status in the physical headset app. |
| [Native MuJoCo cameras](mujoco-live.png) | The same app displaying live paired `zed_left` / `zed_right` shelf images. |
| [Host disconnected](disconnected.png) | Cleared old images, red border and STALE / UNSYNCHRONIZED status after host disconnection. |

The development APK is `org.abstractstack.queststereoviewer`, version0.1.0,
Unity6000.6.9f1 / Meta XR209 / OpenXR1.19.0. The native stream was640x480 per eye,
JPEG quality75, seed0 studio with six dynamic products and the default reset
actuator references. No teleoperation or manipulation controller was active.
The captures precede the final documentation commits; source and timing details
are in [stereo_window.md](../../stereo_window.md) and PR#35's development report.
A still image demonstrates eye routing/status, not binocular fusion, room-anchor
stability under motion, optical latency, hand/body delivery or physical control.
The user separately checked fusion/anchoring, and read-only XR MCP queried the
room anchor; those findings are described separately from these screenshots.

Capture command (replace the device selector):

```sh
metavr --device QUEST_SERIAL capture screenshot --method screencap \
  --output outputs/quest-stereo/capture.png --json
```

The previews only reduce resolution; they do not replace rendered content. See
[preview_hashes.json](preview_hashes.json) for original/preview dimensions and
SHA256 fingerprints. No synthetic XR input was injected for these checks.
