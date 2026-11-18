# Quest stereo window

Issue [#22](https://github.com/cpaxton/abstract_stack/issues/22) adds a receive-only
Quest 3 diagnostic app. It does not command MuJoCo or a physical robot. The
window sits in the room rather than following the head. Its single material
selects the complete left or right texture using Unity's XR eye index; image
geometry stays 4:3. The image uses the cameras' original projection and baseline,
not the headset IPD. This is a stereo window, not an immersive camera reprojection.

The app ID is `org.abstractstack.queststereoviewer`, separate from the tracking
probe. Original tracking scripts/scenes are unchanged. `QuestStereoBuilder`
selects Android, checks the existing OpenXR/Meta configuration and enables
internet permission. Run `ProbeBuilder.Configure` to return project settings to
the tracking probe before building that app. Both helpers use explicit scenes.
The new dependencies declare the already locked Newtonsoft 3.2.2 and Unity Test
Framework 1.8.0 versions explicitly; no SDK version changed.

## Build and check

Open `unity/QuestTrackingProbe` in Unity 6000.6.9f1, then run:

```bash
./scripts/quest-stereo-window.sh configure
./scripts/quest-stereo-window.sh test
./scripts/quest-stereo-window.sh build
```

The APK is `unity/QuestTrackingProbe/Builds/QuestStereoWindow.apk` (ignored).
The default app displays a local synthetic stereo pattern: red/L only in the
left eye, blue/R only in the right, matching motion and pair counters, a grid at
the window plane, and a cross with crossed disparity that should fuse in front.
Move your head to check that the window stays in the room. These patterns are
reproducible diagnostics, not claims that eye routing/depth was verified on a
physical headset. Cover each eye in turn to verify routing before using robot
video. A compositor screenshot alone cannot prove binocular depth.

The startup placement is configurable through `stereo-config.json` in the app's
persistent files directory. Example for USB `adb reverse tcp:8765 tcp:8765`:

```json
{
  "mode": "network",
  "host": "ws://127.0.0.1:8765",
  "window_width_m": 1.2,
  "window_distance_m": 1.5,
  "window_height_m": 1.4
}
```

For LAN use the host computer's reachable IP and the #15/#21 service on port
8765. First use assumes the trusted development network from the v1 contract;
no credentials are written to the config. Remove the file or set
`"mode":"synthetic"` for local diagnostics. Width is bounded to .3–2.5 m,
distance .5–4 m and height .3–2.5 m; image height remains width × .75. Placement
occurs once after a floor-relative head pose appears, at the configured floor
height and distance along horizontal head-forward. It never follows later head
motion. The 4:3 stream can be 4×3 through 1600×1200; initial expected profile is
640×480 per eye. Host resolution is selected on #21, not silently resampled here.

## Receive, decode and presentation

`StereoVideoReceiver` connects to `/quest/video`, sends `video_bind`, checks the
acknowledgement, and accepts complete binary QTP1 pairs. It validates packet,
metadata, per-eye lengths, strict metadata keys/types, finite values, integral
IDs, camera names, unit quaternions, eye order and 4:3 dimensions. JPEG SOF
headers must declare matching bounded dimensions before `LoadImage` runs; this
avoids an encoded image claiming a small metadata size but allocating a huge
texture. Decoded dimensions are checked again. Malformed pairs are counted and
rejected without replacing the previous valid pair.

The network worker has one fixed maximum-size receive buffer, one in-flight
parsed pair and one latest complete pair. Replacements discard both eyes of the
superseded pending pair. IDs must strictly advance for the active session and
generation, including across `Take`; a reset clears pending state and the ID
high-water mark. Unity's main thread decodes at most one pair per `Update` into
two new textures, validates both, then swaps both material references together.
It retains at most the currently displayed texture pair plus the next pair being
decoded (Unity defers destroyed texture reclamation to the frame boundary).
JPEG decoding still uses Unity's synchronous main-thread API. Device timing must
establish whether this is usable at 30 pairs/s before promoting it; H.264 is a
separate negotiated profile, not a silent fallback.

`video_status` runs every 50 ms while active. Before a camera submission, IDs and
presentation time are null and stale is true. `endCameraRendering` records the
complete material pair only for the configured camera, a running XR display, a
visible window intersecting its frustum and an active app. This is **render
submission evidence**, not compositor scanout or optical motion-to-photon time.
Editor-only tests may opt into non-XR submission, disabled in the built scene.
The Android logs label `decoded` and `render_submission` separately, including
pair ID, decode duration, monotonic timestamps, mapped capture age and clock
uncertainty. Counting a decoded pair alone never supplies presentation feedback.

A clock acknowledgement supplies host-minus-client offset and uncertainty.
Until an accepted mapping with uncertainty ≤25 ms exists, or after five seconds
without refreshing it, video is unsynchronized/stale. Capture age is checked
against the last submitted pair, rather than newly decoded but unseen data.
Age plus uncertainty >300 ms, future capture beyond uncertainty, disconnect,
lost focus or pause marks stale. The viewer sends one immediate pause request
through the control owner per stale interval, independent of the 50 ms
`video_status` schedule. On focus loss or suspension it asks for pause before
clearing the session. The last complete pair has a red stale border;
reconnect, suspension and generation invalidation clear both textures and pending
state. Repeated submissions never refresh the original host capture time.

## Integrating with the capture client

`StereoViewer.ConfigureSession(videoUri, sessionId, generation,
videoStatusSend, requestPause)` reuses the existing #15 control session. Call it,
`UpdateClock(offset,uncertainty)`, `SetGeneration` and `ClearSession` on Unity's
main thread. Pass `video_status` to the reliable control sender; the control
owner supplies pause event IDs through `requestPause`, avoiding two independent
control event counters. On a generation change, clear then configure again with
the new generation; discard old session callbacks. Do not start the standalone
launcher when integrating with a tracking client's existing session.

`StereoDiagnosticSession` is a separate receive-only launcher for this demo:
hello, periodic four-timestamp clock exchange, video binding and feedback. It
sends no tracking, calibration, engage, reset or robot command. After control
loss, focus loss or suspension it reconnects using a new UUID and clears all old
frames/clock state. Its pending feedback slot is one latest status, its pause
request is one flag, and its main-thread control callback queue is bounded at 32.

## Validation and remaining acceptance

On 2026-11-17, Unity EditMode ran 15 tests successfully: actual red/blue JPEG eye
regions survive packet parsing/decode; whole-pair overload replacement;
old/repeated IDs, sessions and generations; bad magic, lengths, trailing bytes,
metadata/encoded dimensions, aspect, eye order, unknown fields, fractional IDs,
NaN, quaternion norm and hostile JPEG dimensions; XR shader compilation/default
routing. These establish software framing behavior, not physical eye order.

Physical Quest eye-by-eye labels/fusion, sustained real MuJoCo head video,
decode/submission timing, and focus/re-wear/network recovery require the device
procedure above and the #21 service. Keep this PR draft until that evidence is
recorded. No actuation or teleoperation safety validation is claimed.

Implementation follows Unity's
[single-pass stereo instancing guidance](https://docs.unity.com/en-us/engine/6000.0/manual/xr/graphics/stereo-rendering/single-pass-instancing)
and [LoadImage API](https://docs.unity.com/en-us/engine/6000.7/script-reference/unityengine/imageconversion/loadimage).

### Device check

On2026-11-17 the user confirmed the local L/red and R/blue eye labels, fused
front-of-grid cross and room placement, then confirmed LIVE network video with
advancing matching pair IDs and the correct per-eye labels. The cross is fixed
in the stereo images (the yellow bar animates); its perceived position in front
of a fixed window can shift with viewing angle. It is not a world-rendered 3D
object. This distinction was clarified after the user's observation.

The agent also used the installed Meta VR CLI and XR MCP read-only calls, without
injecting simulated input. Compositor screenshots independently showed LIVE,
matching LEFT/RIGHT pair IDs, and subsequently a LIVE native MuJoCo shelf view.
Repeated `unity_get_world_pose` queries returned an unchanged window transform
with FloorLevel/local_floor tracking origin; `openxr_get_head_pose` reported valid
tracked poses. These limited pose samples establish anchor invariance at the
sampled times, rather than an exhaustive controlled head-motion experiment.
The host recorded2,703 fresh feedback samples for the synthetic USB stream,
with capture-to-render-submission p50/p95 estimates40.7/54.3 ms. Sampled on-device
JPEG decode logs were roughly9–10 ms per640×480 pair. Render submission and optical
display remain distinct. Screenshots/logs/reports are kept in ignored outputs.
Sustained30 Hz real-camera performance, Wi-Fi and the full stale/reconnect/focus
procedure remain draft acceptance gates; the checks above do not establish
robot-control safety.
