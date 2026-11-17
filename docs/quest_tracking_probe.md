# Quest 3 raw tracking probe

Issue [#13](https://github.com/cpaxton/abstract_stack/issues/13) provides a
standalone Unity Android diagnostic, independent of the previous teleop branch.
It captures head, both hand skeletons and the upper-body skeleton. It sends no
network traffic to a robot or simulator and contains no control/retargeting code.

The source project is [`unity/QuestTrackingProbe`](../unity/QuestTrackingProbe).
Its diagnostic scene/configuration builder was adapted from the sibling local
`QuestTrackingProbe`; raw recording is new. Previous reference-app telemetry is
not evidence for this APK. The new package ID
`org.abstractstack.questtrackingprobe` preserves that app and its recordings.

## Build and run

Pinned: Unity **6000.6.9f1**, Meta XR Core **209.0.0**, Unity OpenXR **1.19.0**,
URP **17.6.0**; Android ARM64/IL2CPP/Vulkan, API32 minimum. The project includes
its package manifest, resolution lock, settings, scene and asset metadata.
The copied package lock also retains the reference project’s optional editor-only
Collab/IDE packages; they are not teleop runtime requirements. They remain pinned
for this first tested APK rather than changing package resolution after its build.
Install this Editor's Android SDK/NDK/OpenJDK support and the Unity CLI. Use the
pinned Editor, not an automatic upgrade. Review any Unity software terms yourself.

From the repository root:

```sh
# Regenerate the project scene/settings when needed; saves source assets.
./scripts/quest-tracking-probe.sh configure
./scripts/quest-tracking-probe.sh build

# Select the intended Quest explicitly; set ADB to an absolute adb path if needed.
adb devices -l
./scripts/quest-tracking-probe.sh install DEVICE_SERIAL
./scripts/quest-tracking-probe.sh start DEVICE_SERIAL
```

Build output: `unity/QuestTrackingProbe/Builds/QuestTrackingProbe.apk` (ignored).
Build logs go to ignored `outputs/` for batch builds or the project’s ignored
`Logs/Editor.log` for live builds. The helper detects an open project and uses
its live CLI/Pipeline connection rather than starting a second Editor.
Configuration resets the diagnostic scene. The helper selects both the active
Android target and selected target group before building. Meta’s SDK checks the
selected group separately; a mismatch can produce an APK without tracking
permissions or immersive VR metadata. A post-generation build guard now rejects
missing hand/body tracking declarations or the immersive VR activity category
before Gradle packaging.

Wear the headset, put controllers down, enable hand tracking and allow the body
tracking permission. The room-anchored panel shows runtime availability, hand
confidence/fresh counters, body state-time advances and recorder status. Returned
body-joint dots include estimates. The display is not a calibrated avatar and
pose movement is not an accuracy measurement.

Each process launch writes `raw-SESSION_ID.jsonl` under the app's external files
directory, for at most **180 elapsed seconds**. Relaunch for another recording.
A background writer has a bounded sample queue; overload drops samples and
increments `dropped_records`. Critical session/skeleton/event records have
reserved headroom; exhaustion fails the capture rather than silently accepting
missing metadata. JSON serialization still runs on the Unity main thread; this
probe is not a performance-qualified transport implementation. No automatic file
rotation or deletion occurs; pull and manage captures deliberately.

```sh
./scripts/quest-tracking-probe.sh marker DEVICE_SERIAL hands_visible
# The app consumes marker.txt within about one second and records operator_marker.
./scripts/quest-tracking-probe.sh pull DEVICE_SERIAL outputs/quest-tracking/run-01
# Stop only after the recorder reports stopped if you want a clean ending.
./scripts/quest-tracking-probe.sh stop DEVICE_SERIAL
python3 scripts/summarize_quest_tracking.py outputs/quest-tracking/run-01/files/raw-SESSION_ID.jsonl
```

`force-stop` can truncate the final buffered interval and does not guarantee a
stop event. If disconnected while recording, pull after reconnection. Raw capture
contains personal motion trajectories: keep it in ignored local `outputs/` and
publish only small aggregate reports. Do not commit the APK, Library, Temp or logs.

## Capture profile: quest_tracking_probe_v1

This is a raw evidence format, separate from the planned network protocol. Every
line has `format`, `type`, `session_id`; the session line has `schema_version: 1`.
Raw roots, head and body poses preserve **OVR right-handed tracking-space** axes
(+X right, +Y up, -Z forward), metres and **xyzw** quaternion components. They are
not Unity world poses, robot poses or calibrated operator targets. The origin is
recorded from `OVRManager.trackingOriginType` (configured FloorLevel). Do not mix
samples across `recenter_count` changes without a new alignment.

| Record | Contents |
| --- | --- |
| `session` | UUID, UTC launch time, Unity/SDK/runtime/OS/device/application versions, origin, units and pose/time semantics. |
| `sample` | Sequence, Unity `capture_monotonic_s`, recenter count, presence/focus, drop count and independently queried head/hands/body. |
| `skeleton` | Side/profile and named bone IDs, parent indices and native bind poses; emitted when available and when the body skeleton changes. |
| `event` | Monotonic capture time, name and detail: pause/focus, presence changes, recenter count changes, operator phase marker and recording stop. |

`pose` objects contain `position_m[3]` and `quaternion_xyzw[4]`. The head includes
position/orientation validity and tracking flags plus `pose_target_time_s` from
`GetNodePoseStateRaw`. That time may be predicted. It is **not a fresh physical
sensor sample timestamp**; `source_freshness` is `unknown`. Node validity queries
are separate from the pose API, which exposes no success flag; a returned default
pose cannot establish acquisition success merely because those flags are true.

Each hand records `api_success`, `tracked`, `status`, `confidence`, `root_pose`,
`scale`, `sample_time_s`, `requested_time_s`, five finger confidences/pinch
strengths, pinch flags and `bones`. Each bone has `id`, `name`, `rotation_xyzw`
and `position_m` (empty when the legacy API does not return positions).
`per_joint_validity` and `per_joint_tracking` are explicitly `unavailable`: the
hand-level tracked bit does not establish that every finger joint was measured.

The builder explicitly pins **legacy OVR hands (24 bones)**; SDK 209's default
is OpenXR26, so leaving that setting implicit is incorrect. Both the build and
recorder reject an unexpected hand profile. The strict host analyzer also rejects
unknown profiles and ID/name mismatches against bind metadata. OVR24 rotations use
the SDK's **parent-local hand bone basis**, not the tracking-space basis. Bind
poses and hierarchy are raw, and hand scale is separate. Reconstructing tracking
poses requires the SDK's hand-specific flip-X/wrist correction and forward
kinematics; do not apply the head/body flip-Z rule to these local rotations. No
forward kinematics is claimed here. The runtime skeleton profile and bone names
are recorded; an unexpected profile must be reviewed before adapter use. With
OpenXR26, returned bone positions/rotations instead have tracking-space semantics;
changing that project setting requires renewed device/schema validation.

Hand `source_freshness` is `first`, `advanced`, `repeated`, `regressed` or
`unavailable`. It advances only for API-successful tracked hands whose SDK sample
time exceeds its prior high-water mark. Repeats/invalid results are never counted
as fresh; a regressed clock stays below that high-water mark until caught up.
Do not equate requested timestamps with source sample timestamps.

Body fields include `api_success`, permission, joint set/fidelity/calibration,
confidence, `state_time_s`, `state_time_change`, skeleton revision and named
joints with independent position/orientation valid/tracked flags. Body
`source_freshness` stays `unknown`: advancing body state time is not proof of a
new optical observation. Track validity and estimated status for each signal;
valid body poses can be inferred. The application requests upper-body tracking
only and does not turn inferred leg motion into a control signal.

The OVR runtime, Unity monotonic and UTC clocks are distinct. Their numerical
values are not interchangeable. This capture does not implement host clock sync.

## Device acceptance procedure

Record the device/runtime build and exact commit/APK hash. Use one phase marker
at a time and confirm it appears in the pulled JSONL. Do not treat an unexecuted
or unobserved phase as passed.

1. **Hands visible:** look around/translate the head; move each wrist, flex each
   finger and pinch thumb/index separately. Inspect raw head/root motion, named
   hand rotation changes, tracked/confidence states and nonrepeated sample times.
2. **Upper body:** raise/bend both elbows and lean/turn the torso within a safe
   standing space. Inspect valid/tracked flags and named elbow/spine pose changes.
   The displayed skeleton is an API visualization, not spatial ground truth.
3. **Occlusion:** hide the left hand, then right hand, then restore each. Capture
   confidence/tracked changes and repeated/invalid samples, including stale arrays
   returned alongside invalid status. Do not classify those arrays as fresh input.
4. **Recenter:** use the headset's system recenter action. Confirm a recenter event
   or count change; record any pose discontinuity. If the API does not report the
   change, document that limitation rather than inferring a safe alignment.
5. **Removal/suspension:** remove and re-wear the headset; open the system menu and
   resume the app. Inspect presence/focus/pause transitions and recording gaps.
6. **Permission denial:** use a new installation/test state or headset permission
   settings to deny body tracking. The permission field must be false and body
   tracking unavailable while head/hands remain independently observable. Regrant
   and relaunch for a separate capture. Do not delete an existing recording to
   reset permissions; pull it first.
7. **Duration/overload:** confirm the explicit recording-stopped event at 180s and
   inspect drop/error counts. Report observed loop/frame rates separately from
   fresh hand sample rates. Missing source-time semantics prevent claiming fresh
   head/body sensor rates from this probe.

Device outcomes are recorded in [the device evidence report](quest_tracking_probe_device.md).
This issue remains open until required physical scenarios and usable signals are
supported by retained evidence. Building or launching an APK alone is not tracking
acceptance.
