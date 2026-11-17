# Quest tracking probe: device evidence

November 17, 2026. Issue [#13](https://github.com/cpaxton/abstract_stack/issues/13).
This records the new standalone probe, not the earlier sibling application's
results. Tracking acceptance remains open until the scenarios below have evidence.

## Build and device

- Unity 6000.6.9f1, Meta XR Core 209.0.0, OpenXR 1.19.0; Android ARM64/IL2CPP/Vulkan.
- Quest 3, Android 12 userdebug runtime, build incremental `50709660128600511`.
- Separate app ID `org.abstractstack.questtrackingprobe`; reference app/data preserved.
- Raw trajectories, logs and APKs remain local in ignored `outputs/` or Unity
  `Builds/`. No raw personal motion recording is committed.

## Initial device run: failed build configuration

Source `41925a318251c95b62a816e032c83ebd06a92da9`, application 0.3.0/build 3.
APK SHA256:
`b47f0eaf0d889e5aab50289301b09a4a22c927020b89410606d04e563031a173`.
The APK compiled and installed, but the operator reported a **black screen**.

The generated Android manifest omitted the immersive VR activity category and
hand/body permissions/features. The live Editor had Android as its active build
target but Standalone as its selected target group. Meta's manifest processing
used the latter and was skipped. Building successfully did not establish a usable
Quest application. The builder now selects Android explicitly and rejects missing
VR/tracking declarations before packaging (commit `6894bf5`). Its guard accepted
the generated manifest and rejected five fixtures separately missing each required
hand/body permission, feature or VR category.

The failed run is retained under
`outputs/quest-tracking/device-20261117-run01-final/`, including the raw capture,
`summary.json` and `device.log`. Observations:

| Measurement | Observed |
| --- | --- |
| Sample records / duration | 4,895 / 70.35 seconds |
| Polling rate | 69.57 Hz; not a sensor sample rate |
| Valid head observations | 4,894 |
| Tracked hand observations | 0 for both hands |
| Body API successes / permission-granted observations | 0 / 0 |
| Reported writer drops | 0 |
| End event | `recorder_destroyed` |
| Lifecycle events | Pause and focus changes captured |

These results establish a failed tracking/rendering launch, not hand/body
availability, spatial accuracy, successful occlusion handling or operator acceptance.
The phase marker identifies the requested sequence; it does not prove completion
of its individual steps.

## Second run: rendering and tracking available, schema mismatch rejected

Source `6894bf52de8125b8d41fee356128debd8d4b19d5`, application 0.3.1/build 4.
APK SHA256:
`911d9260fe36796ae89c56e254dc9a67829d28fbfd4b695f4c5fe362991fe10e`.
The operator confirmed **panel visible and body permission allowed**, and reported
completing hand occlusion/restoration, system recenter, headset removal/re-wear
and system menu open/close. A local screenshot independently shows the panel and
body dots in both eye views. Meta XR Operator read-only inspection found an active
world-space canvas and valid/tracked head pose. No synthetic input was injected.

However, SDK 209's serialized hand profile was OpenXR26 while the recorder emitted
OVR24 bind metadata. The strict analyzer rejected the recording at the first named
bone mismatch. This run is **not an accepted capture fixture**, despite rendered
tracking appearing usable. The builder now explicitly pins the declared OVR24
profile, and runtime capture stops rather than mixing profiles unexpectedly.

Retained local data: `outputs/quest-tracking/device-20261117-run02/`; screenshot:
`outputs/quest-tracking/corrected-panel.png`. A diagnostic-only scan of the pulled
169.30-second snapshot found 12,171 samples, 12,171 valid head observations,
11,836 body API successes, 2,094 distinct tracked left-hand samples and 3,058
right-hand samples. These counts do not establish accepted metadata, continuous
tracking, spatial accuracy or fresh head/body sensor rates. Individual physical
phases were not separately marked; the snapshot preceded the recording deadline.
The later full file retained under `device-20261117-run03-final/files/` includes
`recording_stopped` at 184.65 application-monotonic seconds: `duration_limit;
dropped_records=0`, about 180 seconds after recording began. Its invalid hand
metadata still prevents accepting it as a hand-capture fixture.

## Final run: OVR24 recording accepted by the strict analyzer

Source `bed68f005af4bab11b8dde18bb07922180d5a973`, application 0.3.2/build 5.
APK SHA256:
`47a5a8b73bb030eac452cb77dbabfa69d4552787ba3ffcaa13c5b5bb59b6ca0a`.
This APK uses an explicit OVR24 runtime setting and build/runtime profile guards.
The operator performed a short visible-hand, wrist/finger and elbow motion check.
The complete local capture passes the strict analyzer, including session identity,
named bind hierarchy, source-profile agreement and finite/unit pose checks.

Evidence: `outputs/quest-tracking/device-20261117-run03-final/`, with raw
`files/raw-e666b68b28374e6188755d56d6abe0f2.jsonl`, `summary.json` and `device.log`.
The earlier 35.08-second snapshot remains separately under `device-20261117-run03/`.

| Measurement | Final recording |
| --- | --- |
| Sample records / duration | 5,691 / 79.57 seconds |
| Polling rate | 71.50 Hz; not sensor sample rate |
| Valid head observations | 5,691 |
| Distinct tracked hand samples, left / right | 862 / 1,886 |
| Body API successes | 5,443 |
| Skeleton metadata | Left/right hands 24 each; body 70 |
| Reported writer drops | 0 |
| Explicit recenter event | `2->4` at 84.27 application-monotonic seconds |
| Lifecycle events | Focus false, pause true, recorder destroyed |
| Terminal event | `recorder_destroyed; dropped_records=0` |

Observed head/wrist positions, named finger-local rotations and elbow/chest poses
vary during the movement check. Motion ranges are grouped by recenter count so
reference-space jumps do not count as continuous motion. These observations
establish API/data availability and capture consistency, not tracking accuracy,
continuous availability or control suitability. Whole-record hand rates include
untracked intervals (about 10.82/23.69 distinct advances per second); they are not
a steady hands-visible benchmark. Body state advances and head pose-target times
remain distinct from fresh physical sensor samples.

The phase-specific occlusion and full removal/resume checks still need individual
markers and evidence review. In particular, the final file contains no transition
to `user_present:false`; do not accept headset-removal pause behavior from the
operator's reported action alone. A separate denied-permission/regrant capture is
also still required before closing issue #13.

## Required device scenarios

| Scenario | Status |
| --- | --- |
| Readable stereo diagnostic panel | Confirmed in 0.3.1 by operator and screenshot |
| Head pose availability and motion | Available; fresh sensor sample time remains unknown |
| Both wrists and named finger articulation | OVR24 samples and named rotation changes recorded; sustained quality not established |
| Upper-body/elbow/torso availability and motion | Named body poses and validity/tracked flags recorded; estimates remain explicit |
| Individual hand occlusion and reacquisition | Operator completed; phase-specific evidence review pending |
| System recenter and reference-space event | Explicit `2->4` event in final recording; control realignment is a later task |
| Headset removal and application suspension/resume | Operator completed; full phase evidence review pending |
| Body permission denial, independent head/hands, regrant | Pending separate recording |
| Automatic 180-second stop and bounded writer behavior | Duration-limit event observed in 0.3.1 with zero reported drops; no overload stress pass claimed |

## Software checks

- Targeted simulator/model/capture suites: **35 passed, 1 skipped** (optional
  `emet.robots` unavailable). This includes 16 capture evidence tests.
- All 13 pre-commit hooks passed; Python wheel and source distribution built.
- Final APK built with zero errors and manifest validation enabled.
- Independent reviews covered the contract/model conventions and evidence
  analyzer; later profile regressions have explicit rejection tests.

See [the probe procedure](quest_tracking_probe.md) for build, install, phase-marker,
capture and analysis commands. The summary deliberately separates hand sample
timestamp advances from head pose-target and body state timestamps. It does not
convert those observations into an automatic tracking accuracy pass.
