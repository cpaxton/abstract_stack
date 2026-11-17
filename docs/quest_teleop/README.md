# Quest Teleop v1 interfaces

This is the implementation contract for [issue #12](https://github.com/cpaxton/abstract_stack/issues/12),
under [Quest Teleop](https://github.com/cpaxton/abstract_stack/milestone/1).
It starts from current main `c74837384559479ca1b7522e446801e313dd026a`; it does
not depend on the earlier teleop branch or its plan. These are proposed interfaces
and provisional budgets, not implemented transport or demonstrated performance.

The first release controls MABEL's head, arms, fingers, torso and lift in
`ShelfStockingEnv`. The base receives stationary actuator targets. There is no
hardware-robot endpoint. Only `zed_left` and `zed_right` are streamed, into a
room-anchored stereo window on Quest 3. Base driving, wrist video, avatars,
autonomy, demonstration training and a full immersive view are deferred.

The [simulation host adapter](sim_adapter.md) implements the simulator boundary
from #14, including native state/commands, reset ownership and paired head renders.

## Baseline and ownership

[contract.json](contract.json) pins the source/model revision and the reference
Python, MuJoCo, NumPy, Pillow, Unity, Meta XR Core, OpenXR and URP versions.
The Python versions describe the existing lightweight Mac environment;
`uv.lock` remains the package lock. The Unity pins match the sibling tracking
probe's installed packages, not a claim that all SDK features work on its device.
Do not add an SDK or change a pin silently: record the new version and rerun the
relevant fixture/device checks. No dependencies change in this issue.

| Module / issue | Input → output | Ownership |
| --- | --- | --- |
| Quest capture #13 | SDK state → raw JSONL and typed observations | Head, hand and body acquisition, validity, permission/lifecycle events |
| Network #15 | Observations ↔ tracking/control/status messages | Validation, bounded queues, session identity, clock exchange |
| Calibration #16 | Tracking plus neutral capture → calibrated targets | Frame conversion, scale, wrist offsets, alignment generation |
| Arms/head #17 | Calibrated wrist/head targets → nominal actuator proposals | Both seven-joint arms **and both hand-wrist joints**; three neck joints |
| Fingers #18 | Named hand skeleton → nominal actuator proposals | Remaining 16 finger actuators per hand |
| Body #19 | Calibrated torso/height → nominal actuator proposals | `Torso`, `Lift`; disabled until independently checked |
| State/gate #20 | Proposals, measured state, input age → complete applied targets | Engagement, measured hold, final coupled bounds and single command writer |
| Sim adapter #14 | Applied targets → stepped state, RGB pairs | Sole owner of live `MjModel` / `MjData`; reset, stepping and snapshots |
| Video #21 / #22 | State snapshot → stereo pair → headset window | Capture, pair identity, bounded encode/transport/decode/display |
| UI #23 | Operator actions → sequenced control requests | Status and deliberate calibration/engage/pause/reset actions |

Every actuator is listed once in `contract.json`, including its owner, command
kind and physical unit. `Left Wrist` / `Right Wrist` belong to arm retargeting,
not finger retargeting. `Lift` means the native `Total Lift` fixed-tendon length
in metres, combining `Lift Lower` and `Lift Upper`; never command the two stages
independently or modify their `Sync Lift` equality. All other position targets
are radians; drive velocity targets are radians/second. Base steer targets hold
measured actuator lengths and drive speed targets are zero. This is a command
policy, not a fixed-base weld or a promise that contacts cannot move the base.

Only the sim owner writes live `data.ctrl` and steps physics. Solvers may use
scratch data, but cannot write live `qpos`, attach objects, disable contacts,
apply hidden external forces or replace robot geometry. Proposals are composed
once, checked after every correction against native ranges and configured
rate/acceleration bounds, and then applied as a complete vector. Duplicate
ownership, non-finite values or an unsupported transmission fail explicitly.
Keep nominal targets, applied bounded targets and measured results separate.

## Coordinates and named skeletons

All distances are metres, angles radians and durations seconds. Transport poses
use `[w,x,y,z]` unit quaternions and active local-to-parent rotations. Position
and orientation validity are separate; an invalid component is `null`, never an
identity/zero pose presented as a measurement. A valid quaternion must have norm
within `1e-3` of one. Reject non-finite JSON values everywhere.

Three frames must remain distinct:

1. **Raw Unity tracking space:** left-handed, X right, Y up, Z forward, Unity
   `[x,y,z,w]` quaternions. The probe records uncalibrated poses. Native OVR
   right-handed poses must first be converted according to their SDK space;
   this conversion is not robot calibration.
2. **Operator transport frame:** right-handed, X forward, Y left, Z up. Convert
   Unity positions using `p_operator = B p_unity`, where
   `B = [[0,0,1],[-1,0,0],[0,1,0]]`. Convert rotations through matrices using
   `R_operator = B R_unity B^T`, then encode wxyz. `det(B)=-1`; B itself is not
   a quaternion rotation. Do not simply reorder quaternion components.
3. **Robot root:** the native `mabel_chassis` body frame, not world or operator
   space. MABEL's neutral forward direction is native -X; its left is -Y and up
   +Z. The neutral operator-to-root rotation is `diag(-1,-1,1)`. Calibration
   establishes the actual origin/orientation and wrist/palm offsets; never use
   world-frame spawn yaw as a second operator rotation.

Transport head, wrists and named hand/body joint poses are expressed in the
same operator frame at capture time, not individually in bone-local frames.
The acquisition adapter must compose raw root/local skeleton transforms before
frame conversion. Retargeting uses relative motion from calibrated neutral,
with configured scale and bounded reach. An absolute human pose is not a robot
joint command. A single calibration artifact records both neutral poses,
conversion/palm offsets, scale and generation; all consumers use that artifact.

Use explicit `skeleton_profile` and joint **names**, never infer one layout from
an array length. `ovr_hand_24` in the manifest lists the pinned SDK's legacy
`Hand_*` bones in index order. It is different from the 26-joint `XRHand_*`
profile, which requires a new declared mapping before use. Bone tips can lack
measured orientations. `ovr_body_upper_18` lists Root through RightHandWristTwist;
body-derived hand joints must not overwrite the dedicated hand source. Required
body guidance names are `Body_Chest`, `Body_Hips`, `Body_LeftArmLower` and
`Body_RightArmLower`. Unknown names, duplicate names and undeclared profiles
are rejected. Missing joints are explicitly invalid, not copied from the wrist.
Tracking confidence, pose validity and direct tracking are different quantities;
estimated body joints may be valid without being directly tracked.

## Tracking and control messages

Proposed transport: two independently serviced WebSocket connections on the same
host: `/quest/control` (UTF-8 JSON text) and `/quest/video` (binary). Host/port are
configured explicitly; first-use is a trusted local development network. This
issue neither implements access control nor authorizes exposure beyond that
network. A slow video peer must not block physics or control handling.

Every JSON message has `schema_version: 1`, `type` and `session_id`. The client
creates a fresh UUID for each control connection. The video connection binds to
that control session; it does not create an independent session UUID. A control
reconnect requires a new UUID and clears video, calibration and command state.
Unsupported versions and
unknown message types fail the connection before engagement. Integers are exact
nonnegative values below `2^53`, with no wrap within a session. Input payloads
are bounded by the limits in the manifest before allocation/parse. Reject
unexpected fields in v1; protocol additions require an agreed version change.

The first control `hello` is `{schema_version, type:"hello", session_id,
profile:"quest_teleop_v1"}`. The host returns its baseline revision, video
codec/profile and effective budgets in `hello_ack`. Mismatched model/profile
cannot engage. After control `hello_ack`, the video socket sends
`{schema_version,type:"video_bind",session_id}`; the host acknowledges with
`video_bind_ack` and only then sends pairs. Reject orphaned/old sessions and
multiple video writers; control disconnect closes the bound video socket.
Hello/status/control/clock/video-feedback traffic must not be overwritten by
the latest-pose queue.

[examples/tracking.json](examples/tracking.json) is a complete **synthetic**
message. Joint positions intentionally share a placeholder location; it checks
framing/mapping, not anatomical geometry or tracking quality.

| Tracking field | Semantics |
| --- | --- |
| `sequence` | Strictly increasing tracking-message sequence; gaps are allowed; old/duplicate sequences are dropped |
| `calibration_generation` | Host-issued nonnegative generation echoed by client; mismatch cannot control |
| `sent_monotonic_s` | Quest application monotonic send time, not wall time or simulation time |
| `frame` | Exactly `operator_forward_left_up` |
| `worn`, `focused` | Explicit lifecycle state; false pauses control |
| `head`, `left_hand`, `right_hand`, `body` | Independently timestamped sources, not assumed simultaneous |
| Source `observed_monotonic_s` | Quest app monotonic time at acquisition; never replaces SDK source time |
| Source `source_time_s`, `source_clock` | SDK timestamp and declared clock domain, or null / `unavailable` |
| Source `timestamp_kind` | `sensor_sample`, `state_target`, `predicted_target`, or `unavailable`; identifies what the timestamp measures |
| Source `fresh` | True/false only for an available sensor-sample timestamp advancing/repeating; null when the provider exposes only a state target/prediction or no timestamp |
| Body `state_advanced` | True only above the prior timestamp high-water mark; false for repeat/regression, null if unavailable. Regression is logged separately. Advancing state time is not evidence of a new sensor sample |
| Source `pose` | Head/wrist pose with `position_m`, `quaternion_wxyz`, validity/tracked flags and `confidence` |
| Hand/body `skeleton_profile`, `joints` | Named joint pose list under the declared profile, with independent validity/tracked flags |
| `confidence` | SDK confidence normalized to [0,1] where supported, otherwise null; a confidence score is not a tracked flag |

`position_tracked` / `orientation_tracked` are true/false only when that property
is exposed by the provider, otherwise null. The legacy hand API does not supply
all per-joint flags; propagate its limitations instead of manufacturing them.
Body `Time` is a runtime state-target time, not a proven sensor sample. Record
`timestamp_kind:"state_target"`, `fresh:null` and separately `state_advanced`.
A head pose target timestamp, when available, is `predicted_target` with
`fresh:null`; never call a prediction a fresh sensor observation.
The Unity head API may expose no sensor timestamp: use explicit `fresh:null`,
`source_time_s:null`, `source_clock:"unavailable"`. Successful polls establish
observation availability only, not sensor freshness. Head validity, worn/focused
state and observation age are required; actual sensor age remains unknown.
The raw capture profile below is different; its decoder must adapt to this
transport rather than claiming byte-for-byte compatibility.

Control requests contain `{schema_version,type:"control",session_id,event_id,
calibration_generation,action}`. `event_id` increases per session; actions are
`calibrate`, `engage`, `pause`, `resume`, `reset`, `reference_space_changed`.
Acknowledgements echo event ID,
result and active generation. Cache acknowledgements so retries are idempotent.
A reset request is never replayed on reconnect. Pause and
`reference_space_changed` remain available even
when the supplied generation is obsolete; other state-changing actions require
the current generation. `calibrate` captures neutral only while Ready/Paused and
with valid required inputs, increments generation and leaves the system Ready.
`reset` pauses, resets the sim explicitly, increments generation and requires
new calibration. The client sends `reference_space_changed` immediately on
Quest recenter/reference-space change; until its acknowledgement it stops sending
engage/motion updates. The host invalidates calibration, increments generation
and pauses; repeated event IDs return the same acknowledgement. It cannot
silently move robot targets. A session is not allowed to engage until calibration has been acknowledged.

| State | Entry and permitted motion |
| --- | --- |
| Disconnected | No accepted session; bounded stationary/hold policy only |
| Ready | Connected, input/profile valid; no live human command application |
| Engaged | Deliberate engage/resume, current neutral and generation; bounded proposals applied |
| Paused | Tracking/lifecycle/video-age failure or deliberate pause; measured-state hold policy |
| Fault | Invalid command/model/physics or unrecoverable module failure; no live retargeting |

Missing/stale head or either hand pauses both arms and fingers together; a body
failure disables body guidance and pauses torso/lift, leaving arms/head enabled
only in an explicitly selected arms/head mode. No automatic resume on tracking
return. Paused→Engaged requires deliberate resume after realignment and a bounded
transition. Fault recovery uses explicit `reset`, allowed only after host diagnostics show
finite valid model/state and a valid reset path. If those checks fail, remain
Fault until repaired/restarted; a tracking refresh cannot clear Fault. Define the actual loaded hold/effort behavior in #20 before
claiming an object remains held. Video older than its stale budget pauses all
human control because the operator is otherwise moving blind.

[examples/status.json](examples/status.json) separates measured tracking errors
from IK residuals. Status carries state/reason, generation, accepted sequence,
simulation time, host monotonic time and the latest applied command/pair IDs.
All active controllers additionally log nominal/applied/measured actuator values
and native limit/saturation diagnostics. UI must show stale tracking/video and
pause/fault reason without implying that a plausible pose is fresh.

## Clock, queues and provisional budgets

The manifest's budgets are initial engineering targets to measure in #13–#25,
not physics success tolerances or acceptance evidence. Initial targets are
72 Hz capture/send, 50 Hz control, native configured 0.002 s physics steps and
30 distinct stereo pairs/s at 640×480 per eye. Observe the scene's actual
configured timestep; do not silently rewrite it to meet a target.

The sender keeps one newest tracking observation; the receiver keeps one newest
validated observation. Each raw/encoded video queue holds at most one entire
pair. Never enqueue an unbounded history or split eyes across pair IDs. Bound
JSON to 256 KiB, video metadata to 16 KiB and each JPEG eye to 2 MiB. Exceeding a
limit rejects that message/pair and records a diagnostic. Video and control have
independent workers and shutdown paths; the sim owner snapshots state for video.

Clock exchange uses `clock_ping` with client `t0`, `clock_pong` echoing `t0` plus
host receive/send `t1,t2`; Quest records receive `t3`. All four are monotonic
seconds in their own domains. Estimate host-minus-client offset as
`((t1-t0)+(t2-t3))/2` and uncertainty as `((t3-t0)-(t2-t1))/2`. Reject negative
round trips; refresh periodically and after suspend/reconnect. The client sends
`clock_result` containing the same exchange ID and `t0,t1,t2,t3`; the host checks
its own saved `t1,t2`, timestamp ordering and round-trip bounds, recomputes the
offset/uncertainty, then acknowledges the accepted estimate in status. It does
not trust an arbitrary client-provided offset or accept results from old sessions.
No subtraction of SDK-runtime time from app/host time is allowed without a separately measured
mapping and uncertainty. Include offset, uncertainty and mapping age in traces.
Maximum provisional uncertainty is 25 ms; above that pause engagement until a
usable observation-age bound is available.

V1 bounds **observation age**, not sensor acquisition age: use Quest app
`observed_monotonic_s` mapped to host time, plus clock uncertainty, with a 150 ms
limit. For hand streams, update the last accepted observation time only when the
valid SDK sample timestamp advances; duplicate/regressed source timestamps do
not refresh it even if a new network message arrives. The first positive valid
sample establishes the initial reference. There is no runtime-to-app clock
mapping in v1, so no numeric SDK-source-age bound may be claimed.
For head or body providers without actual sensor timestamps, use last
successful valid observation age and label it `observation_age`, never
`sensor_age`; state/prediction timestamp advances are diagnostic only. The
initial profile can bound observation availability, not prove their sensor age.
Reject messages whose mapped observation
lies beyond clock uncertainty into the future; drop stale/reordered messages.
Video stale age uses host capture time translated to Quest time plus uncertainty,
with a 300 ms limit. Until clocks synchronize, video is marked unsynchronized and
human actuation remains paused. Local receipt-age counters alone cannot prove
end-to-end latency.

Measure input acquisition and distinct source rates separately, control ticks,
physics real-time factor, render/encode/decode/present durations, queue drops and
presented pair IDs. Report measured command tracking error alongside IK residual.
Input-to-present timing is an estimate with clock uncertainty; true motion-to-photon
requires independent optical measurement. Store full captures in ignored `outputs/`
with source/model/settings/scene/seed and an event-marked device test procedure.

## Stereo pair contract

Use only the native head cameras: `zed_left` / `zed_right`, 63 mm separation,
83.3° vertical FOV, declared 1600×1200 native resolution. Initial output is
640×480 per eye (same 4:3 aspect). Render both from one forwarded simulation state
without a physics step between eyes. Capture simulation time and both camera
poses relative to `mabel_chassis` at that same state. Do not substitute headset
IPD or a virtual camera baseline.

One binary WebSocket message is one JPEG pair: ASCII magic `QTP1`, then three
unsigned big-endian uint32 lengths `(metadata_bytes,left_bytes,right_bytes)`,
then exactly those UTF-8 JSON / JPEG / JPEG byte regions. No trailing bytes,
partial messages or eye interleaving are allowed. Validate lengths before
allocation/decode and verify decoded dimensions. The pair header is
[examples/stereo_pair.json](examples/stereo_pair.json), with cameras
measured at capture time.

The camera transforms in that example are actual seed-0 reset transforms of the
pinned stocking scene; pair IDs/capture clock values remain synthetic.

Header `pair_id` increases per session. Include `calibration_generation`, capture
host monotonic time, simulation time, codec (`jpeg` initially), pixel format
(`rgb8` before encoding), output dimensions, explicit eye order, camera names,
vertical FOV and root-relative poses. Each camera optical frame has +X image
right, +Y image up, -Z forward, matching MuJoCo `cam_xmat`; quaternions rotate
that frame into robot root. Image rows are top-to-bottom, columns left-to-right.
For the symmetric pinhole profile, `fy = height/(2*tan(fovy/2))`, `fx=fy`, and
principal point in pixel-centre coordinates is `((width-1)/2,(height-1)/2)`.
This declares projection conventions, not calibrated real-camera optics.

Receiver discards old/mismatched generation/session pairs atomically. Display
left only to left eye and right only to right eye in the room-anchored window.
The client reports `video_status` on control at least every 50 ms while active:
`{schema_version,type:"video_status",session_id,sequence,calibration_generation,
presented_pair_id,presented_monotonic_s,reported_monotonic_s,stale}`. Before any
presentation the pair ID/time are null and stale is true. Only actually presented
pairs count; decoder completion is not display evidence. The host retains a
bounded lookup of its sent pair IDs/capture times for this age budget, validates
current session/generation and monotonic feedback sequence, and checks both
reported presentation age with clock uncertainty and conservative age
`host_now - capture_host_time`. Repeated feedback never refreshes capture age.
Engagement requires a reported complete fresh pair; no valid feedback for
300 ms or either age beyond 300 ms pauses control. Losing feedback or rendering
therefore cannot leave actuation active merely because socket writes succeed.
Client stale detection immediately emits a pause request as well.

Keep the last complete pair only while showing its stale status; never combine a
new left eye with an old right eye. Reconnect clears both eyes and decode state.
Benchmark JPEG first; H.264 requires a separately negotiated codec profile with
access-unit/keyframe/recovery rules before replacing this format in #21.

## Raw capture adapter boundary (#13)

`quest_tracking_probe_v1` JSONL is acquisition evidence, not a command protocol.
Every record identifies `format`, `type` and `session_id`; types are `session`,
`sample`, `event` and `skeleton`. It preserves native OVR right-handed
tracking-space head/body/hand-root poses with `quaternion_xyzw`, rather than
silently converting them to the operator frame. Sessions record dependency/runtime
versions and coordinate meaning. Samples identify sequence, app capture monotonic
time, recenter count, dropped records, focus and presence. Lifecycle events make
suspension/recenter gaps explicit.

Head records distinguish runtime pose target time from sensor time and mark
source freshness unknown. Hands carry `ovr_hand_24`, API success, status,
confidence, root pose, scale, sample/request times, source timestamp change,
named bone rotations/positions where returned, and finger confidence/pinch data.
Skeleton records carry ID/name/parent/bind pose metadata. Legacy OVR bone rotations
have the SDK's parent-local special basis; they are **not** global tracking-space
quaternions. A later adapter must implement/verify SDK-compatible forward kinematics
using that metadata before producing the global canonical hand poses above.
Raw capture must mark unavailable per-joint validity/tracking as unavailable,
not invent flags from confidence. Body records carry runtime state time and its
change separately from unknown sensor freshness, joint set/fidelity/calibration,
confidence, skeleton version and per-joint validity/tracked flags. A full-body
returned layout must declare its own profile; the 18 upper-body guidance names
in the transport manifest are selected by name, never by reusing its array size.

Raw timestamps alone do not establish clocks shared with the app, host or camera.
Keep source raw data and later conversions/calibration artifacts separately so a
bad adapter can be corrected without rewriting acquisition evidence.

## Fixtures and checks

The examples are synthetic inputs for independent transport, retargeting, UI and
video implementations. They do not command a robot or prove usable tracking.
Use an actual Quest capture for device validity/freshness tests and actual native
physics for measured control claims. Unit tests check declared axes, ownership,
tendon and stereo geometry against the pinned model when assets are available.

```sh
MUJOCO_GL=cgl ./scripts/test.sh tests/test_quest_teleop_contract.py -q
uv run --no-sync pre-commit run --all-files --hook-stage pre-commit
uv build
```

#13's raw recording profile is documented with its probe. It must retain source
joint names/indices, root/local transform meaning, native-to-Unity conversion,
source timestamps and lifecycle events; the capture cannot be accepted merely
because it resembles the transport example.
