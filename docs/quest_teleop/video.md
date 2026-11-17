# Paired head-camera video (#21)

This component renders only `zed_left` and `zed_right`, encodes one complete
QTP1 JPEG pair and attaches its route to the #15 control host. No robot commands
or engagement state machine are implemented. The native model, cameras, physics
and actuator settings are preserved.

## Run and integrate

From a checkout containing the #15 transport component and its pinned dependency:

```sh
PYTHONPATH=src python scripts/probe_quest_video.py --mode serve --source synthetic \
  --host 0.0.0.0 --duration 120 --output outputs/quest-video/synthetic.json
MUJOCO_GL=cgl PYTHONPATH=src python scripts/probe_quest_video.py --mode serve \
  --source mujoco --host 0.0.0.0 --duration 120 --seed 0 \
  --output outputs/quest-video/native.json
```

Use `MUJOCO_GL=egl` on a configured Linux host. `--mode benchmark` needs no network
component. `--width 640 --height 480 --fps 30 --quality 75` are defaults;
`--save-pair outputs/quest-video/first.qtp` stores one diagnostic packet.
The Unity #22 viewer uses the same host on `/quest/control` and `/quest/video`.
USB `adb reverse tcp:8765 tcp:8765` provides a baseline; LAN tests must explicitly
bind the intended interface and configure that computer's reachable address.

```python
service = VideoService(worker.encoded)
registry = SessionRegistry()
registry.add_disconnect_callback(service.disconnect)
server = ControlServer(
    registry=registry,
    extra_routes={"/quest/video": service.handle},
    on_video_status=service.on_video_status,
)
```

`VideoService` accepts the registry/session API by duck typing; encoding and
rendering do not import the transport. The route imports the explicitly declared
#15 WebSocket library only when serving. `health(session)` provides conservative
presentation health for the future #20 controller to consume. It never enables
actuation itself. Control disconnect closes video, clears sent-pair lookup and
feedback, and a new control UUID is required on reconnect.

## Ownership and bounds

The live owner reads detached `SimState` and calls `ProcessVideoWorker.submit`.
Two capacity-one IPC queues carry snapshots and encoded pairs to/from a private
spawned process. There is one final encoded latest slot plus bounded in-flight
work. Updates never wait for a slow renderer or receiver; IPC feeder races may
drop an update instead of blocking. The render process owns its own environment,
model, data and graphics context. Before accepting snapshots it verifies the
same generated XML hash and rejects unsupported actuator activation state.
It replays qpos/qvel/ctrl/time into **scratch data only**, forwards once, and
renders both cameras without stepping between eyes. Live model/data never cross
threads or processes, and replay writes do not constitute physical control.

Snapshot capture time precedes rendering/encoding; it is never relabelled with
completion time. Reset generations invalidate old queued or in-flight output.
Pair IDs do not reset while the worker lives. Worker startup and crashes are
reported, and a zero-pair benchmark fails rather than reporting success.
`VideoWorker` is a thread-only reference/test harness; the runnable native probe
uses `ProcessVideoWorker` to isolate graphics/GIL scheduling from physics/control.
Scheduling still shares this computer's CPU/GPU; it is not hard real time.

A slow send has a 250 ms deadline and closes that video peer. There is never a
queue of pending send tasks. Send selection drops pairs already older than
300 ms. At most 128 sent-pair identities/capture times are retained per session.
Generation changes clear presentation provenance. Feedback sequence, current
generation, sent-pair identity, mapped presentation/report chronology, clock
uncertainty/expiry, conservative capture age and feedback timeout all gate
health. Repeated presentation of the same old capture cannot refresh its age.

## Images and geometry

Pixels are RGB8 before JPEG, with rows top-to-bottom and columns left-to-right.
The paired JPEG quality is75, chroma subsampling4:2:0; no independently interleaved
eye messages are used. Strict v1 metadata, camera names, finite normalized poses,
dimensions and JPEG dimensions are validated. Limits are16,384 metadata bytes,
2,097,152 bytes per eye and4,210,704 bytes total; diagnostic decode additionally
bounds dimensions to2048 per axis and3,145,728 pixels.

Real images use native63 mm baseline,83.3 degree vertical FOV and640×480 initial
output (4:3). Camera poses are measured relative to `mabel_chassis` from the same
forwarded snapshot: optical +X right,+Y up,-Z forward, wxyz quaternion. Intrinsics
follow the shared contract. The synthetic source has explicit LEFT/RIGHT labels,
pair IDs, host capture timestamps, color bars, checker grid and common motion with
opposite eye disparity. Its neutral illustrative camera pose is labelled
synthetic; it is not a native MABEL geometry measurement.

## Validation and current codec decision

Tests check malformed/oversized/trailing/duplicate-key packets, nonfinite or
unknown fields, wrong eye order, wrong JPEG dimensions, full paired roundtrip,
bounded overwrite work, timeout behavior and clock/provenance/generation/age
rejection. Native tests compare scratch images and camera transforms exactly with
a live same-state render and prove physics progresses while a separate worker
renders. Process failure/reset invalidation are covered. The optional
`test_quest_video_transport.py` suite exercises real localhost sockets with #15:
bind, complete pair, four-timestamp clock, presentation feedback, generation
invalidation, orphan rejection and control-disconnect video cleanup. It skips
when that independently reviewed component is absent; run it after combining the
branches.

On this Mac (macOS27.0.1 arm64, Python3.11.15, MuJoCo3.17.0, NumPy1.26.4,
Pillow13.0.0), seed0 studio with6 dynamic products and untouched reset actuator
references, a20 s process-isolated run measured354 pairs and849 native control
steps. At640×480/quality75, packet p50/p95 was34,927/35,062 bytes; encode
p50/p95 was1.40/1.92 ms and Mac decode1.48/2.21 ms. Rendering averaged51.4 ms per
pair; this development run, with Unity and test work also active, **missed the
provisional30 pairs/s target**. Control-step p50/p95/max was9.17/11.08/20.81 ms.
The original thread-only experiment had a107.6 ms maximum step; the production
process boundary removes shared graphics/GIL work, without claiming a fixed
wall-clock scheduling guarantee. Full machine-local reports remain in ignored
`outputs/quest-video/` with settings/hashes. These measurements precede the final
error-reporting changes and are performance observations, not frozen acceptance
results; rerun the commands from the reviewed commit for a matched comparison.

Retain JPEG for this prototype: native packets are approximately1.05 MB/s at30
pairs/s and Mac encode/decode each take about1–2 ms; rendering is the measured
host bottleneck. Quest timing and network/render feedback are recorded during
integration with #22. Hardware H.264 is not justified by host JPEG timings alone
and is not implemented or silently substituted. A later choice requires target
Wi-Fi/Quest evidence and a separately negotiated access-unit/keyframe/recovery
profile. Render submission feedback is not compositor scanout or optical
motion-to-photon evidence. The sustained30 Hz target, LAN behavior, and complete
wearer lifecycle checks remain explicit acceptance gates until measured.

## Parallel merge plan

This PR owns only `video.py`, video tests/probe and this document. #15 owns Python
network dependencies and its transport files; #16 owns calibration files; #22
owns stereo Unity assets/settings. The common development base contains existing
PRs#26/#27/#28 only. After those foundations are reviewed and squash-merged with
authorization, rebase each task's own commits onto the actual main squash
history and retarget the four disjoint PRs. No task needs another task's code in
its diff. Video serving requires #15 at runtime; the viewer's network test needs
#15+#21. Codec/synthetic/core tests remain independently runnable.

### Physical Quest integration, 2026-11-17

The #15/#21 host and #22 APK streamed640×480 synthetic pairs over USB reverse.
There were4,493 pair sends, no send timeouts, and2,703 fresh presentation feedback
samples out of2,706 (initial stale/unsynchronized samples were retained). The
clock-mapped capture-to-render-submission estimate p50/p95/max was
40.7/54.3/82.4 ms; conservative capture age at host receipt61.0/80.0/111.4 ms.
Clock uncertainty p50/p95 was0.79/19.97 ms. This is application submission timing,
not optical motion-to-photon or a Wi-Fi result. The user confirmed LIVE,
advancing pair IDs, correct per-eye labels and an anchored window. Meta VR CLI
compositor capture independently showed those labels/matching IDs/LIVE;
read-only XR MCP queried valid floor-space head poses and an unchanged room
anchor. No synthetic device input was injected. Real native-camera streaming
also produced a LIVE stereo shelf view in the compositor capture. Full local
reports, logs and screenshots stay ignored; no personal tracking trajectories
are committed. The initial local cross is fixed in its stereo images; apparent
motion in front of the window is not evidence of a separately tracked 3D cross.
