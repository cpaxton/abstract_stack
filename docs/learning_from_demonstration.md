# Demonstrations and teacher–student data

The first collection contract and local export adapter exist. A native MABEL
hold fixture collected ten causal control transitions with virtual wrist and
tactile measurements. This is pipeline evidence, not a stocking demonstration,
a trained student, verified VR teleoperation or hardware-ready tactile sensing.
No dataset was uploaded and no training job was started.

## Current evidence and gaps

| Item | Status |
| --- | --- |
| Physics diagnostic traces and review videos | Existing; preserve failures and unknown process outcomes |
| Causal control-rate collection | New explicit recorder and native hold fixture |
| Nominal teacher / applied servo separation | Stored as distinct arrays with a native action contract |
| Privileged teacher / observable student separation | Explicit separate namespaces and student-only frame projection |
| LeRobot | Pinned official 0.7.0 local export/reload passed in a separate locked worker |
| Student training or DAgger | Not implemented |
| VR demonstration loop | Not integrated with the new recorder |
| Raw synchronized camera collection | Opt-in native first-/third-person RGB; real-image SDK roundtrip passed |
| F/T and tactile | Native simulated signals from `sim_sensors.py`; not calibrated hardware measurements |

The existing `trace.npz` records post-physics-step task metrics and applied
actuator targets every 2 ms. A terminal nominal-reference dictionary does not
recover the missing nominal action history. Those arrays are useful diagnostic
evidence, not correctly aligned behavior-cloning pairs. Review MP4s also contain
overlays and a paused final verdict; they must not become raw student camera
observations or be aligned by frame index to 50 Hz decisions.

An audit of the retained v17 failure inspected 72,440 diagnostic rows and refused
BC import. It preserved the reported failure and matching process status.
Results are in `outputs/lfd-v1/legacy-v17-audit/audit.json`.

```bash
.venv/bin/python scripts/export_lfd.py audit-legacy \
  outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode \
  --output outputs/my-fresh-legacy-audit
```

Exit 2 means the bundle is not training-ready; it does not delete or rewrite it.
Native replay could reconstruct some proprioception, but cannot recover unknown
nominal teacher references or turn a continuation into a fresh episode.

## Collection clock and action contract

One record has the following order:

1. Capture observable sensors at physics time `t` before querying the policy.
2. Query a teacher using that student view plus a separate privileged context.
3. Record the teacher's nominal native reference at `t`, before compensation.
4. Apply the low-level controller's compensated native servo targets and step
   physics for exactly one declared control period.
5. Record the next observation at `t + 1/fps`, together with those applied targets.

The recorder observes this transaction; it neither applies controls nor steps
MuJoCo. All interaction and all-product safety/task guards remain the caller's
responsibility. Native target bounds, finite data, causal sensor timestamps,
declared maximum sensor age and a fixed control period are checked.

`ActionContract` names every native actuator in order and declares its mode,
units and bounds. For MABEL this is 59 channels: arm/finger/wrist/steering
position references in radians, the native common lift in metres and wheel
velocity references in rad/s. Unlimited native steering controls are represented
explicitly with null bounds rather than an invented zero-width interval.
Measured transmission position, speed and effort are distinct from control
targets; the fixture records their exact order and units in provenance.

The initial action representation is the complete nominal native reference
vector. Changing later to palm goals, deltas, force targets or primitive actions
requires a new explicit action contract and compatible runtime adapter. It must
not reinterpret an existing policy's output units or apply compensation twice.

## Student and teacher observations

`ObservationField` declares each channel's shape, units, frame, source and maximum
age. `Snapshot` contains only those measurements at a control boundary.
`student_observation` emits values, availability and sensor sample timestamps.
Undeclared object/world-state fields are rejected. `teacher_observation` nests
the same student view beside a separate `privileged` dictionary; teacher context
is written to `teacher.jsonl` and never enters `student_frames()`.

Supported numeric modalities are proprioception, wrist reaction wrench,
contact-only hand wrench and coarse tactile proxies. Raw camera channels use
`observation.images.<camera>` and require HWC uint8 RGB. Unavailable channels use
zero values, false availability and sample time -1. Zero force with true
availability is a valid zero measurement. The first collector aggregates native
per-wrist/taxel validity conservatively: a modality is unavailable if any entry
is unavailable. Per-element validity can be added in a later schema revision.

The wrist reaction channel follows MuJoCo parent-interaction F/T conventions;
the contact-only channel is the net external contact wrench **on** the distal
hand and excludes gravity/inertial loads. Both use native wrist-child axes at
the wrist anchor. Tactile values are forces in configured collision-geom frames,
not optical tactile images, soft skin, contact patches or calibrated taxels.
The sensor suite's exact metadata is retained. All are explicitly simulator
measurements/approximations; no noise, bias or hardware equivalence is implied.

## Neutral episode format and outcomes

A fresh collection directory contains:

- `episode.json`: contracts, clocks, scene/model/controller provenance and outcome.
- `frames.npz`: N nominal/applied actions and N+1 pre/next observations, masks
  and timestamps; no pickled objects.
- `teacher.jsonl`: N privileged contexts, indexed to pre-action observations.

Payload hashes bind the saved arrays and teacher contexts. A reader rejects
changed payloads, malformed shapes, nonfinite observations and inconsistent
clocks/outcomes. The manifest begins as unknown before collection. A handled
interruption preserves complete earlier transitions and the unfinished action;
an unhandled crash leaves an unknown manifest, not a fabricated successful demo.
This recorder does not independently verify a claimed task outcome: a real task
integration must supply the evaluator's measured terminal result and process
status. The hold fixture deliberately reports failure/no stocking task attempted.

Known failures remain available but require explicit inclusion in export.
Unknown/incomplete outcomes cannot become teacher-training episodes. A successful
demonstration may be marked training-eligible; fixture and replay records are
never promoted by that flag. The episode schema's `fresh_task_credit` is zero;
task evaluation remains separate from collection and serialization.

## Native collection fixture

```bash
MABEL_ROOT=third_party/open_mabel \
  .venv/bin/python scripts/export_lfd_fixture.py \
  --output outputs/my-fresh-lfd-fixture --steps 10
```

The standalone fixture settles the native robot, captures measured proprioception
and the virtual sensor suite before each decision, commands clean nominal holds
through `MabelController`, and records applied compensated targets separately.
`BaseContactGuard` samples each 2 ms physics step. It performs no pose writes,
hidden attachments or external-force actions. By default its camera channel is
explicitly unavailable and no renderer runs.

The frozen ten-transition probe completed in 1.47 s wall time with a 30 s budget,
at 50 Hz control and 500 Hz physics. It captured 11 snapshots of 177-dimensional
proprioception, 2x6 reaction wrenches, 2x6 contact wrenches and 56x3 tactile proxy
forces. All those sensor modalities were available; the camera was not. Its
model revision is `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`. Exact copied sources,
source hashes, provenance, arrays and the zero-credit process result remain in
`outputs/lfd-v1/native-fixture/`. Nominal and applied controls differ in that
record, confirming that the dataset does not silently rename applied targets.

This is not a hook in the stocking loop: that loop applies finger compensation
before some common-control hooks, so attaching a recorder afterward could lose
the actual teacher reference. The later integration must capture the clean
decision before **all** compensation and retain per-control action timing.

## Opt-in synchronized raw cameras

On a Linux EGL setup, a bounded collection command is:

```bash
flock -w 30 /tmp/abstract-stack-render.lock timeout 60s env MUJOCO_GL=egl \
  MABEL_ROOT="$PWD/third_party/open_mabel" \
  .venv/bin/python scripts/export_lfd_fixture.py \
  --output outputs/my-fresh-camera-fixture --steps 5 --cameras \
  --camera-width 160 --camera-height 120
```

The camera option adds raw `observation.images.zed_left` and
`observation.images.stocking_third_person` HWC uint8 RGB at every control
boundary, including the final next-observation. It leaves original camera
poses and all physical geometry unchanged. Both views render sequentially
while physics is paused, before the corresponding nominal action. Their
sample timestamps equal that boundary's native physics time. This represents
instantaneous ideal cameras; no exposure, readout, latency or noise is modeled.
Rendering blocks collection, so 50 Hz simulation decisions do not imply 50 Hz
wall-clock operation.

`camera_captures.jsonl` records every attempted view's native capture/sample time,
monotonic start/end times, availability and error. Monotonic times are local
process diagnostics and never substitute for simulation timestamps or become
student features. A render failure produces a zero image, false availability
and sample time -1; it never reuses the last good frame. Invalid camera names
or dimensions fail configuration. Framebuffer limits are respected without
modifying the native model. The collector detects physics advancement during
capture and aborts rather than labeling that frame synchronized.

Camera provenance includes native IDs, parent bodies, local poses, camera modes,
vertical FOV, image dimensions, renderer/version/backend and visibility options.
The scene/model and source hashes bind the remaining camera configuration.
Raw images contain no review overlays or target/annotation sites, and use no
display cutaways. Default camera orientations may occlude useful task geometry;
this fixture validates capture, not useful gaze or view selection.

A source-frozen native probe collected five hold transitions and six paired
snapshots. All 12 raw frames were available at 160x120; every image sample time
matched its proprioception boundary. Collection completed in 4.93 s under its
60 s budget. Copied sources, hashes, logs, raw PNG previews and the first strict
SDK verification failure remain under `outputs/lfd-cameras-v1/frozen-run/` in
the camera feature worktree. The corrected CPU-only export/reload each completed
in about 2 s under separate 45 s budgets; results remain in
`outputs/lfd-cameras-v1/export-fixed/verified/audit.json`, with a combined report
at `outputs/lfd-cameras-v1/report.json`. Native rendering held the shared lock.
The fixture outcome stays failure/no stocking attempted, training-eligible stays
false and task credit stays zero. Synthetic dropped-frame tests cover masks;
this native probe did not experience a camera dropout. No MP4 codec was tested.

## LeRobot adapter and local validation

The adapter targets the official
[LeRobot 0.7.0 dataset API](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/datasets/lerobot_dataset.py).
It calls `LeRobotDataset.create`, `add_frame`, `save_episode` and `finalize`,
writing locally with no Hub method. The official
[dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
explains why finalization is needed to close Parquet writers.

```bash
PYTHONPATH=src outputs/envs/lfd-export/bin/python \
  scripts/export_lfd.py lerobot outputs/my-fresh-lfd-fixture \
  --output outputs/my-fresh-local-lerobot --include-failed
PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  outputs/envs/lfd-export/bin/python scripts/export_lfd.py verify-lerobot \
  outputs/my-fresh-local-lerobot --output outputs/my-fresh-roundtrip-audit
```

Run this in the dedicated worker with admitted `lerobot[dataset]==0.7.0` and
`PYTHONPATH` pointing to this checkout's `src`. The adapter imports no simulator
when reading neutral episodes. It rejects other SDK versions until validated.
It exports nominal actions as `action`, applied commands as a diagnostic field,
and declared student sensors with availability/sample-time features. Failure,
terminal and training-eligibility fields stay explicit. A local sidecar retains
source contracts/provenance/outcomes and states that privileged data was not
exported. It never calls Hub upload.

LeRobot is **not installed in the simulator environment**. Official 0.7.0 needs
NumPy >=2, while this project and the rollout group require NumPy <2/1.26.4.
The older 0.4.4 package would instead downgrade several existing ML dependencies.
No package downgrade, override or unsupported no-dependency install was used.
The dedicated worker keeps its dependency lock in `tools/lfd_export/`; it uses
Python 3.12 and NumPy 2.2.6 without changing the simulator environment.

Create that worker using the committed lock:

```bash
UV_CACHE_DIR=outputs/cache/uv \
  UV_PROJECT_ENVIRONMENT="$PWD/outputs/envs/lfd-export" \
  uv sync --project tools/lfd_export --locked --python 3.12
```

The official SDK exported and reloaded the ten-transition native fixture locally.
The roundtrip verifies nominal/applied actions, numeric sensors, image dtype,
availability, failure outcome and training exclusion. The final frozen-source
export and reload each completed in 2.07 s with a 45 s per-process budget.
Copied sources, hashes, process records and the report remain in
`outputs/lfd-v1/lerobot-roundtrip-v3-run/`. The first rejected SDK
attempt and the initial precision diagnostic are retained alongside it. No video
was encoded or decoded: the raw camera feature is a masked, unavailable image,
and this does not validate camera collection or codec operation.

The later camera probe above separately validates nonzero raw image storage.
The SDK writes embedded PNG pixels losslessly; `verify-lerobot` checks every
stored uint8 pixel against the neutral episode. In LeRobot 0.7.0 the default
`get_raw_item()` image path returns float32 CHW RGB divided by 255 even when
the dataset constructor receives `return_uint8=True`. Verification checks that
explicit normalization exactly, separately from lossless stored pixels. The
older zero-only probe could not establish this image representation contract.

The SDK owns its relative `timestamp = frame_index/fps`. Native absolute clock
values use the separate float64 `diagnostics.observation_time_s` feature and
float64 sensor sample-time features. Parquet stores these doubles exactly, as
checked against the original neutral episode. The SDK's default HF-to-Torch
reader casts floating features to float32; verification checks that explicit
cast separately. Use the neutral episode or raw Parquet for precise timing;
do not reconstruct causal alignment from the cast reader clock. Native recorder
timestamp admission remains unchanged. Terminal next-observations remain in the
neutral episode's N+1 snapshots; the LeRobot rows contain N pre-action frames.

`home_robot_v3` revision `6c2ce9d841e7de36cd518d3a71f48960e97ad377` was inspected
read-only. Its MolmoBot exporter identifies itself as `emet-molmobot-jsonl`;
that is not a real LeRobot v3 writer. No upstream source was copied or modified.

## Next learning milestones

The foundation has 17 LfD tests covering causal clocks, action contracts,
privilege separation, interrupted outcomes, payload tampering and float32
overflow rejection. Five camera tests add synchronized views, dropped-frame
masks, configuration admission, physics-advancement rejection and exact image
storage/reader checks. All 22 LfD tests pass with native MABEL assets available.
Sensor convention tests remain separate. The official SDK roundtrip is retained
process evidence; it is not replaced by an adapter mock.

Collect actual successful pick/carry/insert/retreat demonstrations with raw
synchronized views, plus retained failures and recovery annotations. Freeze
controller and sensor contracts before training, split whole episodes by scene
seed/configuration and keep a fresh frozen evaluation suite separate. Begin with
supervised imitation of the declared action representation, then consider teacher
queries on student-induced states. That is a proposed teacher–student/DAgger
workflow, not an implemented student or an RL-training claim.
