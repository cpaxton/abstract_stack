# MuJoCo teleoperation host adapter

Issue [#14](https://github.com/cpaxton/abstract_stack/issues/14), implementing the
[interfaces contract](README.md). This branch is based on the interfaces PR #26;
it does not depend on the tracking APK or the earlier teleop branch.

`abstract_stack.teleop.sim.MujocoTeleopAdapter` constructs and privately owns a
`ShelfStockingEnv`. All live operations, including rendering and close, must run
on the creating Python thread. Cross-thread calls fail before accessing MuJoCo.
Network/encoder workers receive detached snapshots and return commands through
their own future queues. This adapter contains no networking, retargeting,
engagement logic or hardware robot endpoint.

## Interface

```python
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.sim import MujocoTeleopAdapter

with MujocoTeleopAdapter(ShelfConfig(product_count=0), seed=0) as sim:
    state = sim.read_state()
    targets = state.applied_targets.copy()  # complete native actuator order
    targets[state.actuator_names.index("Neck Yaw")] += 0.03
    sim.apply_actuator_targets(targets)
    measured = sim.step()
    pair = sim.render_stereo_pair(width=640, height=480)
```

| Operation | Contract |
| --- | --- |
| `read_state()` | Detached qpos/qvel, actuator names, applied targets, measured actuator lengths/velocities, modeled actuator forces, world root pose and terminal flags. |
| `apply_actuator_targets(vector_or_complete_mapping)` | Validate all channels before writing `ctrl`; no physics step or pose assignment. Partial named updates fail. Return a detached copy of the applied vector. |
| `stationary_command()` | Measured actuator-length hold with reset-measured base steering references and zero wheel speed; validates native command ranges and can fail. |
| `step()` | One configured control interval using the existing environment's native stepping and task evaluation; returns measured state. |
| `reset(seed=...)` | Explicit episode reset, previous commands discarded; increments `reset_generation`, resets step index. |
| `render_camera(name, ...)` | Named-camera RGB8 copy, native-root-relative optical pose, FOV and pinhole intrinsics. |
| `render_stereo_pair(...)` | Native `zed_left` and `zed_right` from one simulation state, one monotonic capture time and one pair ID. |
| `describe()` | Scene config/seed, counts, timestep/frame skip, model/source XML hashes, MuJoCo version and effective base policy. |
| `close()` | Close renderer on its creating thread; repeated close is harmless. Other operations after close fail. |

`reset_generation` describes simulation resets; it is **not** the operator
`calibration_generation`. The later control owner must invalidate alignment on
reset and attach its own current calibration generation at the wire boundary.
Stereo pair IDs increase across resets for the adapter's lifetime. They are not
reused when a new generation begins.

## Native semantics and stationary base commands

The three drive channels always command zero rad/s. The three steering channels
hold their measured actuator lengths at episode reset. These six channels are
reserved; an attempted change is rejected atomically. The free base joint,
geometry, masses, inertias, collisions, equality constraints, gains and native
actuator ranges are preserved. Contacts can still move the robot.

Actuator position targets and `actuator_lengths` are not interchangeable with
arbitrary joint qpos entries. In particular, `Lift` is the native `Total Lift`
tendon length in metres, combining both slide stages. Other position commands
are radians; wheel commands are rad/s. State exposes actuator velocity separately
so wheel position is never compared with a velocity command. Raw qpos/qvel retain
the full native model layout, including free-joint quaternions and scene objects.
Root position/quaternion are world-frame; camera optical poses are relative to
`mabel_chassis`, with image right/up/-Z forward and wxyz rotation conventions.

Applying a validated command refreshes MuJoCo's derived state with `mj_forward`
without integrating qpos/qvel. Forces are MuJoCo modeled actuator forces, not
hardware measurements or contact-force verdicts. Physical tracking appears only
after stepping. Returned arrays can be modified by callers without altering live
physics; no public method returns live model/data handles.

`stationary_command()` is a low-level helper, not a proven pause/loaded-hold
policy. The pinned fixture has a reproducible counterexample: repeatedly applying
measured holds can yield a `Right Index Abd` position outside its native servo
command range. The helper raises without writing anything. It does not silently
clip the pose or enlarge limits. Selection of bounded pause references and
loaded-hold behavior belongs to #20.

## Stereo and ownership

Both eye renders execute synchronously on the simulation owner thread, with no
step or command change between them. Encoding can run elsewhere after receiving
the complete pair. This correctness-first implementation blocks the owner while
rendering; it does not establish the later video pipeline's rate/latency targets.
The rendering tests verify unchanged qpos/qvel/commands, native eye order, RGB8
dimensions, 63 mm baseline, metadata and detached image buffers. Unknown camera
names and nonpositive/noninteger dimensions fail before renderer allocation.

## Reproduce the baseline

With the lightweight environment and pinned model installed:

```sh
PYTHONPATH=src MUJOCO_GL=cgl uv run --no-sync python scripts/probe_quest_sim.py \
  --scene empty --seed 0 --seconds 2 --render \
  --output-dir outputs/quest-sim-baseline
PYTHONPATH=src MUJOCO_GL=cgl ./scripts/test.sh \
  tests/test_quest_sim_adapter.py tests/test_quest_teleop_contract.py \
  tests/test_shelf.py tests/test_mabel.py -q
```

On Linux use `MUJOCO_GL=egl`; headless baseline capture without `--render` does
not need a graphics context. Set `MABEL_ROOT` for an existing pinned checkout.
Artifacts include `baseline.json`, a native control-sample `trace.jsonl`, and
optional initial/final left/right PNGs. The report records source revision and
Python hashes, actual upstream checkout revision, composed/source model hashes,
scene/seed/configuration, effective command policy and simulation timing.

Peaks are measured at control observations, not every physics substep. Tracking
errors are reported per actuator with metres/radians/rad/s kept separate. Wheel
commands of zero do not establish zero base displacement. The empty preset still
contains the shelf/table and native contact geometry; it is not a certified
unloaded fixture. No manipulation, loaded hold or stationary-body acceptance is
inferred from the baseline.

## Clean-source measurement

November 17, 2026, implementation commit
`8a2d4ab4914fe785cfed97107f8dd161ed0f440f`, `dirty_sources:false`, MuJoCo 3.17.0.
Actual upstream checkout: `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`.
Scene `empty`, seed 0, 0.002-second physics timestep, 10 substeps per control tick,
67 position values / 66 velocities / 59 actuators. Native reset targets were held
without compensation, with no geometry/contact/constraint modifications.

| Measurement | Observed |
| --- | --- |
| Simulated duration / control samples | 2.00 seconds / 101 including reset |
| Maximum absolute wheel-speed command | 0 rad/s |
| Peak free-base displacement at control samples | 0.013978 m |
| Peak free-base rotation at control samples | 0.058564 rad |
| Peak measured lift tendon error | 0.003742 m |
| Stereo pair capture times | 0.00 and 2.00 simulation seconds |
| Native stereo baseline | 0.063 m at both captures |

Full measurements and native trace remain under
`outputs/quest-sim-clean-8a2d4ab/`; both initial/final eye previews were generated.
The native head-camera preview was inspected. No acceptance thresholds were
introduced or tuned. The observed displacement is retained as a limitation of
the raw stationary-command policy.

Validation: 38 targeted adapter/contract/simulator tests passed, with one optional
Emet integration test skipped. Tests include real stereo rendering, native model
preservation, deterministic seeded resets, command atomicity, detached snapshots,
thread ownership and the measured-hold command-range counterexample. Repository
hooks and wheel/source-distribution builds passed.
