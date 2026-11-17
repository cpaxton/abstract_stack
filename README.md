# Abstract Stack

Bimanual, contact-rich shelf stocking in simulation. The goal is to stock an
entire shelf in one continuous episode, including recovery when placements go
wrong.

**Status:** initial MuJoCo shelf environment with the full upstream MABEL model,
seeded resets, contact-based stocking checks, a native viewer, and an Emet ZMQ
adapter. A world-state solver completes initial seeded carton scenes and records
dual-view MP4s. VR control, demonstration replay, generalized recovery and mjlab
training remain future work.

## Planning and gripper design

- [Technical feasibility plan](docs/technical_feasibility.md): proposed experiments
  and acceptance gates for dense stocking, recovery, teleoperation, human pace,
  realistic sensing, learning and bimanual mobile execution.
- [Proof-of-concept critique](docs/poc_critique.md): what current results establish
  and the technical gaps still to close.
- [Gripper design tracker](docs/gripper_design_tracker.md): GET, Dream Gripper,
  build assets, sensing/actuation choices and matched UMI/tooling experiments.
- [Experiment index](docs/experiments.md): supported workflows and measured results.

Track candidate grippers and their next experiments in the design tracker;
acceptance targets in the feasibility plan are proposals, not achieved results.

## Current manipulation results

The optional locked `rollout` group adds MuJoCo Warp to the normal environment
(`uv sync --locked --group rollout --inexact`). Initial
[CPU/GPU rollout measurements](docs/reports/rollout_backend_development.md) favor
CPU batching for 32 short MABEL forecasts; this is separate from task success.
A [stationary whole-body prototype](docs/whole_body_control.md) coordinates both
arms and their shared lift, with native unloaded tracking evidence. Mobile and
contact integration remain open.

The [demonstration pipeline](docs/learning_from_demonstration.md) separates
observable student inputs, privileged teacher context, nominal actions and
applied controls. A native hold fixture passed local LeRobot export/reload;
stocking/VR recording and student training remain open. The
[simulated sensors](docs/sim_sensors.md) provide wrist F/T, contact-only wrenches
and coarse tactile force proxies. A
[compliant replanning continuation](docs/reports/compliant_lookahead_development.md)
advanced a stalled carton to backstop contact; release and full stocking remain
unverified.

The [first physical stocking script](docs/scripted_interactions.md) attempts
box pickup and push-to-contact with per-step force traces. Development trials
have completed two-carton shelves with verified pickup, lift, back contact,
release and final stability. These are development cases, not a held-out success
rate. The prefilter tries bounded
orientation alternatives, checks sampled joint paths, and records predicted
position/orientation errors separately from physical execution error.

The [compliance follow-up](docs/reports/compliance_followup.md) completes
**10/10 scenes in each of four two-carton suites**: all 80 cartons pass pick,
placement, positioning and clean-retreat gates. Thumb preshaping clears an
approach obstruction; lower grip/push targets improve insertion. Task criteria
and the 25 N force limit remain fixed. These small stationary, left-arm pilots
do not establish crowded-shelf, bimanual or jam-recovery performance.
The [thumb-preshape report](docs/reports/thumb_preshape_followup.md) retains the
intermediate controller's measurements and rejected development probes.
The earlier [contact-control report](docs/reports/contact_control_followup.md)
retains the 8/10 and 7/10 results, rejected experiments and unknown native crash.
The [historical ten-scene report](docs/reports/stocking_ten_scenes.md) breaks down
pick, placement, final positioning and retreat without knocking objects over.
Run `./scripts/benchmark_stocking.sh` for the baseline, bounded compliant-control
hill-climb on separate development seeds, paired validation and report generation.
Task acceptance and force abort limits stay fixed during the search.

The [Quest Teleop v1 interfaces](docs/quest_teleop/README.md) define the new
simulation-only Quest 3 work: stationary base, upper-body tracking and a stereo
head-camera window. This is an interface contract; live VR control is not yet
implemented.

[Web teleoperation](docs/web_teleop/README.md) drives the simulation from a
browser, phone or tablet: an interpolated 3D twin, draggable palm targets with
IK, stereo head-camera video, keyboard and touch controls, and the same
engage/pause safety model. Run
`abstract-stack teleop` after building `web/`.

See [Shelf scene setup](docs/scene_setup.md) for dimensions, editable scene
configurations, MABEL import, scene exports, and validation. Upstream concerns
are tracked locally in [home_robot_issues.md](home_robot_issues.md).

For a furnished kitchen/living area, bedroom and bathroom with MABEL and the
stocking objects, use the [small house scene](docs/house_scene.md):

```bash
abstract-stack build-scene --scene house --fetch-mabel --render --output-dir outputs/house
abstract-stack sim --scene house
```

Development rules and hook/CI setup are in [CONTRIBUTING.md](CONTRIBUTING.md)
and [AGENTS.md](AGENTS.md).
The [workflow and experiment index](docs/experiments.md) links the supported
controller and historical studies. The [dense-row prototype](docs/dense_stocking.md)
targets 10 mm or 5 mm total unused shelf width: seed-0 trials complete one or two
insertions beside prefilled cartons, while full stocking and fresh-scene robustness
remain open. The [cardboard investigation](docs/cardboard_simulation.md) documents
separate material fixtures and engine limitations.

The [research draft and Agentic GenTAMP review](paper/README.md) describe the
task, controller interface and planned experiments. Build with `./paper/build.sh`.
For the current physical solver attempt, run:

```bash
./scripts/solve_shelf.sh --render --output-dir outputs/adaptive-shelf
```

For [seeded random scenes and synchronized third-/first-person MP4s](docs/stocking_solver.md):

```bash
./scripts/solve_shelf.sh --random-scene --objects 2 --seed 0 --video \
  --output-dir outputs/stocking-seed1
```

The [MABEL tracking controller](docs/mabel_controller.md) supplies bounded
bimanual joint/lift references with robot-gravity feedforward and measured wrist
damping. The solver tries alternative object/grasp candidates and verifies
physical stages. Bimanual task coordination, broader generalization and MABEL RL
control remain unverified.

## Initial scope

- Start in pure MuJoCo simulation with rigid cartons and bottles.
- Stock multiple products without resetting the scene between placements.
- Handle progressively tighter gaps, slide products into place, and rearrange
  earlier placements when needed.
- Recover from jams, tipped objects, and failed grasps while continuing the task.
- Record demonstrations and evaluate complete-shelf performance.

MABEL is the first platform to evaluate: validate its robot model,
omnidirectional base, bimanual control, VR teleoperation, and mjlab integration
before committing to it. RB-Y1 is the fallback. Begin with the existing hands;
two versus three fingers remains an open decision informed by task performance
and ease of control.

## Responsibilities

| Owner | Initial work |
| --- | --- |
| Anuj | VR app and MuJoCo connection; demonstration recording and replay; comfortable teleoperation. |
| Chris | Shelf environment, task and evaluation design, data generation, and initial autonomy. |
| Together | Robot/control interfaces, platform validation, and hand selection. |

Teleoperation quality is a core deliverable: responsive tracking,
clutching/recentering, useful camera views, intuitive grasp control, and quick
retries. Define the robot/control interfaces together so teleoperation, replay,
and autonomy can drive the same environment.

## Setup

Use Python 3.10–3.12 (the installer defaults to 3.11), Git, and
[uv](https://docs.astral.sh/uv/). From this checkout:

On macOS, use the [native Mac simulator setup](docs/macos.md) to avoid Emet's
Linux GPU dependencies and select the Mac renderer/viewer:

```bash
./scripts/install-macos.sh
./scripts/run-macos.sh sim
```

For the full Emet stack on a compatible machine:

```bash
./install.sh
source .venv/bin/activate
```

The installer synchronizes the project and development tools into `.venv`, then
checks package imports. It can be run from any working directory and forwards
arguments to `uv sync`:

```bash
PYTHON_VERSION=3.12 ./install.sh
./install.sh --no-dev
```

The upstream dependency is
[home_robot_v3](https://github.com/cpaxton/home_robot_v3), whose Python
distribution and import name are `emet`. We install its `sim` extra directly
from the Git repository. This currently follows upstream `main`; a generated
`uv.lock` records the resolved revision and should be committed once the initial
environment has been validated. Later installs can use `./install.sh --locked`.

Emet includes a substantial robotics and learning dependency stack. Native build
dependencies and graphics setup may be needed; follow its
[installation documentation](https://github.com/cpaxton/home_robot_v3#installation-astral-uv)
for your machine. This installer installs Python packages only. Emet's optional
third-party simulators/assets are separate. MABEL's model is fetched using the
command below; its controller, VR app, and mjlab stack are not installed.

The installer sets `GIT_LFS_SKIP_SMUDGE=1` because upstream documentation media
includes unavailable LFS objects. These images are not needed by the simulator.
Use the installer again after an LFS checkout failure. For direct `uv` commands
that need to fetch Emet, set the variable explicitly:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Run the shelf environment

After installation:

```bash
# Fetch only the model/mesh directories, with upstream license files.
abstract-stack fetch-mabel

# Full MABEL, three shelf levels, supply table, three cartons and three bottles.
abstract-stack sim

# Build an inspectable scene with robot assets fetched if necessary.
abstract-stack build-scene --scene stocking --fetch-mabel --render \
  --output-dir outputs/stocking

# Empty geometry or a shelf with existing products and tighter gaps.
abstract-stack sim --scene empty
abstract-stack sim --scene tight
```

The assets come from
[robotmabel/open_mabel](https://github.com/robotmabel/open_mabel) at revision
`d0181f04025c06e1790b7ab41fe44d602cc2ee5b`. They are downloaded into
`third_party/open_mabel` in a source checkout and excluded from Git. Existing
checkouts are never overwritten. Set `MABEL_ROOT=/path/to/open_mabel` or use
`--mabel-root /path/to/open_mabel` with simulation commands to use your own
checkout. Upstream models and meshes retain their
[upstream licenses](https://github.com/robotmabel/open_mabel/blob/main/THIRD_PARTY_NOTICES.md).

The viewer starts on the overview camera. Use the native MuJoCo camera and
actuator controls to inspect the scene and move joints; press **R** to start a
fresh episode. MABEL faces world -X and the shelf opens toward +X. Three fixed
task cameras (`overview`, `shelf_front`, `supply_overhead`) complement the nine
upstream robot cameras.

```bash
# Physics-only smoke test: no display or renderer needed.
abstract-stack sim --headless --steps 500 --json outputs/smoke.json

# Offscreen rendering requires EGL or another supported MuJoCo GL backend.
abstract-stack sim --headless --steps 100 --screenshot outputs/shelf.png

# Inspect all 59 named actuator channels and their limits.
abstract-stack inspect
abstract-stack sim --actuator 'Left Arm 1=-0.1'

# Export MJCF for MuJoCo or upstream MABEL scene tools.
abstract-stack export --output outputs/shelf.xml

# Test the shelf/products without downloading MABEL.
abstract-stack sim --robot none --headless --steps 100
```

The default physics timestep is 2 ms with 10 physics steps per control tick
(50 Hz). Position targets persist until changed; wheel drive commands are
angular velocities in rad/s, steering/arm/hand commands are radians, and `Lift`
is total extension in metres across its two stages. Reset holds the assembled
lift at 0.3175 m. The robot base remains dynamic. LiDAR/sensor evaluation is off
by default for CPU speed; pass `--enable-sensors` to enable it.

### Emet connection

The adapter extends Emet's robot and stationary-control interfaces locally and
reuses its ZMQ simulation server. It handles MABEL's tendon-driven lift and
velocity wheel actuators explicitly:

```bash
abstract-stack serve --headless
# State/action transport without camera rendering:
abstract-stack serve --headless --no-cameras --steps 1000
```

The server binds localhost ports 4401–4404; `--port-offset 100` selects
4501–4504. State messages include a `shelf_stocking` dictionary with progress
and episode metrics. State publication defaults to 50 Hz and camera/servo
publication to 15 Hz; Emet's `EMET_ZMQ_STATE_HZ`, `EMET_ZMQ_FULL_HZ`, and
`EMET_ZMQ_SERVO_HZ` variables override these rates. `--steps` counts physics steps in `serve` and control
ticks in `sim`. The generic Emet navigation path drives the base pose/velocity;
it is not MABEL's physical swerve controller. The local environment's named
wheel channels are available for physical drive experiments. Use explicit hand
joint targets through this generic adapter; the local stocking solver supplies
its own grasp feedback. VR retargeting remains future work.
Emet simulation helpers that set poses or attach/detach objects are recorded as
interventions so assisted runs cannot count as unassisted success.

In a Python client process, register the adapter before requesting an Emet
backend:

```python
from abstract_stack.robots.mabel import register_mabel
from emet.robots import get_robot_backend

register_mabel()
client = get_robot_backend("mabel").create_client("127.0.0.1")
```

### Environment API and current success criterion

```python
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig

env = ShelfStockingEnv(ShelfConfig(time_limit=120.0), seed=7)
try:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step({"Left Arm 1": -0.1})
finally:
    env.close()
```

`step` accepts a partial mapping of actuator names to absolute targets, a full
actuator vector, or `None` to hold targets. It returns the usual five values but
is not a Gymnasium `Env` subclass. Observations contain robot/product state,
target positions, and stocking progress. RGB is available through `env.render()`.
Invalid, non-finite, or out-of-range commands are rejected before changing
controls.

Each product has a marked target on an assigned shelf level. A placement counts
only after 0.5 continuous simulated seconds with shelf support contact, an
upright orientation within 15 degrees, low linear/angular speed, the complete
bounding box inside the shelf, and its centre within 5.5 cm of the target in
each horizontal axis. Dislodging an item removes its progress. Reward is the
change in the number of stocked products. All six stocked ends the episode;
120 simulated seconds without success truncates it. `ShelfConfig` exposes
geometry and timing settings for experiments.

Summaries report stocked products, tipped/dropped products, simulation/wall time,
and logged interventions (`env.record_intervention(reason)`). These are initial
task checks; recovery-event detection and the broader benchmark below remain to
be implemented. There is no autonomous stocking policy yet.

Development checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests run contact/stability, timeout, reset and action validation checks. Tests
marked `mabel` use the actual downloaded model and skip if it is absent. A static
placement fixture tests evaluation; it does not demonstrate robot grasping or
autonomous task completion.

## First milestones

1. **Validate the platform:** load MABEL in MuJoCo, exercise both arms, hands,
   and base, and establish the control interface. Check the VR and mjlab paths;
   evaluate RB-Y1 if MABEL cannot support the initial workflow.
2. **Make teleoperation usable:** tune tracking, clutching, cameras, grasp
   control, and retries. Record and replay a multi-object sequence with
   timestamps, robot state, actions, and environment state.
3. **Build the shelf task:** define shelf geometry, product sets, spawn
   distributions, target placements, collision/contact settings, success
   tolerances, and time limits. Add tight gaps and controlled failure cases.
4. **Collect data and add autonomy:** collect continuous stocking demonstrations,
   establish an initial policy, and include rearrangement and recovery behavior.
5. **Evaluate whole episodes:** compare teleoperation and autonomy on fixed
   evaluation seeds, product sets, and packing densities. Report failures as well
   as successful runs.

## Evaluation design

An episode starts with the configured shelf and product set and ends when all
products are successfully stocked, a time limit is reached, or the run is
aborted. Preserve placements and contact history between objects. Log resets,
human assistance, and retries explicitly.

| Metric | Proposed definition |
| --- | --- |
| Complete-shelf success | Fraction of all evaluation episodes with every required product stably placed within the task's placement tolerances. Report unassisted success separately from assisted completion. |
| Completion time | Wall-clock task duration; report successful-episode times alongside timeouts and aborted runs. |
| Interventions | Count and duration of human assistance or resets, including the reason for each intervention. |
| Recovery | Success rate and elapsed time for detected or injected failure events, broken down by jam, tipping, failed grasp, and other causes. |

Before collecting benchmark results, fix placement/stability tolerances, event
definitions, time limits, and train/evaluation splits. Record scene seeds,
product geometry and poses, robot/hand configuration, control frequency,
simulator settings, policy version, and dependency revisions so runs can be
reproduced. Recovery should restore progress toward completing the shelf within
the same episode.
