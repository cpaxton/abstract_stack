# Native MABEL base repositioning

The new `NativeSwerve` interface demonstrated actuator-only lateral translation
in three isolated development episodes. This is a base-action result, with
**zero pick or complete-stocking credit**. It does not establish improved shelf
reach, a successful base-plus-insertion sequence, or a full ten-carton episode.

## Native interface and frames

The pinned full MABEL model is `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`.
Its three modules are `FL`, `FR`, and `B`; each exposes `Steer` and `Drive` channels.
Steering targets are radians for native position servos (`kp=350`, `kv=12`,
32 Nm effort bounds). Steering joints and targets are continuous; applying an
artificial +/-2pi clamp would break shortest-path control. Drive targets are
rad/s for native velocity servos (`kv=2`, +/-126 rad/s, +/-11 Nm).

The interface derives mount positions, wheel-axis orientation and collision-mesh
radius from the actual model. All three wheel radii are approximately
**0.03812355 m**. The native wheel frame introduces a quarter-turn offset:
positive drive rolls along `drive_axis_world × world_up`, rather than assuming
that a zero steering angle points along robot X. The observed zero-angle rolling
headings in the chassis frame are approximately pi, pi/2 and -pi/4.

`targets([vx_world, vy_world, yaw_rate], dt=...)` transforms world velocity into
the current chassis frame, includes module motion from yaw, selects continuous
steering within +/-pi/2 and reverses wheel drive when shorter. It holds drive at
zero while steering is unaligned, with existing drive commands ramped toward
zero. Once aligned it applies cosine scaling, explicit
steering/wheel acceleration limits from retained servo references and native
actuator bounds. It returns six
named targets; it never assigns base pose or velocity. Defaults limit translation
to 25 mm/s, steering reference speed to 1 rad/s and wheel acceleration to
0.8 rad/s². Diagnostics expose alignment gating, rate limiting and native target
clipping. Native masses, collision geometry, gains, masks and limits are unchanged.

## Library integration

```python
from abstract_stack.base_motion import (
    BaseContactGuard,
    BaseMotionConfig,
    NativeSwerve,
    screen_base_translation,
)

settings = BaseMotionConfig()
forecast = screen_base_translation(env, [0.0, 0.1], config=settings)
# Only an accepted forecast permits attempting the action.
steering = NativeSwerve(env.model, env.data, settings)
guard = BaseContactGuard(env, settings, max_product_displacement_m=0.002)

# In the existing per-physics-step callback, after the ordinary task observer:
guard.step(env)

# Feed targets through the existing MABEL tracker and native env.step path:
targets = steering.targets([0.0, 0.02, 0.0], dt=env.dt)
```

`BaseContactGuard.step(env)` does not reset or mutate episode observers, controls
or physical state. It monitors every robot link against non-robot fixtures and
products, sums all carton-pair contact normals before the 25 N guard, and records
all product displacement plus per-step tip/drop incidents. Ordinary wheel/floor
support contacts are expected and excluded from that fixture limit; other robot
floor contacts remain guarded. It also enforces 2 degree base tilt, the screened
yaw envelope and zero external-force channels. `report()` contains peak contacts,
native steer/drive efforts and product disturbance. `BaseMotionFailure` inherits
`InteractionFailure`: a measured force/tilt failure terminates the action; it does
not authorize another drive attempt. The optional displacement bound is relative
to guard construction, allowing a wrapper to keep its existing episode observer.

The scratch forecast translates the entire robot, not just its chassis. It
samples translation at 5 mm spacing, steering sweep fractions 0/0.5/1 and yaw
poses -0.03/0/+0.03 rad with 8 mm non-floor clearance. Floor clearance is checked
for non-wheel robot geometry. These are sampled kinematic checks, not continuous
collision or tire-dynamics guarantees. All live execution remains force guarded.

## Retained native development evidence

Every attempt has complete copied sources, model/source hashes, start/terminal
MuJoCo integration state, all actuator targets, per-physics-step pose/velocity and
contact traces, terminal verdict and a declared 300 s process budget under
`outputs/base-reposition-v1/`. Model/scene snapshots are explicit; no pose or
velocity assignment occurs in live execution. All external-force channels stayed
zero and interventions stayed empty.

| Attempt | Cartons prefilled / on table | Requested +Y | Measured +Y | Final XY error | Simulated duration |
| --- | --- | --- | --- | --- | --- |
| `positive-y-seed0-v1` | 9 / 1 | 100 mm | 98.090 mm | 1.910 mm | 9.64 s |
| `positive-y-seed1-v2` | 9 / 1 | 120 mm | 118.085 mm | 1.916 mm | 10.38 s |
| `full-table-seed0-v2` | 0 / 10 | 100 mm | 98.090 mm | 1.910 mm | 9.64 s |

Each met the predeclared 2 mm final XY tolerance, 0.01 rad yaw tolerance, stopped
speed below 3 mm/s and 0.3 s dwell followed by 0.5 s braked hold. Final X drift was
below 0.08 mm and yaw error below 0.00032 rad. Non-floor robot fixture/product and
carton-pair forces were zero. No product tip/drop incidents occurred; product
motion during the base action was below 1.3e-13 m. Wheel/floor physics remained
active throughout.

The two v2 attempts share identical full runtime sources, copied from the same
immutable snapshot. Peak v2 steer effort was below 6.35 Nm (native bound 32 Nm),
and drive effort below 1.36 Nm (native bound 11 Nm). The v1 attempt preceded guard
factoring; its successful trace remains preserved. Later diagnostics fields do
not change actuator behavior, but historical claims refer to recorded sources.
These three probes are not a broad success-rate campaign.

## Reproduction and remaining work

```bash
PYTHONPATH=src MABEL_ROOT=/path/to/open_mabel \
  python scripts/probe_base_motion.py --output outputs/new-base-probe \
  --seed 0 --dy 0.1 --initially-stocked 0
```

Use a fresh output directory. The current runner keeps its task observer through
settling and motion, records a separate base-action guard reference, and closes
the loop on measured base pose rather than wheel odometry. This matters because
the pinned full model uses motor-mesh wheel collisions; their physical rolling
and slip are not guaranteed by ideal wheel kinematics.

Unit checks cover shortest steering/reversal, nonzero world/chassis heading,
native control/rate bounds without state writes, future whole-chassis collision
rejection in scratch data, and guard rejection for native robot/carton force,
yaw-envelope violation and product tilt. Task integration must retain nominal
upper-body targets and existing observers, re-screen from the measured state,
and restrict fallback to failed planning; a force abort remains a failed episode.
