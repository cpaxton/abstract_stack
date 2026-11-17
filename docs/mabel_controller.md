# Reusable MABEL tracking controller

`abstract_stack.mabel_controller.MabelController` accepts named **nominal**
references for both seven-joint arms, both wrists, and the shared lift. It
implements robot-gravity feedforward and measured wrist velocity feedback
through the canonical position servos. It changes no model masses, gains,
transmissions, collision masks or native limits and applies no external forces.
The same tracker now executes the stocking script's joint/IK references.

```python
from abstract_stack.mabel_controller import MabelController

tracker = MabelController(env.model, env.data)
for _ in range(100):
    action = tracker.targets({"Left Arm 1": 0.08, "Right Arm 1": -0.08}, dt=env.dt)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(tracker.last_report)
```

Angles and angular rates use radians and radians/s. Lift references use total
tendon length in metres and a default rate limit of 0.08 m/s. Arm/wrist nominal
and applied reference changes are limited to 1 rad/s. Partial dictionaries hold
omitted nominal dimensions; they do not add feedforward repeatedly to `ctrl`.
The returned full actuator vector contains **applied servo targets**, which are
different from nominal references. Do not feed that vector back as a new nominal
goal. Fingers and wheels pass through native controls and require their own
executors/limits. This tracker is a simulator component, not a hardware API.

## Compensation and feedback

Scratch MuJoCo data at the measured configuration and zero velocity supplies
robot gravity. A least-squares transmission solve produces forces only for
driven arm/wrist/lift coordinates. The common lift tendon drives two stages:
its fitted force supports the load once, rather than summing both generalized
gravity loads. The tracker converts force to a native servo reference offset by
dividing by the position gain. Native force bounds, where defined, are retained.

The native wrists have `kp=1`, `kv=0`; the wrapper adds configurable measured
velocity feedback (default 0.2 N m s/rad) through bounded references. Native arm
and lift position/velocity feedback remains active. Offsets are capped at 0.1 rad
for joints and 0.01 m for the lift, then clipped to joint/control bounds and rate
limited. Reports expose force, offset, range and rate limiting. No integrator
learns an opposing contact force as a persistent bias.

Only robot gravity is modeled. Unknown payloads, contact loads, changing grasp
geometry and dynamic coupling can leave residual error. A controller must never
fight a jam merely to reach a reference. The task executor retains physics-step
force and tilt aborts, validates actual tracking/grasp/lift/release, and must add
whole-body support and payload checks before claiming robust contact execution.
Cartesian error correction and bimanual motion planning are separate work.

## What is verified

`tests/test_mabel_controller.py` runs the real dynamic robot with fixtures placed
explicitly away from both hands to isolate unloaded tracking. It checks joint
reference changes on both arms and the lift, compensation/rate limits, state
immutability during prediction, unchanged gains, and zero external forces.
It requires less than 0.2 mm lift error and 0.006 rad arm/wrist error in that
fixture. Those are regression tolerances, not a workspace-wide guarantee.

The stocking bench has a separate initial right-thumb/shelf contact. Its right
wrist offset persists under gravity compensation because contact contributes
load. Do not report it as an unloaded servo calibration result or remove the
contact to claim successful manipulation. In one left-hand bench hold probe,
gravity feedforward reduced hand error from approximately 13 mm to 0.13 mm;
this is a single-pose diagnostic, not task success.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_mabel_controller.py
./scripts/solve_shelf.sh --render --output-dir outputs/adaptive-shelf
# Disable gravity feedforward as an ablation; wrist damping remains enabled:
./scripts/solve_shelf.sh --no-gravity-compensation --output-dir outputs/no-gravity
# Preserve the original object order/centre grasp as a separate ablation:
./scripts/solve_shelf.sh --fixed-grasps --output-dir outputs/fixed-grasps
```

The solver screens object/height/orientation candidates, currently preferring
objects nearer the chassis. It attempts retract, raise, pregrasp, approach,
contact-based closure, lift, transfer, supported push and release. It stops closure early
when opposing hand contacts reach a 6 N balanced squeeze estimate; the 25 N
physics-step abort remains active. Force-target closure is a heuristic and must
still pass measured lift/retention checks. Failed stages exit 2 and keep evidence.
Carry predictions check a rigidly transported product at sampled configurations,
while physical grasp retention is monitored during execution. Placement includes
a bounded seating correction from measured product/board clearance when needed.
The insertion retains the grasp, regulates balanced squeeze separately from
net push force, and corrects product orientation/height from world state. It
holds at upright back contact before a checked opening/withdrawal. These are heuristics,
not a validated contact controller for every product and friction condition.
During carry, bounded closure feedback (0.04 fraction/N/s, capped at 0.25
fraction/s) maintains the 6 N target; the gain and rate are recorded in the
script configuration. Placement and back-contact targets use measured rotated
product extents. The IK lift interval now comes from native joint/tendon bounds,
instead of the old narrower hand-written range; collisions and measured physical
guards continue to decide whether a pose/path is acceptable.
Two-carton development scenes now complete all physical gates. A native-physics
regression covers one such scene, but a waypoint sequence does not establish
success on a new scene or a general success-rate claim.

## RL coordination and admission contract

The inspected `home_robot_v4` manipulation profiles support Mars and Stretch,
not MABEL. Thread messaging was unavailable during this change, so no request
has been delivered to that agent and no MABEL RL integration is claimed.
The standalone handoff below contains only upstream robot/controller concerns.

For a MABEL profile, agree on:

- The pinned model revision, gains/transmissions, collision geometry and control
  settings. Fingerprint all of them; reject bundles with incompatible layouts.
- Named left/right arm and wrist coordinates and one shared total lift reference.
  Do not duplicate lift actions or reinterpret another robot's arm/gripper layout.
- Existing multi-joint hands, their aperture/closure mapping, opposing contact
  and load-retention verification. MABEL does not fit a single-gripper-joint
  assumption without an explicit adapter.
- Actor observations versus privileged training/evaluation diagnostics; reference
  frames, normalizers, action semantics, update period and history cadence.
  This simulator runs control at 50 Hz; a 20 Hz policy needs an explicit adapter.
- Common conventional and learned-controller tracking/contact evaluation,
  physical base support, force/impulse and tilt aborts, held-out seeds, and
  unknown/unsupported outcomes when admission checks are not met.

No MABEL weights are trained, downloaded, or promoted by this patch. A future
adapter should keep the deterministic tracker as a baseline and retain the same
physical acceptance gates for a learned residual controller.
