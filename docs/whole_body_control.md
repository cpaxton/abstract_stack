# Stationary whole-body control foundation

`StationaryWholeBodyController` is an opt-in reference coordinator for the two
MABEL palms, two wrists and shared lift. It is not wired into the shelf-stocking
controller. Mobile-base coordination, grasp/contact transitions and loaded
whole-body tracking remain open work.

## Implemented contract

Both arms enter one bounded 17-dimensional least-squares solve: seven arm joints
and one wrist per side, plus one common lift coordinate. The two palm Jacobians
share that same lift column. When a palm goal is omitted, its previous nominal
world pose becomes a hold objective; the other arm cannot simply consume shared
lift motion without accounting for the held palm.

The native lift actuator drives a 1:1 tendon spanning two sliding joints. Model
admission verifies this transmission. Scratch forecasts retain the measured
stage imbalance and distribute a proposed change in total extension equally.
This is a kinematic forecast, not independent control of the stages. Their
individual native bounds also restrict the common coordinate.

The coordinator reuses `MabelController` for gravity feedforward, optional named
load estimates, wrist damping and bounded native position-servo offsets. Nominal
and compensated applied targets stay separate. By default, nominal planning
reserves the configured compensation offsets inside native ranges. A current
nominal outside that reserved interval must be staged before creating this
coordinator; initialization cannot teleport it into the new interval.

Every prediction:

- Builds a local Cartesian step with explicit linear/angular speed bounds and
  a single joint/lift rate box, then checks the final nominal FK step bounds.
- Includes all native position actuators below both palms in its finger
  posture, including thumb DIP, PIP, MCP and abduction references.
- Checks five coupled arm/finger interpolation samples for nominal targets and
  five for the final compensated applied targets. The original 2 mm collision
  threshold and 1 mm existing-contact allowance are retained; wheel/floor contact
  is excluded as in the existing IK. No product/hand contact whitelist is added.
- Returns position residuals in metres, orientation residuals in degrees,
  solver diagnostics, active limits, reference bounds and tracking-offset limits.

Collision/path checks are sampled kinematics, not continuous collision or
dynamics certificates. A kinematic goal-within-tolerance flag does not grant
grasp, measured tracking, release or stocking credit.

`predict_step` changes scratch data only. `accept` requires the latest prediction
and the exact unchanged integration state, rejects failed admission even if a
reporting dictionary is edited, then returns a native control vector. It does
not step MuJoCo. The caller owns actuator application, monitoring at every
physics step and measured acceptance. Keep the model immutable for the
coordinator's lifetime.

The native base must already have zero drive commands, measured linear speed
below 2 mm/s, angular speed below 0.01 rad/s and tilt below 2 degrees. Steering
targets pass through unchanged. The existing `NativeSwerve` controller and
screened empty-base action remain the base interface; no base pose/velocity
write or loaded-base action is introduced here.

## Minimal free-space use

```python
from abstract_stack.base_motion import BaseContactGuard
from abstract_stack.whole_body_control import PalmGoal, StationaryWholeBodyController

# Start from a settled native MABEL scene with clear unloaded arms.
controller = StationaryWholeBodyController(env.model, env.data)
guard = BaseContactGuard(env)
poses = controller.nominal_poses()
goals = {
    side: PalmGoal(position + [0, 0, 0.003], rotation)
    for side, (position, rotation) in poses.items()
}
step = controller.predict_step(goals, dt=env.dt)
if not step.report["accepted"]:
    raise RuntimeError(step.report)
env.step(controller.accept(step), on_physics_step=guard.step)
```

This demonstrates one bounded native command. It is not a grasp script. A shelf
integration still needs task-specific support/slip/contact guards, explicit
contact permissions, endpoint/path admission over longer actions and fresh
measured tracking evidence under loads.

## Reuse and validation

Reviewed `home_robot_v3` revision
`6c2ce9d841e7de36cd518d3a71f48960e97ad377`, particularly its stationary MuJoCo
control adapter and Pinocchio IK solver. Those provide useful interface context,
but no upstream code was copied. The adapter's generic wheel/tendon assumptions
and the URDF/Pinocchio solver do not supply this native MABEL bimanual/common-lift
mapping. The implementation uses the local NumPy box solver and native MuJoCo
Jacobians, adding no dependencies and changing no upstream checkout.

The final focused snapshot passed **6 tests in 5.26 s**, including native
bimanual tracking, an omitted-palm hold objective, stale/one-use acceptance,
compensated-geometry rejection and measured base-motion rejection. Its frozen
package, tests, hashes, process result and log are in
`outputs/whole-body-control-v1/final-focused/`.

A separate frozen final-source native probe ran 900 physics steps at 2 ms after
settling, with an unloaded hold, both palms moving 3 mm upward, then only the left
palm moving another 2 mm. It recorded zero external forces, native warnings or
robot/fixture contact force. Sampled model arrays were unchanged: actuator
gains/biases/limits, body masses/inertias, collision masks and equality parameters.
MABEL's one native equality was preserved. Final palm
position errors after the two-arm phase were about **0.20 mm left / 0.37 mm
right**; after the left-only phase they were **0.29 mm / 0.31 mm**. This tiny
free-space movement does not establish larger-workspace, contact or payload
performance. Controls, integration states, the exact source snapshot, dependency
and scene provenance and traces are retained in
`outputs/whole-body-control-v1/native-tracking-final/`.

One earlier test process segfaulted in environment construction before the
whole-body controller was created. Its unknown result/source and Python stack
excerpt remain in
`outputs/whole-body-control-v1/focused-native-crash-after-base-admission/`.
The subsequent focused pass and tracking probe do not replace that failure or
establish a fix for the open native process-stability issue. No whole-shelf
success or learned controller is claimed.

Next work is a guarded mobile-base admission contract, joint-headroom objectives,
contact-mode-specific force/torque feedback and native loaded-physics evaluation
before considering stocking integration.

## Guarded task adapter and corrective deltas

`WholeBodyTaskAdapter` exposes the stationary coordinator to a phase executor.
Build it with the current environment and a task guard that runs at every native
physics step. Pass an uncorrected absolute world goal for each active palm.
A teacher, policy, or teleoperator can produce a `PalmDelta` directly. Optional
`WristCompliance` adds a proportional offset from measured reaction-wrench
error along selected world axes. The two offsets are summed and capped together
before the shared-lift solver runs. Reuse the original uncorrected goal on each
call so the offsets do not accumulate. The controller still checks native rate
limits, coupled finger geometry and final compensated targets.

```python
from abstract_stack.whole_body_task import PalmDelta, WholeBodyTaskAdapter, WristCompliance

# phase_guard checks task-specific contacts and all products each physics step.
adapter = WholeBodyTaskAdapter(env, guard=phase_guard)
result = adapter.advance(
    {"left": left_world_goal},
    deltas={"left": PalmDelta(translation_m=[0, 0, 0.001])},
    compliance={
        "left": WristCompliance(
            translation_m_per_N=[0.0003, 0, 0],
            rotation_rad_per_Nm=[0, 0, 0],
        )
    },
    finger_targets={"Left Index MCP": 0.2},
)
```

Translation uses metres. Rotation vectors use radians around world axes and
compose on the left of the uncorrected palm orientation. The sensor input is
the net contact-only wrench ON the distal robot subtree at the wrist anchor,
in world axes with force before torque. Gains are metres per newton and radians
per newton-metre. Positive reaction error moves the attractor along the
reaction, reducing penetration along a constrained axis. Desired wrench is
zero by default. This is a bounded proportional offset, without integration
or a direct torque command.

`advanced=False` means the scratch preview failed and no physics step ran.
A successful interval reports measured tracking separately from the preview.
Guard failures propagate immediately and may leave the interval partially
executed. The phase executor must use its own measured support, grasp,
contact, and retreat gates. Recreate the adapter at an environment reset
or after any other owner has run the actuator interface. Stage the base through
the existing navigation/settling path before construction.

This API supports free-space repositioning and corrections that pass the
coordinator's existing collision admission. Dense insertion and loaded
manipulation retain their existing task-specific path and contact guards.
A guard can detect failures, while the adapter's collision checks still
reject newly penetrating intentional contact.
