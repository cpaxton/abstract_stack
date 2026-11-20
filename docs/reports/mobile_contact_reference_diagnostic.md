# Why the first mobile insertion lost contact

Recorded episode `outputs/contact-rich-replanning-v1/mobile-table-ten-v1/episode`
failed with `push: lost shelf support or hand contact`. The carton retained shelf
support; hand contact was lost. Its earlier native base reposition succeeded.
This investigation preserves that failure and assigns no new task credit.

The independent diagnostic replayed all recorded native actuator commands from
the saved initial integration state. It reproduced the terminal integration
state **bit for bit**, and reproduced the final nominal arm/lift command exactly.
Its source comes from that episode's immutable package. New diagnostic script
and bounded-step source copies are retained separately under
`mobile1-contact-diagnostic-v1/diagnostic_sources`.

## The dominant error was in the sent nominal reference

At failure, the requested tool height was 0.828317 m. FK of the actual **sent
nominal joint/lift reference** was 0.807874 m: **20.443 mm too low**. The measured
reference point was 2.378 mm above that nominal FK. The last IK solution itself
had only **0.379 mm** endpoint position error. Consequently, neither endpoint IK
residual nor measured servo sag alone explains the failure.

The online IK request required much more joint motion than one 20 ms step. The
controller independently clipped each joint and lift increment afterward. This
changed the coupled Cartesian path. On the first three force-control ticks:

| Tick time | IK endpoint miss | Sent nominal FK miss | Nominal Z error | New bounded-step miss from the same prior reference |
| --- | --- | --- | --- | --- |
| 78.40 s | 1.736 mm | about 7.7 mm | +4.11 mm | about 0.28 mm |
| 78.42 s | 1.592 mm | about 15.3 mm | +8.43 mm | about 0.27 mm |
| 78.44 s | 1.447 mm | about 22.2 mm | +12.25 mm | about 0.28 mm |

The original requested advances were only about 0.8 mm per tick. The sent FK
first overshot forward and upward, contacting the carton, then swung backward
and downward. All recorded hand contacts were on `left_middle_ip`. The episode
had no wrist-posture replans. Existing `PoseIK.predict` returns the first feasible
candidate, so this result does not establish a global best-residual seed switch.
It does establish an incompatible instantaneous IK motion followed by independent
rate clipping. Joint velocities were still substantial at failure: Arm 4 about
+0.758 rad/s, wrist -0.616 rad/s, total lift +0.0795 m/s.

Finger joint errors were small at the terminal fixture: MCP about -0.003 rad and
PIP +0.008 rad. Recalibrating the fingertip offset from the terminal pose shifts
its local X coordinate by about 2.1 mm because the abduction posture changed.
The exact decomposition above uses the **original saved tool offset**, not a
newly calibrated reference. It therefore matches the controller's own trace.

## Retained Cartesian paths also hid an initial joint gap

The original approach's first retained waypoint differed from the measured start
by -0.664 rad at the wrist, +0.396 rad at Arm 2 and +0.401 rad at Arm 6. Its
minimum rate-limited initial transition was **1.245 s**, omitted by a duration
calculated only from differences between retained waypoints. Solving the
Cartesian sample at t=0 with a new posture constraint does not guarantee that
its joint vector equals the measured starting vector.

Strict 1 mm terminal-state searches found accepted endpoints and sampled paths
for tilt/wrist pairs `(20°,0.4)`, `(30°,0.2)`, `(30°,0.4)`, `(40°,0)`, `(40°,0.2)`
and `(50°,0)`. Alternate wrist/posture paths had initial joint gaps equivalent
to **1.10–2.36 s**; even the current wrist-zero case retained a 0.056 s gap.
These are scratch kinematic results. They do not establish that another posture
tracks under native contact, or that switching posture instantly is safe.

## Coordinated bounded Cartesian correction

`cartesian_feedback.py` adds a separate pure computation:

```python
step = bounded_cartesian_step(
    hand.ik,
    previous_nominal_q,
    desired_position,
    desired_rotation,
    dt=env.dt,
    rates=joint_and_lift_rates,
    reference=prediction.reference_qpos,
)
```

It solves a nine-variable damped least-squares problem with joint/native bounds
and the complete per-tick rate box included in the optimization. A small NumPy
active-set solver redistributes motion among free joints rather than clipping an
unbounded answer afterward. Multiple linearizations share the same original
rate box. Existing posture goals are ramped in time without changing the saved
posture constraints. The returned FK and residual describe the **returned nominal
joint reference**, not the unconstrained target.

The caller must still check collisions, coupled finger geometry, reference
bounds and all force/incident guards before actuation. Force-integrator
anti-windup can use returned nominal FK; measured servo sag is a separate error.
No native model, gain, task criterion or abort limit is changed by this module.

The counterfactual column above compares one bounded step from each original
previous nominal reference; it is not a simulated new episode. Once the original
trajectory has already drifted, one bounded step cannot erase accumulated error.
A fresh native rollout remains required to evaluate contact retention and speed.

Tests cover coordinated box redistribution, separable reference solutions, native
FK/rate bounds without live state writes, smoothly ramped wrist goals, exact
nominal holding and local progress without restarting from measured sag. An
additional numerical check of 100 random nine-dimensional boxes found projected
KKT residual below 1.7e-16 and at most seven active-set solves. A local scratch
microbenchmark averaged approximately 0.32 ms per bounded step; this is not a
full simulator or hardware throughput claim.
