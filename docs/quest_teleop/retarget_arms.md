# Bimanual arm and head retargeting (#17)

`ArmHeadRetargeter` converts calibrated native-root-relative palm poses and head
orientation to named native position-servo proposals. It owns a separate
`ShelfStockingEnv`, model, data and scratch collision state. It never writes a
live robot pose/control, steps the live simulator, changes model parameters or
adds external forces. The #20 gate owns final composition/application. A valid
proposal is kinematic admission, not an engagement or physical tracking pass.

## Inputs and outputs

```python
retargeter = ArmHeadRetargeter(
    config,
    seed=0,
    expected_model_sha256=sim.describe()["model_xml_sha256"],
)
references = retargeter.measured_poses(sim.read_state())
proposal = retargeter.propose(
    state,
    {"left": calibrated["left_hand"], "right": calibrated["right_hand"]},
    calibrated["head"],
    elbows={"left": left_elbow_root_position},  # optional; invalid => explicit fallback
)
if proposal.valid:
    # Submit proposal.targets, not live qpos, to the session/generation/time gate.
    pass
```

Use #16's calibration artifact and wrist-local palm offsets before this call.
The targets name the `left_palm` / `right_palm` body origin and orientation;
`measured_poses` supplies the robot-side neutral reference. Do not substitute
raw OVR bone-local rotations, an absolute human wrist or a world-space pose.
The head target is the calibrated `head` body orientation; translation is not
commanded. Native neck joints are solved from their actual transformed axes,
so no assumed Euler axis order or quaternion component reorder is used.
Metres, radians, active wxyz rotations and `mabel_chassis` coordinates apply.

Default output is exactly16 arm/wrist plus3 neck named actuator targets.
`channels=("head",)` returns only the three neck channels;
`channels=("arms",)` returns both eight-channel arms. One-arm staging supplies
the other arm's neutral palm target. Missing either required palm/head pose,
unknown source names, invalid state, episode completion, excessive timing gaps,
failed IK or blocked geometry returns `valid=False`, an empty target map and
an explicit reason/diagnostic. No closest failed candidate becomes a command.
A failed deterministic search does not prove global unreachability.

Session/calibration generation, source validity/freshness, clock age and lifecycle
are supplied by #15/#16/#20. Call `reset()` on reference/session/generation changes,
tracking loss, pause/realignment and gate rejection. Internal state also resets
when the simulator's independent reset generation changes. Repeated snapshots
are rejected; accepted proposals require advancing simulator steps/time.

## Search and bounds

Each arm uses damped least-squares on its own eight native hinge channels, with
fixed measured torso/lift/base and the other arm's current/candidate geometry.
The nominal search tries the requested orientation, then at most six axis
alternatives within0.08 rad. Position/orientation residuals are always reported
against the **original** request; an alternative never relaxes the configured
15 mm /0.15 rad admission thresholds. The selected-orientation residual is
reported separately. Native actuator and joint ranges are intersected; admitted
transmissions must be direct unit-gear native position servos. Lift/body/base and
finger ownership are excluded from arm/head proposals.

A finite usable elbow point biases posture through a damped Jacobian null-space
term; it never replaces the wrist target. Missing/nonfinite/out-of-workspace elbow
points explicitly fall back to the measured-posture seed. Default workspace is
within0.90 m of the corresponding shoulder mount and palm height at least0.20 m
in the root frame. These are configured conservative bounds, not a reachability
oracle. The solver uses at most80 iterations,0.02 damping and0.10 rad bounded
iteration steps.

After IK, joint-reference filtering (60 ms time constant),0.8 rad/s speed and
3 rad/s² acceleration limits intersect with native bounds. Rate history derives
from the **observed applied references**, rather than assuming a prior proposal
was applied. An unexecuted proposal therefore cannot integrate into a target
jump. A rate/acceleration/native-bound intersection that is empty fails closed.
Nominal and bounded targets remain separate in diagnostics.

The complete nominal endpoint and sampled nominal/bounded paths are checked
against robot/self/environment contacts, including non-grasping links and the
measured finger posture. Default settings match the existing IK guard: reject
new penetration deeper than2 mm or existing penetration worsened by more than
1 mm. Wheel/floor support is excluded. Existing overlap is reported, not called
clearance. Eight samples per segment are a kinematic prefilter, not continuous
collision checking or prediction of inertial/contact behavior.

## Final composition hook

```python
check = retargeter.check_composed(state, complete_named_targets)
if not check["valid"]:
    # #20 faults/retains its previous qualified references; do not apply this map.
    pass
```

The hook requires every native actuator exactly once. It replays the proposed
arm, neck and **finger** posture into scratch qpos, then checks the whole coupled
path. All32 pinned MABEL finger actuators are direct unit-gear hinge servos.
This catches a newly colliding finger posture even with unchanged arm targets;
checking the arm against old fingers alone is insufficient.

Base references must remain unchanged. Until #19 is qualified, torso/lift accept
only unchanged references or a measured-state hold. The owner can pass
`body_hold={"Lift": ..., "Torso": ...}` containing its latched measured hold
references to admit a bounded transition from current commands to that latch.
Values outside that transition interval still reject; this does not authorize
human body retargeting. Native `Total Lift` fixed
tendon admission verifies its two unit-coefficient stage joints; scratch total
length changes split equally while preserving the measured stage difference.
No live lift pose, equality or tendon is changed. The hook validates native
control/joint bounds after composition; #20 must still apply final global
rate/acceleration and lifecycle bounds. It returns explicit collision evidence
or a transmission/bounds reason. The caller must use the same scene/model hash.

## Reuse assessment

The existing `PoseIK` is specialized to the left palm and searches shared lift,
with left-hand carrying/object policies. It cannot be reused wholesale for
independent bilateral, fixed-body ownership. This module reuses its tested
`rotation_error` convention and native Jacobian/scratch approach while keeping
bilateral ownership local. Existing `MabelController` supports bounded gravity
feedforward through position references, but owns lift and maintains a different
nominal/applied state. It is not embedded here: proposals remain nominal, and a
future qualified compensation stage belongs before #20's final bounds/geometry
check. Low-gain wrist/gravity and contact tracking error remain visible in the
measured results; no load retention is claimed.

## Verification and native evidence

Run with the project's installed environment and pinned MABEL assets:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_quest_retarget_arms.py
PYTHONPATH=src python scripts/probe_quest_retarget_arms.py \
  --seed 0 --ticks 50 --output outputs/quest-retarget-arms/probe.json
```

The probe records startup source revision/hash, model/source-model hashes,
configuration/seed, settings/timestep, per-tick nominal/bounded/measured values,
separate IK and measured errors, native contact-force peaks, timings and all
scheduled outcomes. It is an isolated command fixture, without networking/human
actuation. Full traces remain ignored under `outputs/`.

Development seed0 used the unmodified robot in an explicit empty-space studio
fixture: product_count0, shelf front_x=-1.4, timestep0.002/frame_skip10. The default
front_x=-0.62 starts with robot/shelf overlap up to about25 mm; it is separately
retained as a blocked composed-finger fixture instead of hiding those contacts.

| Scheduled case | Observed result after50 native ticks (1 simulated second) |
| --- | --- |
| Left20 mm upward palm target | Executed; measured palm error8.56 mm /0.0362 rad |
| Right20 mm upward palm target | Executed; measured palm error8.56 mm /0.0362 rad |
| Both20 mm upward palm targets | Executed; measured left/right error8.56/8.56 mm |
| Head orientation0.12 rad about each root axis | Three executed cases; final measured orientation error0.0033–0.0074 rad |
| Far palm target | Rejected before stepping, outside configured workspace |
| Default shelf + proposed left-index closure | Rejected before stepping, composed sampled-path collision |

These are six executed and two rejected cases out of eight scheduled; neither
rejection counts as tracking/manipulation success. The accepted-case measured
errors include physics response; nominal IK residuals remain separate. In these
development runs proposal p95 was about2.8–7.0 ms, without a hard real-time claim.
The probe supplies exact code/model/settings for reruns; source fingerprints
identify development measurements when the worktree was dirty.

Regression tests cover snapshot detachment/native-model preservation, both arms,
three neck axes, native physics arm/neck response, calibrated-frame behavior,
required/optional inputs, native/workspace limits, local search failure,
collision rejection including proposed fingers, rate/acceleration/observation
history, reset/timing/thread ownership and malformed settings.

Keep draft until live canonical Quest/calibration input, broader workspace/load
and complete #20/#24 lifecycle composition are qualified. This module does not
establish contact manipulation, optical accuracy or object retention.

## Parallel ownership and merge order

This task owns only `retarget_arms.py`, its named tests/probe and this document.
#18 owns finger retargeting; #20 owns command/lifecycle gating. No shared module,
manifest, Unity scene or package file is edited. All task PRs target the shared
`codex/quest-control-base`, containing existing provider PRs#26/#27/#28/#32–#35,
including #27's bind-validation fix. After authorized provider squash merges,
rebase each task's own commits onto actual main and retarget. No main merge is
performed by this implementation.
