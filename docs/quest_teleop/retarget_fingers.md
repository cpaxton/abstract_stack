# Quest finger retargeting (#18)

`FingerRetargeter` maps canonical global OVR24 hand positions to proposals for
all32 native finger actuators. It owns no simulator and writes no live state.
The #20 gate owns freshness, session/generation, engagement, final composition,
collision checks and complete command application. `Left Wrist`/`Right Wrist`
remain with #17. This component is simulation-only and adds no robot endpoint.

## Input and API

```python
mapper = FingerRetargeter(native_descriptor_model)
mapper.reset(dict(zip(state.actuator_names, state.applied_targets)))
proposal = mapper.propose(tracking["left_hand"], tracking["right_hand"], dt_s=0.02)
if proposal.valid:
    # Submit to the session/generation-aware final gate, never apply directly
    # from a network callback or call this an authorization to engage.
    finger_targets = proposal.targets
```

The constructor copies the model's descriptors and retains no `MjModel`/`MjData`.
It requires the pinned32 limited hinge-position transmissions with matching joint
names and unit gears. It does not retune gains, force limits, joints, mass,
collision masks, native equality constraints or tendons. `reset` takes the caller's
complete validated applied references (a full59-channel mapping is accepted).
Measured positions outside native command ranges are rejected rather than
silently clipped into a new reference. A failed reset clears the previous filter.

Input is the `ovr_hand_24` canonical list with all24 unique named entries. The
root must have valid position/orientation flags; every position used by the
geometric mapping must be finite and valid. ForearmStub/Pinky0 are not used by the
features. Tip orientations may remain unavailable; no identity orientation is
manufactured. Global joint positions are in the common operator frame. Raw
parent-local SDK rotations are not accepted as positions; #15's SDK-compatible
acquisition/FK adapter must compose them before conversion. Hand side comes from
the trusted left/right source slot and is not inferred from global position.
Transport validation/age and confidence policy remain upstream/gate concerns.

`FingerProposal` contains `valid`, the complete32-entry `targets`, native-bounded
pre-filter `nominal`, feature/rate/saturation `diagnostics`, and a failure `reason`.
If either hand is invalid, the whole proposal has empty targets and nominal maps;
filter state is invalidated. Restored tracking alone cannot restart proposals:
the caller must explicitly reset from newly approved references. The component
does not itself decide that calibration, freshness or deliberate engagement are
valid. It supports no identity or last-known-pose substitution for invalid input.

## Explicit geometry and native projection

A palm frame is constructed from WristRoot→Middle1 (forward) and Pinky1→Index1
(radial, projected perpendicular to forward). The palm-normal cross product uses
an explicit left/right reflection sign. Features are therefore invariant under
proper rigid transforms and mirrored left/right fixtures produce the same
anatomical commands in the native mirrored hands.

| Native suffix (both sides) | Canonical OVR24 geometry |
| --- | --- |
| Index/Middle/Ring/Pinky MCP | Proximal1→2 flexion in the palm forward/normal plane |
| Index/Middle/Ring/Pinky PIP | 0.7 × hinge flexion at joint2 +0.3 × hinge flexion at joint3; the native distal/IP unit has one actuator |
| Index/Middle/Ring/Pinky Abd | Signed angle of proximal1→2 out of the forward/normal plane, added to the native open reference with gain0.7 |
| Thumb MCP/PIP/DIP | Segment bend angles at Thumb1/2/3 respectively |
| Thumb Abd | Radial→forward sweep of Thumb0→1, mapped to a bounded native opposition reference |

The defaults normalize human MCP/PIP flexion by1.4/1.5 rad, thumb bends by1.3 rad
and thumb opposition by1.2 rad. Native open references come from the pinned
model's `qpos0`, including thumb and abduction offsets. Closed defaults are
MCP1.4/PIP1.6 rad for the four fingers; thumb MCP0.55, Abd−0.1, PIP0.8 and DIP1.1
rad. Constructor checks reject endpoints outside actual native control limits.
Feature saturation and final native saturation are reported separately. These
are declared geometric projection settings, not a fitted anatomical calibration.
Degenerate palm/segment geometry rejects the complete proposal.

The pinned simulation model has **32 direct finger joint servos and no finger
fixed tendon/equality coupling**. Its sole fixed tendon is `Total Lift`, owned by
body control. No nonexistent finger tendon channels or independent non-thumb DIP
actuators are invented. Real ORCA hardware tendon behavior is not validated here.
A scratch FK check of native thumb Abd showed that increasing−0.73304→−0.1 rad
reduces thumb-DIP to index-PIP distance in the neutral model, supporting the
selected opposition direction; this is geometry, not pinch/contact success.

## Filter and final gate

When the final gate applies stricter bounds, pass the actual complete applied
references on every tick as `applied_references=...` to `propose`. This reconciles
filter position/velocity with the executed references instead of accumulating
unexecuted proposals. External jumps or rate violations invalidate the mapper;
reset explicitly after pause, generation changes or gate rejection. The legacy
no-feedback call is for isolated characterization where proposals are applied
unchanged.

The exponential nominal filter has tau0.08 s. At each valid step, velocity is
bounded to2 rad/s and acceleration to8 rad/s², using the supplied control dt;
gaps over0.05 s invalidate the filter. Native-limit braking reserves stopping
distance with

`v_cap = sqrt((a_max*dt)^2 + 2*a_max*distance_to_bound) - a_max*dt`.

The feasible velocity interval intersects that cap, absolute velocity limits and
the prior velocity ± acceleration×dt. Empty feasible intervals fail explicitly;
final position clipping cannot hide an acceleration violation. Small nominal
overrun under reversal/braking is possible while rate/native bounds remain
satisfied. `nominal`, filtered targets and measured servo response are distinct.

Final combined arm/finger geometry must be checked using the **proposed** finger
posture before #20 writes any command. For these identity-gear direct joints,
scratch projection uses actuator→joint→qpos address. The mapper does not certify
collision freedom. Running its isolated characterization below intentionally
bypasses the future engagement/collision gate to retain native failure evidence;
it must not be substituted for the integrated teleoperation entry point.

## Verification and retained native counterexample

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_quest_retarget_fingers.py
MABEL_ROOT=/path/to/open_mabel PYTHONPATH=src python scripts/probe_quest_retarget_fingers.py
# Optional actual native close-up snapshots; Mac graphics context required:
MUJOCO_GL=cgl MABEL_ROOT=/path/to/open_mabel PYTHONPATH=src \
  python scripts/probe_quest_retarget_fingers.py --render
```

Tests cover named/mirrored features, rigid transform invariance, side ownership,
open/fist/pinch/gradual proposals, velocity/acceleration/native-bound braking,
invalid joints/time gaps/reset behavior, model/data non-mutation, native response
and the contact-blocked default-shelf case. `retarget_fingers_fixtures.py` contains
explicitly synthetic landmarks; they are not personal device recordings or
spatial ground truth.

The probe runs seed0 with no products, unchanged native dynamics and zero drive
commands. It records source hashes/settings/model provenance plus nominal,
applied, measured actuator positions/forces and native contacts at50 Hz under
ignored `outputs/quest-retarget-fingers/`. Contact peaks are the maximum observed
at the final substep of each control tick, not a claim to capture every physics
substep peak. The robot base remains free. It records two declared scene presets:
front_x−1.4 m for shelf clearance, and the default−0.62 m shelf to retain contact
failure. The clearance run can still have native finger/self contacts.

Initial development measurements (Python3.11.15, MuJoCo3.17.0, NumPy1.26.4,
model revision`d0181f04025c06e1790b7ab41fe44d602cc2ee5b`) at the closed sample:

| Case / channel | Applied rad | Measured rad |
| --- | --- | --- |
| Shelf clearance / Left Index MCP | 1.4000 | 0.9572 |
| Shelf clearance / Right Index MCP | 1.4006 | 0.9569 |
| Shelf clearance / Left Middle PIP | 1.3760 | 1.2648 |
| Shelf clearance / Right Middle PIP | 1.3764 | 1.2617 |
| Default shelf / Left Index MCP | 1.4000 | −0.3598 |
| Default shelf / Right Index MCP | 1.4006 | 0.5085 |

Both hands show continuous measured response in the clearance case, with lag and
native self-contact errors retained. In the default case the shelf obstructs
left index links: the maximum sampled finger-contact normal force was455.8 N,
versus7.60 N in the clearance scene. This is a **failed contact/tracking case**,
not a usable manipulation posture. Bounded commands alone cannot establish
tracking, grasp, object retention or safety. The combined #17/#20 guard must
reject such proposals; no hidden pose correction or contact suppression is used.

Live canonical Quest finger qualification, fitted human/robot mapping,
collision-approved combined motion and object grasp/lift/release are not complete.
Keep this component draft until the required integration/device acceptance is
met. Native fixture tests do not establish accurate or safe live teleoperation.

## Stack and ownership

Branch`codex/quest-finger-retargeting` targets`codex/quest-control-base`, which
contains existing pending providers including the latest #27 bind-validation
fix. This task owns only finger implementation/fixtures, its tests/probe and this
document. #17 owns arm/head code; #20 owns the single final gate. Shared provider
files and Unity settings are unchanged. After authorized foundation squash merges,
rebase this task's own commits onto the resulting main history and retarget.
