# Relative pose calibration (#16)

`abstract_stack.teleop.calibration` converts global Quest poses and captures a
single immutable alignment artifact for later retargeters. It produces root-frame
pose proposals; it never sends actuator targets or modifies MuJoCo state.

Unity global poses use metres and xyzw. `unity_to_operator` applies the reflected
basis `B=(z,-x,y)` to position and `B R B^T` to rotation, yielding canonical wxyz.
`ovr_to_operator` first follows Meta XR Core 209.0.0 `OVRCommon.cs`
`FromFlippedZVector3f` / `FromFlippedZQuatf`: `(x,y,-z)` / `(-x,-y,z,w)`.
It accepts native **global tracking-space** head and hand-root poses only.
Legacy hand bone-local rotations need acquisition-side SDK-compatible forward
kinematics; applying this helper directly to those rotations is incorrect.
The network adapter (#15) owns that skeleton composition.

Capture requires valid head and both wrists, current usable input and a
Ready/Paused caller. The caller supplies measured root-relative robot neutral
poses and explicit per-source `TargetSpec` values. No human absolute position is
used as a robot joint command. Optional `Body_Chest` and `Body_Hips` capture their
own neutral references; omitting a body source later omits its proposal. Head and
both wrists are always required together.

By default, horizontal neutral head-forward aligns with native MABEL forward
(-X); gravity stays +Z. Looking almost straight up/down cannot establish heading
and fails calibration. An explicit proper alignment quaternion can override
heading estimation when another measured reference is available. Robot world
spawn yaw never enters the mapping. At an identity neutral head pose, alignment
is `diag(-1,-1,1)` as specified in the interface contract.

For each source, let `P0,R0` and `P,R` be neutral/current poses after composing its
configured wrist-local palm transform, `A` the operator-to-root alignment, `s` the
translation scale, and `Pr,Rr` the supplied robot neutral:

```text
target_position = Pr + s A (P - P0)
target_rotation = A R R0^T A^T Rr
```

The same formula supplies relative head/body references. Palm translations rotate
with the wrist; palm orientation offsets participate in both neutral and current
compositions. Each side has a separate offset/scale and retains its source name.
There is no assumed human arm length, palm frame or automatic anthropometric fit.
A caller may derive an arm scale from measured robot/operator reach, then persist
that explicit value in the artifact. Translation and angle bounds reject the
whole proposal; there is no clipping or substitution of an unavailable pose.
Neutral input reproduces the supplied robot references exactly, avoiding a
calibration target discontinuity. This algebra does not bound a later IK solver's
joint transition or prove physical command continuity.

## Integration

```python
from abstract_stack.teleop.calibration import Calibration, TargetSpec

alignment = Calibration(session_id, generation=host_generation)
# neutral/current: named validated Pose values in operator_forward_left_up.
# measured_robot: named Pose values in mabel_chassis_native, never world poses.
specs = {name: TargetSpec(pose) for name, pose in measured_robot.items()}
artifact = alignment.capture(neutral, specs, ready_or_paused=True, inputs_usable=True)
# Host publishes artifact.generation and artifact.to_dict() once capture succeeds.
alignment.acknowledge_engagement(
    session_id=session_id,
    generation=artifact.generation,
    deliberate=True,
    inputs_usable=True,
)
proposals = alignment.propose(
    current,
    session_id=session_id,
    generation=artifact.generation,
    inputs_usable=True,
)
```

The host is the single generation authority. Synchronize its generation with this
object before capture, then publish the returned generation; do not independently
increment in both modules. `invalidate("reference_space_changed")` and
`invalidate("reset")` advance generation, erase the artifact and require a new
capture followed by deliberate engagement. On control reconnect, discard this
object and create one for the new session. Failed captures also invalidate old
alignment. Input loss or invalid/out-of-bounds proposals clear local engagement;
restored tracking does not automatically resume. An obsolete session/generation
cannot produce proposals.

`inputs_usable` is an explicit precondition supplied by the integration gate,
which must validate schema, profile, independent observation ages, clock
uncertainty, focus/presence, video feedback and required tracking. Calibration
cannot infer those from a bare pose. Its engagement acknowledgement means only
local alignment eligibility. #20 remains responsible for global state,
realignment, finite native commands, loaded hold, rate/acceleration/collision
bounds and the sole live command writer. Artifact serialization includes personal
neutral coordinates: keep it in ignored session outputs, never source control.

## Validation and remaining device gate

```sh
PYTHONPATH=src python -m pytest tests/test_quest_calibration.py -q
PYTHONPATH=src python scripts/probe_quest_calibration.py /path/to/raw.jsonl \
  --output outputs/quest-calibration/replay.json
```

Known-axis tests cover Unity and native OVR forward/left/up, positive rotation,
reflected-basis round trips, neutral heading, side identity, scale/palm offsets,
optional body inputs, generation changes and failure/re-engagement behavior.
No physics settings or model assets change.

Replay of the prior physical probe 0.3.2 recording
SHA256 `907438216b54db5ba2c69b1b8eb834543d57e89caf6f842378713a3080840732`
validated 8,439 global root conversions from 5,691 samples over 79.575 s: head
5,691, left hand 862, right hand 1,886. There were no conversion errors and one
recorded reference change. **No sample had both hands tracked simultaneously**;
therefore no complete neutral was captured and no complete-neutral pass is
claimed. The replay stores only aggregate diagnostics; the raw recording remains
in ignored outputs. It does not treat source target timestamps as sensor times.

Keep the PR draft until a labelled device run shows both hands simultaneously,
captures a neutral and checks each wrist separately forward/up/left, head yaw,
head/body movement, then recenter/reset/new neutral. Compare numerical proposed
deltas with known movement directions; confirm neutral target equality before
and after each capture. Actual bounded actuator continuity must subsequently be
checked with #17/#19/#20 and native stepping. There is no live engagement or
manipulation success evidence in this change.
