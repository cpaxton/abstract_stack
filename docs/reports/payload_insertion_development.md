# Payload tracking and supported insertion development

These are changing-policy development probes, not a validation success rate.
Each episode schedules ten cartons from the table, zero prefills, seed 0, and
10 mm **total** free row width including both walls. No full-ten success has
been demonstrated. Earlier results remain in
[the development ledger](replanning_development.md).

## Carry diagnosis and controller changes

The v11 failure reached a nominal wrist reference near -0.853 rad with an
applied servo target at the native -0.8972 rad lower bound. Measured wrist error
was about 0.180 rad. The native wrist range is asymmetric: -0.8972 to +0.6736 rad.
No native model range, gain, inertia or collision mask has been changed.

Opt-in `--payload-load-compensation` reserves 0.4 rad of nominal wrist headroom
at each end of the native range and 10 mm of positive lift headroom. The latter
matches the existing lift compensation limit. Planning copies these narrower
reference ranges into its scratch solver; applied targets still obey the native
limits. Fixed-posture constraints must also lie inside the reserved ranges.

Measured active-carton contact wrench is transported to the native wrist axis.
The opposite wrist torque enters the existing position-servo compensation path,
bounded to 0.3 Nm and 0.5 Nm/s. A native contact test verifies its sign against
MuJoCo constraint torque. Nominal references and applied compensated targets
remain separate; coupled arm/finger geometry is checked before actuation. This
prototype does not provide general payload feedforward for every arm joint.

In v12 the measured wrist error at failure was about 4.8 mrad, without wrist
target clipping. This is a different pose and phase from v11, so it is not a
controlled tracking-improvement comparison. V12 still lost its grasp: recorded
controls replayed to a bitwise-identical terminal integration state, showing
30.01 mm relative slip since verified lift, mostly lateral (21.29 and 20.37 mm;
vertical -5.68 mm). That replay earns zero fresh-task credit.

V13 raises the explicitly recorded grasp-force target from 4.5 to 6 N in an
unpromoted development config. It reached the existing compensated-payload
collision guard against another table carton. The 25 N abort limit and task
tolerances were unchanged. V14 adds vertical-first inventory transit: measured
product extents determine an elevated route with a requested 20 mm clearance.
Every retained waypoint receives a scratch rigid-payload clearance check.
This sampled nominal forecast does not guarantee continuous live clearance or
predict grasp slip. Closely spaced initial states may conservatively reject a
route, which does not establish global infeasibility.

## Episode outcomes

Artifacts live in `outputs/contact-rich-replanning-v1/mobile-table-ten-vN/`.
Each directory retains exact source/config snapshots, controls, initial state,
events and terminal/process outcomes. All stage counts use ten as denominator.

| Attempt | Pick / shelf place / final / retreat | Result |
| --- | --- | --- |
| v11 | 2 / 1 / 1 / 1 | Second carton exceeds unchanged 30 mm carry-slip guard |
| v12 | 1 / 0 / 0 / 0 | Carry slip despite wrist compensation |
| v13 | 1 / 0 / 0 / 0 | Proposed compensated payload collides with table inventory |
| v14 | 1 / 0 / 0 / 0 | Pick/carry/table release complete; pusher approach rejected |
| v15 | 1 / 0 / 0 / 0 | Three native base moves verified; full insertion path rejected |
| v16 | 1 / 1 / 0 / 0 | Partial shelf push, unloading and clearing verified; deep-push search rejected |
| v17 | 1 / 1 / 0 / 0 | Alternate finger admits deep push; progress stalls near backstop and times out |
| v18 | 1 / 1 / 0 / 0 | Settled partial stop verified; empty withdrawal stopped after bounded staging |
| v19-video | 0 / 0 / 0 / 0 | Launcher constructor argument error before episode; unknown outcome |

V14 and v15 delivered carton 8 to the existing table beside the shelf. Table
support earns no shelf-place credit. V15's checked 50 mm base backoff and lateral
repositioning admitted some approach candidates, but none could also complete
the full-depth push. A local search failure is not global unreachability.

## Supported two-stage push

`--split-insertion` attempts an intermediate object center 5 mm inside the
shelf front after accounting for the measured carton extent. It requires loaded
routes, checked base repositioning, bounded contact IK and payload compensation.
The first push must measure shelf support and an upright carton. Unloading and
retreat must leave it supported without hand contact. The robot then clears its
hand, restores the longitudinal base backoff if needed, and plans the remaining
push from the actual supported state. All native force, contact, final reference
and all-product monitoring guards remain active. A force abort is terminal.

Partial support is not final positioning, back contact or verified final
retreat. The existing evaluator continues to require those later gates. This
mode is a prototype; full physical completion has not yet been demonstrated.
V16 verified the intermediate push and unloaded shelf support, with no recorded
tip/drop incidents. The remaining insertion candidates were blocked principally
by a folded pinky against the shelf side. The nominal choice of middle finger
was inappropriate even though this carton was not the end carton in the row.
V17 searches all three supported native pushing fingers
under the same 45 s planning budget, coupled geometry and clearance checks.
The ordinary controller retains its original single-finger choice.
V17 reached about 12 mm short of the intended backstop position. Its final
bounded controller had arm joints 2 and 5 at planning limits: requested X
increments near 0.31 mm became about 0.008 mm of nominal X progress per 20 ms.
Measured tool tracking remained about 0.4 mm. Thus this stall is primarily a
constrained nominal-motion problem, separate from servo tracking. No final
back contact or final-position credit was awarded.

The next candidate brakes before the partial stop, requires X error within
5 mm and measured linear speed below 5 mm/s for the existing contact dwell,
and rechecks unloaded support/position before and after the base handoff.
These stricter intermediate gates address review feedback; they do not change
the task's final tolerances or retrospectively certify v16/v17's handoffs.
V18 passed the new dwell/velocity/position gate, then the original empty-hand
clearance move stopped with a 7 mm predicted remaining miss. This preserves the
ordinary staging failure instead of treating its closest pose as a completed
retreat. The next prototype searches three checked withdrawal displacements
before executing one; execution failures remain terminal.

Use a fresh directory for every supervised run:

```bash
.venv/bin/python scripts/run_frozen_replanning.py \
  --output outputs/my-fresh-ten-carton-probe --wall-seconds 600 -- \
  --objects 10 --seed 0 --total-free-space 0.01 --search-seconds 30 \
  --loaded-route --admit-loaded-route --table-front --guarded-approach \
  --base-reposition --finger-load-compensation --bounded-contact-ik \
  --square-backstop --contact-recovery --payload-load-compensation \
  --split-insertion \
  --control-config configs/controllers/payload_grip_development.json
```

Add `--video --video-fps 10` to the worker arguments for synchronized MP4s from
the world camera and native robot-mounted camera. Run video jobs with
`MUJOCO_GL=egl flock --no-fork /tmp/abstract-stack-render.lock` before the Python
command on this Linux workstation. Both views share physics timestamps; neck
gaze uses recorded native actuator commands. Failed and partial episodes retain
their verdicts in the overlays and metadata.
The first recording launch (`v19-video`) passed an unsupported constructor
keyword and failed before physics. It retains zero credit. The launcher now
uses the existing `video`/`video_fps` constructor arguments, with native CLI
construction tests for recording enabled and disabled. No video was produced
by the failed launch.

The supervisor freezes package, runner and explicit control configs, then runs
one process group with a declared wall budget. It records source hashes, base
revision, effective command, package versions and terminal process result.
Missing, malformed, contradictory, signaled and timed-out outcomes earn zero
successful stages. A planning-only result always earns zero manipulation credit.
The native crash gate remains [open](native_view_crash_diagnostic.md); these
changes are not promoted defaults or a full-suite stability fix.

Validation: the frozen `split-insertion-v2` snapshot passed all 250 tests in
205.03 s, including native MABEL physics and rendering. The later settled-handoff
gate and surrounding changes have 38 focused passing tests; synthetic gate observations alone
are not physical insertion evidence. Pre-commit and an offline wheel/source
build passed. The earlier signaled suite remains an unresolved stability result.
