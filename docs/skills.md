# Skill curriculum and initial distributions

Start with perfect state, a fixed single-level shelf, rigid 70 x 50 x 240 mm
cartons and the existing supply layout. No policy learning or scene/dynamics
randomization is required to establish the scripted teacher skills. Reset
jitter is the existing +/-2 mm with a recorded seed.

The objective is reliable pickup on this **declared, task-conditioned** initial
distribution. A 100% observed score on a small development set is not a
guarantee over all poses or a statistical robustness result. Keep unsafe,
unreached, failed and unknown cases in their scheduled denominators.

## Decomposition

| Skill | Initial-state distribution to collect | Measured exit gate |
| --- | --- | --- |
| Grasp/lift/hold | Fresh front carton, plus remaining cartons after preceding stocks: object pose/velocity, neighboring inventory, empty-hand posture and table support | Opposing contact, actual rise, retained hand contact, no table support, then 0.5 s loaded hold |
| Transfer | Outputs of verified grasps: actual object-to-hand transform, payload yaw/tilt, arm/lift posture, inventory and shelf clearance | Retain the load through the checked shelf-approach route |
| Shelf placement | Outputs of transfer: measured rotated extents, carried pose/velocity, shelf height and grasp offset | Native shelf support within the original bounded seating range |
| Contact finishing | Supported placements with measured depth/lateral/yaw error, contact partners and wrench | Task-specific final position/alignment/contact dwell; unchanged force guards |
| Release/retreat | Verified supported finishing states, including residual hand load | Unload, withdraw, and preserve every required product's support/pose/stability |

`StockingScript._grasp_and_lift()` is shared by the complete stocking sequence
and `GraspSkillProbe`; transfer/placement is separately callable. Placement,
contact finishing and release already have separate control entry points. These
are five skill boundaries, not five trained policies. Transfer is still a
single composed route and does not yet have an independent learned controller.

## Composable skill library and per-carton task plan

[`skill_library.py`](../src/abstract_stack/skill_library.py) makes those
boundaries explicit and reusable. A `Skill` is a named unit with a declared
measured entry precondition, a declared measured exit gate, optional upstream
dependencies and a callable `execute` adapter. `TaskPlan` validates and orders a
sequence of skills; `run_plan` executes it and records reach/entry/exit evidence
at every boundary. `build_stocking_plan()` returns the canonical released
per-carton sequence `pick → place → insert → push_to_contact → retreat`, where
`pick`/`place` reuse the existing grasp and transfer skills, `insert` releases
and registers the carton, `push_to_contact` advances to measured back contact,
and `retreat` verifies squared contact and withdraws.

Two default-preserving seams expose the finer steps without changing physics:
`_release_and_insert_front(grip, push=False)` stops after front registration,
and `_insert_from_front(retreat=False)` stops with back contact established.
The defaults keep the fused release/insert/retreat behavior. `--plan-driven`
(pending a CLI flag; `ScriptConfig(plan_driven=True)`) composes the five skills
through `run_plan`; the grasped supported-slide protocol keeps its existing
fused path. The plan is always recorded in `summary.json` under `task_plan`,
with each phase event annotated with its `skill`. Success criteria, force
guards and tolerances are unchanged.

## Why reset poses are insufficient

In the original fixed two-carton failure, carton 1 was picked first. Before the
second pickup, carton 0 had moved and reached about 63.3 degrees yaw. The pinky
hit its top during descent. IK had permitted all hand/active-carton contact and
reported a tiny endpoint residual; that did not certify a safe grasp entry.

The corrected teacher specifies grasp yaw from the measured object frame, checks
bounded centering offsets (at most 8 mm in each horizontal grasp-frame axis),
requires 2 mm open-hand clearance, sets the orientation above the object, and
executes retained Cartesian entry waypoints. During entry the active object has
no blanket hand-contact permission. Joint-rate cost times the retained path;
the code does not interpolate straight to the final joint configuration.
Actual closure, opposing contact and lift still require native physics.
Each closing command checks the proposed coupled arm/finger posture before
actuation. Failure rolls back the unexecuted nominal command and stops the skill.

In the initial composed seed-0 follow-up, the grasp entries have about 0 and 63.58
degrees yaw. Both cartons lift about 56 mm with zero measured table support
and approximately 4.7 N hand load. The first stock completes; the second
transfer is rejected by IK. The transfer entries themselves have roughly
-31.7 and +31.5 degrees payload yaw. Thus that skill must start from actual
grasp outcomes, not a nominal rigid attachment assumption.

A later opt-in transfer change uses world-aligned carton orientation after
verified lift, rather than preserving the table yaw as a placement goal. It
completes two stocks in 90.7 simulated seconds on the fixed bench. A bounded
shelf-front staging search preserves that native trajectory exactly. Three
cartons still reject loaded staging at native shoulder limits. These transfer
versions are distinct from the frozen standalone grasp teacher.

Trying the rear carton while the front carton is still present is a different
entry condition: native wrist/tower geometry intersects the front carton.
Those scheduled probes remain rejected failures in the broader test report.
The task planner currently admits front-first order; the rear carton is tested
again from the state after that first stock. Do not silently discard the blocked
cases or claim arbitrary-order 100% pickup.

## Record and inspect skill starts

```bash
MABEL_ROOT=third_party/open_mabel flock /tmp/abstract-stack-render.lock \
  .venv/bin/abstract-stack script --scene pick_place --seed 0 \
  --grasp-skill --record-skill-starts --output-dir outputs/my-skill-sequence

MABEL_ROOT=third_party/open_mabel flock /tmp/abstract-stack-render.lock \
  .venv/bin/python scripts/evaluate_grasp_skill.py \
  --output outputs/my-grasp-development --seeds 0 1 --products carton_1
```

`skill-entries/index.json` lists every reached entry, zero-sample skills, and
empirical position ranges/yaw samples. These ranges describe observed states;
they are not independently sampled uniform priors. Each `entry.json` contains
**every** object's pose, velocity, dimensions and support/hand forces, plus tool
and base position. A full native checkpoint includes distinct hand/tracker
nominal references, applied controls, observer incident/retreat history and task
memory. Capturing an entry does not step the live world. Identical native MJBs
within an episode share storage only after full checkpoint verification; each
checkpoint still has its complete model file and original hash.

The isolated runner also accepts `--checkpoint PATH --seeds 0 --products carton_0`
for a saved `grasp_lift` entry. It validates the entry identity, restores known
nominal/applied references and task/observer history, and performs no reset.
Retained entries earn no new full-episode or stocking credit.

These files are privileged distribution/checkpoint evidence. They are marked
training-ineligible and do not replace causally aligned pre-action observations
and nominal actions. Next, wire the existing LfD recorder into these skill
boundaries and collect verified demonstrations before training students.

## Rear clearance task

[pick_place_rear_clearance.json](../configs/scenes/pick_place_rear_clearance.json)
is a separate fixed layout with a **60 mm desired rear gap** behind stocked
cartons and zero prefills. Shelf and robot geometry are unchanged; target poses
move forward. This interprets “behind” as the space between stocked cartons and
the shelf back wall. Supply-table clearance is unchanged.

The existing backstop controller is unsupported for this task: it deliberately
seeks wall contact. A rear-gap teacher needs pose-based depth registration and
supported unloading/retreat, and must report actual rear clearance after release.
Do not relabel back-wall contact as success or apply the old contact evaluator
to claim the new task complete. Keep the old backstop test as a separate skill
fixture. The new layout itself establishes no manipulation success.

Current evidence and videos: [grasp development](reports/grasp_skill_development.md).

## Base relocation and 2D navigation

When a loaded shelf route is out of the left arm's reach (its `Left Arm 2`
adduction range bottoms out at `-0.175 rad`), the opt-in `--base-reposition`
mode relocates the base through native swerve wheels. The route is planned on a
scene occupancy grid with `emet` A* ([`navigation.py`](../src/abstract_stack/navigation.py)),
which re-marks every measured product footprint as an obstacle, so randomized
object positions are reflected in the map. The idle right hand is swung outward
first, and the pregrasp move uses a checked Cartesian entry. See the
[three-carton report](reports/navigation_three_carton.md).

## Contact skill cases for later training

With `--record-skill-starts`, the episode's `skill-entries/index.json` catalogs
restorable cases. Each entry stores measured poses, velocities, mass, support
and hand loads, completed neighbors, tool and controller state, and the seed.
Checkpoints share one native model binary per episode through hard links. The
native state and validation witness remain specific to each entry.

| Case | Entry variation to retain | What to measure during execution |
|---|---|---|
| Acquire and lift | Mass, footprint, grasp skew, support material | Opposing contacts, lift, slip |
| Transfer and seat | Grasp offset, obstacle and fixture clearance | Load retention, shelf support |
| Side registration | Exposed side, wall or neighbor, lateral error | Signed advance, yaw, unload |
| Front squaring | Skew, projected clearance beside stocked items | Yaw, support, displacement |
| Forward insertion | Depth error, mass, friction, free space | Signed advance, wrench, support |
| Loaded push recovery | Force without motion, depth remaining | Renewed advance, load, recovery choice |
| Recovery exhausted | Remaining gap, force and elapsed stall time | Failure evidence for alternative skills |
| Contact refinement and retreat | Pose error, preload, neighbors and support | Final pose, stable release, safe clearance |

Preload changes are one recovery. Unload and re-contact at a different height
or offset, a side push, and a pivot are separate candidate skills. Record
their outcomes under the same evaluation criteria when implemented. The entry
catalog describes reached states and should retain failures and unreached
cases. Split later evaluation by scene seed and object properties to avoid
putting correlated checkpoints from one episode into both training and
testing. Entry snapshots by themselves carry zero fresh-task credit and are
ineligible as causal training demonstrations.

### Plan the landing and contact moves together

![Flow from observation through free landing, hand preshape, compliant pushing, recovery and withdrawal](media/contact_skill_flow.svg)

The dashed box shows the proposed joint search. Solid boxes show the skill
sequence and its measured outcomes. The current full withdrawal forecast
runs after contact execution. The earlier tight insertion supplies the paired,
folded and side-push postures. A planner candidate must check its own hand
transition, approach, push and exit.

Regenerate both formats from `docs/media/contact_skill_flow.dot` with Graphviz
using `dot -Tsvg` for the docs and `dot -Tpdf` for the paper.

The current controller now verifies and records the free landing after the
grasp hand clears. The landed carton must be upright, supported, and unloaded
before side registration or forward insertion can start. The landing and final
task targets appear separately in the event log.

The remaining planner gap is landing selection. For each candidate landing
pose, check the carried approach, opening and grasp-hand exit, then check
access for a separate pusher and its exit at the proposed final pose. Compare
direct forward advance, forward-then-side registration, and a pre-squaring
step only when measured geometry calls for it. Use remaining free space as
working room. Choose a target depth that leaves room behind the object unless
the task calls for backstop contact. Reobserve at every handoff and replan the
next step from the actual pose. A candidate without an exit should be rejected
before choosing the landing. Keep each phase independently measurable and
retain failed candidates for the skill catalog.

### Hill-climb consecutive cartons from verified milestones

A run with `--record-skill-starts` saves a `stocking_milestone` entry after
every verified retreat with an empty hand and no tip/drop incidents. Its
entry lists completed products. Keep each run in its own output directory so
different policy branches can resume the same measured state.

```sh
python scripts/probe_front_release.py \
  --checkpoint outputs/my-run/episode/skill-entries/NNN-carton_ID-stocking_milestone/checkpoint \
  --output outputs/next-carton-probe --next-cartons 1 --record-skill-starts \
  --control-config configs/controllers/adjacent_insertion.json --wall-seconds 180
```

Legacy `grasp_lift` entries with an unloaded hand can also start the next
carton from their recorded completed set. For other retained motion states,
use the matching skill continuation. A native snapshot can preserve arbitrary
motion state, but resuming a partially executed motion requires its controller
references, progress and guard state. Do not silently treat an in-contact
snapshot as a cleared milestone. All continuations earn zero fresh credit.

### Decide the next motion from measured preconditions

| Observed state | Candidate next motion | Required postcondition |
|---|---|---|
| Held load and a checked free landing | Carry, seat, open, and clear | Upright supported carton and unloaded hand |
| Released carton with a clear projected forward path | Preshape and advance | Measured depth progress under force and support guards |
| Projected footprint conflicts with a stocked neighbor | Square in front or choose a lateral route | Upright support and a newly checked approach and exit |
| Push force is present without object motion | Bounded preload increase, then unload and recontact if needed | Renewed advance without disturbing stocked objects |
| Hand contact is lost before reaching the target | Recontact from a checked pose | Measured safe contact and renewed advance |
| Hand contact is lost at the intended final pose | Check an unloaded clear path | Stable support, final pose, and verified retreat |
| Loaded at final pose and withdrawal clips a neighbor | Change hand posture or exit direction | Force, clearance, and stocked-neighbor stability pass |
| No admitted motion or a guard trips | Stop and snapshot the state | Failure is retained with phase, reason, and selected inputs |

The unloaded-clear branch now handles a failed refinement when the carton
already satisfies placement and the robot is unloaded. It requires stationary
square support, checks the full exit in a native forecast, and verifies
support and retreat in the live scene. Its event does not award backstop
contact credit. The recontact branch for a carton still short of its goal
remains planner work. Different cartons can take different branches based on
their measured state.

With skill recording enabled, fresh runs and skill probes now add a
`motion_failure` entry for a handled controller stop before the controls
are changed. The entry stores the failed phase and reason. If checkpoint
creation itself fails, the run retains the original failure and reports
the checkpoint error separately. A native process crash can prevent terminal
capture. Use the last valid entry and native process log for those cases.

For a long continuation that reaches its wall budget, start from its latest
valid skill entry. `--insert-only` accepts a verified post-squaring entry and
replans final advance. `--withdrawal` accepts contact-refinement and
withdrawal-planning entries and revalidates contact before trying the exit.
`--unloaded-clear` accepts a terminal failure entry and rechecks unloaded
support before forecasting retreat. These are skill-level replans from exact
physics states. They do not resume a partially consumed controller loop.

### Angled entry development case

The shelf sequence under test is free landing, front squaring, a short angled
advance alongside the stocked neighbor, then final registration and guarded
exit. `probe_front_release.py --insert-only --guided-entry-deg=3` enables the
entry probe from a released or post-squaring checkpoint. The opposite sign
is a matched candidate. The angle is a fingertip-plane reference, not a
command to rotate the carton by that angle.

The fixed-lateral-position collision model screens only the first 12 mm of
this opt-in advance. The live controller checks each next motion and monitors
support, wrench, signed progress, and stocked-neighbor displacement. The
reference tapers to zero near the target. Preserve failed cases, including
those rejected before motion.

A future reusable skill contract should take measured object geometry,
pose, support, motion direction, named contact partners and clearance
limits as inputs. It should report progress, measured wrench and whether
the resulting state admits the next skill. The current millimetre and angle
settings are development probes for this shelf. Before adopting them for
other objects, validate progress, jams and pattern preservation across
different dimensions, masses, friction and pose error.

### Sampling recorded stocking states

Use the checkpoint inventory to choose reproducible inputs for local skill
tests. For example, this lists all recorded insertion failures with a
compatible start state, then samples distinct checkpoint manifests for a
smaller development run.

```bash
.venv/bin/python scripts/sample_stocking_states.py outputs/adjacent-insertion-v1 insert --failed
.venv/bin/python scripts/sample_stocking_states.py outputs/adjacent-insertion-v1 insert --failed --count 3 --seed 17
```

The other sampling groups are `grasp-lift`, `transfer-seat`,
`supported-release`, `side-registration`, `push-to-contact`, and `retreat`. Use
`--terminal` to list stopped recovery states, including failures without a
compatible start state in the archive. Each record
links the native checkpoint and its manifest SHA-256 with the run that
provides outcome evidence and the later skill entries reached by that
carton. In a continuation, the parent checkpoint is
included only when its manifest matches the recorded input digest.
`followed_by_failure` and `followed_by_milestone` describe what happened later
in that run. `unverified` means neither outcome is available in the entry
archive. Interpret the terminal phase before assigning a failure to a skill.
The failure filters match recorded execution phases for each group.

Keep both the originating full episodes and zero-fresh-credit continuations
in this development corpus. Compare policy variants from identical
checkpoints and score measured stage reach separately from final retreat.
Keep a fresh frozen-policy suite for end-to-end validation. A last-slot
wall-guided insertion may use a distinct motion with the same support,
placement, and retreat gates.

For a native composed run with paired fingers, positive-wall packing and
guarded withdrawal, pass `--plan-driven --guided-wall-entry` to `script`.
Guided entry uses the same measured-neighbor lateral goal as final
registration. The square phase reserves extra room for the measured corner,
then checks that clearance survives unloading. The first carton uses the
original entry path.

The checkpoint probe can stop at a measured handoff with `--stop-after`.
Use `seat` with `--transfer`, `release` with `--insert`, and
`registration` or `backstop-staging` with `--insert-only`. Automatic
registration uses a free-landing entry. For a native contact-dwell test,
use `--withdrawal --stop-after=squared-contact` with a contact-refinement
entry. A bounded backstop staging handoff carries its own evidence and can
enter guarded refinement. Actual back contact and the squared dwell have
separate fields. These standalone probes receive zero fresh credit.
