# Skill curriculum and initial distributions

Start with perfect state, a fixed single-level shelf, rigid 70 x 50 x 240 mm
cartons and the existing supply layout. The fixed scripted teacher uses the
existing +/-2 mm reset jitter. The isolated development environments below
exercise additional measured start distributions before any policy training.

The objective is reliable pickup on this **declared, task-conditioned** initial
distribution. A 100% observed score on a small development set is not a
guarantee over all poses or a statistical robustness result. Keep unsafe,
unreached, failed and unknown cases in their scheduled denominators.

## Decomposition

| Skill | Initial-state distribution to collect | Measured exit gate |
| --- | --- | --- |
| Grasp/lift/hold | Fresh front carton, plus remaining cartons after preceding stocks: object pose/velocity, neighboring inventory, empty-hand posture and table support | Opposing contact, actual rise, retained hand contact, no table support, then 0.5 s loaded hold |
| Transfer | Outputs of verified grasps: actual object-to-hand transform, payload yaw/tilt, arm/lift posture, inventory and shelf clearance | Retain the load through the checked shelf-approach route |
| Place at staging pose | Outputs of transfer: measured rotated extents, carried pose/velocity, shelf height and grasp offset | Native shelf support within the bounded seating range, with residual pose error allowed |
| Release and register | Supported front placement, measured hand load and open space | Unloaded hand and supported carton inside the insertion approach corridor |
| Side press for alignment | Released carton with a reachable exposed face, yaw error, or insufficient corner clearance to square | Retained lateral shift after unloading. If squaring is needed, verify its separate released yaw and clearance gates |
| Insert | Released supported carton ahead of its neighbors, with measured lateral and yaw error | Verified intermediate depth and square after the pusher unloads |
| Forward push to contact | Supported carton in the insertion corridor, measured depth/lateral/yaw error and contact wrench | Existing backstop force, support, and strict square dwell |
| Lateral row compaction | Supported partially filled row, known wall and neighbors, measured deficit in the remaining opening | Recover the opening after unloading, including a reserve for measured overlap, while keeping the row supported |
| Strict row finish | Compacted row with measured alignment error and sufficient opening | Strict alignment and envelope limits after unloading, with measured contacts during the press |
| Release/retreat | Verified supported finishing states, including residual hand load | Unload, withdraw, and preserve every required product's support/pose/stability |

`StockingScript._grasp_and_lift()` is shared by the complete stocking sequence
and `GraspSkillProbe`; transfer/placement is separately callable. Placement,
contact finishing and release already have separate control entry points. These
are callable controller boundaries. The row press can run between cartons
when the measured opening is too small or gaps are distributed across the row.
Final alignment is evaluated after contact and unloading.

## Seeded skill environments

[`StockingSkillEnv`](../src/abstract_stack/skill_env.py) exposes `reset(seed=...)`
and `step(action)` for `grasp-lift`, `insert`, `push-to-contact`,
`row-compaction`, and `row-press`.
Its observations expose MuJoCo state. Each action selects a checked native
primitive or hand preshape, runs one bounded decision, and returns a binary
reward from measured skill evidence. The same seed reproduces the reset and
settle. Use separate public development and private held-out seeds.

| Environment | Randomized reset factors | Action choices |
| --- | --- | --- |
| `grasp-lift` | Table pose, active yaw through about ±7°, and neighboring positions | Guarded grasp and loaded hold |
| `insert` | Carton width, whole-row free space, target slot, staging depth, offset, and yaw | Paired or single fingertip |
| `push-to-contact` | Carton width, whole-row free space, target slot, and initial lateral offset | Paired or single fingertip, followed by existing strict refinement |
| `row-compaction` and `row-press` | Press direction, two to four supported cartons, carton height, inter-carton spacings, back clearance, and 2–4 mm open-span deficit | Center or counter-yaw front-face press, exposed-side press, centered side press, or side press to the wall |

Row compaction measures the available opening against the sum of all remaining
carton widths, after subtracting the total measured envelope overlap. This
reserves enough space to separate the compressed cartons later. Both row
environments require measured force at every neighboring pair during the
press, internal gaps at most 1 mm after release, both arms unloaded, and
stable shelf support. The strict `row-press` environment also requires
0.15 degree yaw and the existing 0.2 mm projected overlap bounds. The two
arms and every object remain under native guards.
The press fixtures keep the supply inventory outside the shelf contact workspace.
Some reset rows have a rear gap so the policy can choose a side push where
the backstop cannot react against a front press.
The experimental `row-press --action side-square` policy first verifies the
compaction handoff. For cartons already near the backstop, it checks the full
shelf exit, reshapes the hand, and tries the existing strict depth finish
with a narrow tool. It keeps the same final row gates.

```bash
MABEL_ROOT=third_party/open_mabel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_skill_curriculum.py \
  --seed 0 --seed 4 --case-wall-seconds 130 --output outputs/my-skill-curriculum
```

The evaluator writes the frozen policy identity, each fixture and native reset
state, actuator traces, measured outcomes, and process failures. These episodes
earn zero fresh stocking credit. Loaded transfer, front release, and retreat
still use the retained-checkpoint probes below. They need real grasp and tool
load states from previous skills.

The comparison below uses separately supervised row stages. Keep the strict
finish among the reported outcomes when a staging handoff passes.

```bash
MABEL_ROOT=third_party/open_mabel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_skill_curriculum.py \
  --skill row-compaction --action side-centered --seed 9 --seed 18 \
  --case-wall-seconds 125 --output outputs/my-row-compaction
```

The two-carton `side-centered` press passed the compaction handoff on both
public seeds. The initial 3.19 mm and 3.71 mm deficits became 1.45 mm and
1.79 mm of reserve after subtracting the entire projected overlap. Both runs
recorded contact between the cartons, stable shelf support, zero arm load
at exit, and zero tip/drop incidents. Strict final alignment remained open
because the compressed cartons retained measured yaw and envelope overlap.
The repeated physics traces are under `outputs/skill-curriculum-pilot-v1/compaction-final/`.

The first paired-tip development trials also verified grasp/hold and
intermediate insertion on both seeds. Strict backstop push passed on seed 1
and stopped at the bounded refinement budget on seeds 0, 3, and 4. Those
case reports remain separate from the staged row results.

The composed strict finish progressed from the staged row to native back
contact with a narrow tool and the measured lateral target held fixed. The
longer 400 ms refinement rollouts reduced the first carton's yaw magnitude
from 1.26 to 0.27 degrees. The second carton ended at 0.36 degrees and the
row retained 0.66 mm of projected overlap. Refinement exhausted its 20-replan
budget before the strict gates passed. The earlier staged handoff stays
verified and this strict outcome stays failed. See
`outputs/skill-curriculum-pilot-v1/compose-square-forecast/report.json`.

### Interleaving a lateral press with stocking

`--interleave-row-press` runs an experimental sequence on a single spacious
shelf. MABEL stocks and withdraws from two cartons, physically closes their
exposed seam, unloads and clears the hand, then replans the third carton.
The press selects an exposed end from the measured row, checks the base route
and fingertip approach, and stops after sustained carton-pair contact. The
exit gate requires a reduced seam, both cartons still within their placement
criteria, an unloaded robot and no tip or drop incidents.

```bash
mkdir -p outputs/interleaved-stock-inputs
jq '.target_gap = 0.029' configs/scenes/tight_three_cartons.json \
  > outputs/interleaved-stock-inputs/scene.json
.venv/bin/python scripts/run_stocking_case.py \
  --config outputs/interleaved-stock-inputs/scene.json \
  --output outputs/my-interleaved-stock --wall-seconds 360 -- \
  --grasp-skill --base-reposition --interleave-row-press --record-skill-starts
```

The summary records the seam before and after, candidate base workspaces and
the measured press exit alongside the scheduled carton stages. Successful
boundaries and terminal failures have their own checkpoints. To test the
press and third carton from the second verified `stocking_milestone`, pass its
checkpoint to `scripts/probe_front_release.py` with
`--next-cartons 1 --interleave-row-press --record-skill-starts`. The probe
retains the original observer history and earns zero fresh episode credit.

Add `--post-press-release-reapproach` to the fresh run to test a second
contact sequence after a verified row press. It seats the last carton with the
same grasp and carry, opens the hand while checking shelf support, and approaches
again with paired fingertips for the push and withdrawal. The
`release_reapproach` skill entry captures the retained grasp immediately before
opening, and `contact_finish` captures the subsequent fingertip push start.

With the scene above and seed 0, this sequence verified all four gates for all
three cartons after a 30.1 mm reduction in the measured seam. The earlier
sequence verified two withdrawals and failed during the third push's feedback
path check. The actuator traces match through the third carton seating at
213.76 simulated seconds. The fresh run and failed predecessor are recorded in
`outputs/runs/agent-harness/three-cartons-post-press-v1/` and
`outputs/runs/agent-harness/three-cartons-v1/`.

The same seed with the 50 mm/s development preset also passed all gates and
finished 2.07 simulated seconds sooner. A baseline-controller repeat on seed 1
passed all three picks and placements, then exhausted the bounded contact
refinement search on the third carton. Its terminal checkpoint is retained in
`outputs/runs/agent-harness/three-cartons-post-press-seed1-v1/`.
The wider contact search recovered sustained squared contact from that stop.
The withdrawal forecasts then rejected the release because squareness did not
survive unloading. These recovery attempts are recorded in
`outputs/runs/curriculum/post-press-seed1-recovery-v1/`.

## Gymnasium skill curriculum

`StockingCurriculumEnv` presents one guarded primitive per decision. It uses
the [Gymnasium environment API](https://gymnasium.farama.org/api/env/) with a
discrete action space and fixed-size, privileged observations. Observations
include carton poses, velocities, sizes, masses and targets, presence and
active-carton masks, measured contact normals, and actuator lengths, velocities
and targets. Each decision returns reward 1 for the measured skill gate and 0
otherwise. The episode terminates at the skill result or truncates at the
simulator time limit.

| Primitive | Success evidence | Reset sources |
| --- | --- | --- |
| Pick | Measured lift and 0.5 s hold with opposing contact and table clearance | Procedural or captured |
| Place | Retained load transferred and seated at the shelf front with measured shelf support | Captured loaded handoff |
| Insert | Supported opening and front registration with the hand unloaded | Captured or recovery |
| Push | Native back contact and upright dwell, with the hand retained for retreat | Procedural, captured or recovery |
| Slide | Lateral contact and a measured opening after unloading, with neighboring cartons supported | Procedural or captured |
| Retreat | Verified withdrawal with the required cartons still stocked | Captured or recovery |

The [development corpus](../configs/curricula/development.json) pins the
checkpoint manifest hashes and groups related states by episode lineage. The
corpus records historical recovery evidence where available. A success on an
older controller stays separate from a success on the current action. Captured
resets preserve native physics state, recorded nominal and applied controls,
and the all-product observer. A released legacy push can explicitly initialize
a new tool reference from its recorded nominal joints. The reset records this
choice before any physics steps. Isolated episodes earn only skill credit.
A captured case can declare `skill_sim_seconds` to run a bounded skill near
or beyond its source episode deadline. The reset report records the source
deadline and the local budget. The native clock and checkpoint stay intact.
Without an explicit `case_id`, the Gymnasium reset samples a lineage uniformly,
then one of its matching checkpoint references. Adding several boundaries from
one episode gives that lineage the same sampling weight as a single-boundary
episode. Keep lineages intact when selecting development and validation cases.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --inexact --extra curriculum
.venv/bin/python scripts/evaluate_skill_curriculum.py --gym --seed 9 --seed 18 \
  --output outputs/runs/curriculum/procedural-01
.venv/bin/python scripts/evaluate_skill_curriculum.py --gym \
  --manifest configs/curricula/development.json --artifact-root outputs \
  --output outputs/runs/curriculum/captured-01
```

Use `--case` to select one or more corpus entries. The supervisor freezes the
controller once per campaign and records every scheduled result, including
reset rejections, timeouts and unknown process outcomes. An archive checkout
needs the referenced checkpoint files under its artifact root. Captured cases
with different physical handoffs can have different legal action masks.

Use `--workers 4` to run independent cases concurrently on CPU. Each case
keeps its own native process, controller state, timeout, and output directory.
The report retains the scheduled order and updates as cases finish. The
per-case wall budget starts at launch. The default `--workers 1` runs serially.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_skill_curriculum.py --gym --skill push \
  --seed 9 --seed 18 --seed 3 --seed 12 --action recorded --agent-attempts 1 \
  --workers 4 --case-wall-seconds 120 --output outputs/runs/curriculum/push-parallel
```

`--agent-attempts 1` also records the pre-action observation, action mask,
and measured result for the skill-option dataset builder. Independent seeds
and captured handoffs can use this parallelism. Within a case each action,
recovery and physics step still depends on the previous native state. When
comparing worker counts, keep the source bundle, case schedule and thread
settings fixed and check the recorded outcomes. Concurrent load can affect
searches and episodes with wall-time budgets.

The four-seed native push batch took 83.4 s serially and 29.9 s with four
workers, a 2.79× throughput gain. Both runs verified three pushes. The fourth
reached the stocked-neighbor displacement guard. Their reset states, physics
traces and option transitions were bit-identical. The parallel result exported
four complete option transitions, including the failed push.
[Benchmark and parity witness](../outputs/runs/curriculum/parallel-push-v1/native/independent-parity.json)

For procedural push, the action mask admits `recorded` for the single-finger
fixture policy and `paired` for the paired-finger policy. Use the executed
action in each attempt when comparing outcomes. The agent can select another
admissible action when the requested preference is masked. Compare both
policies on the same reset seeds to collect paired outcomes.

The paired collection produced eight transitions from four identical starts,
with bit-identical pre-action observations for each pair. The single-finger
option verified three pushes. The paired-finger option exhausted contact
refinement on all four starts. This small batch covers procedural development
seeds and contributes zero fresh stocking credit.
[Paired option dataset](../outputs/runs/curriculum/parallel-push-v1/native/paired-option-dataset/manifest.json)

```python
from abstract_stack.skill_curriculum import StockingCurriculumEnv, load_curriculum

cases = load_curriculum("configs/curricula/development.json", artifact_root="outputs")
env = StockingCurriculumEnv("retreat", "outputs/runs/curriculum/local-retreat", cases=cases)
try:
    observation, info = env.reset(seed=9, options={"source": "recovery"})
    action = env.action_space.sample(mask=info["action_mask"])
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

The next evaluation should hold out whole episode lineages and procedural
geometry seeds, then check each primitive together with its preceding and
following handoffs. A single-action primitive can already serve as a reset,
benchmark and scripted teacher. Learning continuous corrections requires a
control-rate action contract with the same force and motion bounds.

### Agent retries and mid-motion starts

`--agent-attempts` gives the agent a bounded number of distinct admissible
options. Each retry restores the same native state and checks its digest.
The zero-credit ledger retains each failure and its artifacts. `--rewind`
allows one declared earlier checkpoint from the same skill and episode lineage
after the original options are exhausted or a loaded checkpoint fails measured
entry admission. The report preserves both outcomes. For example, the close
stop beside the shelf panel still carries hand contact. The agent records that
boundary as rejected and retries from the earlier empty-hand grasp checkpoint.

```bash
.venv/bin/python scripts/evaluate_skill_curriculum.py --gym --agent-attempts 4 \
  --rewind --manifest configs/curricula/development.json \
  --case lift-body-push-stop --output outputs/runs/curriculum/my-recovery
```

The original lift/body stop rejects both tested actions. Its pre-push rewind
can release the retained grasp, use the checked finger approach, and verify the
backstop gate. The fourth-carton case reaches contact refinement with the
single-finger option and still exhausts the refinement budget.

Use `replay_skill_cut.py --attempt <attempt-directory> --physics-step <n>
--output <checkpoint>` to materialize only the state you need. Step zero is the
original reset. Every physics step through the end is addressable for a
retained pick, place, push, slide or retreat. The command checks the clock and
terminal state before saving. `replay_skill_cut.py --resume <checkpoint>`
loads the cut and verifies the original suffix again. Arbitrary cuts preserve
native physics. A new closed-loop decision at an interior cut requires an
explicit controller initialization and measured skill preconditions.

Agent attempts save `decision.npz` before selecting actuator references and
`outcome.npz` after the skill. `scripts/build_skill_option_dataset.py` combines
completed transitions from one or more `--run` directories and keeps unfinished
decisions in the manifest. These data support a small policy that selects
primitive options from privileged observations and action masks. Use the
lineage field for splits so retries on one state stay together.

### Tuning a captured skill

Run the same case and action for each candidate, and pin the checkpoint hash,
initial native state digest and candidate controller configuration. The
`--control-config` option accepts the bounded settings in
`abstract_stack.tuning.CONTROL_BOUNDS`. It keeps the native model, exit gate,
abort limits and collision policy fixed. For a push, start with the normal
preload target, yaw correction gain or approach motion. Change one coordinate
at a time so the observed effect is attributable.

```bash
.venv/bin/python scripts/evaluate_skill_curriculum.py --gym --agent-attempts 1 \
  --manifest configs/curricula/development.json --skill push \
  --case fourth-carton-stall --action single --verify-retreat \
  --control-config configs/controllers/fingertip_advance_development.json \
  --output outputs/runs/curriculum/push-candidate
```

Retain every attempt and rank verified, incident-free completion first. Use
contact dwell, clearance, pose error and force history to diagnose failed
attempts and choose the next search direction. Compare time and peak loads
among successful candidates. Test a selected candidate on held-out lineages,
including the handoff into retreat, before promoting its settings.

Add `--verify-retreat` to a captured push to execute withdrawal from the live
push terminal state with the native gates. The attempt lists the skills that
verified, including a successful push followed by a failed retreat. The
controller overrides `insertion_advance_speed_m_s` and `base_speed_m_s` change
the motion references within the existing force feedback and motion guards.
Compare candidates across the whole handoff before choosing a speed based on
push duration.

On the two released-carton development starts, the 50 mm/s preset finished
push and retreat in 43.78 and 33.95 simulated seconds. The earlier 40 mm/s
preset with a yaw lever gain of 0.24 took 46.53 and 34.91 seconds on the same
starts. Raising advance to 60 mm/s failed the squaring gate on the tight-three
start. The presets remain opt-in pending a larger suite with varied carton
properties. See `outputs/runs/curriculum/push-finalists-v1/` for the frozen
candidate comparisons.

The 50 mm/s preset also passed a captured release-and-reapproach in 32.22
seconds versus 35.65 seconds with its recorded controller. On a ten-slot shelf,
paired fingertips completed the push and retreat of one carton beside an
already-stocked neighbor. The single-finger option failed its posture search
on that start. The paired trial earns only saved-skill credit.

For recovery search, `--rewind-action release-reapproach` changes the first
admissible action only on a declared earlier boundary. Keep `--action recorded`
for the first branch to test a healthy grasped slide without spending time on a
tool change. Both branches retain their attempts, failures and simulated time.

For an option selector, collect repeated alternatives from the same reset with
measured outcomes and legal-action masks. For control-rate learning, record
pre-action relative poses and velocities, contact forces and torques,
tracking residuals, joint and actuator limits, and the nominal commands before
compensation. Vary mass, friction, clearances, and observation error across
episodes. Keep domain parameters that would be unavailable to the deployed
robot in the teacher context.

## Inspecting the scripted agent

Full stocking runs and Gymnasium runs with `--agent-attempts` write an
append-only `agent-history.jsonl`. The journal records the scheduled cartons,
selection candidates, chosen carton and protocol, skill and native phase,
measured gate outcomes, and checkpoint paths. Explicit task plans also record
their entry checks and exit gates. Fused controllers identify skill progress
from their native phases and use the existing per-carton measurements for
success. Large force and IK traces stay in their existing artifacts.

```bash
.venv/bin/python scripts/inspect_stocking_agent.py outputs/runs/agent-harness/three-cartons-v1
.venv/bin/python scripts/inspect_stocking_agent.py \
  outputs/runs/agent-harness/saved-push-recovery-v1/lift-body-push-stop --json
```

Use `--events` for all journal records in their original order. The inspector
can read a rollout while it runs. A stopped process retains its last complete
record and the supervisor's exit status. A rewind starts a distinct branch
with its own checkpoint, attempted actions, and outcome. Every new stocking
checkpoint references the corresponding journal sequence through its event
boundary.

The scripted recovery runner checks the current action mask, retains each
attempt, and tries a distinct admissible action before taking a declared
rewind. Its history records the available and previously tried actions so a
later policy can make the same decision from the same evidence.

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
composes the five skills
through `run_plan`; the grasped supported-slide protocol keeps its existing
fused path. The plan is always recorded in `summary.json` under `task_plan`,
with each phase event annotated with its `skill`. Success criteria, force
guards and tolerances are unchanged.

The [selection guide](#decide-the-next-motion-from-measured-preconditions)
also covers the callable recovery and between-carton skills. The active-carton
side press is available through the checkpoint probe. The row press has a
separate between-carton entry point and appears as `slide` in the Gymnasium
curriculum. The opt-in live side-square policy selects the active-carton
press from measured clearance.

Agents can read the catalog before motion:

```python
from abstract_stack.skill_library import describe_stocking_skills

catalog = describe_stocking_skills()
for skill in catalog["skills"]:
    print(skill["name"], skill["description"])
    print(skill["preconditions"], skill["postconditions"])
```

`plan_order` lists the five per-carton steps. Each descriptor includes
its entry point or a `run_plan` adapter and its upstream dependencies.
`callable` identifies adapters accepted by the Python task plan. Other
entries name their controller, probe or curriculum entry point.
`preconditions` and `postconditions` are strings for choosing and explaining
actions. The measured controller checks and journaled outcomes determine
whether a particular action is admitted or succeeds. Completed stocking runs
save this same catalog in `summary.json` under `skill_catalog`.

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

New live skill entries record the head controller's command, desired angles
and next update time. A continuation restores that state when head tracking
is enabled. Older probes serialize native neck targets and hold them during
continuation. Those targets can differ from a live rollout's commands even
when both begin at a bit-identical native state.

### Decide the next motion from measured preconditions

| Skill | Use when | What to verify |
| --- | --- | --- |
| Base relocate or turn | The inventory, shelf or contact patch needs a different workspace | Admitted base route or swept turn, feasible next arm posture, and stable payload and shelf contents |
| Pick and hold | The hand is empty and a checked approach reaches a supply carton | Opposing contact, actual lift and retained hold |
| Transfer and seat | The robot has a verified grasp and a checked free landing | Retained load through transfer, then upright shelf support |
| Supported release and registration | The carton is shelf-supported and the narrow pusher needs access | Stable landing and a cleared, unloaded grasp hand |
| Side press | A released carton can be reached from the exposed side. Use the center to reduce off-axis torque or shift at the front patch to open room for a square | Signed displacement survives unload, with support and neighboring placements preserved |
| Front square | The released carton needs less yaw for its projected footprint to fit | Depth, support, clearance and yaw pass after the pusher unloads |
| Sidewall registration | A wall-adjacent slot affords a checked side approach and a wall reaction for alignment | Wall reaction and alignment, then clearance for the narrow forward tool |
| Forward insert and push | The next depth segment fits. Use a supported grasp in a spacious route or a released single/paired fingertip in a tight slot. Guided entry is available beside a neighbor or wall | Signed depth progress, supported staging, and separate evidence of back contact when required |
| Breakaway or recontact | Depth progress stalls under load, or a released carton loses finger contact before reaching its goal | Renewed advance under the same bounds. Recontact is available in the opt-in receding controller |
| Contact refine | The supported carton is near its final depth and still needs yaw or lateral correction | Measured backstop contact and squared dwell where required, with a checked release |
| Retreat or unloaded clear | The carton already satisfies the relevant placement gate | Check the hand posture and exit under native force guards. Verify support and preserve the row through the exit. An unloaded clear grants retreat credit only |
| Row compaction (`slide`) | The partial row has internal gaps and too little usable opening for the remaining widths | Measured carton-pair contacts during press, usable opening after unloading, and stable supported cartons |
| Strict row finish (`row-press`) | Compaction yielded the opening, and yaw or projected overlap still needs correction | Required alignment, envelope and contact gates, followed by unloaded stability |
| Stop and snapshot | No candidate is admitted, a guard trips, or bounded recovery makes no progress | Retained failure with the phase, measured state and inputs for another attempt |

Select row pressure from measured motion and contact. Increasing the target
is a candidate when a checked path has room to advance, the intended contact
chain is loaded, and signed measured force is near the current target while
the row remains supported. First change the workspace, contact patch or
approach if the hand cannot reach, the contact chain is incomplete, or the
reference is saturated. Keep force aborts fixed, including the combined
robot–product load and summed per-pair carton and wall normals. A force
setpoint specifies the admittance controller's target. Verify actual force
and usable opening after unloading. See the
[side-press traces and whole-shelf outcomes](reports/whole_shelf_development.md#lateral-alignment-from-saved-release-states).

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
`supported-release`, `side-registration`, `push-to-contact`, `row-press`, and `retreat`. Use
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

For the opt-in lateral insertion probe, unload before squaring and verify
the displacement on the carton bottom. The center can move while the
carton rocks and return when the hand opens. The `side_slide_unloaded`
checkpoint records a retained shift after the tool exits. Its resume
path performs a bounded square ahead of the occupied row and checks
yaw and clearance again before advancing. This is a distinct staging
boundary with zero stocking credit.

Use the exposed side of a supported, released carton for an alignment
press. `--side-press centered` contacts the middle of the side face and
`--side-press corner` uses the existing front patch. Select the signed
distance from measured clearance. Both verify that the carton retained its
shift after the hand unloaded.

```bash
.venv/bin/python scripts/probe_front_release.py \
  --checkpoint "$released_entry" --output outputs/side-staging \
  --insert-only --record-skill-starts --side-press centered \
  --side-offset-m=0.003 --stop-after registration

.venv/bin/python scripts/probe_front_release.py \
  --checkpoint "$released_entry" --output outputs/side-and-square \
  --insert-only --record-skill-starts --side-press corner \
  --side-offset-m=-0.004 --interleave-side-square --base-turn-deg=-60 \
  --guided-wall-entry --wall-seconds 235
```

The composed probe parks the fingers and checks the return base turn
before squaring. In-place squaring holds a measured 0.10° yaw goal under
load to allow for drift during unloading, then verifies the released
carton at 0.15° before advancing. Use the saved `side_slide_unloaded`
checkpoint with `--resume-side-square` to evaluate the front sequence
from that handoff. Choose the press direction and base turn for the
available side clearance. For a supported row, use the row-compaction
environment and verify measured open span after unloading.

`--side-square-recovery` enables clearance-based dispatch in the live
positive-wall policy. It requires `--guided-wall-entry`, paired fingertips
and base motion. The selector reobserves the released pose, projects the
measured carton footprint, and computes the shift needed away from the
occupied side. It reserves 2.5 mm beyond the measured shortfall and caps
the shift at 10 mm. It also checks shelf support, arm load, wall clearance,
and room for the short square ahead of the row. An eligible shift uses
the checked side press, return turn and front square before forward
insertion. It selects the side center when the front patch would add
torque in the direction of the observed yaw. The side tool's nominal
60° approach turn applies to the tested
positive-wall layout. Physical approach, displacement, yaw, and support
still have independent gates.

`--release-pose-feedback` keeps bounded position and yaw feedback active
while the fingers unload and open on the shelf. The continuous opening
schedule and coupled finger, force and support checks still apply. Arm
correction stops when measured hand contact falls below 0.2 N. Start a
matched release trial from a `front_release` checkpoint with
`probe_front_release.py --insert --stop-after release --release-pose-feedback`.
Verify the unloaded pose and test the following insertion and retreat.

The [early registration profile](../configs/controllers/early_registration_development.json)
starts returning from the open-side buffer before the carton reaches the
backstop. Its 25 mm gap is a motion reference for the start of aligned
advance. The planner screens the proposed fingertip and arm geometry and
the native controller checks each applied motion. Pair this profile with
`--balance-contact-alignment` to rank bounded finishing prefixes by their
combined lateral and yaw error, scaled by the corresponding measured gates.
After contact, the live dwell and release still determine the result.

`--screen-supported-unload` checks candidate unloads beside stocked
neighbors in independent native states. It compares paired-plane yaw
correction with contact-lever feedback when the first candidate fails
admission or loses strict yaw during unloading. The forecast covers the
unloading prefix. The live guarded withdrawal verifies the subsequent exit
and records the released pose after the hand clears.

The [development corpus](../configs/curricula/development.json) includes
the seed-study front-square stops and their earlier starts, plus subsequent
turn, square, unload and contact-refinement failures. For a side-unloaded
entry, Gymnasium `push` exposes its `recorded` action. It starts with the
short square and then runs the existing push gate. If the exact stop fails
the measured reset gate, use its declared rewind or the matching native
probe. Keep the stop and the earlier handoff in the same episode lineage.

The experimental `--center-last` order packs alternately from the two
ends. It reserves measured space for unplaced widths and targets the
outer boundary until both immediate neighbors are present. The
center-slot advance checks the carton footprint against both neighbors.
Use the matched center-last supply fixture if comparing the order without
an early far-side inventory reach. The stock order requires paired
insertion and a single dense row.

`probe_front_release.py --bimanual-clear` moves both unloaded palms from
a saved stocking milestone through one stationary shared-lift controller.
It checks coupled geometry and measures both palm poses while monitoring
every product. It is an empty-hand clearance primitive and earns zero
stocking credit.
