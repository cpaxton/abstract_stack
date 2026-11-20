# Dense shelf insertion

This experimental task shares **10 mm or 5 mm total unused lateral width** over
an entire shelf row, including both wall gaps. Ten cartons occupy about half a
metre for seed 0. Initial targets divide this budget over eleven gaps; these are
initial geometric targets, not ten independent allowances.

```bash
abstract-stack dense-scene --seed 0 --objects 10 --initially-stocked 9 \
  --free-space 0.01 --output outputs/dense-10mm.json
abstract-stack build-scene --config outputs/dense-10mm.json --render \
  --output-dir outputs/dense-preview
abstract-stack solve --config outputs/dense-10mm.json \
  --output-dir outputs/dense-insertion
# Once evaluating video, add:
# --video --third-person-camera dense_front --fps 10
```

Use `--free-space 0.005` for the harder budget. Use `--initially-stocked 0` for
the full task. Nine prefilled cartons plus one insertion is a curriculum, not a
ten-object stocking success. Prefilled products remain under physics-step
tip/drop monitoring and receive no manipulation stage credit. Seeds determine
product dimensions, shelf depth/height and supply jitter; pass the same seed to
generation and execution when changing the default seed.

Batch campaigns declare the same curriculum up front, preserve process exits
and unknown outcomes, and keep each fresh scene in its own directory:

```bash
abstract-stack batch --seeds 0 1 2 --objects 10 --initially-stocked 9 \
  --free-space 0.005 --episode-timeout 300 --output-dir outputs/dense-campaign
```

The original `batch` command without `--free-space` retains its spacious task.

## Scene and control assumptions

The scene provides a coplanar transfer table flush with the shelf opening.
This is an explicit fixture aid. Reset-only shoulder/elbow parking avoids
initial collisions between the unmodified native robot and the denser fixtures.
The initial base is explicitly 4 cm closer to the shelf than the spacious bench;
the table legs are inset to clear the chassis. Exported XML includes a first
`scene_start` keyframe with the configured parked pose and native stationary
references; reset to keyframe 0 when loading it directly. Native joint references,
geometry, inertias, collision masks and gains remain unchanged.

The script picks a carton, aligns it on the empty shelf front using measured pose, unloads
the grasp, opens in place, withdraws, and plans a selected-finger push. It
searches bounded tilt, wrist-posture and approach alternatives with scratch
data. The supported-slide prediction translates the carton along -X after
closing the measured front-face gap, keeping its current Y/Z/orientation. This
forecast stops carton translation at geometric backstop contact while the tool
can take preload. This assumption checks predicted collisions; it is not an attachment or a dynamics
guarantee. Actual motion comes exclusively from native actuator targets and
MuJoCo contact. Contact-force feedback and measured wrist-load compensation
remain bounded and logged. Carton/carton contact has the same 25 N abort as the
robot/product guard.

## Acceptance and current status

Success requires the physical shelf goal, completed pick/place/final-position/
retreat evidence for **every scheduled manipulation**, and no transient tip or
drop. Final projected carton envelopes must fit the whole row for the configured
stability dwell: no individual negative gap or sum of negative gaps may exceed
0.2 mm. Numerical overlap is not counted as free space. Bounds and force aborts
are fixed while refining control.

Dense insertion is under development. Earlier tests exposed invalid fixture
clearance, grasp-release drift, wrist deflection and yaw-induced jams. A fresh
ten-carton/10 mm trial with nine prefilled products advanced the active carton
about 15 cm, then timed out before back contact. Subsequent 10 mm and 5 mm trials
completed all four stages for the single scheduled insertion, with no transient
tip/drop or final projected overlap. These are development cases, not a frozen
multi-scene success-rate study. See the [development report](reports/dense_insertion_followup.md).
The seed-0 two-insertion trial also completed all stages for both scheduled
cartons. Three fresh 5 mm scenes subsequently failed at transfer, pregrasp force,
and insertion dwell: **0/3** complete scenes under the preceding source
fingerprint. All three episode fingerprints match their campaign source;
the subsequent backstop-scope refinement is recorded separately. All failures
remain in the report. The table spans several controller variants and must not
be combined into an overall frozen-policy rate.

The early all-ten-from-table trial failed its first transfer IK filter, after one
verified pick. Later empty-shelf development reached eight verified withdrawals
with the 20 mm fixture. Full strict stocking and robustness across fresh scenes
remain open.
The separate spacious-shelf 40/40 result does not establish dense success.
Further tests stage on the shelf
front to shorten the ungrasped push and replan wrist posture when a corrected
target fails the unchanged IK filter. Initial base placement is explicit scene
configuration. The recovery profile below uses guarded native base motion.

![Native MABEL releasing a carton in the dense row](images/dense_insertion.png)

The 10 mm seed-0 recording is at
`outputs/dense-ten-10mm-video-v1/videos/{third_person,first_person}.mp4`.
Both H.264 views are 640×480 at 10 fps with 774 encoded frames: 764 physics
frames plus 10 final paused frames. They render the same native physics states;
samples and actuator controls match the successful headless trial bit for bit.
The preview alone is not a success test.

Native process stability is also unresolved: a serial pre-push run segfaulted
during the existing spacious two-carton regression, after a full 139-test pass.
Its model/start state, unknown outcome and traceback excerpt are retained in
`outputs/native-crashes/dense-prepush-abd1297/`. See local issue SIM-001 in
[home_robot_issues.md](../home_robot_issues.md). A successful retry does not
replace the failed process or explain its cause.

Full traces, rejected plans, source fingerprints and failed outcomes stay in
ignored `outputs/`. Saved-state continuation probes are development diagnostics
and cannot replace fresh episodes in a success-rate denominator.

## Fresh ten-carton recovery profile

The `--dense-recovery` profile uses ordered released insertion, guarded
withdrawal and native base motion. A compensated geometry stop can trigger
a checked unload and up to two reapproaches. The retry selects a new standoff
or a folded single-finger posture locally. After a side press, the stance
planner checks the native base sweep and prospective tool approach together.
If the short square hands off a supported carton with residual yaw, the forward
push uses measured lateral registration before the carton overlaps its
neighbors. The post-square handoff applies both during uninterrupted execution
and when resuming the saved motion.

The pickup planner can move an empty robot to an aisle stance after a swept
pregrasp failure. It checks the base sweep, approach and descent before moving.
Loaded shelf approaches use the measured grasp transform to check candidate
stances and the complete approach, transfer and seat. If the nominal route
fails, the planner can use a landing shifted toward the center in the verified
free front band, then reobserve the carton after opening and clearing the hand.
A relocated empty pusher also checks supply-table clearance along its
gravity-compensated approach.

A provisional placement requires board support, a cleared and verified
retreat, yaw within 1°, enough measured room for the remaining cartons, and
screened approaches for finishing and for the next carton. If capacity
is exhausted, the planner can press the exposed end of a supported group
toward its wall and measure the opening again. Deferred placements are
finished before the final row dwell. The strict stage still requires a
measured back-contact witness and released yaw within 0.15°.

```bash
PYTHONMALLOC=malloc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/run_stocking_case.py --seed 5 \
  --config configs/scenes/tight_ten_cartons_20mm.json \
  --output outputs/runs/agent-harness/my-dense-recovery \
  --wall-seconds 1800 --idle-seconds 150 -- \
  --grasp-skill --base-reposition --ungrasped-insertion --paired-finger-insertion \
  --wall-packing --live-guarded-withdrawal --guided-wall-entry --plan-driven \
  --record-skill-starts --screen-pusher-approach --side-square-recovery \
  --release-pose-feedback --balance-contact-alignment --screen-supported-unload \
  --dense-recovery --control-config configs/controllers/early_registration_development.json
```

For saved-state development, `scripts/probe_front_release.py --insert-only
--dense-recovery --strict-finish` checks the complete insertion and strict
withdrawal from an authenticated skill entry. The saved native states and
their live fresh episodes have separate ledgers. Use `--withdrawal --strict-finish`
with a saved withdrawal-planning entry to exercise only the guarded exit.
Near the end of a long rollout, `--skill-sim-seconds` assigns a local
simulation budget measured from the saved native clock. The continuation
reports both budgets and zero fresh-task credit.

Add `--defer-final-alignment` to stock the entire shelf before scheduling
fine alignment. A packed-first run requires all scheduled picks, shelf
support and withdrawals, the original measured envelope and packed dwell,
and zero tip/drop incidents. Its summary reports `packed_success`, the strict
grade and the names of cartons still needing alignment.

When an arm withdrawal reaches its measured corridor limit after unloading,
the packed-first controller can replan from that state. If the measured hand
already clears the shelf, the robot screens a short chassis retreat and
monitors product displacement, contact and support while moving. The shelf
clearance and row support must still pass after the retreat. The controller
can also use this recovery after the applied motion check rejects an arm
reference on step size alone. The rejected reference is never applied.

For the final wall slot the packed-first controller tries a checked shorter
stow above the carton, then approaches diagonally from a screened base stance.
It can combine the side withdrawal and lift when a single coupled Cartesian
path passes. During the final-carton sequence the idle right arm can skip a
repeated stow wait when it holds the current stow reference and its measured
posture has settled.

If the measured pose rejects a planned turn before rotation starts, the
workspace planner can screen one more stance from the actual base pose. It
reserves two extra millimetres of predictive clearance, then applies the
original live turn and force guards. A force abort remains terminal.

Use a separate recentering probe when a shallow, unloaded carton still misses
its reserved slot after squaring. The carton must have at least 12 mm of
free depth ahead of the occupied row. The measured footprint must fit after
the side press, unloading and turn back toward the shelf.
The `front_square_deferred` checkpoint records this measured handoff separately
from the earlier grasp release, so it can be replayed directly.

```bash
.venv/bin/python scripts/probe_front_release.py \
  --checkpoint "$DEFERRED_SQUARE_CHECKPOINT" --output outputs/my-front-recenter \
  --insert-only --dense-recovery --recenter-front --stop-after registration \
  --record-skill-starts
```

With skill checkpoints enabled, a completed packed dwell saves a
`packed_row_ready` entry. Use it, or a previously saved complete-row stocking
milestone, to run corrective passes and recheck the packed dwell and strict
poses:

```bash
.venv/bin/python scripts/probe_front_release.py \
  --checkpoint "$PACKED_ROW_CHECKPOINT" --output outputs/my-row-correction \
  --dense-recovery --finish-row --record-skill-starts
```

Use `--align-product carton_0` to limit the correction to a named carton.
Repeat the flag to select several cartons. The row still passes the full
packed hold before reporting the outcome of the requested corrections.

See the [empty-shelf results](reports/whole_shelf_development.md#dense-recovery-from-an-empty-shelf)
for the 20 mm development trials and remaining gates.

For cardboard deformation, see the separate [material investigation](cardboard_simulation.md).
Its panel indentation probes do not establish deformable grasping or insertion.
