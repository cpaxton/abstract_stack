# Whole-shelf development

Target: ten cartons from the supply table, zero prefills, one fresh native MABEL
episode, and **10 mm total free row width including both walls**. The 5 mm row,
observation/actuation noise and private held-out validation remain later gates.
The [development ledger](whole_shelf_trials.md) and
[machine-readable evidence](whole_shelf_trials.json) retain scheduled objects,
failures, unknown processes, controller inputs and exact source identities.
These successive development policies do not constitute a frozen success rate.

The earlier [three-carton baseline](adjacent_insertion_development.md) completes
all stages with spacious walls. `tight-four-13` picks and supports four cartons
and completes three insert/release/retreat cycles. The best retained fresh ten-carton trial, `tight-ten-11`, completes two stocks
and stops at the next inventory navigation. Later trials expose release,
registration and passive contact drift. Their outcomes remain unchanged.

## Videos and completed three-carton baseline

`tight-three-14` completed all three picks, shelf placements, final positions,
and retreats in one fresh episode, with zero prefills and zero tip/drop
incidents. It used 3 mm target gaps between cartons and extra room at the walls.
Both videos replay the recorded actuator commands and reproduce the terminal
MuJoCo integration state bit for bit.

- [Third-person video](../../outputs/adjacent-insertion-v1/tight-three-14-video-gpu/videos/third_person.mp4)
- [First-person video](../../outputs/adjacent-insertion-v1/tight-three-14-video-gpu/videos/first_person.mp4)
- [Replay verification](../../outputs/adjacent-insertion-v1/tight-three-14-video-gpu/result.json)

The ten-carton run `tight-ten-20` started with an empty shelf and 10 mm total
free row width. It picked and placed two cartons and verified final positioning
and retreat for one. Carton eight passed front squaring, then stopped advancing
about 70 mm short of the backstop. The push timed out with zero recorded
tip/drop incidents.

## Edge-slot skill

`ten-edge-transfer-11` restores the failed second transfer, then physically
executes an interior front handoff, about 49 mm of sideways registration against
the sidewall, unloading/retreat, an outer-finger push, squared backstop contact,
and full shelf exit. Final yaw is **-0.07586 degrees**, with zero recorded
tip/drop incidents. This continuation earns **zero fresh-task credit**.

The sidewall skill measures about **0.837 N** wall normal force before unloading.
Its final lateral error from the wall registration goal is **0.0849 mm**; that
intermediate goal leaves the canonical task target unchanged. The final native
withdrawal clears 218 mm horizontally and 20.8 mm vertically. The model's
geometry, inertias, gains, limits and contact masks are unchanged.

The control changes address observed physical limits:

- Check empty-hand transit around the roof, rather than treating a reachable
  pregrasp endpoint as a complete route. Keep the folded pusher until shelf exit.
- Unload a wrapping grasp in the front interior, where it fits. Use the existing
  exposed-side skill to reach an edge slot and the wall to supply squaring contact.
- Fold unused fingers without self-collision; use the native outer finger near
  a wall. Both commanded preshapes and predicted geometry include that posture.
- Reserve wrist reference headroom: a 0.6 rad reference leaves too little room
  below the native 0.6736 rad actuator target ceiling. Do not change that ceiling.
- Add selected-finger gravity/contact compensation through native position
  targets, bounded to 0.15 Nm and 0.15 rad feedforward offset at 0.5 rad/s.
  Measured loaded MCP error decreases from about 0.09 rad to about 0.0004 rad.
  Nominal preshapes, applied targets, rate transitions and measured joints remain
  separate, with final coupled geometry checks before every command.
- Distinguish backstop force from total hand force. Bounded measured support
  resistance feedforward prevents board friction from consuming the backstop
  setpoint. Complete native release forecasts include the whole shelf exit.

All 25 N guards, collision admission, native contact dwell and final task/row
criteria remain fixed. The 0.15 degree contact/release gate is unchanged. A
tighter controller reference may return a bounded square staging result only
when the native contact dwell passes; full guarded release forecasting and fresh
live verification remain required. Failed local planning does not establish
global unreachability.

## Interior registration and fresh episodes

The best fresh ten-carton episode so far, `tight-ten-11`, completes two cartons
and stops during the third pickup's inventory navigation. A stopped-pose latch
passes that retained route with all five waypoints inside the original 2 mm
bound; its final measured error is 1.33 mm. That recovery earns no fresh credit.
`tight-ten-12` exposes another wrist posture's release and lateral sensitivity.

General selected-finger load compensation now follows the two active MCP/PIP
pairs through insertion, native lookahead and complete withdrawal. The first
probe, `ten-finger-load-insert-01`, releases square but remains 12.03 mm sideways
from the canonical slot. A bounded midpoint integral and native lateral forecast
ranking reduce that error to **0.179 mm** in `ten-finger-load-insert-02`: final yaw
**-0.0265 degrees**, full withdrawal, and no recorded incidents. Both restore
one entry and earn zero fresh-task credit. Their source IDs differ.

The corresponding wall probe, `ten-finger-load-edge-01`, exhausts its squaring
search and remains a failure. Inspection shows a reference/ranking mismatch:
its single-pad reference controls the yaw lever, but its optimizer ranks lateral
translation. Only the two-pad controller should switch to lateral ranking;
wall-side refinement retains the square-backstop objective. Fresh complete-shelf
acceptance remains open.

## Wall-first assembly

The active protocol opts in with `--wall-packing`. It selects the positive-wall
carton first and then works inward, regenerating the grasp and navigation plan
from measured state each time. Interior contact goals use the measured occupied
neighbor's envelope. The remaining unstocked widths must still fit between the
goal and the opposite wall. Canonical task targets stay unchanged; complete
shelf success still uses the original placement, whole-row overlap, support,
stability and no-knock-over checks. Controller goals may form a contacting row;
10 mm free width is a whole-row budget, not a mandatory gap between each pair.

`ten-wall-first-opening-01` recovers the first wall carton from `tight-ten-17`'s
front-release entry using the original contact model: measured release, lateral
wall registration, outer-finger insertion, squared backstop and full withdrawal.
Its final yaw is **-0.02650 degrees**, with no recorded incidents. This is zero
fresh-task credit. The previous opening retained 0.2097 N hand contact, above the
unchanged 0.2 N gate. A checked 5 mm departure clears pressure before verifying
release. Neither finger opening nor a fixed waiting period counts as release.

Lateral admittance now projects its reference inside the existing X/Y/Z motion
corridor before IK, retaining final checks after nominal projection. It does not
expand the corridor. A projection-only probe on the alternative fixture model
still fails its displacement/support dwell and remains in the ledger.

## Numerical-contact ablations

An exact actuator replay of `tight-ten-13` shows its first interior carton rotates
from roughly zero to 5.3 degrees during the next cycle without any robot contact.
`scripts/diagnose_product_contacts.py` attributes native contacts on a replay and
requires a bit-equal terminal integration state when parameters are unchanged.
Explicit counterfactual overrides earn zero fresh credit and cannot claim an
exact reproduction of the original episode.

The original MABEL model already uses three NoSlip iterations. Thirty reduces
rotation over the same approximately 110 seconds to about one degree; adding
impratio 100 reduces it further but does not establish sticking. A 5 ms procedural
contact time constant with impratio 10 settles to a fixed pose. That broader
recipe also changes hand/carton interaction and does not pass the first fresh
insertion. It is retained in `tight_ten_cartons_rigid_contact.json` as an ablation.

A narrower `rigid_fixture_contacts` recipe changes only explicitly named
carton/shelf pairs to 5 ms, impedance `.99 .999 .001 .5 2`, with the original
friction coefficients and contact dimensions. Native geoms, inertias, actuators,
masks, hand/carton contacts and global solver options stay unchanged. The
corresponding unforced probe settles to a fixed pose; fresh manipulation with
this recipe remains unsuccessful. `tight_ten_cartons_fixture_contacts.json`
retains that experiment. **The current wall-first trial uses the original scene
and contact model**, relying on the environment to constrain the packed row.

These are numerical/material fixtures, not stocking successes. Parameters and
source identities remain in the raw records. The recipes follow MuJoCo's
[contact-model guidance](https://mujoco.readthedocs.io/en/stable/modeling.html)
and [NoSlip option documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-noslip-iterations).

## Reproduction and validation

```bash
PYTHONDONTWRITEBYTECODE=1 MABEL_ROOT="$PWD/third_party/open_mabel" \
  flock /tmp/abstract-stack-render.lock .venv/bin/python scripts/run_stocking_case.py \
  --config configs/scenes/tight_ten_cartons.json \
  --output outputs/my-fresh-whole-shelf --wall-seconds 1800 -- \
  --grasp-skill --base-reposition --ungrasped-insertion \
  --paired-finger-insertion --wall-packing \
  --control-config configs/controllers/adjacent_insertion.json \
  --record-skill-starts
```

The scene declares a 1,800-second simulation limit and this command declares a
1,800-second wall budget; earlier trials used shorter declared budgets. Time is
reported, not hidden by checkpoint retries. `front_insertion` snapshots preserve
the post-registration state, nominal arm/tool settings, applied finger tracking
state and observer history. `probe_front_release.py --insert-only` restores that
entry; `--record-skill-starts` can capture entries during a zero-credit probe.

A focused scene/insertion/lookahead/withdrawal/navigation suite passes **58
tests**, including native fixture settling and preservation of the MABEL subtree.
This check preceded the last opening/separation change; final checks remain
required. A full frozen suite at `fcee9d0` reached 95% progress before a
native signal 11 while constructing primitive MuJoCo data. A later expanded
process also failed during native model construction; subsequent focused passes
do not replace either failure. Logs/source records remain under
`outputs/adjacent-insertion-v1/validation-fcee9d0/` and
`native-validation-20261118/`. Native process stability and full-shelf validation
remain open. No upstream issue or message has been sent.

Checkpoint-model cleanup preserves every path and checksum: SHA-256 verified
hardlinks reclaim 11.59 GiB of identical binary models. The audit is retained at
`outputs/adjacent-insertion-v1/model-dedup-20261118/audit.json`; no recording,
failed result, checkpoint state or manifest was removed.

The second-carton recovery from `tight-ten-18` validates the loaded route
through transfer and seating. A paired push squares the released carton before
its swept volume overlaps the stocked neighbor. A single finger then failed
the final advance because the carton rotated into the neighbor. In
`ten-front-square-09`, the paired pusher keeps its midpoint 1 mm toward the
unoccupied side during advance, then uses the backstop to finish registration.
The measured release and full retreat pass at -0.0393 degrees yaw, with no
incidents. This checkpoint continuation earns zero fresh-task credit.

Fresh trial `tight-ten-19` completed carton nine and picked, transferred,
and seated carton eight. During its front squaring, the native geometry
check rejected a load correction predicting 2.07 mm of self-overlap between
the middle and ring proximal links. The controller now checks smaller
corrections between the previous applied finger targets and the new proposal,
keeping the arm command, existing collision criteria, and rate bounds.
Continuation `ten-front-square-10` from that trial's released entry finished
carton eight with verified withdrawal at 0.0281 degrees yaw and no incidents.
It earns zero fresh credit. Trial `tight-ten-20` uses the updated controller.

The partial ten-carton run has synchronized [third-person footage](../../outputs/adjacent-insertion-v1/tight-ten-20-video-gpu/videos/third_person.mp4) and [first-person footage](../../outputs/adjacent-insertion-v1/tight-ten-20-video-gpu/videos/first_person.mp4).
The [replay record](../../outputs/adjacent-insertion-v1/tight-ten-20-video-gpu/result.json)
verifies a bit-for-bit match to the terminal integration state.

After front squaring, carton eight advanced about 11 mm, then stayed near
X = -0.6554 m for about 40 seconds under contact. A checkpoint retry
(`ten-front-square-11`) raised the push setpoint from 2.5 N to 3.5 N and
completed measured backstop contact and full retreat with 0.0101 degrees of
final yaw and zero incidents. It earns zero fresh credit. The next controller
tries 3.5 N once during final paired insertion when sustained measured push
force exceeds 1 N and the carton advances less than 1 mm in one simulated
second. Backstop force control and the 25 N guard remain in place.

The first checkpoint regression with the stall detector
(`ten-front-square-breakaway-01`) completed withdrawal at 0.0281 degrees yaw
with zero incidents. The carton advanced at the original setpoint throughout
this retry, so the higher-force branch did not fire. Fresh trial
`tight-ten-21` tests whether the stall recurs and triggers the new branch.

### What the ten-carton video shows

- At simulation time 133.9 s, the first carton (`carton_9`, red) advances
  to the backstop. It starts verified release and retreat at 146.05 s.
- At 278.01 s, the second carton (`carton_8`, yellow) starts a short push to
  square it before its footprint overlaps the one stocked neighbor. The hand backs away to set up the
  final insertion.
- At 293.53 s, the yellow carton starts its final push toward the same backstop.
  It stalls roughly 70 mm short. The controller ends the episode at 338.53 s
  without authorizing final release or retreat.

The progress guard now allows at most three seconds below 1 mm of forward
progress once push contact has been established, with a fresh three-second
window when the breakaway setpoint is selected. A momentary loss of push force
does not restart the clock. The failed outcome records the carton, remaining
backstop gap, measured force, setpoint and whether breakaway was attempted.

Carton eight faced one stocked neighbor and open space on its other side. Its
pre-squaring move was a conservative response to the measured 4.83 degree yaw,
not a requirement imposed by a slot bounded on both sides. Future planning
should select pre-squaring when the measured swept footprint or hand clearance
requires it.

Backstop contact is the current benchmark's temporary squaring reference.
The next placement policy should target a measured depth with clearance behind
the carton and verify support, pose, stability and safe retreat there. It can
use a backstop or neighboring carton when contact helps registration.

### Compliant pushing across object masses

The stalled carton remained upright and supported. The controller was already
using measured-force admittance: forward reference velocity decreased as
measured push force approached its 2.5 N target. With almost no carton motion,
the controller could settle near that force without overcoming resistance.
A 3.5 N checkpoint retry succeeded under the original safety guards. That
suggests insufficient net forward force in that contact state. It does not
isolate static friction, pad loading, or neighbor contact as the cause, and
3.5 N is not a material parameter or a universal recipe.

The skill objective should be measured carton advance to the requested pose,
with a slow contact approach and force feedback governing how the hand moves.
Detect progress over a short window. If the carton remains supported and the
wrench indicates a clean front-face push, try a small bounded preload increase.
If the wrench indicates downward load, side load, or growing yaw, unload and
re-contact with a different height or offset. Use a side push to register
lateral pose and a separate square or pivot skill when orientation needs
correction. Stop when the bounded recoveries cease making progress. Keep force
guards and release checks fixed.

For mass and friction variation, measure distance advanced, peak wrench,
load at first sustained motion, support, and recovery outcome. Compare the
same motion-based policy across declared mass and friction settings, using
separate development and held-out scenes. Target insertion depth directly
when leaving clearance behind the carton.

Fresh trial `tight-ten-21` triggered the 3.5 N breakaway on carton eight and
completed its retreat. Across ten scheduled cartons it verified three picks,
three shelf placements, two final positions and two retreats. The third carton
(`carton_7`) reached release planning. All forecast candidates failed. The
first predicted a 2.30 mm overlap between a ring-finger collision geometry
and the already-stocked carton eight on the first unload tick. This is a
withdrawal-planning case, distinct from a loaded push stall. The opt-in
checkpoint catalog now records both.

For the carton-three exit, wall packing now admits a narrowly scoped distal
finger contact with a named stocked neighbor in both native forecast and live
withdrawal. The scratch admission is limited to 3 mm predicted overlap.
The native monitor still guards loads and tip/drop on every step, and the
withdrawal aborts if a named neighbor translates more than 2 mm from its
entry position. Final placement and release gates still apply. Shoulder,
fixture, self and unnamed contacts retain their existing checks.

Withdrawal tests smaller measured-load corrections before rejecting a
compensated command. A sustained measured square dwell can also advance to
a full withdrawal forecast before the tighter yaw reference destabilizes
contact. Neither staging nor a forecast earns task credit.

The revised carton-three checkpoint trial
`ten-third-neighbor-contact-02` passed native release and full retreat, ending
at -0.00039 degrees yaw with zero incidents and 5.60 N peak summed robot/product
normal force during withdrawal. This trial combines early forecast handoff,
bounded compensation search and named-neighbor admission, so it does not
isolate which change was necessary. It earns zero fresh credit. The fresh
ten-carton validation is `tight-ten-22`.

Fresh trial `tight-ten-22` picked and seated two cartons. The second triggered
breakaway and then advanced only 0.22 mm over three seconds, aborting 69 mm
short of the backstop. It captured both loaded-stall and exhausted-recovery
entries. The next controller can try a second bounded preload step at 4.5 N
before aborting. `ten-second-breakaway-02` completed full withdrawal from the
saved second-carton entry without triggering either preload step. Fresh trial
`tight-ten-23` tests the combined sequence and recovery behavior.

### Four-carton finishing push

The four-carton episode `tight-four-13` picked and seated all four and
completed three retreats. The fourth carton satisfied the environment's
placement check but did not reach the backstop dwell required by this
controller. Its push ended about 25 mm short, before final withdrawal.

The updated selected-finger push can enter a finishing phase within 35 mm of
the backstop. It uses the same measured progress window and bounded preload
steps as the longer insertion, followed by the same contact and withdrawal
checks. The historical fourth-carton entry used a wrist reference outside the
current reserved range. Its first recovery attempt failed during controller
initialization. The retry explicitly replans the tool from the retained native
state and records that adoption in the continuation report.

The longest ten-carton recording is `tight-ten-21` (456.68 s of actuator
replay). It shows three placements and two completed retreats. The synchronized
[third-person](../../outputs/adjacent-insertion-v1/tight-ten-21-video-gpu/videos/third_person.mp4)
and [first-person](../../outputs/adjacent-insertion-v1/tight-ten-21-video-gpu/videos/first_person.mp4)
videos reproduce the terminal physics state bit for bit.

The [four-carton third-person replay](../../outputs/adjacent-insertion-v1/tight-four-13-video-gpu/videos/third_person.mp4)
and [first-person replay](../../outputs/adjacent-insertion-v1/tight-four-13-video-gpu/videos/first_person.mp4)
match the original terminal physics state bit for bit. They show three complete
stocking cycles and the fourth push failure.

Continuation `fourth-finish-push-06` from the retained fourth-carton push
entry reached the backstop and verified full retreat. Final yaw was -0.0137
degrees with no reported tip/drop incidents. The push and contact refinement
used the measured neighbor envelope for the same lateral goal. Named
carton-face and distal finger contact were admitted under the native force
guards, with a 2 mm neighbor displacement limit during the push. This
continuation earns zero fresh credit. Trial `tight-four-15` tests the complete
four-carton sequence from reset.

### Verified four-carton recovery replay

The combined replay follows `tight-four-13` up to the exact native state
before the fourth push, then replays `fourth-finish-push-06`. Both the switch
state and terminal integration state match bit for bit. The fourth retreat
passed and all four cartons satisfy the native final placement conditions.
This is a continuous 381.34-second recorded trajectory with a controller
switch at the retained boundary. It earns zero fresh episode credit.

- [Third person](../../outputs/adjacent-insertion-v1/four-carton-recovered-video/videos/third_person.mp4)
- [First person](../../outputs/adjacent-insertion-v1/four-carton-recovered-video/videos/first_person.mp4)
- [Replay verification](../../outputs/adjacent-insertion-v1/four-carton-recovered-video/result.json)

A separate fresh attempt, `tight-four-15`, failed at the first carton's
contact refinement. The recovered replay does not replace that failure.

### Four-carton regression checkpoints

| Saved state relative to `outputs/adjacent-insertion-v1/` | Diagnosis | Focused next motion |
|---|---|---|
| `tight-four-13/episode/skill-entries/022-carton_0-contact_finish/checkpoint` | Fourth carton stalled about 25 mm short of the backstop | Check measured neighbor registration, final push and retreat. Recovered in `fourth-finish-push-06`. |
| `tight-four-15/episode/skill-entries/005-carton_2-front_insertion/checkpoint` | First carton reached placement but refinement forecasts lost push contact | Replay from the same released landing and isolate contact refinement and unloaded clear. |
| `four-first-regression-low-force-01/episode/skill-entries/003-carton_2-motion_failure/checkpoint` | Reduced push force reached release planning, but no exit forecast passed | Diagnose contact still present during unload and preserved geometry limits. |
| `tight-four-16/episode/skill-entries/008-carton_2-motion_failure/checkpoint` | Unloaded clear withdrew straight, then the final lift predicted 2.06 mm of torso to hand-tower overlap | Use `--unloaded-clear` to compare straight-then-lift and simultaneous retreat/lift from the exact stopped state. |

Use `--insert-only --continue-after-insert 1 --record-skill-starts` to finish
a retained insertion, save its verified milestone, and attempt one more
carton in the same native state. The first ten-carton probe of this composition
is `ten-chain-three-to-four-01`, starting from the third-carton released entry
of `tight-ten-21`. The original fresh-run outcome remains unchanged.

The exact first-carton stopped state from `tight-four-16` reproduced the
torso/hand-tower rejection with both original clear paths. A third candidate
retracted 35 mm less while lifting. `four-first-clear-exact-02` verified live
retreat after the forecast and checked that the collision geometry of the
whole hand was at least 10 mm outside the shelf front. The pose and support
checks still apply. `tight-four-17` tests the change in a fresh sequence.

The ten-carton continuation `ten-chain-three-to-four-01` saved a verified
three-carton milestone, picked and seated carton four, then timed out during
front squaring. `ten-chain-four-finish-01` resumed at its saved released
entry, verified retreat, and saved the four-carton milestone with six
cartons still in inventory. `ten-chain-five-01` resumes that milestone.
These are checkpoint transitions and earn zero fresh episode credit.

The ten-carton scene's fourth recovery is available in synchronized
[third-person](../../outputs/adjacent-insertion-v1/ten-chain-four-video-02/videos/third_person.mp4)
and [first-person](../../outputs/adjacent-insertion-v1/ten-chain-four-video-02/videos/first_person.mp4)
video. This 45.27-second segment uses the saved model and starts at the
released fourth carton. Both the initial boundary and terminal native state
match the saved evidence exactly. The first render attempt rejected the
original episode XML's model signature. The successful retry uses the
continuation checkpoint binary and verifies exact native replay.

Carton five was picked, placed, and advanced to withdrawal planning before
the combined continuation timed out. Its latest valid test state is
`ten-chain-five-01/episode/skill-entries/010-carton_5-withdrawal_planning/checkpoint`.
`ten-chain-five-exit-01` resumes there and revalidates contact before exit.

### Live-guarded exit rollout

`ten-chain-five-exit-01` resumed the fifth carton's withdrawal-planning
checkpoint, verified retreat, and saved a five-carton milestone. Five
unstocked cartons remain. The following development rollout uses
`--live-guarded-withdrawal` to execute the ordinary loaded exit once, without
a full native lookahead. It still requires verified entry contact and checks
the commanded arm and hand geometry, native forces, neighbor displacement,
support, final placement and retreat. The already-unloaded branch keeps its
checked path alternatives. Successful earlier forecasts and trials retain
their original evidence.

```sh
python scripts/probe_front_release.py \
  --checkpoint outputs/adjacent-insertion-v1/ten-chain-five-exit-01/episode/skill-entries/003-carton_5-stocking_milestone/checkpoint \
  --output outputs/ten-live-guarded-from-five --next-cartons 5 \
  --live-guarded-withdrawal --record-skill-starts \
  --control-config configs/controllers/adjacent_insertion.json --wall-seconds 900
```

This is a checkpoint continuation and earns zero fresh credit. Compare force,
displacement and stage outcomes before drawing conclusions about speed.

The first live-guard rollout exposed a checkpoint-resume issue before pickup.
The grasp hand inherited the saved pusher spread instead of the recorded
grasp spread. Restoring the grasp references after constructing the hand
cleared that issue. `ten-chain-live-guards-from-five-04` picked and seated
carton six, then stopped before withdrawal. It advanced 0.23 mm during the
last three seconds at a 4.5 N push target. The terminal native wrench shows
about 3.1 N of opposing X reaction from carton five and 1.1 N from the shelf
board, so this case is a corner jam against a stocked neighbor. It has
retained post-squaring, loaded-stall, and terminal-failure checkpoints.

A forward push with a 4 mm lateral reference offset did not provide a
measured 4 mm of clearance and still jammed. The next focused probe
`ten-six-side-offset-01` uses the exposed-side push before forward insertion
to move the released carton 4 mm away from the occupied side. Until a
rollout reaches live withdrawal, it provides no evidence about exit speed.

The exploratory live exit uses a 1 degree release yaw window and retains the
original 0.15 degree dwell as a separate benchmark indicator. The coarse
loaded entry additionally requires current backstop contact, upright support,
low measured carton velocity, and placement for every completed carton. It
does not award the strict backstop dwell credit. Force, displacement, native
collision, tip/drop, and release checks remain active.

In matched fifth-carton checkpoint probes, strict live exit stopped at
-0.1675 degrees after unloading. Full forecast and exit took 85.02 s wall.
Exploratory live exit passed in 24.67 s wall, finished at 0.3938 degrees and
saved its milestone. The strict yaw indicator was false. Preserving the
open-side offset through front squaring also got carton six past the
corner jam. Its exit is evaluated separately from the saved refinement state.

`ten-six-coarse-exit-01` completed the sixth retreat with no recorded
incidents and -0.2817 degrees final yaw. It saved a six-carton milestone
with four still in inventory. Its strict yaw indicator was false.
`ten-chain-live-guards-from-six-01` starts from that milestone and attempts
the four remaining cartons under the same exploratory release rules.

Carton six clips, replayed from exact native states:

- [Corner jam, third person](../../outputs/adjacent-insertion-v1/ten-six-jam-video/videos/third_person.mp4)
  and [first person](../../outputs/adjacent-insertion-v1/ten-six-jam-video/videos/first_person.mp4).
  The forward attempt stalls about 69 mm short of the backstop.
- [Recovered exit, third person](../../outputs/adjacent-insertion-v1/ten-six-recovered-exit-video/videos/third_person.mp4)
  and [first person](../../outputs/adjacent-insertion-v1/ten-six-recovered-exit-video/videos/first_person.mp4).
  This clip starts from the recovered contact-refinement checkpoint and shows
  live release and retreat. Both clips match their terminal native states.

The follow-on `ten-chain-live-guards-from-six-01` completed carton seven and
saved a milestone. It stopped on carton eight when the stationary opening's
coupled geometry check rejected the finger sweep. That is a distinct front
handoff issue.

### Base access, guided entry and the last slot

By carton eight, the base already moved roughly 0.48 m laterally to the
inventory approach and 0.40 m back with the load. Both moves completed.
The failure was a predicted ring-finger sweep into the side panel during
stationary grasp opening. The saved front-release entry isolates a bounded
opening retry that can hold the wall-side digit at its last admitted target
while opening the other fingers. The next departure must still pass geometry
and the carton must remain supported.

Use the measured neighbor as a guide only when the contact normal steers the
carton toward the target rather than opposing forward motion. A candidate
angled approach should leave room for the leading corner, use the measured
wrench and displacement to distinguish guiding from wedging, and square at
the desired depth before retreat. For the final slot, compare a precision
entry with a wall-guided entry and check the base and tool exit as part of
the choice. These are candidate policies; the current sixth-carton recovery
used open-side clearance through squaring and advance.

The [six-carton third-person edit](../../outputs/adjacent-insertion-v1/six-carton-edit/third_person.mp4)
and [first-person edit](../../outputs/adjacent-insertion-v1/six-carton-edit/first_person.mp4)
run 1:46 at 3–6× speed. They show the first two cycles from the original run,
then labeled cuts to the verified saved insertion or exit segments for
cartons three through six. Missing portions of timed-out continuations are
omitted. Each input clip was checked against its recorded terminal integration
state. The edit earns zero fresh credit.

`ten-eight-opening-01` cleared the wall-side opening failure by holding the
ring digit at its last admitted reference. `ten-eight-forward-square-01`
then squared farther forward but found the projected insertion still placed
the active carton about 5.3 mm into its neighbor. `ten-eight-open-side-01`
tests a more lateral landing before final advance.

The farther-offside landing trial `ten-eight-open-side-01` moved the
wrapping hand too close to the side panel and failed opening again. That
landing offset is not retained in the default live-guarded policy.
`ten-eight-side-before-advance-01` instead starts after the previously
verified opening and tries the exposed-side shift with an empty hand.

Two angled-entry probes started from the identical post-square state
`ten-eight-forward-square-01/episode/skill-entries/002-carton_2-front_squaring_complete/checkpoint`.
The -3 degree pad-plane command advanced carton eight by 7.1 mm and moved it
3.1 mm toward the occupied side. The +3 degree command advanced 4.8 mm and
moved it only 0.15 mm toward the open side. Both stopped on projected
carton-to-neighbor overlap (3.63 and 3.34 mm respectively) and saved terminal
failure fixtures. Neither angle established a successful guiding contact.
For this shelf, the remaining sequencing problem is to produce measured
lateral separation before advancing the leading corner into the occupied
row. Angle alone did not create enough separation from this entry state.

The next probes tested lateral clearance as a separate motion. From the
verified front release, moving the base 80 mm or 140 mm closer to the shelf
still produced no checked exposed-side approach. Both stopped before
touching the carton. Starting the same side push after forward squaring
reached only a bounded staging pose, leaving a 559 mm predicted tool miss.
The carton stayed at the post-square pose.

Landing 6 or 8 mm farther toward the open side reached measured lateral
positions of -254.63 and -256.43 mm, respectively. Both stationary openings
failed the coupled hand sweep, with predicted ring-finger penetration
of 2.44 and 2.11 mm into the side panel. Holding the implicated finger
commands was insufficient to clear the whole motion. These are geometry
rejections before execution of the rejected commands. All five probes
recorded zero observer incidents and preserved terminal failure fixtures
under `outputs/adjacent-insertion-v1/ten-eight-*`.

The handoff to guided advance should require verified shelf support, an
unloaded hand with a checked exit, and measured clearance for the leading
corner along the first forward segment. The clearance check should use the
current carton and neighbor footprints so it applies to other carton
dimensions. This shelf still needs a tool pose that can achieve those
conditions together. The seven-carton continuation remains the longest
verified chain.

### Fresh live-guarded run and diagonal front push

`tight-ten-fresh-live-01` ran from the initial ten-carton scene (seed 0)
with the command in Reproduction and validation plus
`--live-guarded-withdrawal`. It picked and placed two cartons, verified
final position and retreat for one, and recorded zero incidents. Carton two
(`carton_8`) squared at the shelf front, but the final pusher search exhausted
its base/posture candidates. The fixed-lateral advance projected about 7.1 mm
of overlap with `carton_9`. The failure fixture is
`episode/skill-entries/020-carton_8-motion_failure/checkpoint` under that run.
The scored rollout advanced 287.55 s of simulation in 256.64 s of wall
time. The continuous [third-person replay](../../outputs/adjacent-insertion-v1/tight-ten-fresh-live-video-02/videos/third_person.mp4)
and [first-person replay](../../outputs/adjacent-insertion-v1/tight-ten-fresh-live-video-02/videos/first_person.mp4)
are sampled at 3 fps. Replay matched the recorded terminal integration
state bit for bit.

`ten-eight-diagonal-01` is an independent, zero-fresh-credit probe from the
saved post-square state for the eighth carton. It uses the existing paired
front-face pusher with `--insert-only --guided-entry-deg=3 --diagonal-entry`
and live-guarded withdrawal. The probe limits commanded forward speed to
4 mm/s while seeking 6 mm of measured lateral travel toward the open side.
It moved 6.00 mm forward and 0.84 mm sideways before the compensated geometry
check projected 3.06 mm of active-to-neighbor overlap and stopped motion.
The measured lateral gate never passed and there were zero observer incidents.
For samples with forward force magnitude above 0.3 N, the median forward
and open-side force components were 1.32 N and 0.28 N.
Together these trials isolate lateral registration ahead of forward insertion
as a problem at both ends of the row.

### Insertion checkpoint regression

`insert-regression-01/cases.json` freezes six replayable insertion starts
from failures at positions two, six, and eight in the ten-carton order. It
includes cases from the fresh seed-0 episode and the sequential recovery
chain. The earlier and later boundaries of a failing insertion are distinct
inputs. The comparison uses the same guided diagonal probe flags across all
six, with a 90 s wall budget per case and per-case source snapshots. None
earns fresh task credit.

The two sixth-carton inputs show why both boundaries matter. From the released front-insertion
entry the probe squared and advanced, but timed out 14.7 mm before the
nominal final center. From the saved post-square state it reached bounded
backstop staging, verified native refinement and retreat, and saved a
six-carton milestone. The 6 mm lateral target was never reached on that
successful case. The diagonal motion cannot be credited with establishing
the intended clearance.

The separate `fresh-second-clearance-before-square-01` probe computed a
lateral target from the measured neighbor footprint before square/advance.
The carton briefly crossed that target (58.90 mm versus a 58.94 mm maximum
world Y), then drifted to 66.55 mm and finished at 1.39 degrees yaw. It
timed out at the intermediate depth despite retaining shelf support. The
clearance gate checks measured pose through the dwell and again after
unloading, so a transient crossing cannot pass the handoff.

| Carton in order | Earlier insertion entry | Post-square entry |
|---|---|---|
| 2 | Reached forward push, predicted 3.03 mm neighbor overlap | Reached forward push, predicted 3.05 mm neighbor overlap |
| 6 | Reached forward push, contact dwell timed out | Refined contact and verified retreat |
| 8 | Reached forward push, predicted 3.06 mm neighbor overlap | Reached forward push, predicted 3.06 mm neighbor overlap |

All six reached the forward push. One verified retreat and five retained
terminal failure fixtures, with zero observer incidents across the batch.
The turns that started at the earlier insertion entry all reached the
post-square boundary, so that handoff remains reachable under the comparison
policy even when the subsequent motion fails.

`fresh-second-clearance-before-square-02` removed the pad-plane angle after
the lateral target was first observed. It ended at -0.41 degrees yaw and
71.03 mm world Y. Relative to the first trial, squaring improved and lateral
registration worsened. The measured clearance and squaring gates never
held together through dwell. Both variants leave an exact stopped state
for testing a separate sideways correction before any further advance.

The exposed-side retry from the fresh second carton
(`fresh-second-exposed-side-01`) found no checked approach after the 140 mm
base offset. A separate 8 mm off-side landing, `fresh-second-open-side-landing-01`,
held the carton on the open side but timed out in the subsequent squaring push.

`fresh-second-clearance-before-square-03` uses the current rotated footprint
to choose between steering toward open space and squaring. It can return
to steering if the corner loses clearance. The same start state passed
the square/unload handoff with zero hand load and 1.77 N shelf support,
reached the backstop refinement entry, then verified the existing guarded
withdrawal. Strict contact refinement exhausted its release-goal search,
so the withdrawal used the existing coarse loaded exit after checking
placement. Final yaw was -0.37 degrees. The episode saved its two-carton
milestone, recorded zero incidents, and earns zero fresh credit.

The same footprint policy passed the earlier insertion entries for the
sixth and eighth cartons in `insert-clearance-regression-01`. Together with
the second-carton probe, all three previously failing early entries verified
retreat without observer incidents. The eighth run passed the square/unload
handoff and saved an eight-carton checkpoint milestone. Its end yaw was
0.149 degrees. The strict contact optimizer exhausted its search and the
existing coarse loaded exit completed the retreat. The final yaw alone
does not establish a strict contact dwell.

#### Composed plan and standalone checks

`plan-from-first-milestone-01` restored the fresh run's first milestone and
executed carton two through the composed plan, starting at grasp. It verified
all five handoffs and saved a two-carton milestone. The run used
`--plan-driven --guided-wall-entry` and recorded zero incidents. It is a
checkpoint continuation.

`skill-seams-01` exercises the same controllers from retained boundaries.
The push-only replay admitted near-backstop staging for carton two, with its
direct back-contact flag false. The standalone squared-contact replay on
carton six measured 0.242 s of strict dwell against the 0.12 s requirement.
Retreat for carton eight, supported release for carton eight, transfer/seat
for carton two, and side registration for carton one passed their respective
native gates. The side registration moved 47.61 mm. An isolated grasp/lift
probe passed from a sampled grasp entry that had led to an earlier close
failure on carton six. Each completed probe recorded zero incidents.

`plan-from-eight-milestone-01` reached the row-packing check for carton nine
after a checked unloaded base move. The last placed carton's measured lower
edge was -280.37 mm and the shelf wall was -375 mm, leaving 94.63 mm open for
two 50 mm cartons. No ninth grasp was attempted. There was 7.47 mm of
measured space on the occupied side of carton eight, making lateral
rearrangement a candidate before retrying. The unmodified reserve check
requires at least 100 mm open; its failure event now reports the measured
shortfall. The run preserved all eight stocked cartons and zero incidents.

The repeat `plan-from-eight-reserve-evidence-01` logged a 5.37 mm
open-span shortfall and retained its terminal checkpoint. The planner
reported the same stop before the ninth grasp.

### 20 mm whole-row fixture

`configs/scenes/tight_ten_cartons_20mm.json` sizes the clear shelf span as
ten 50 mm cartons plus 20 mm of total free space. The 520 mm fixture keeps
the positive wall at +135 mm and extends the free end 10 mm. With a fixed
copy of the observed eight-carton arrangement, that yields 104.63 mm open
for the final two cartons. It specifies a new scene, so that calculation is
a geometric comparison and the actual run must measure its own open span.
Nominal targets divide the free-space budget across the eleven carton and
wall gaps (1.82 mm each). The 510 mm scene remains the 10 mm benchmark.

`tight-ten-20mm-fresh-01` ran from reset at seed 0 using the composed plan,
guided wall entry, and live-guarded withdrawal at source revision `8b72b50`.
It saved four retreat milestones. The fourth checkpoint retains strict
0.15 degree release evidence for cartons one, three, and four, and a
coarse release for carton two. There were zero incidents through that
checkpoint. The supervised process then stopped with SIGSEGV during the
unloaded base move for carton five. The fresh terminal outcome is unknown
and receives no full-task credit.

The traceback points at the carton-pair contact iterator. The follow-up
uses MuJoCo's indexed contact-geom column in the same force-summing guard
and resumes from the fourth milestone in
`tight-ten-20mm-from-four-indexed-01`. This is a checkpoint continuation.

The indexed continuation passed the prior crash boundary, picked and seated
carton five, and reached its backstop. It recorded 0.455 N backstop contact
and 1.685 N shelf support at the stop. The squared-contact search exhausted
20 replans without verifying the dwell, so retreat never completed.
Four cartons remain verified and there were zero incidents in the
continuation. The terminal fixture is
`episode/skill-entries/010-carton_5-motion_failure/checkpoint` under the
continuation directory.

### Carton-five refinement retry

`tight-ten-20mm-fifth-retry-01` resumed the terminal contact-refinement
failure with the existing force and motion guards. The second bounded
search still did not verify strict contact dwell. The existing guarded
development exit became eligible and saved a five-carton milestone with
final yaw -0.182 degrees and zero incidents. This continuation earns zero
fresh credit and the strict release flag is false.

The matched `tight-ten-20mm-fifth-square-first-01` trial ranked yaw
correction ahead of lateral registration while outside the strict yaw
gate. It also saved a development retreat, with final yaw -0.313 degrees
and no strict release credit. This alternate candidate ranking was not
promoted to the default policy. Its pinned source remains with the probe.

`tight-ten-20mm-from-five-01` picked and seated carton six, then stopped at
the intermediate squaring depth with 2.59 degrees of yaw. The active
carton's rotated corner still exceeded the lateral clearance limit, so
the controller continued steering sideways and did not complete the
square/unload handoff.

`tight-ten-20mm-sixth-square-at-stop-01` gave yaw correction priority in
the existing 5 mm depth band around that intermediate stop, ahead of the
occupied row. It reduced end yaw to 0.208 degrees, but still finished about
4.1 mm toward the occupied side of the lateral goal. Both measured gates
remain required. A combined run (`tight-ten-20mm-sixth-offset-square-01`)
started from the saved loaded release entry, landed 6 mm farther toward
the open side, and enabled the same squaring priority. It passed both gates,
then passed strict release, ending at 0.0013 degrees yaw with zero
incidents. The new priority branch did not activate on the successful
trajectory, so the exploratory branch was removed from the working
controller. The offset trial saved the six-carton milestone and earns
zero fresh credit.

The next continuation (`tight-ten-20mm-from-six-01`) passed carton
seven's insertion and then stopped in contact refinement at 1.067 degrees
yaw. The guarded restart at the terminal failure state
(`tight-ten-20mm-seventh-retry-01`) passed strict release at -0.00009
degrees final yaw, with no incidents, and saved a seven-carton milestone.
The 520 mm chain through that milestone has five strict-release cartons
and two development-only withdrawals (cartons two and five).

The eighth carton in `tight-ten-20mm-from-seven-01` stopped at the unloaded
stationary opening check. The predicted ring-finger sweep intersected the
negative side panel. The explicit 4 mm front-depth offset in
`tight-ten-20mm-eighth-shallow-01` resumed the same supported release entry.
It passed opening, measured shelf support, insertion, and the development
withdrawal at 0.656 degrees final yaw. It saved a milestone with eight
verified retreats, including five strict releases, and zero incidents.

The final reserve replay `tight-ten-20mm-from-eight-01` stopped before the
ninth grasp. It measured 92.60 mm of open span for two 50 mm cartons, a
7.40 mm shortfall. This is a measured capacity stop after placement drift
in the 520 mm fixture. Cartons nine and ten were not attempted.

### Pressing the stocked row

At the 520 mm eight-carton milestone the measured gap between cartons
seven and eight is 7.93 mm, versus a 7.40 mm reserve shortfall at the open
end. Pressing carton eight toward the occupied wall can recover the
internal gap in principle. Once the cartons touch, lateral contact can
transmit load through the row. The native guards still sum all contact
normals for each carton pair and wall and check every product for tips
and drops.

The exposed-side press probes targeted 7.9 mm of lateral travel.
`tight-ten-20mm-row-press-01` and `-03` rejected base approaches 140
and 80 mm closer to the shelf. `-02` completed the lateral reposition
at the existing standoff, then found no checked side-push approach.
The front-face variant `tight-ten-20mm-front-row-press-02` reused the
force-guided tangent press and rejected its raised precontact poses.
These are reach and path failures before a press.

A higher-preload comparison `tight-ten-20mm-row-press-loaded-01`
started at carton eight's retained loaded release entry. Increasing the
normal target from 2 N to 3 N passed strict release and ended at -0.020
degrees yaw, with no incidents. It did not recover the required lateral
space. Normal preload and lateral compaction have separate outcomes.

Three more comparisons used the identical loaded eighth-carton checkpoint.
`tight-ten-20mm-row-press-loaded-02` reduced backstop target to 0.6 N and
ended in the development exit at 0.387 degrees yaw. Its lateral gain
override applies to insertion feedback and does not change the
refinement controller's fixed lateral gain. `-03` kept the 0.6 N target
and used 400 ms bounded refinement prefixes. `-04` also enabled the
existing pitch candidates, allowing up to 16 candidates per refinement
cycle. Both longer-horizon probes passed strict release, with final yaw
below 0.001 degrees in magnitude and zero incidents.

The final open span remained approximately 92.7 mm. The broader search
selected a pitched candidate for depth advance, then held in 18 of 20
cycles. The moving lateral candidates typically forecast displacement
toward the open end, opposite the desired compaction. The unchanged
force and geometry guards evaluated each applied prefix. These trials
validated squaring from the loaded pose; lateral compaction and resumed
carton-nine stocking remain unverified.

A later lateral-only comparison starts from the strict eight-carton
milestone `tight-ten-20mm-row-press-loaded-04`. The native side approach
targets the leading part of the exposed face from both the shelf standoff
and a 40 mm closer base position. Neither found a checked approach.
A scratch endpoint screen across five base poses and both arms admitted
none of its 90 candidates. Right-hand tips in that screen use a mirrored
reference, so this is approximate reach evidence. Expanding the left
search over roll, tilt and height found precise side-contact endpoints at
the deep carton, but those placed the palm or thumb through the side
panel. Endpoint screening does not certify a swept path.

At the earlier post-square boundary the left arm has more room.
`tight-ten-20mm-side-interleaved-03` cleared the retained pusher, moved the
base, and executed an exposed-side push. It moved 3.06 mm in the intended
direction, but yaw was -1.75 degrees and the existing loaded dwell rejected
the handoff. Trial `-04` shifted the contact patch toward the carton
center. It met loaded dwell at 2.51 mm displacement and 0.071 degrees
yaw, but the straight unload exceeded bounded Cartesian reach. Both runs
recorded zero incidents and retain the stopped states.

The two-leg unload in `tight-ten-20mm-side-interleaved-06` cleared the
tool, but the center displacement vanished. An exact replay reproduced
the terminal integration state bit for bit. At the loaded handoff the
center had moved 2.51 mm and the carton had rocked 1.22 degrees. The
center of its bottom face had moved less than 0.01 mm.

The low contact in `tight-ten-20mm-side-interleaved-bottom-01` used the
projected center of the carton bottom for braking and kept the existing
center, yaw and unload checks. It translated the bottom about 4.1 mm
against a 3 mm goal. The slower 1.2 N target trial
(`tight-ten-20mm-side-interleaved-slow-01`) reached the 3 mm bottom-face
goal, but retained 1.69 degrees yaw. Neither passed the lateral skill's
dwell and unloading contract. The direct lateral motion occurs before
forward insertion, where the side panel allows the required hand pose.

The centered and intermediate contact-point retries (0.30 and 0.50
of the carton half-depth forward of center) stopped at path admission.
For the centered trial, the strongest otherwise feasible candidate had
2.28 mm of measured side clearance against the pre-existing 4 mm
admission margin. The 0.65 offset is retained in the opt-in checkpoint
probe because it reached and translated the carton. The plan does not
automatically run this experimental correction. A loaded 3 mm shift
with 1.69 degrees yaw is evidence of lateral sliding, with no successful
lateral handoff or recovered row capacity.

### Lateral slide, unload, square, advance

The opt-in `--interleave-side-square` probe separates the sideways staging
motion from the strict square gate. It verifies the translated bottom,
shelf support, unload, and tool exit before attempting a short square ahead
of the occupied row. A saved `side_slide_unloaded` entry can resume the
latter sequence with `--insert-only --guided-wall-entry --resume-side-square`.
The square must still meet the 0.15 degree requirement and the clearance
gate before advancing. These are development continuations.

`tight-ten-20mm-side-square-advance-01` retained 2.61 mm translation
with zero hand force after unloading. The compensated fingertip exit
would have recontacted the carton. Trial `-02` increased side clearance
before exiting and retained 2.58 mm after the tool cleared. The next
paired-finger shape was blocked by a Middle/Ring self-contact forecast.
Trial `-03` verified the folded middle finger, but opening and narrowing
the paired fingers simultaneously still stalled. Trial `-04` opened
the pair in a wider posture before narrowing it. Both shape gates
passed and the short square ran, then timed out at 2.41 degrees yaw.
No incidents were observed. The shelf count and acceptance gates
remain unchanged.

Trial `tight-ten-20mm-side-square-advance-05` gave paired-plane yaw
correction priority during the bounded short square. It retained a
2.58 mm side slide with zero hand load, squared to 0.088 degrees after
unloading, advanced, and completed guarded development withdrawal.
Its eighth-carton milestone has 100.014 mm of projected open span for
the two remaining 50 mm cartons. This is only 0.014 mm of geometric
reserve and is not evidence that both remaining insertions can execute.
It earns zero fresh credit and no strict release credit for carton eight.

The continuation `tight-ten-20mm-from-interleaved-eight-01` admitted
row capacity for carton nine and verified its grasp and lift. Its
loaded shelf approach had no checked candidate. A staged loaded base
reposition (`tight-ten-20mm-ninth-transfer-staged-01`) retained the
grasp but did not solve that approach. Scratch planning found the
nearest routes blocked by the middle finger at the negative side
panel. Moving the proposed landing toward the shelf center admitted
scratch seat routes. Live lead trials at 6 and 12 mm reached shelf
support, then stopped on different coupled opening checks. The lead
mode stops before forward insertion and does not modify the
canonical target or the row-capacity gate.

The guarded 24 mm lead release passed when the opening planner retained
the offending wall-side digits at their last applied references and
checked departure with the same posture. Carton nine remained supported
and the hand cleared in `tight-ten-20mm-ninth-held-opening-02`. The
ninth-carton end-to-end attempt still stopped there. No checked exposed-side
approach was found for the 24 mm correction
(`tight-ten-20mm-ninth-side-square-advance-01`), and the paired guided retry
stopped at front-square dwell with 2.66 degrees yaw
(`tight-ten-20mm-ninth-guided-01`). A repeat of the held opening at the
closer 12 mm lead failed its own coupled departure check. These stops
recorded no incidents. Carton ten remains unreached.

A scratch left/right endpoint screen for the ninth lateral correction
ran 160 candidates across five navigable base poses. It admitted none.
The smallest position misses were 13.87 mm on the left and 25.59 mm on
the right. The right tip reference was mirrored for screening. No
right-arm motion was commanded.

### Outside-in order and shared-lift hand clearance

The opt-in `--center-last` policy alternates shelf ends and leaves the
last opening near the center. A measured interval reserves enough space
for the unplaced widths on both sides. When both immediately neighboring
slots are occupied, the controller targets the interval midpoint and holds
forward motion if the current rotated footprint leaves the corridor.
The original positive-wall order remains a separate mode.

Fresh seed-0 trial `tight-ten-20mm-center-last-fresh-01` completed one
strict retreat, then stopped at the opposite-end carton's inventory
grasp. The explicitly reordered supply scene assigns the same ten
starting positions to slots in outside-in order, keeping the shelf
geometry and clearance budget fixed. Trial
`tight-ten-20mm-center-last-fresh-02` got past that grasp and lift and
stopped at the loaded shelf approach. Both runs recorded zero incidents.

The final-slot reset comparison `final-slot-fixtures-20mm-02` prefilled
nine cartons explicitly and began with the active carton already
supported at the shelf front. Under the same width and 20 mm budget,
the negative end slot completed strict release and retreat. The center
slot stopped on paired fingertip-to-neighbor geometry. A narrow-tool
retry avoided that particular sweep but stopped on projected carton
overlap. The slower measured-corridor retry also stopped on geometry
after 1,817 held-forward cycles.

The follow-up `final-slot-fixtures-20mm-compacted-01` reset the nine
prefilled cartons toward both walls. This concentrated the same 20 mm
whole-row budget into a 68 mm center opening for the 50 mm carton. The
paired fingertips reached back contact and the hand cleared the shelf
without incidents. The carton missed the strict 0.15 degree release
dwell. With release forecasting enabled, the matched
`final-slot-fixtures-20mm-compacted-strict-01` stopped at the bounded
release search with 0.476 degrees of yaw. These resets test the final
insertion from shelf support and earn zero fresh stocking credit.

The unloaded `--bimanual-clear` probe on the saved eight-carton state first
stages arm references inside the shared controller's planning headroom
using the checked single-arm path. In
`tight-ten-20mm-bimanual-clear-06`, both palms then moved about 12 mm
forward from the shelf and 12 mm up under one shared-lift solve.
Measured final errors were 0.47 mm on the left and 0.68 mm on the right.
The per-step guard verified no significant product contact, displacement,
or incidents. This verifies a guarded two-hand clearance motion.
Coordinated two-hand carton contact remains unverified.

The seed-0 negative-end transfer was later resumed with a scratch-checked
lead landing toward the shelf center. It reached shelf support, then
stopped at the coupled opening/departure gate. A Cartesian prefix
retry also failed admission. The fresh count remains one retreat. The
exact release stop is saved separately from the end-slot reset fixture.

The repeated two-hand clearance probe (`tight-ten-20mm-bimanual-clear-07`)
also measured orientation errors of 0.010 degrees on the left and
0.217 degrees on the right. It reproduced the position errors and
recorded no incidents.

### Full development timeline

The 520 mm scene with 20 mm of total row clearance has ten scheduled
cartons and zero prefills. The original fresh process lost its unflushed
actuator trace while repositioning the unloaded base for carton five.
The frozen original controller,
scene, seed, and settings reproduced the settled reset and the first four
stocking milestones bit for bit. The reconstructed episode stopped after
the fourth milestone. Its terminal state matches the starting state of
the first saved continuation.

The recovered section replays fourteen recorded continuations, cutting
branches at the exact physics step of the chosen saved skill entry.
Every cut matches the native integration state. The final sequence shows
eight verified withdrawals, including five strict releases. Carton nine
reaches shelf support and the hand clears before the guided insertion
times out. Carton ten is unreached. The differently initialized
center-slot fixture is available as a separate clip. The development
timeline earns zero fresh task credit.

- [Full dense-row timeline, third-person and robot cameras](../../outputs/whole-shelf-video-v2/dense-shelf-full-timeline.mp4)
- [State verification at the reconstructed milestones](../../outputs/whole-shelf-video-v2/prefix-verification.json)
- [Recovered segment verification](../../outputs/whole-shelf-video-v2/recovered-tail/result.json)
- [Complete three-carton stocking episode](../../outputs/whole-shelf-video-v2/three-carton-complete-3x.mp4)

Use `scripts/render_stocking_timeline.py` with either `prefix-recipe.json`
or `tail-recipe.json` under `outputs/whole-shelf-video-v2/`. Each segment
selects a recorded episode or checkpoint continuation. The optional
`stop_at_initial_of` chooses an exact branch boundary from the next
continuation's archived starting state. The renderer checks the replay
clock, the terminal integration state, and that rendering leaves that
state unchanged. The final movie joins the two verified MP4s at the
shared fourth-carton state.

The first geometry sweeps after this run report all scheduled cartons
across shelf and carton dimensions in the [domain randomization guide](../domain_randomization.md).

### Fresh ten-carton comparison

Each case ran from reset at seed 0 with zero prefills and the same frozen
policy, including the existing live feedback and planning guards. No case
used a checkpoint restore or a manual correction. Carton counts apply to
all ten scheduled products. Final position is the strict task-stage
metric, and the retreat column counts measured withdrawals.

| Whole-row clearance and layout | Pick | Shelf support | Final position | Retreat | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| 20 mm, original 520 mm shelf | 5 | 5 | 1 | 4 | Compensated pusher approach rejected for carton five |
| 100 mm, extended toward the supply table | 0 | 0 | 0 | 0 | Side panel blocks first empty-hand descent |
| 100 mm, extended away from the supply table | 2 | 2 | 1 | 1 | Second carton times out at the intermediate alignment gate |

All three completed processes recorded zero knock-overs. The 20 mm run
matched the first 319,767 archived actuator commands and the fourth
milestone exactly. The first rejected reference for carton five predicted
2.06 mm of ring-finger penetration into the active carton during pusher
approach. This is a pre-actuation geometry rejection. In the roomier
layout that preserved inventory-side clearance, the second carton reached
the staging depth within 0.63 mm and held shelf support. Its 1.47 degree
yaw and lateral position left the alignment gate open.

- [Single-run dense video](../../outputs/fresh-ten-carton-v1/20mm-timeline/fresh-dense-run.mp4)
- [Fresh comparison and stage ledger](../../outputs/fresh-ten-carton-v1/report.json)

The repeated seed-0 prefix shows repeatability at one initial state. Broad
ten-carton reliability still needs fresh runs on other reset seeds after
the compensated approach and alignment gates are addressed.

### Ten-carton failure curriculum and approach search

`outputs/runs/agent-harness/ten-cartons-20mm-v2/` reproduces the fresh
five picks, five placements, one strict final position and four retreats from
zero prefills. The fifth carton stops during reapproach after passing front
squaring. The reference predicts 2.06 mm of overlap between the active carton
and ring finger. The full observer recorded zero tips or drops.

In the restarted push from the saved front-square boundary, a 45 mm standoff
completed the fifth retreat in 27.50 simulated seconds at 0.0015 degrees yaw.
A 30-degree pusher tilt with the recorded 20 mm standoff also completed the
retreat in 32.52 simulated seconds at -0.0831 degrees. The default approach and
raised approach hit the applied geometry guard. A 35 mm standoff moved beyond
approach but stopped at a joint limit during the push. The trials retain their
controls, force traces and exact checkpoint hashes under
`outputs/runs/curriculum/ten-carton-late-v1/`.

The 45 mm standoff passed on the separate tight-three and eighth-carton
starts. To try it in the composed sequence, merge
`configs/controllers/post_square_standoff_development.json` with the run's
control overrides. It applies to reapproach after verified front squaring. An
isolated saved start can also use `--pusher-standoff-m 0.045` with
`probe_front_release.py --insert-only`. The 20 mm preset still serves as the
comparison condition. The composed trial from the fourth milestone passed
carton five and stalled on carton six's front-square phase, so the profile
remains opt-in.

Continuing directly from the successful fifth-carton probe reached six and
seven verified withdrawals. The eighth carton reached support before a
coupled opening rejection. Two retries moved the release reference 3 or 4 mm
toward the shelf front. Both then passed strict release and withdrawal from
that saved entry. They left 110.74 mm and 107.92 mm of measured opening for
the last two 50 mm cartons. The 4 mm branch also had zero projected envelope
overlap at its eighth milestone. The chain through eight includes two
strict final positions and earns zero fresh episode credit.

From the existing eight-withdrawal checkpoint, the ninth carton again stalled
during the guided sideways correction. Filling the outer slot first instead
exposed a supply-table access constraint. The nearer carton intersects the
planned grasp of the farther one. The inventory base screen now checks the open
grasp endpoint as well as the hover. It ruled out 15 navigable poses before
driving the base to a blocked grasp. Swapping the assignments of the two
interchangeable pending cartons allowed the next pickup. For both the inner and
outer destination, the loaded route screen exhausted its candidates before
seating. It now uses the actual front landing target, checks base navigation,
and considers lateral offsets in both directions. The unrelated spacious and
tight placement controls passed.

The [development corpus](../../configs/curricula/development.json) retains the
earlier and terminal boundaries, including the blocked inventory grasp, loaded
transfer, fifth approach, ninth front square, and seed-1 retreat failure. The
roomier run also exposed a finger closing sweep against the side panel, and a
forward-table variant failed the empty-hand route before pickup. The curriculum
groups branches derived from the same settled start in the same lineage. It
samples lineages uniformly before sampling a checkpoint, so successive stops
from one episode have the same aggregate weight as a single checkpoint from
another episode.

### Nine withdrawals and the last wall slot

From the newer eight-withdrawal milestones, the ninth carton passed pickup and
seating. The direct alignment clipped the negative side panel. Native opening
checks held obstructed digits at their last applied reference, rescreened any
additional blocked digits, and verified departure with the same posture. A
35 mm approach then completed the ninth contact and retreat at about -0.0002
degrees yaw. The 45 mm branches also reached a ninth milestone.

The tenth carton was picked, seated and released in the front band via a
checked lead landing. Three strict final positions and nine withdrawals are
verified in this chain. Several workspace translations kept the fixed-heading
side approach outside its admissible geometry. A scratch orientation screen
found 40 of 384 candidates with all three tested endpoints feasible. The native
turn uses a swept geometry screen, the existing force and carton displacement
guards, and a base drift bound. The -60 and -90 degree turns at the original
workspace reached side contact and moved the tenth carton about 42 mm laterally.
The front-offset pushes rotated the carton by about 33 degrees and exhausted the
displacement/support dwell. Their terminal states retain the measured motion.

- [Nine verified withdrawals and the tenth carton staged](../../outputs/runs/agent-harness/ten-cartons-20mm-v2/nine-and-tenth-staged.mp4)
- [Bit-exact playback evidence](../../outputs/runs/agent-harness/ten-cartons-20mm-v2/nine-and-tenth-staged-evidence.json)
- [Fresh runs, continuations and curriculum comparison](../../outputs/runs/curriculum/ten-carton-late-v1/summary-report.json)

### Checkpoint-assisted ten-carton packed hold

The successful tenth-carton route turns the base -60 degrees, makes a short
front-corner shift, then pushes under the measured wall brace. The carton was
0.11 mm from the wall registration target and 0.064 degrees off square when the
hand clearance plan stopped. After an additional checked sideways unload,
the hand cleared, reapproached with a 45 mm standoff, inserted the carton,
and completed the tenth withdrawal. Its released yaw was 0.164 degrees, which
earns development withdrawal credit. Six releases passed the strict yaw gate.
Three cartons also produced the back-contact evidence required by the strict
final-position stage metric.

The final packed-row probe held the saved actuator references and stepped
native MuJoCo for 0.70 s. All ten cartons stayed stocked, with no tips or drops.
The row had 17.84 mm of projected free space and 0.0044 mm of summed envelope
overlap. The packed dwell passed with the 20 mm nominal fixture. The chain
starts from zero prefills and earns zero fresh episode credit. The fresh
strict ten-carton run and the 10 mm fixture remain acceptance cases.

- [Complete checkpoint-assisted replay](../../outputs/runs/agent-harness/ten-cartons-20mm-v2/ten-carton-verified.mp4)
- [Packed-row hold and forces](../../outputs/runs/curriculum/ten-carton-late-v1/ten-hold-retained-controls-0/summary.json)
- [Stage counts and exact branch paths](../../outputs/runs/curriculum/ten-carton-late-v1/completed-chain.json)

Use `probe_front_release.py --verify-row` with the final `stocking_milestone`
to repeat the dwell without issuing a Cartesian reposition. The turned-back
ablation also saved a ten-carton checkpoint and passed a separate packed hold.
Its live continuation failed while serializing its summary, so its original
process and strict release outcome remain unverified.

### Empty-shelf repeat and live approach retry

Three supervised MABEL runs started with bit-identical MuJoCo states from
seed 0 in the ten-carton fixture with 20 mm nominal whole-row free space.
Each began with zero prefills. All three summaries report zero assistance
and zero tips or drops. Skill-entry checkpoints were recorded as diagnostics.
The stage counts cover all ten scheduled cartons, including unreached stages.

| Policy | Pick | Shelf support | Strict final position | Verified retreat | First stop |
| --- | ---: | ---: | ---: | ---: | --- |
| Current guarded controller | 5 | 5 | 1 | 4 | Carton five, compensated finger approach geometry |
| 45 mm approach after front squaring | 4 | 4 | 2 | 3 | Carton four, front-squaring dwell timeout |
| Guarded retry from the live approach stop | 5 | 5 | 1 | 4 | Carton five, retry finger preshape geometry |

The baseline reproduced all 387,057 recorded actuator commands and physics
samples from the earlier fresh 20 mm run, including its terminal integration
state. The guard rejected a proposed ring-finger overlap of 2.06 mm before
actuation. The live retry planned a new approach with 45 mm standoff from the
measured stop. Its first preshape command predicted up to 3.03 mm of finger
overlap. The guard rejected it before stepping, leaving the integration state
identical to the baseline stop. The retry implementation is archived in the
run's frozen source bundle.

Applying the wider standoff routinely after front squaring passed carton
three's strict final-position gate. Carton four's initial front-square push
then reached the target depth within 0.013 mm and timed out at 2.63 degrees
yaw against the 0.15 degree gate. Its released start and terminal failure
are in the development curriculum under the same seed-0 lineage. These runs
measure repeatability and policy differences for one reset condition.

- [Fresh trial results, source revisions, controls and stage evidence](../../outputs/runs/agent-harness/ten-cartons-fresh-v3/report.json)
- [Fourth-carton squaring failure](../../outputs/runs/agent-harness/ten-cartons-fresh-v3/postSquare/episode/summary.json)
- [Fifth-carton live retry evidence](../../outputs/runs/agent-harness/ten-cartons-fresh-v3/liveRetry/episode/summary.json)

### Carton-five approach geometry

A scratch replay of the last 1,090 physics steps reproduced the earlier
carton-five stop bit for bit. The nominal approach had 0.074 mm of IK position
error and 0.011 degrees of orientation error at the goal. A 501-sample sweep
of the nominal joint path found 19.69 mm minimum clearance for the affected
ring-finger mesh. At the stop, the measured hand had zero carton contact
force and the shelf carried 1.766 N.

| Pose evaluated at the rejected tick | Ring-finger mesh distance to carton |
| --- | ---: |
| Measured configuration | +18.22 mm |
| Nominal arm and fingers | +19.82 mm |
| Compensated arm and proposed fingers | −2.057 mm |
| Compensated arm with the previously applied fingers | −2.059 mm |
| Compensated arm and fingers with nominal wrist | +12.49 mm |

The planner used 20 degrees of pusher tilt for cartons two through four.
At carton five, joint limits prevented the lower-tilt candidates from
meeting the +3 degree yaw correction reserve. The selected 40 degree
posture made the nominal approach 3.12 s, compared with 2 s for the
earlier cartons.

The controller planned with a nominal wrist reference of 0.200 rad and
measured 0.211 rad. It generated a 0.361 rad servo target to compensate
holding torque and damp wrist motion. The pre-actuation guard checked the
arm and fingers at the combined servo target as a geometric pose. The
wrist offset supplied the largest isolated contribution. The projected
overlap exceeded the 2 mm limit even when the finger compensation update
was held at its previous value.

The opt-in `--screen-pusher-approach` previews settled gravity-compensated
arm and finger targets on scratch data after front squaring. It checks the
planned approach at multiple samples, then considers a wider standoff when
the preview finds insufficient clearance. On the captured starts for cartons
two through four, the original approaches retained more than 10 mm of
projected carton clearance. For carton five, the gravity-only preview
projected 1.44 mm of overlap at 20 mm standoff and 17.73 mm of clearance at
45 mm with the same posture. The live controller checks every command with
its measured-rate, load and collision guards during execution.

- In the isolated native diagnostic, the 20 mm approach failed the new
  preview. Replanning at 45 mm selected a 20 degree pusher posture with
  27.29 mm projected clearance. The carton completed release and
  withdrawal in 27.50 simulated seconds, with 0.0015 degrees final yaw
  and zero tips or drops. Backstop contact occurred, but the strict squared
  dwell remained open. The diagnostic started from the recorded
  post-square state and earns zero fresh task credit.
- The first fresh seed-0 process segfaulted during alignment for carton
  three. A fresh restart reached carton five's loaded base move after
  four logged withdrawals, then segfaulted during contact sampling.
  Neither process produced a terminal episode score.
- Seed 1 last logged the second carton's base move at 185.218 simulated
  seconds. The outer process ended with status 143 near its 30-minute wall
  budget, leaving its raw process record scheduled and its outcome unknown.

- [Scratch geometry audit and reproduced guard result](../../outputs/runs/agent-harness/ten-cartons-fresh-v3/carton-five-geometry-audit.json)
- [Isolated native approach and withdrawal diagnostic](../../outputs/runs/agent-harness/ten-cartons-servo-screen-v1/carton-five-local-screen/summary.json)
- [Fresh trials, frozen source and process outcomes](../../outputs/runs/agent-harness/ten-cartons-servo-screen-v1/report.json)

### Fresh runtime comparison with the approach screen

The runtime comparison uses the same 20 mm nominal row budget, no
prefilled products, the existing force and placement gates, and the
opt-in servo approach screen. The seed-0 development trials compare
MuJoCo 3.17 with the system allocator and a cached MuJoCo 3.16 build with
the default allocator. The older build runs in a separate temporary
environment with the other dependencies shared by path. These runtime
trials probe process robustness and can produce different grasp or
contact trajectories.

The seed-1 trial adds an atomic phase and physics heartbeat with
`--idle-seconds 120`. The heartbeat captures the current simulation time,
product, phase, measured tracking error and contact loads once per
simulated second. An idle termination records the last observed phase
and elapsed time. Completed episode summaries still govern all stage
scores.

| Fresh trial | Pick | Shelf support | Strict final position | Verified retreat | First stop |
| --- | ---: | ---: | ---: | ---: | --- |
| Seed 0, MuJoCo 3.17 and system allocator | 8 | 8 | 1 | 7 | Carton eight, front-square dwell timeout |
| Seed 0, MuJoCo 3.16 | 2 | 2 | 1 | 1 | Carton two, front-square dwell timeout |
| Seed 1, MuJoCo 3.17 and system allocator | 2 | 2 | 1 | 1 | Carton two, front-square dwell timeout |

All three completed trials had zero tips or drops. The MuJoCo 3.17 seed-0
run began from the same native reset state as the preceding fresh baseline
and matched the first 386,017 issued commands exactly. After front
squaring carton five, it rejected the 20 mm approach and selected a 45 mm
approach with a 20 degree pusher tilt and 27.29 mm of projected clearance.
Five through seven then completed guarded withdrawal. The initial run in
the 3.16 environment stopped at a missing dependency. The recorded retry
loaded the dependency through the project environment and compiled a
distinct settled state.

Carton eight reached within 0.54 mm of staging depth, stayed supported,
and ended at 2.94 degrees yaw. A 3.05 mm lateral shortfall kept the
paired-pad controller in its open-side guidance and prevented the
strict square dwell. The two shorter trials reached the same control
mode with 2.60 and 2.46 degrees of remaining yaw. In all three stops,
the commanded pusher offset reached its 6 mm lateral limit and the
carton moved less than 0.02 mm laterally in the final second. In the
eighth-carton stop, the measured lateral force averaged +0.091 N while
the desired shift was toward negative Y. The two second-carton stops
averaged −0.255 N and −0.214 N.

The development curriculum retains the fresh seed-0 eighth-carton and
seed-1 second-carton entries and terminal states. The 3.16 checkpoint
remains with the corresponding runtime artifacts.

The four isolated yaw-first retries did not verify front squaring.
The existing side-push-then-square routine exhausted its exposed-side
approach search on two saved starts. A checked 60-degree clockwise
base turn enabled an 8th-carton side shift of 6.08 mm before unloading.
The subsequent squaring plan exhausted its posture search. A separate
turn-back attempt stopped at the hand clearance check before turning.
The isolated runs earn zero fresh task credit. The two primitive allocator probes both completed
100,000 iterations. A successful seed-0 process under the system allocator
adds a completed development episode, and the intermittent native crash
remains open.

- [Dual-camera fresh replay at 32×](../../outputs/runs/agent-harness/ten-cartons-runtime-v1/fresh-video/stocking-timeline.mp4)
- [Bit-exact replay of all 635,757 physics steps](../../outputs/runs/agent-harness/ten-cartons-runtime-v1/fresh-video/result.json)
- [Exact reset, policy, runtime and stage report](../../outputs/runs/agent-harness/ten-cartons-runtime-v1/report.json)
- [Seed-0 fresh terminal summary](../../outputs/runs/agent-harness/ten-cartons-runtime-v1/mujoco317-malloc/episode/summary.json)

### Matched seed study with supply position jitter

The eight fresh runs use the same ten-carton fixture, MuJoCo 3.17,
system allocator, frozen source, controller settings, and servo approach
screen. Seeds 0 and 1 repeat the earlier development cases. Seeds 2–7
extend the comparison. Each starts with an empty shelf and no restores.

The reset samples each supply carton's X and Y independently within 2 mm
of its nominal location. Start yaw, carton dimensions, mass, friction,
target poses, shelf, and robot start are fixed. The largest observed
planar shift of the same carton between two seeds is 5.08 mm. The row
budget is still 20 mm across both walls and nine inter-carton gaps, or
1.82 mm per nominal gap.

| Seeds | Runs | Picks per run | Shelf placements per run | Strict final positions per run | Verified withdrawals per run |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 8 | 8 | 1 | 7 |
| 1–7 | 7 | 2 | 2 | 1 | 1 |

All eight processes produced terminal outcomes. The suite recorded zero
completed rows and zero tips or drops. Seed 0 reached the earlier
eighth-carton stop, reproducing all 635,757 samples and actuator
commands and its terminal integration state bit for bit.

Seeds 1–7 each placed carton two and stopped at its front-square dwell.
They reached within 0.11 mm of staging depth and ended 1.17–2.20 mm short
of lateral corner clearance, with 2.46–2.63 degrees of yaw. The paired
fingertips continued to command the 3-degree open-side steering posture.
The lateral movement in the final second was at most 0.055 mm. Seed 1
reproduced all 170,509 samples and actuator commands from its earlier
run bit for bit.

The difference emerges during unloading and opening. On all eight seeds,
carton two aligned within 0.07 degrees before unloading. It landed on
seed 0 at 1.70 degrees and Y = 53.66 mm. On seeds 1–7 it landed at
4.90–5.08 degrees and Y = 57.36–57.67 mm, closer to the occupied side.
That yaw range increases the planar Y footprint of the 50 mm carton by
5.80–6.00 mm. The free-landing check verifies support, uprightness and
hand clearance. The following squaring mode must recover the lateral
and angular error left after release.

This study tests a single geometry under supply position jitter. The
policy plans from measured state and uses contact feedback during
execution. The release and front-squaring sequence remains sensitive
to the resulting landing pose.

- [Measured second-carton yaw through release and squaring](../../outputs/runs/agent-harness/ten-cartons-seed-study-v1/carton-two-yaw.png)
- [Seed comparison, provenance and per-carton stage evidence](../../outputs/runs/agent-harness/ten-cartons-seed-study-v1/report.json)

### Lateral alignment from saved release states

Contacting the exposed side gives the front pusher room to turn the
carton. At the second-carton stops on seeds 1 and 4, a 3 mm centered
side press reduced yaw from about 5° to 1.65° and 1.26° after unloading.
A corner press in the opposite direction created about 4 mm of extra
clearance for a front square. The square now holds a 0.10° loaded
control goal before testing the existing 0.15° released yaw gate.

| Saved second-carton start | Loaded side displacement | Yaw after square attempt | Verified front square |
| --- | ---: | ---: | --- |
| Seed 1 | 4.12 mm | 0.080° | Yes |
| Seed 4 | 4.09 mm | 0.196° | Dwell timed out |
| Seed 6 | 3.84 mm | 0.099° | Yes |

The already stocked carton moved at most 0.022 mm in these three
diagnostics. Seed 1 stopped during final release-goal search on the
forward continuation. Seed 6 reached a two-carton stocking milestone,
with verified release and 0.39° final yaw. Its strict final release yaw
gate remained open. Repeating seed 6 with the unloaded-entry guard
reproduced all 72,040 physics samples and actuator commands and the
terminal integration state bit for bit. On the late seed-0 eighth-carton
start, the side shift held after unloading and the swept geometry
rejected the return base turn. A single-finger front-corner nudge timed
out on all three starts where it was tried.

Whole-row pressing was exercised separately. The four centered-side
trials on three- and four-carton fixtures did not verify compaction.
Two stopped before side contact. One reached the compensated-geometry
guard and one could not sustain the measured open-span and support
dwell. In the latter case, yaw left 3.92 mm of projected overlap in
the row. The side-to-wall variant also timed out on the two four-carton
starts where it was tried. The two-carton centered control reproduced
the 1.45 mm open-span reserve after unloading. Its strict alignment
and contact gate remained open.

The centered row controller requested 12 N and the side-to-wall
variant requested 16 N. The attempts on larger rows that reached
contact averaged 3.16–3.42 N along the push direction over their last
simulated second, with repeated reference saturation. The four-carton
seed 6 recorded force at only two of three seams in both variants.
The side-to-wall variant also requires wall contact and changes the
travel budget, so these traces are evidence about the contact chain
and motion limits. A paired force comparison would use the same motion
variant and held reset state.

These are isolated development probes with zero fresh stocking credit.
The side-and-square sequence is opt-in through the checkpoint probe.

- [Per-trial stage results and frozen source links](../../outputs/runs/agent-harness/side-compression-v1/report.json)
- [Seed-6 forward continuation](../../outputs/runs/agent-harness/side-compression-v1/official-side-square-6/summary.json)

### Measured side recovery and supported opening

In the seed study, seeds 1–7 reached less than 0.05° yaw before
unloading carton two. They moved to 0.84–1.14° during pressure reduction
and 4.90–5.08° after opening. The resulting lateral overhang activates
clearance steering instead of the squaring motion. The front pusher
reaches staging depth while the corner still falls short of clearance.

The live side-recovery policy computes a bounded shift from the
measured footprint relative to the occupied side. It requests about
3.8–4.3 mm on the second-carton development starts, unloads, turns back,
and squares ahead of the row. Fresh seeds 2 and 6 matched the archived
native state bit for bit at the release-to-insertion handoff. In the
initial fresh comparison, seed 2 stopped at the approach base turn.
Its yaw repeatedly drifted across the 0.01 rad arrival boundary
during zero-twist hold. A tighter internal heading for entering
settling verified the base turn and completed carton two from the same
saved release. The pose and stop dwell gates were kept fixed.

The other initial fresh trial, seed 6, passed the side shift and
stopped at a front-square dwell timeout with 0.156° yaw. Saved seeds
3 and 7 completed strict release and withdrawal with the initial
policy. Seed 5 still timed out after a supported shift. A bounded yaw
integral changed the loaded alignment but degraded release or square
on two of three replayed handoffs. That variant was dropped.

Five paired release trials start from the same seated, still-grasped
carton for each row below. The feedback variant keeps pose correction
active under measured load and uses the original continuous finger
schedule. The earlier opening variant also changes the finger
interpolation and timing. All releases passed the physical unloaded
support gate.

| Seed and ordinal | Continuous opening yaw | Continuous opening with pose feedback | Earlier feedback-guided opening |
| --- | ---: | ---: | ---: |
| 0, eighth | 4.76° | 0.33° | 0.28° |
| 2, second | 5.10° | 0.33° | 0.32° |
| 4, second | 4.64° | 0.45° | 0.38° |
| 5, second | 5.23° | 0.32° | 0.40° |
| 7, second | 4.97° | 0.56° | 0.40° |

From the improved landings, seeds 4 and 5 with continuous pose feedback
completed strict release and withdrawal from the saved start. Seeds 2
and 7 with the earlier opening schedule verified placement and
withdrawal with their relaxed development release-yaw outcome.

The fresh seed-5 run with continuous pose feedback reached five picks,
five shelf placements, one strict final position, and four verified
retreats. Its original seed-study run reached two placements and one
retreat. The new run stopped on the fifth carton's side press with zero
tip or drop incidents. The corner press moved the carton center 4.95 mm
against a 4.94 mm request. The bottom-face error was 0.26 mm, inside the
0.5 mm displacement gate. Yaw grew from −1.62° to −5.84°, outside the
5° side-staging gate. The fresh seed-7 run with the earlier release
schedule reached second-carton withdrawal, where the applied geometry
check rejected a move. Both runs retained the measured stops.

On a continuation from the fifth carton's released checkpoint, the
selector chose the side center because the corner force would add yaw
in the measured direction. The centered press moved 4.99 mm and
verified its unloaded handoff. Front squaring then passed, and the run
verified the fifth retreat with no incidents. Its final yaw was 0.66°,
outside the strict final release gate. This saved-state recovery earns
zero fresh credit.

Comparing the fresh and saved seed-2 recovery from a bit-identical
integration state found a replay difference at the first tick. The
first differing control values were neck yaw and pitch. The archived
probe held the saved targets, and the fresh run continued to track the
carton. New skill entries retain the gaze controller reference and update
clock. A native checkpoint test verifies identical neck commands and
integration state through the next gaze update.

Continuations earn zero fresh credit. Every source bundle and effective
controller mode is linked in the report. The development curriculum
pins original stops, recovered handoffs and later failures by manifest
hash and episode lineage.

- [Matched release, side recovery and fresh runs](../../outputs/runs/agent-harness/measured-side-recovery-v1/report.json)

### Earlier registration and checked unloading

The guided advance held up to 4 mm of open-side clearance until the
last part of the approach. In the archived second-, fourth-, and
fifth-carton pushes, the carton moved farther from its lateral goal
between 40 mm and 10–15 mm from the backstop. Returning to that goal
after loading the backstop left the pusher trying to slide the carton
sideways under constraint.

The matched continuation below starts at the same recorded front-square
handoff for each pair. Both variants rank guarded finishing prefixes by
the combined lateral and yaw residual. The early registration profile
brings the intended lateral reference back to the slot 25 mm before
the backstop. Coupled arm/finger admission still applies during the
advance. Post-square restores now preserve the phase used by the live
pusher approach screen.

| Saved start | Backstop-entry lateral error with 3 mm setting | Error with 25 mm setting | Strict contact, release and retreat with 25 mm setting |
| --- | ---: | ---: | --- |
| Second carton | 3.86 mm | 0.13 mm | Verified |
| Fourth carton | 2.61 mm | 0.09 mm | Verified |
| Fifth carton | 1.86 mm | 0.38 mm | Verified |

The independent seed-6 continuation verified retreat with 0.33° final
yaw. The independent seed-7 continuation stopped at the compensated
arm/finger geometry check. Both are retained in the development report.

The earlier seed-7 withdrawal guard rejected a distal ring-finger
overlap of about 3.01 mm beside a stocked carton. From the loaded entry,
the unload screen rejected the paired-plane correction and selected
contact-lever feedback. Subsequent native execution verified strict
release and full hand retreat. Starting the alternative at the later
geometry stop failed. The screen therefore checks the unload prefix
before committing to motion. The final yaw gate is measured after the
hand clears. A fifth-carton continuation passed the unload prefix and
later drifted out of strict yaw during retreat.

The fresh early-registration trials on seeds 5 and 7 both reached a
base waypoint that passed its stopped dwell and drifted outside the
2 mm goal during the following hold. The final errors were 2.13 mm and
2.21 mm, with no robot–carton force. A screened short correction from
each exact stop recovered the same goal, with final errors of 1.999 mm
and 1.89 mm. The retry maintains a measured 0.8 s dwell.

At the later seed-7 supported opening stop, the thumb group had fallen
to 0.17 N while the other fingers held 0.86 N. The pressure-relief motion
used the same checked finger targets and moved the arm away from the
measured load. It reached the existing 0.6 N per-group gate, then
completed supported opening with 0.25° released yaw. The saved opening
and base retries earn zero fresh stocking credit.

The seed-7 continuation from its second milestone reproduced the base
settle error, ran the checked correction, then verified native strict
contact, release and retreat for both subsequent cartons. The seed-5
continuation from its third milestone verified a fourth retreat and
then exited with native signal 11 during fifth-carton refinement. Its
terminal task outcome is unknown. The fresh seed-5 wrapper exited with
status 143 after seven logged retreat checkpoints and before a terminal
summary. Those checkpoints identify real resumable stages. The
interrupted episode earns zero fresh stage credit.

The contact retries from those saved entries reached the fifth and
eighth retreat milestones with zero incidents. Both passed released
yaw, at −0.117° and +0.012° respectively. Both entered the exit through
the coarse loaded handoff, so they receive withdrawal evidence only
under the existing strict final-position scoring. The pressure-relief
retry on seed 7 also completed a separate fifth-carton continuation
with strict contact, release and withdrawal. The Gymnasium recovery
case rewound from the pressure stop and verified supported release.

- [Native trials, control variants, and failure checkpoint paths](../../outputs/runs/agent-harness/combined-stocking-v1/report.json)

### Fresh integrated runs and pose variation

The integrated candidate ran from empty shelves on public seeds 5 and 7 with
ten scheduled cartons, zero prefills and 20 mm of total lateral free space.
Each reset matched its earlier early-registration run bit for bit. Both runs
used controller source `b26c3a744102`, MABEL revision `d0181f04025c`, and
native MuJoCo 3.17.0. The waypoints and force guards ran online from measured state.

| Reset seed | Picks | Shelf support | Strict final positioning | Retreat | First terminal failure |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 | 6/10 | 6/10 | 4/10 | 5/10 | Sixth carton. A compensated ring-finger command would have overlapped carton 5 by 3.00 mm. |
| 7 | 7/10 | 7/10 | 4/10 | 6/10 | Seventh carton. The return base turn brought the predicted left tower within 15.84 mm of the supply table. |

Continuous monitoring recorded zero tips or drops in either episode. Strict
positioning requires back contact, released placement and released yaw. One
seed-5 retreat and two seed-7 retreats lacked at least one of those checks.
The retained start before the sixth front-square handoff on seed 5 verified
push and retreat in isolation. The original geometry stop is also retained.
A 35° return turn cleared the native guards from the saved seed-7 stop.
Its subsequent front square failed the compensated finger check.

The pose pilot translated the shelf by (−10, +15) mm in world X/Y on seed 5.
Its first side push passed the nominal intermediate reference, then held
pressure about 2.8 mm beyond it until timeout. The updated skill uses measured
side progress, support and yaw to decide when the wall lies within one more
bounded push. After clearing the hand it replans the wall approach from the
current tool posture. The revised fresh run reached the wall and backstop and
then exhausted the withdrawal decision budget on the first carton. A native
withdrawal from the retained release entry verified retreat in isolation.

The independent supply pilot increased per-axis spawn jitter from ±2 to ±6 mm
on seed 7, moving an individual carton by up to 4.72 mm from its matched start.
It reached the first closure before a coupled finger/shelf-side check rejected
the command. The first generated pose-suite case also stopped at closure.
A 3.1 N grasp force setpoint verified both isolated picks from their earlier
saved entries. A fresh supply-pilot run with that setpoint failed on the first
lift. The first generated layout passed strict placement and retreat for
carton one, then failed to lift carton two. The lower setpoint remains a
development probe. All eight generated development scenes passed native reset
clearance and a guarded 0.2 s hold.

- [Frozen runs, mode flags and failure checkpoint paths](../../outputs/runs/agent-harness/fresh-integrated-v1/report.json)
- [Fresh seed 7 video, six retreats](../../outputs/runs/agent-harness/fresh-integrated-v1/seed-7-video/stocking-timeline.mp4)
- [Fresh seed 5 video, five retreats](../../outputs/runs/agent-harness/fresh-integrated-v1/seed-5-video/stocking-timeline.mp4)
- [Generated pose variant video, one strict placement and retreat](../../outputs/runs/agent-harness/fresh-integrated-v1/pose-000-force31-video/stocking-timeline.mp4)
- [Pose-only suite setup](../domain_randomization.md#pose-only-carton-rows)
- [Eight native pose-suite reset checks](../../outputs/runs/agent-harness/fresh-integrated-v1/pose-suite-native/report.json)
- [Two isolated grasp setpoint probes](../../outputs/runs/agent-harness/fresh-integrated-v1/wider-pose-pick-target-probe/report.json)
- [Isolated shifted-shelf withdrawal](../../outputs/runs/agent-harness/fresh-integrated-v1/shelf-shift-isolated-retreat/report.json)

### Dense recovery from an empty shelf

Each trial starts with ten cartons on the table and 20 mm of nominal free
width across the row, including the two wall gaps. The controller logs all four
stages, keeps exact failed motion states, and measures incidents at every native
physics step. These rows use the same policy snapshot, whose full source hash
and effective flags are in the [campaign record](../../outputs/runs/agent-harness/dense-recovery-v4/campaign-report.json).

| Seed | Pick | Shelf support | Strict final position | Verified withdrawal | First terminal blocker |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 | 9/10 | 8/10 | 4/10 | 8/10 | No checked loaded shelf approach for carton nine |
| 7 | 7/10 | 7/10 | 4/10 | 7/10 | Seventh-carton corrective contact refinement exhausted |

Neither episode had a tip or drop. Some placements passed support and withdrawal
and still needed another contact pass for final alignment. For seed 7, the short
square did execute, but its caller retained the earlier guide and the carton
ended up about 11 mm off the controller's lateral goal. Repeating from the saved
front-insertion entry with the handoff corrected passed strict positioning and
withdrawal at −0.097° released yaw. That replay starts with six cartons already
on the shelf and earns zero fresh task credit.

Seed 5 ran for 23.9 minutes and seed 7 for 19.3 minutes. Native base
repositioning consumed 8.1 and 5.6 minutes respectively, their largest
wall-time cost. These timings include the per-step guards and recording.

The sixth-carton geometry stop replay also passed strict placement and
withdrawal after a guarded unload and reapproach. Separate saved insertion
entries for cartons nine and ten passed strict finishing at approximately
0.0002° and −0.112° released yaw, respectively. They start well inside those
skills. The latest four replays reproduced those successes, using a 35 mm
approach on the known ninth entry and 45 mm on the tenth. The default ninth
approach exhausted contact refinement and remains a development case.
A separate eighth-carton replay recovered its grasp path and carried
the carton to the shelf, then a folded finger touched the supply table during
the empty approach. The saved failure recorded 26.8 N across the contact
patch. Adding a measured-clearance reserve to the compensated approach let the
robot continue to a supported square contact. Strict unloading remains
sensitive to that entry state.

From the ninth-carton loaded stop, a checked front landing 6 mm toward the
shelf center restored the transfer and verified a free shelf-supported release.
The released carton was about 9.8° off square. A separate insertion replay
stalled in the short-square phase at 2.6°. Both the loaded transfer stop and
the later support and insertion stops remain in the development curriculum.

In a subsequent frozen pair, seed 7 recorded 9 picks, 8 shelf-supported
placements, 5 strict final positions and 8 withdrawals before exhausting
its ninth loaded approach. Seed 5 reached eight withdrawal checkpoints
before an intermittent native signal 11 during the ninth lift. The run has
no terminal grade. Its last native boundary and crash trace remain in the
campaign record. [SIM-001](../../home_robot_issues.md#sim-001--intermittent-native-mujoco-signal-during-mabel-control)

Later saved-state replays on both lineages reached supported release for
carton nine. Seed 7 needed retained finger postures through pressure taper
and departure. On seed 5, the short square reduced yaw to about 1°, and a
native side approach moved the carton 7.7 mm into the lateral opening.
Returning from that side approach to a reachable insertion posture remains
part of the handoff verification.

The tight-three regression and the pre-change control replay took identical
native trajectories and both stopped in the second carton's contact refinement
with one verified withdrawal. The 10 mm frozen holdout and the 5 mm trials
await complete packed fresh episodes at 20 mm. Strict positioning remains
a separately reported grade when using the packed-first profile.

The packed-first profile can defer fine alignment until after all withdrawals.
The completed 20 mm packed-first trial on seed 5 verified all ten picks and
shelf placements, nine withdrawals and five strict final positions. The tenth
carton stayed supported while a finger retained 0.42 N of contact after opening.
Seed 7 reached nine supported placements, eight withdrawals and five strict
final positions before its ninth-carton finger opening stopped at the side
panel. Neither trial reached the packed-first success gate.

A subsequent matched 20 mm pair reproduced the same stage counts from reset.
Seed 5 picked and placed all ten cartons, verified nine withdrawals and
stopped at the tenth carton's side registration. Seed 7 picked and placed
nine, verified eight withdrawals and stopped at the ninth carton's side
registration. Both passed five strict final positions and recorded zero tip
or drop incidents. Both native episodes reported terminal results.

A later matched pair achieved the first fresh packed success on seed 5.
The run started from reset with zero prefills and verified all ten picks,
shelf placements and withdrawals. Four cartons passed strict positioning.
The packed row had 18.93 mm projected free width and 0.070 mm aggregate
envelope overlap at 1799.11 simulated seconds. Continuous observation recorded
zero tips and zero drops. Seed 7 verified eight withdrawals and stopped
during the ninth lateral shift. The frozen pair and new pose pilot therefore
retain separate scores for packing, strict alignment and generalization.

On seed 7's saved ninth-carton entry, a 10 mm checked approach to the side
face completed the lateral shift. A coupled two-finger square then achieved
0.092° yaw under contact and settled at 0.181° after unloading. The carton
was still ahead of the occupied row and needed roughly 2.6 mm of lateral
registration. The shallow-handoff replay corrected that offset and advanced
to about 6.5 mm from its backstop depth. The compensated path check rejected
the ring finger's approach to the stocked neighbor before commanding it.
The exact unload and geometry stops remain available for practice. A single
finger either failed the checked servo approach or stalled during lateral
registration. A separate 4 mm side correction reached the lateral target
region and left 2.76° yaw, outside its stage gate.

A dedicated recentering retry on the same seed 7 state reached a base stance
where the measured turn check rejected the swept hand near the shelf. The
one-shot recovery rescreened the measured stance, backed off and completed
the turn through native wheels. The next side approach still had no checked
arm path. The two stops retain separate failure records.

The standalone recentering skill passed on seed 5's saved ninth-carton
handoff. The measured center finished inside the reserved slot and the
projected footprint still fit after unloading and turning back. The
continuation recorded zero incidents and saved the newly verified handoff.

Continuations of seed 5 seated the tenth carton at the backstop with
−0.0122° yaw and no remaining hand force. A fixed-orientation arm withdrawal
left its measured lateral corridor. From that exact stop, a swept and
force-guarded 40 mm chassis move advanced the measured hand 38.6 mm farther
out of the shelf. The full row then passed a 0.70 s native packed hold with
all ten cartons supported, 18.69 mm projected free width, 0.070 mm aggregate
overlap, and no tip or drop incidents. The hold used a checkpoint-local time
budget and earns zero fresh-task credit.

Late-wall trials on the saved seed 5 landing found checked 45° and 55°
approaches that both seated the tenth carton. Both arm exits reached the
applied Cartesian step bound after unloading. Each saved stop passed a
screened chassis retreat, and both stops are now independent Gym cases.
From an earlier saved landing, the shorter checked stow, direct side exit
and guarded chassis retreat completed the packed hold at 1789.08 s on the
original 1800 s simulation clock. A separate replay starting before the
tenth release exercised the same skill sequence with a skill-local deadline
and passed all ten supported placements and the packed hold. These runs
earn zero fresh-task credit.

An experiment that skipped settled stow waits throughout the episode failed
on seed 5's second carton. The changed trajectory entered a short side square
and drifted 9.4 mm in shelf depth, beyond its 5 mm motion gate. The controller
keeps the original stow wait through the first nine cartons and tries the
shorter sequence only for the last carton.

The best completed uninterrupted trace spent 655 s in base translations
out of 1691 s of recorded simulation. Repeated right-arm stows took
another 45 s. This breakdown keeps the timing work focused on the motions
that dominate the full episode.

The retained ten-carton fixture passed a new 0.70 s native hold, with
17.84 mm projected free width and 0.0044 mm envelope overlap. The saved
episode remains a continuation with zero fresh credit.
From that fixture, a targeted correction changed carton zero's released
yaw from 0.164° to 0.023°. Its strict contact and withdrawal passed,
the row remained packed, and all ten cartons stayed supported with no
tip or drop incidents.
Repeating the correction from the new post-dwell checkpoint reached 0.009°
released yaw and saved another verified packed boundary.

The full test suite with native MABEL passed 543 tests after switching
placement checks to the native contact arrays.

- [Fresh seed 5, synchronized third-person and robot cameras](../../outputs/runs/agent-harness/dense-recovery-v3/seed-5-video/stocking-timeline.mp4)
- [Bit-exact replay and artifact checksums](../../outputs/runs/agent-harness/dense-recovery-v3/seed-5-video/result.json)
- [Longest uninterrupted seed 5 video, nine verified withdrawals](../../outputs/runs/agent-harness/dense-recovery-v9/seed-5-video/stocking-timeline.mp4)
- [Longest-run bit-exact replay and artifact checksums](../../outputs/runs/agent-harness/dense-recovery-v9/seed-5-video/result.json)
- [Repeated uninterrupted seed 5 terminal grade](../../outputs/runs/agent-harness/dense-recovery-v10/seed-5/process.json)
- [Repeated uninterrupted seed 7 terminal grade](../../outputs/runs/agent-harness/dense-recovery-v10/seed-7/process.json)
- [Fresh packed success on seed 5](../../outputs/runs/agent-harness/dense-recovery-v14/seed-5/process.json)
- [Full fresh ten-carton two-camera video](../../outputs/runs/agent-harness/dense-recovery-v14/seed-5-video/stocking-timeline.mp4)
- [Bit-exact verification of the fresh ten-carton replay](../../outputs/runs/agent-harness/dense-recovery-v14/seed-5-video/result.json)
- [Matched seed 7 terminal grade](../../outputs/runs/agent-harness/dense-recovery-v14/seed-7/process.json)
- [Seed 7 short square and measured unload](../../outputs/runs/agent-harness/dense-recovery-v15/ninth-coupled-square/summary.json)
- [Measured registration and forward progress from the saved unload](../../outputs/runs/agent-harness/dense-recovery-v15/ninth-shallow-square-handoff/summary.json)
- [Measured turn recovery and subsequent arm-planning stop](../../outputs/runs/agent-harness/dense-recovery-v16/ninth-dedicated-side-turn-recovery/summary.json)
- [Two-camera base-turn recovery clip](../../outputs/runs/agent-harness/dense-recovery-v16/base-turn-video/stocking-timeline.mp4)
- [Verified standalone recentering on seed 5](../../outputs/runs/agent-harness/dense-recovery-v16/seed-5-recenter-skill/summary.json)
- [Two-camera carton recentering clip](../../outputs/runs/agent-harness/dense-recovery-v16/front-recenter-video/stocking-timeline.mp4)
- [Saved ninth-carton transfer and exact stops](../../configs/curricula/development.json)
- [Tenth-carton chassis withdrawal from the exact arm stop](../../outputs/runs/agent-harness/dense-recovery-v9/tenth-unloaded-retreat-swept/summary.json)
- [Tenth-carton finish with synchronized external and robot views](../../outputs/runs/agent-harness/dense-recovery-v9/tenth-packed-finish-video/stocking-timeline.mp4)
- [Packed hold on the completed seed 5 continuation](../../outputs/runs/agent-harness/dense-recovery-v9/fresh-tenth-verified-row-swept/summary.json)
- [Repeatable Gym retreat control from the same exact stop](../../outputs/runs/agent-harness/dense-recovery-v9/tenth-unloaded-retreat-curriculum-local/report.json)
- [Two additional Gym retreat cases](../../outputs/runs/agent-harness/dense-recovery-v11/steep-retreat-curriculum/report.json)
- [Packed hold under the original simulation deadline](../../outputs/runs/agent-harness/dense-recovery-v11/tenth-compact-earlier-verified-row/summary.json)
- [Composed tenth-carton finish from the saved grasp release](../../outputs/runs/agent-harness/dense-recovery-v11/tenth-composed-local-row/summary.json)
- [Bit-exact two-camera replay of the original-deadline finish](../../outputs/runs/agent-harness/dense-recovery-v11/tenth-composed-video/stocking-timeline.mp4)
- [Early side-square stop from the broader stow change](../../outputs/runs/agent-harness/dense-recovery-v12/seed-5/episode/skill-entries/018-carton_8-motion_failure/checkpoint/manifest.json)
- [Saved ten-carton packed hold](../../outputs/runs/agent-harness/dense-recovery-v9/retained-verify-row/summary.json)
- [Targeted correction and packed-row check](../../outputs/runs/agent-harness/dense-recovery-v9/retained-finish-one/summary.json)
- [Bit-exact targeted correction video](../../outputs/runs/agent-harness/dense-recovery-v9/retained-finish-one-video/stocking-timeline.mp4)
