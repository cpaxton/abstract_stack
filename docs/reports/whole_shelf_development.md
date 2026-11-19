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
