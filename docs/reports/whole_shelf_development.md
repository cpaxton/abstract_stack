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
