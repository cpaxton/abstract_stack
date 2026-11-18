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
and completes three insert/release/retreat cycles. The ten-carton trials reach
one complete stock, then expose the next pickup's roof transit and the next
transfer's wrapping-hand/sidewall collision. Those failures remain unchanged.

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

## Reproduction and validation

```bash
PYTHONDONTWRITEBYTECODE=1 MABEL_ROOT="$PWD/third_party/open_mabel" \
  flock /tmp/abstract-stack-render.lock .venv/bin/python scripts/run_stocking_case.py \
  --config configs/scenes/tight_ten_cartons.json \
  --output outputs/my-fresh-whole-shelf --wall-seconds 1800 -- \
  --grasp-skill --base-reposition --ungrasped-insertion \
  --paired-finger-insertion --control-config configs/controllers/adjacent_insertion.json \
  --record-skill-starts
```

The scene declares a 1,800-second simulation limit and this command declares a
1,800-second wall budget; earlier trials used shorter declared budgets. Time is
reported, not hidden by checkpoint retries. `front_insertion` snapshots preserve
the post-registration state, nominal arm/tool settings, applied finger tracking
state and observer history. `probe_front_release.py --insert-only` restores that
entry; `--record-skill-starts` can capture entries during a zero-credit probe.

The latest focused insertion/lateral/navigation/lookahead checks pass **35 tests**
and all-file pre-commit passes. These checks preceded the one-line wall-ranking
correction; a final rerun remains required. A full frozen suite at `fcee9d0` reached 95% progress before a
native signal 11 while constructing primitive MuJoCo data. A later expanded
process also failed during native model construction; subsequent focused passes
do not replace either failure. Logs/source records remain under
`outputs/adjacent-insertion-v1/validation-fcee9d0/` and
`native-validation-20261118/`. Native process stability and full-shelf validation
remain open. No upstream issue or message has been sent.
