# Scripted shelf-stocking evaluation

Native MABEL MuJoCo physics, privileged world state, 2 cartons per scene. No RL, live pose writes, attachments, external object forces or interventions.

| Metric | Baseline | Hill-climbed |
|---|---:|---:|
| Complete scenes, all stages, no knock-overs | 6/10 | 6/10 |
| Geometric full-shelf success | 6/10 | 6/10 |
| Observed runs without recorded tip/drop | 10/10 | 10/10 |
| Pick | 16/20 | 16/20 |
| Place | 16/20 | 16/20 |
| Final position | 14/20 | 14/20 |
| Retreat | 14/20 | 14/20 |
| Mean time of successful scenes (sim s) | 87.83 | 87.83 |
| Maximum measured hand-normal force (N) | 25.26 | 25.26 |

All scheduled objects remain in each stage's denominator, including objects unreached after an earlier failure. Stage gates are measured events; complete-scene success additionally requires every gate and no transient knock-over.
No-tip/drop counts cover observed execution only; an aborted prefix does not establish a completed retreat, recovery, or post-abort stability.

| Seed | Baseline pick/place/final/retreat | Tuned stages | Baseline / tuned result |
|---|---|---|---|
| 0 | 2/2/2/2 | 2/2/2/2 | pass / pass |
| 1 | 2/2/2/2 | 2/2/2/2 | pass / pass |
| 2 | 2/2/2/2 | 2/2/2/2 | pass / pass |
| 3 | 2/2/2/2 | 2/2/2/2 | pass / pass |
| 4 | 0/0/0/0 | 0/0/0/0 | approach: contact force exceeded 25.0 N / approach: contact force exceeded 25.0 N |
| 5 | 2/2/2/2 | 2/2/2/2 | pass / pass |
| 6 | 2/2/1/1 | 2/2/1/1 | release: opening/retreat path rejected before actuation / release: opening/retreat path rejected before actuation |
| 7 | 0/0/0/0 | 0/0/0/0 | approach: contact force exceeded 25.0 N / approach: contact force exceeded 25.0 N |
| 8 | 2/2/1/1 | 2/2/1/1 | push: contact force exceeded 25.0 N / push: contact force exceeded 25.0 N |
| 9 | 2/2/2/2 | 2/2/2/2 | pass / pass |

Stage definitions:

- **pick:** Verified physical lift, hand contact and table unloading.
- **place:** Measured shelf support before the supported slide.
- **final position:** Back-contact/upright dwell, then unloaded release satisfying unchanged task geometry/velocity/support conditions.
- **retreat:** Release and clear completed; current and previously placed cartons stable for task dwell, no product tipped/dropped during retreat.
- **clean success:** Process exited normally, complete task, all four stages, no product tipped >45 degrees or dropped below 0.25m at any monitored physics step.
- **unreached:** Counted as zero in the scheduled-object denominator; not silently omitted.

Selected explicit controller overrides: `{}`.

The paired scene configurations, initial robot/object states, model revision and MuJoCo version have matching SHA-256 fingerprints for every verified pair.
Seeds 0 and 1 were used in earlier development; this ten-scene suite is public pilot validation, not a blind held-out test.

This is a narrow, ten-scene simulation benchmark, not a generalization claim. The JSON companion retains failures, timings, force peaks and incident evidence.

Development seeds: `[100, 101]`. Search budget: 7 evaluated configurations, including the default; each runs on every development seed.

| Trial | Overrides | Audited score | Accepted |
|---|---|---|---|
| 0 | `{}` | `[2, 0, 0, 16, -40.0795, -176.62]` | True |
| 1 | `{"grasp_force_target": 4.5}` | `[2, 0, 0, 16, -40.0795, -189.28]` | False |
| 2 | `{"grasp_force_target": 7.5}` | `[1, 0, 0, 10, -40.0795, -127.44]` | False |
| 3 | `{"push_force_target": 1.5}` | `[1, 0, 0, 14, -65.9488, -170.962]` | False |
| 4 | `{"push_force_target": 2.5}` | `[2, 0, 0, 16, -40.0795, -178.88]` | False |
| 5 | `{"payload_orientation_speed": 0.22499999999999998}` | `[1, 0, 0, 10, -41.423, -128.102]` | False |
| 6 | `{"payload_orientation_speed": 0.375}` | `[1, 0, 0, 14, -53.8749, -171.662]` | False |

Gross hand-normal impulse is integrated over the physics samples in the JSON. Successful-time means are conditional on completion; all failure timings remain in the per-scene data.

The bounded search found **no improvement**. The selected controller is the default
(6 N squeeze target, 2 N push-force target, 0.30 rad/s payload uprighting), so the
second ten-scene campaign is a reproducibility check, not evidence of a new policy.
All ten scene/model/initial-state fingerprints, stage counts and terminal outcomes
match. The success estimate remains 60%, with a 95% Wilson interval of 31.3–83.2%;
repeating the same seeds does not double the number of independent scenes.

The audited score is lexicographic: clean scenes, negative unknown outcomes,
negative knock-over scenes, achieved stages, negative sum of per-scene peak normal
force, negative total simulated time. The original search record used the same
success/stage/force/time priorities before review added explicit unknown/knock-over
tie penalties. Re-scoring all seven retained trials left every acceptance decision
and the selected controller unchanged. Neither safety thresholds nor task criteria
were tuned. This is one bounded coordinate sweep, not evidence that the default
is globally optimal.

An initial development probe used seeds 100, 101 and 102. Seed 102 exceeded five
minutes of CPU-bound IK without a terminal summary and was stopped as **unknown**;
the completed 100/101 outcomes and partial 102 artifacts remain under
`outputs/compliance-search-v1/`. The bounded search used 100/101 only. This compute
exclusion limits development coverage; it does not remove any of the ten evaluation
scenes. Subsequent physical campaigns use a 180-second episode wall limit; video
campaigns default to 900 seconds. Neither ten-scene campaign had an unknown outcome.
The baseline began before the wall-budget implementation, but every baseline episode
finished below 180 seconds. The second campaign used two concurrent, isolated,
physics-only workers; the reproduction script runs serially.

Failure diagnosis and next scripted changes:

- **Approach, seeds 4 and 7:** gross hand-normal forces reached 25.15 and 25.26 N
  before closure/lift. Collision checks permit intended hand/object contact; accepted
  kinematic IK does not predict the contact load. Add a force-aware descent/stop and
  choose another checked grasp height when the open hand loads the box too early.
- **Release, seed 6:** the second carton's checked opening/retreat path predicted
  2.039 mm penetration between the thumb distal geometry and the earlier carton,
  exceeding the unchanged 2 mm allowance. It stopped before actuation. Search
  alternate opening/withdrawal directions or grasp orientations with the earlier
  carton included in collision checks; keep the rejection threshold fixed.
- **Supported push, seed 8:** replaying the saved integration state and every native
  actuator command reproduced both final product poses exactly. The terminal
  contact included 25.72 N between `left_tower` and `shelf_board_0`, plus 0.56 N
  between the idle right thumb and shelf side. The summed arm/fixture force hit
  26.28 N and triggered the 25 N guard. This is a palm/shelf clearance problem;
  box push-force tuning alone does not address it. Add per-link fixture-contact
  unloading and replan the wrist/hand height while preserving carton support.

No tip/drop was detected in the observed portions of any episode. Failed runs
stopped at their guard; a future free evolution or recovery after abort was not
measured. The final report and JSON establish the stage counts above, not dense full-
shelf packing, whole-house navigation, bimanual operation or recovery performance.

This is the historical controller at commit
`82c869d421549afada0d3121ef53c23a79ef46d6`. In a separate checkout of that
revision, reproduce with `./scripts/benchmark_stocking.sh outputs/new-benchmark`.
The current script has a different controller and stronger force coverage; see
[the contact-control follow-up](contact_control_followup.md) for its matched
baseline, source revisions and fresh-scene results.
Full traces, initial MuJoCo integration states, applied controls and logs are in the
referenced `outputs/` directories. The compact [JSON evidence](stocking_ten_scenes.json)
contains all scene specifications, controller settings, hashes, force peaks/impulse,
search/audit records, the unknown development probe and replay diagnosis.

For context, 18 cartons reached grasp precheck and 16 lifted (16/18); two cartons
were never attempted after an earlier failure. All 16 cartons entering placement
reached measured shelf support. Fourteen of 16 entering final insertion passed
final positioning. Fifteen opening/retreat phases began, including the rejected
release plan; 14 completed (14/15). These conditional counts supplement the fixed
20-object denominators above; they do not replace them.
