# Receding-horizon stocking plan

The first goal is ten cartons from the supply table, zero prefills, in one
continuous native-MABEL episode. Use privileged world state initially. Begin
with 10 mm total unused shelf width, then 5 mm. The width budget includes both
walls and every inter-carton gap. We do not yet have a working ten-carton baseline.

## Planning and replanning

Plan a task-level suffix to the goal; refine only the next action, execute it,
observe, and replan. Contact feedback operates inside that action. A completed
carton must not be repeatedly displaced just because a new plan looks cheaper.
Recoveries that move completed cartons need an explicit reason and revalidation.

SS-Replan uses deterministic, cost-sensitive belief-space planning, observation
updates and reuse of the remaining discrete plan. It re-estimates continuous
arguments and relaxes reuse constraints if needed. It is not MCTS. See the
[reviewed v2 paper](https://arxiv.org/html/1911.04577v2), especially Section VI,
and the [authors' implementation](https://github.com/caelan/SS-Replan).

Our initial implementation borrows that execution structure, with deterministic
beam search instead of PDDLStream. It does not implement the paper's belief
filter, observation actions or probabilistic costs. MCTS remains a candidate
once contact outcome distributions and rollout costs are useful enough to test.
Running thousands of full native-physics episodes inside every planning call
would defeat our speed goal; cheap geometric screening and deferred motion
checks come first. This is our engineering choice, not a result from the paper.

| Layer | First implementation | Next extension |
| --- | --- | --- |
| Task search | Bounded tree over object order to fixed target slots | Placement side, staging, temporary moves and explicit compaction |
| Admission | Optional empty-approach and nominal loaded-corridor forecast | Release/retreat, actual grasp uncertainty and tracking margins |
| Replanning | At completed-carton boundaries; reuse discrete tail only | At every primitive boundary, changed geometry and failed contact |
| Contact control | Existing native servo/load compensation; isolated lateral probe | Common bounded force/moment feedback and jam recovery |
| State | Explicit privileged observation adapter | Seeded pose bias and actuator error; delays later |

`replanning.py` has no environment handle and cannot move the world. Its cost
uses travel and pickup-crowding heuristics. Hypothetical suffix placements are
local nominal forecasts, not collision-free motion plans or physical successes.
`receding_stocking.py` integrates the search as an opt-in dense-script variant.
Its basic mode checks the first grasp endpoint; opt-in route admission also
forecasts the empty approach, nominal closure and loaded corridor. Loaded
execution recomposes goals from actual grasp transforms, checks each segment
again and retains sampled Cartesian joint waypoints. All original force and
evaluation gates remain in place. A later transfer failure remains a failed
episode. No automatic recovery from a force abort exists.

The search retains at least one branch per candidate first object. If the
cheapest first action is rejected it can check another root. Each new observation
rechecks its first action; only object names/order survive tail reuse. Failed
checks are not permanently cached. Stop reasons, expansion/check counts, state
fingerprints, reuse rejection and total planning time are logged. Budgets apply
between native IK calls; a single non-preemptible call can overrun the wall budget.

## Contact skills

The immediate skill test is a middle-row insertion followed by a lateral push
toward a neighbor or wall. Test both directions and record actual object motion,
normal force, torque, neighbor motion, unloading and retreat. An exposed carton
reset at the shelf front is a useful fixture probe; it earns no pick/insertion
credit and does not establish the combined task. A fully packed row may leave
no room for a hand beside a carton; search staging/contact postures before
assuming that a geometrically possible object displacement is robot-reachable.

Build one outer compliant controller for normal force, tangential motion and yaw
correction through bounded native position-servo reference offsets. Preserve
gravity/load feedforward, separate nominal/applied references, rate limits,
anti-windup and final bounds after every correction. Declare force frames/signs.
Do not call this direct torque control: the current native interface uses position
servos. Rising force without object progress triggers unload/backoff and a
bounded alternate action once that recovery is physically checked.

## Experiment sequence

1. Preserve a current-controller full-ten attempt and source/config provenance.
   Run the tree-search variant headless, retaining its first failure and timing.
2. Establish exposed-side pushing, then middle insertion plus lateral registration.
   Add a grasped/supported lateral-shift alternative if side access is blocked.
3. Add loaded-route admission and primitive-level replanning, then expand from
   several consecutive placements to all ten cartons without reset or assistance.
4. Add object/shelf pose errors of 2–10 mm and 1–5 degrees, and actuator reference
   bias. Keep clean evaluator state separate; label any privileged planning guards.
   Delay, correlated noise and dynamics uncertainty are later work.
5. Establish a complete-task baseline before optimizing speed. Log simulated task
   time separately from planner and renderer wall time. Remove unnecessary holds,
   shorten checked routes, and increase free-space speed with bounded braking.
6. Freeze the candidate for ten scenes separate from development. Compare clean,
   perception-only, actuation-only and combined conditions. Aim for 2x faster
   median simulated completion at matched success, not faster selected videos.

Acceptance stays fixed: all scheduled cartons count for pick, place, final pose
and retreat; continuously monitor every product; retain unknown/crashed outcomes;
keep the 25 N guards and 0.2 mm individual/total dense-overlap limits. Report
prefills separately. These opt-in prototypes do not change defaults or promote a
controller based on isolated skill success. See [TODO](../TODO.md) for status.

## Development commands

Use a fresh output directory for each attempt. From the feature checkout:

```bash
python scripts/run_replanning.py --plan-only --objects 10 --seed 0 \
  --search-seconds 15 --output-dir outputs/order-probe
python scripts/run_replanning.py --objects 10 --seed 0 \
  --search-seconds 15 --output-dir outputs/replanning-ten
python scripts/run_replanning.py --objects 10 --seed 0 --loaded-route \
  --admit-loaded-route --alignment-seconds 10 \
  --search-seconds 15 --output-dir outputs/replanning-corridor
```

`--table-front` instead releases on the existing flush table before the finger
push. Table support earns no shelf-place credit; actual shelf support during
insertion is required. `--guarded-approach` enables the existing force-triggered
descent mode. Both remain experimental; the ten-carton goal is open. JSON
`--control-config` overrides include bounded staging/timing settings and insertion
force/yaw gains, while task acceptance and abort limits remain fixed.

Pass `--mabel-root` if the native model is elsewhere. `--plan-only` performs no
physics step and earns no task credit. Full attempts produce the normal scene
bundle, traces, terminal state, per-stage evaluation and summary. Search budgets
do not limit episode runtime; use an external process supervisor when needed and
retain a missing terminal summary as an unknown process outcome.
