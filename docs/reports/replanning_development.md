# Initial receding-horizon development evidence

Ten cartons, zero prefills, seed 0, 10 mm total free row width. These are two
development attempts, not a success-rate campaign. The source-visible controller
uses privileged MuJoCo state. Native geometry, force guards and task criteria are
unchanged. Neither attempt completes the shelf.

| Attempt | Pick / place / final / retreat, each out of 10 | Simulated time | First failure |
| --- | --- | --- | --- |
| Reviewed controller at `e8679d5` | 1 / 0 / 0 / 0 | 31.70 s | `align_shelf` prefilter; 94 mm predicted position miss |
| Opt-in order search, source snapshot | 1 / 0 / 0 / 0 | 31.46 s | `align_shelf` prefilter; 26 mm predicted position miss |

The initial full MuJoCo integration-state arrays are bitwise equal. Both attempts
returned verified failure summaries, with zero product tip/drop incidents. Each
had a declared 600-second process budget and finished in approximately 20 seconds
wall time. They overlapped other development checks, so wall times are not a speed
benchmark. Predicted IK position misses are not measured tracking errors and do
not establish global unreachability; collision and orientation gates also apply.

Order search expanded 1,252 nodes and selected `carton_7` first, checking its
grasp endpoint freshly in about 0.62 seconds. It produced an order for all ten
cartons, but its nominal suffix cost does not certify loaded routes. The physical
result confirms that checking a grasp endpoint alone is insufficient. Next work
is loaded-route admission before pickup and replanning at primitive boundaries;
the current implementation replans only after a completed carton.

Artifacts are under `outputs/contact-rich-replanning-v1/`:

- `baseline-ten-seed0`: immutable `e8679d5` checkout, normal rollout evidence.
- `tree-ten-seed0/episode`: opt-in controller rollout evidence and search logs.
- `tree-ten-seed0/source_snapshot`: exact development sources and runner; hashes
  in `source_snapshot.json`. Subsequent lateral-push acceptance checks were
  tightened; reproduce this attempt from its snapshot, not later source.
  A snapshot outside a Git checkout reports a null revision inside its bundle;
  the supervisor records the feature base revision `037c61e` separately.
- `order-probe`: reset-state planning only, no physics step/task credit; about
  0.69 seconds, same proposed order, unchanged live qpos/ctrl/time.

The isolated lateral skill has separate [retained failures](lateral_contact_development.md).
Neither a full middle-insertion-plus-side-push skill nor ten-carton stocking is
validated. The complete ten-scene robustness/speed comparison remains pending;
these results do not promote new defaults.

## Subsequent development, same ten-carton seed

Every attempt starts all ten cartons on the table, with zero prefills. Source
changes between attempts; exact copied packages/configuration are retained per
run. [Machine-readable outcomes](replanning_development.json) include unsuccessful
and unverified processes. These probes do not establish a population success rate.

| Attempt | Pick / shelf place / final / retreat, each out of 10 | Terminal result |
| --- | --- | --- |
| loaded-route v1 | 1 / 1 / 0 / 0 | Fine-alignment timeout |
| loaded-route v2 | 1 / 1 / 0 / 0 | Pusher approach search rejected |
| loaded-route v3 | 1 / 1 / 1 / 1 | First complete stock; next clearance rejected |
| loaded-route v4 | 2 / 1 / 1 / 1 | Second actual loaded route rejected |
| loaded-route v5 | 2 / 2 / 1 / 1 | Second coupled opening rejected |
| loaded-route v6 | 0 / 0 / 0 / 0 | Configuration validation error before physics; unknown task outcome |
| loaded-route v7 | 2 / 2 / 1 / 1 | Shallower front staging still rejects opening |
| table-front v1 | 1 / 1 / 1 / 1 | Next pickup exceeds unchanged force guard |
| table-front v2 | 1 / 1 / 1 / 1 | Guarded contact outside bounded grasp region |
| table-front v3 | 2 / 1 / 1 / 1 | Next pusher search rejected |

The low corridor executes retained Cartesian-IK joint waypoints instead of
replacing them with an unchecked direct joint interpolation. It recomposes tool
goals from measured object/tool transforms and searches bounded transit yaw
alternatives, restoring the object's original orientation before placement.
Nominal pickup admission now also forecasts the empty approach. It still cannot
certify actual tracking, contact forces or retention, and release/retreat
admission remains incomplete. The second opening rejection was localized to
thumb/shelf-side geometry, not a neighboring carton.

The table-front variant releases on the existing coplanar transfer table, then
slides the carton onto the shelf. This changes the control protocol, not the
fixture. Temporary table support earns no shelf-place credit; the finger-push
stage records that gate only after measuring shelf support. Both protocols still
require the real pickup, final physical placement, unloaded retreat and unchanged
dense-row criteria.

An initial tail recheck took about 13 ms with endpoint-only admission; complete
geometric admission is more expensive (roughly 1.5–3.5 seconds in initial probes,
26.6 seconds in one difficult later replan). Soft per-root budgets were added;
native IK/refinement calls are non-preemptible. These are measured development
latencies, not a real-time planning claim. Frozen packages identify which version
and admission scope produced each number.

## Standard-controller regression

The required benchmark ran in a separate frozen source copy: ten legacy scenes,
seven two-scene development trials, then ten selected-controller scenes. All 34
initial bundles share one Python-source fingerprint; all ten baseline/selected
initial integration-state pairs match exactly. Development retained the default
force settings (`{}`). Results were baseline **6/10** and selected **9/10**.
The selected seed-2 process terminated with native `SIGSEGV`, without a terminal
summary, so it earns zero successful stages. The other nine selected runs
completed. No retry replaces that unknown outcome; SIM-001 remains open.

This standard two-carton benchmark does not exercise the opt-in order search,
loaded corridor or contact fixtures, and is not ten-carton validation. Its
Markdown/JSON report and complete logs are under
`outputs/contact-rich-replanning-v1/standard-regression/benchmark/`.
