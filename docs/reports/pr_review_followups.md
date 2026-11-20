# PR review follow-ups

The five controller/planning findings were validated together on
`fix/review-followups`, based on PR #7. During the subsequent authorized merge
pass, each fix moved into its originating PR. PRs #2 through #6 are now
squash-merged; #7 has been restacked with its planning fix and remains draft
while its experimental scope/acceptance decision is open. PR #8 retains this
audit report and an additional native-crash observation for review.

The benchmark below remains pinned to its original immutable source `e8679d5`.
Historical outcomes have not been rewritten. Archive refs preserve original
source revisions; merge instructions require squash merges and rebasing each
child onto the actual parent squash commit.

| Original PR | Change |
| --- | --- |
| #2 | Capture the prediction's measured qpos baseline, preserve its allowed-object permission on staging rechecks, and reject stale or mismatched cached contact modes. |
| #3 | Validate joint bounds after composing partial commands with retained nominal references. |
| #4 | Retain launcher help and Linux shell tests, preserve renderer overrides, and run installation, native physics and scene export on macOS CI. |
| #5 | Remove the redundant arm component from the combined force maximum; retain hand/arm diagnostics and the combined guard. |
| #6 | Add named selected/legacy force profiles with validated JSON overrides and provenance; retain the historical low-force preset. |
| #7 | Replace the shallow environment copy with a documented planning snapshot: independent MuJoCo data, shared read-only model, and timestep only. |

## Validation

The local diagnostic suite passed **149 tests**. After the final cached-mode
guard, the extended native staging regression and bounded-staging tests passed,
and the full pre-push physics hook passed with the same 149-test collection.
Pre-commit and distribution builds passed. The independent read-only review
found no remaining actionable correctness concerns after that guard.

[PR #8 CI at `e8679d5`](https://github.com/cpaxton/abstract_stack/actions/runs/44005638415)
passed 148 tests with one expected Emet adapter skip, rendered the house and
built both distributions.

A known seed-0 dense regression with **5 mm total row free space**, nine prefilled
cartons and one scheduled insertion passed pick, place, final positioning and
retreat, with no tip/drop incidents. Initial state, terminal state, native
actuator commands and physics samples are bitwise equal to the prior
`outputs/dense-ten-5mm-insertion-v2` reference. This is a regression case, not
fresh multi-scene success evidence. Comparison and raw runs are retained in
`outputs/review-dense-regression-v1/`.

At macOS revision `046f228`, [all three CI jobs passed](https://github.com/cpaxton/abstract_stack/actions/runs/44005175164).
The native macOS suite passed 62 tests, with one expected Emet adapter skip;
the complete local Linux suite passed 63 tests. The
[first hosted Mac attempt](https://github.com/cpaxton/abstract_stack/actions/runs/44004661921)
passed physics but failed CGL rendering with `invalid pixel format`. It remains
recorded. Mac CI checks physics and scene export; native CGL rendering and the
interactive viewer remain hardware checks. Linux CI still requires rendering.

One root full-suite invocation exited 139 after 36 completed tests, with a
faulthandler frame at native model construction. Its log and exit evidence are
retained in `outputs/native-crashes/review-followups-pytest-v1/`. The successful
149-test diagnostic rerun does not resolve this intermittent native crash;
see SIM-001 in `home_robot_issues.md`.

## Stocking regression benchmark

The complete baseline/search/validation workflow ran from immutable source
`e8679d5`. Raw artifacts, paired reports and the provenance audit remain under
`outputs/review-followups-benchmark-v1/`; the
[JSON companion](pr_review_followups.json) retains the numeric evidence and
all seven search configurations, including rejected candidates.

| Metric | Legacy baseline | Current controller |
| --- | ---: | ---: |
| Complete scenes | 6/10 | 10/10 |
| Pick | 16/20 | 20/20 |
| Place on shelf | 16/20 | 20/20 |
| Final positioning | 14/20 | 20/20 |
| Retreat | 14/20 | 20/20 |
| Unknown outcomes | 0/10 | 0/10 |

These are spacious two-carton scenes, seeds 0 through 9. All scheduled products
remain in each denominator. Baseline uses `--force-profile legacy
--legacy-contact-control --no-thumb-preshape`; the current controller uses the
selected profile and the existing contact control/thumb preshape. This compares
the combined modes, not an isolated force-setpoint change. Both retain the
25 N force guard and unchanged task criteria.

The bounded search evaluated seven configurations on separate development seeds
100 and 101: 12/14 scene successes. Both failures were rejected lower-grasp-force
probes (3.375 N), which failed the measured lift gate. The incumbent defaults
remained selected: 4.5 N grasp and 1.875 N push, with no explicit overrides.
No controller defaults were changed by this review follow-up.

All 34 scheduled benchmark episodes have verified terminal evidence and zero
interventions. All nine campaign manifests and their episode bundles have the
same frozen Python-source fingerprint, no dirty sources, and matching effective
settings. All ten baseline/validation pairs match scene/model/reset and settled
integration-state hashes. No failed attempt was replaced by a retry. This public
suite is regression validation, not blind generalization evidence; complete
dense-row stocking remains unsolved.

Reproduce from the frozen source with pinned MABEL assets and the installed
Python environment:

```bash
./scripts/benchmark_stocking.sh outputs/review-followups-reproduction
```
