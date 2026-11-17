# Stocking workflows and experiment index

New development: [CPU/GPU rollout evaluation](backend_evaluation.md),
[measured throughput report](reports/rollout_backend_development.md), and
[stationary bimanual/shared-lift control](whole_body_control.md). These prototypes
do not establish full-shelf success or close the native process-stability gate.
The [compliant lookahead study](reports/compliant_lookahead_development.md)
retains failed proposals and native checkpoint continuations, including measured
backstop contact. The [LfD pipeline](learning_from_demonstration.md) and
[simulated sensors](sim_sensors.md) document causal collection, local LeRobot
roundtrip evidence and the remaining student/camera integration work.

## Supported starting point

Install with `./install.sh`, fetch native MABEL with `abstract-stack fetch-mabel`,
then run:

```bash
abstract-stack solve --random-scene --seed 0 --video --output-dir outputs/my-run
```

This is the stationary two-carton controller: native thumb preshape, bounded
grip-force feedback, checked insertion and checked withdrawal. Start with the
[solver guide](stocking_solver.md) and [controller guide](mabel_controller.md).
For scene editing and house previews, use [scene setup](scene_setup.md) and
[house scenes](house_scene.md). A house render does not validate navigation or
manipulation in that house.

## Results, in order

| Report | What it establishes |
|---|---|
| [PR review follow-ups](reports/pr_review_followups.md) | Controller/planning contract fixes, native Mac CI and dense regression equivalence; no new dense success-rate claim. |
| [Selected compliance study](reports/compliance_followup.md) | Current default; four ten-scene suites, 40/40 complete; matched force comparison and retained unknown search outcome. |
| [Thumb preshape](reports/thumb_preshape_followup.md) | Intermediate 6 N / 2 N policy; 30/30 selected runs, top-edge diagnosis, rejected probes and native dual-view recording. |
| [Contact control](reports/contact_control_followup.md) | Earlier 8/10 and 7/10 results; release/clearance changes and one unknown historical native crash. |
| [Original ten-scene study](reports/stocking_ten_scenes.md) | Historical 6/10 baseline, fixed criteria and earlier compliance search. |

Historical commands require the named source revision and effective settings.
Current defaults intentionally differ. Raw artifacts remain under ignored
`outputs/`; committed JSON files retain the exact source paths, fingerprints,
scene definitions, denominators and failures. Unknown runs are not replaced by
successful diagnostic retries.

## Next task families

See the [dense stocking guide](dense_stocking.md) for the implemented scene
generator, insertion curriculum, commands and open acceptance gates.
Dense stocking is a separate task: unused lateral width summed over the entire
occupied shelf row is **10 mm or 5 mm**, including wall gaps. It needs explicit
contact modes and insertion order; the spacious two-carton results above do not
measure it. Deformable cardboard is a separate material-model investigation;
rigid-box success does not validate carton crushing, folding or elastic recovery.
The [cardboard investigation](cardboard_simulation.md) compares native material
options and retains numerical failures from independent panel fixtures.

Keep material fixtures, task geometry, control logic and evaluation separate.
Store full experimental traces/videos under `outputs/`, record every scheduled
failure and keep small rendered previews and numeric summaries with the report.
