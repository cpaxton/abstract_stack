# Fixed-shelf videos and predictive control baseline

2026-11-17. Development is back to perfect state and fixed single-level shelves.
Two separate fixed scenarios are shown: a fresh spacious two-carton task, and
the existing dense-row contact fixture. They are not a randomized evaluation or
a combined success-rate campaign. Native MABEL geometry, actuator parameters,
collision masks, task gates and force abort limits are unchanged.

## Videos

All videos are local ignored artifacts under `outputs/fixed-shelf-video-v1/`.
Each recording has native third-person and robot-mounted first-person views.
The contact/replay overviews also show an overhead debug view and measured
forces. Videos play at simulation-time speed; rendering is passive.

| Recording | Native duration | Outcome and scope |
| --- | --- | --- |
| `fresh-two-carton/overview.mp4` | 56.524 s task physics, plus labeled final pause | Fresh `pick_place`, seed 0, two scheduled cartons, zero prefills. Carton 1 completes all four stages; carton 0 aborts its next approach on the 25 N guard. |
| `square-and-retreat/overview.mp4` | 20.166 s | Saved dense fixture: square the existing carton, unload and retreat. No new pick or complete-task credit. |
| `long-sequence/overview.mp4` | 153.386 s | Continuous saved-fixture replay through the next pickup, shelf support and unfinished backstop recovery. No fresh episode or complete-task credit. |

The fresh overview is a passive side-by-side composition of the synchronized
native videos in `fresh-two-carton/episode/videos/`. Full original videos and
the failure ending remain intact. The two replay directories also contain
`third_person.mp4`, `first_person.mp4` and `overhead_debug.mp4` at 24 fps.

## Fresh fixed task: one of two complete stocks

The existing `pick_place` configuration has one shelf at 0.80 m, 0.50 m width,
two 70 x 50 x 240 mm cartons and 2 mm supply jitter fixed by reset seed 0.
This is a spacious task, not the 5/10 mm whole-row dense-packing benchmark.

| Scheduled carton | Pick | Shelf support | Final position | Retreat |
| --- | --- | --- | --- | --- |
| carton 1 | Yes | Yes | Yes | Yes |
| carton 0 | No | No | No | No |

The episode stops with `approach: contact force exceeded 25.0 N`; the peak
hand/active-product normal is 25.1667 N. Native process exit is 2, wall time
47.53 s against a declared 240 s budget. All products were monitored at 2 ms
physics steps; no tip/drop incident was recorded. One episode is insufficient
to estimate a success rate; complete-scene success is false.

The remaining carton moved and rotated during the earlier manipulation. Before
the second approach it was near [-0.4073, -0.4375, 0.9199] m with about
63.3 degrees yaw. The approach IK reported approximately 0.0000007 m position
error and 0.00086 degrees orientation error with accepted path checks, then
encountered excessive physical contact. IK feasibility is separate from
contact dynamics and successful manipulation. The next planner work is to
predict/admit the intended contact from the changed pose and replan the grasp,
without raising force limits or changing task tolerances.

## Dense fixture: reproducible contact finish and longer continuation

A fresh contact-finish run restored the existing single-level fixture and used
the current source. Supported squared contact and measured release/retreat
passed: 39.924 mm back, 20.011 mm up, final yaw about 0.0000594 degrees and zero
hand/product load. This run's complete actuator sequence and terminal native
integration state are bit-identical to the earlier selected withdrawal run.
There were no recorded tip/drop incidents.

The short video combines the three existing squaring segments with that fresh
finish. The longer replay then continues through `continuous-next-stock-v5`,
`continuous-next-stock-v8-direct` and `continuous-next-stock-v10-compliant-square`.
All four/seven segment boundaries and terminal full integration states matched
exactly. Neither rendering nor chain boundaries reset or change native state.
The longer replay still ends at 3.228 degrees yaw and about 50 mm lateral error;
its final positioning and retreat remain unverified. The explicit carton 9/5
target-slot permutation remains disclosed in the
[continuous development report](continuous_stocking_development.md).

Native render workers passed within their declared budgets. The tool's outer
long-render session reported exit 143 after the worker's successful result;
that status is retained separately in the evidence ledger. The worker's saved
result, complete MP4s and exact native replay checks establish this replay,
not a new successful manipulation episode.

## Controller direction and reproduction

The [two-layer plan](../receding_horizon_stocking.md#fixed-shelf-two-layer-controller)
uses a slower measured-state planner for predicted object motion and short
references, and faster compliant execution. Log reference validity, predicted
versus measured pose at the **same prediction horizon**, contact force, nominal
versus applied targets and measured tracking separately. A tiny IK residual must
not be presented as a small physical execution error.

Fresh fixed-task command, with frozen source/provenance recorded in its parent:

```bash
MUJOCO_GL=egl MABEL_ROOT=third_party/open_mabel \
  flock /tmp/abstract-stack-render.lock .venv/bin/python -m abstract_stack.cli \
  script --scene pick_place --seed 0 --force-profile selected \
  --video --render --fps 24 --output-dir outputs/my-fixed-two-carton
```

Use the contact-finish command from the
[withdrawal report](squaring_withdrawal_development.md#reproduce-and-continue),
then `scripts/render_checkpoint_chain.py --fps 24` with the ordered segment
directories recorded in each video manifest. Use fresh outputs. Full inputs,
source IDs, mode flags, native traces, failures, process outcomes and video
timestamps remain archived. New run directories contain no Python source trees.
All three overview MP4s passed complete decoding and 24 fps stream checks.
See the [compact evidence ledger](fixed_shelf_videos.json).
