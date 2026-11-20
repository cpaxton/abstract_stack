# Independent review: Quest PRs 27 and 26

Reviewed PR 27 at 98222f70feb6268868513c4c6ab8bfe4a52b4729 and companion PR 26 at
768a7aa82154cabed3667e3ccc164fed852f1e5d. No repository changes, branch changes,
device actions, public comments or merges were performed.

## Finding

P2, PR 27, scripts/summarize_quest_tracking.py:184-193:
The skeleton parser retains only id/name pairs. It never checks parent_index or
profile and checks bind-pose array shapes/finiteness without checking whether a
bind orientation is usable. Consequently it accepts an OVR24 hand hierarchy
with a parent_index of 999, a self-cycle, a mismatched OpenXR26 bind profile, or
a zero bind quaternion, and reports the same 24-bone metadata count as the valid
case. The device report says that passing this analyzer validates the named bind
hierarchy; this check currently validates names only.

Reject incompatible profiles and invalid parent references/cycles, validate the
SDK-defined root sentinel and usable bind transforms (mark genuinely unavailable
SDK components explicitly rather than inventing rotations), or downgrade the
report to hierarchy-unverified and remove the stronger evidence claim. Add
negative fixtures covering these cases before downstream forward kinematics
depends on the recorded hierarchy. This is a verified evidence-validation gap,
not an observed robot-control failure: this PR contains no robot control.

Reproduction files: outputs/reviews/quest27/reproductions/ contains valid.jsonl and four
mutated captures. All five successfully return from the exact published
summarize() function with skeleton_metadata_bone_counts.left_hand == 24.

## Validation and scope

- PR 27 capture/analyzer tests: 16 passed in an isolated copy.
- PR 26 coordinate/fixture tests: 2 passed; 2 native MABEL tests deliberately
  deselected to avoid contending with the main agent's native/checkpoint work.
- Inspected raw recorder, display/lifecycle code, builder, manifest guard,
  device/pull helper, analyzer, tests, procedure and device evidence.
- Compared the measured APK source bed68f0 to PR 27 head: only the device report
  was added, so the reviewed runtime code matches the reported APK source.
- PR 26's frames, actuator ownership, split SDK/app/host clocks, source freshness,
  session/generation gating and paired-image contract have no additional concrete
  correctness finding from this review.

Hardware/Unity build was not repeated. Persistent hand rate/accuracy, verified
unmount detection, global hand-skeleton reconstruction and actual teleop
pause/engagement behavior remain documented downstream acceptance work. The
later operator confirmation in the PR body is functional evidence, not a new
instrumented measurement. No finding assumes those deferred features exist.
