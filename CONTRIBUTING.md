# Contributing

## Development environment

Use Python 3.11 and uv. The default install includes the Emet dependency stack:

```bash
./install.sh
source .venv/bin/activate
uv run pre-commit install --install-hooks
abstract-stack fetch-mabel
```

Keep `GIT_LFS_SKIP_SMUDGE=1` set when updating Git dependencies; unavailable
upstream documentation media must not prevent a simulation-only checkout.
Never run `git reset --hard` or clean an existing upstream checkout to fix assets.

## Required checks

```bash
uv run pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
uv build
```

Disabling pytest plugin auto-loading keeps unrelated system/ROS plugins out of
the tests; all required tests use built-in pytest facilities. Tests requiring
downloaded MABEL assets are marked `mabel` and skip when those assets are absent.
Fetch the model and run them before reporting robot integration as validated.
Pre-commit runs formatting, lint, file hygiene and configuration checks. Physics
tests run in CI and can also be invoked with the pre-push hook.

CI installs the locked `ci` dependency group and the local package without its
heavy Emet dependencies. It tests the real MABEL model and renders the house;
Emet-specific adapter tests skip in this minimal environment. For the complete
Emet integration, run the full development install locally. `scripts/test.sh`
uses the active checkout's environment without synchronizing it again.

## Scene and physics changes

- Use metres, kilograms and radians. Document the world frame, object origins,
  full versus half dimensions, collision proxies and support heights.
- Provide a small scene preset and seed. Reject layouts that do not fit rather
  than silently clipping geometry or moving the robot.
- Keep task objects dynamic and fixtures explicit. Preview cutaways change
  visibility only; walls and ceilings must still collide.
- Preserve the upstream robot's mass, joints, transmissions and torque limits.
  No live qpos writes, object teleportation, hidden welds or disabled contacts in
  physical-controller evaluation. Scratch IK and episode resets are allowed.
- For manipulation, measure palm/object tracking, hand contact forces, actual
  object lift, support/release, and final contact. A closing hand command is not
  a verified grasp. Log aborted phases and timeouts as failures.
- Before commanding a waypoint, predict its position/orientation residuals and
  check sampled paths against scene and robot geometry. Keep position error in
  metres, orientation error in degrees, and physical execution error separate.
  Report joint-limit and collision rejection reasons. Bounded alternative poses
  and closest-pose staging must not silently relax grasp acceptance. Retain the
  original target, remaining error and explicit replan/abort decision.
- Seed resets; record actual dependency/model revisions. Save traces and summaries
  under `outputs/`, with commands that reproduce the result. Document failed or
  partial runs as carefully as successful ones.

## Pull requests

Dense-row work follows the [dense stocking guide](docs/dense_stocking.md).
Declare total whole-row free space and prefilled count before a campaign.
Count each scheduled manipulation once; a prefilled carton earns no pick credit.
Keep native contact guards and final overlap tolerances fixed. Sum contact
normals per carton pair before applying force limits. Cardboard panel fixtures
are separate numerical experiments, described in the
[material investigation](docs/cardboard_simulation.md).

During cleanup, keep failed raw runs and immutable historical reports. Put large
recordings under ignored `outputs/`, link current guides from the experiment
index, and keep task geometry, control, evaluation and material probes separate.

For scripted stocking changes, run `./scripts/benchmark_stocking.sh` in a fresh
output directory. It evaluates ten two-carton scenes, hill-climbs compliant
control on separate development seeds, repeats the ten scenes, and writes a
paired Markdown/JSON report. Preserve failed and unverified runs. Physical-only
campaigns have a 180-second wall budget per episode; video campaigns default to
900 seconds. Use `batch --episode-timeout` to declare a different budget.
An unknown process outcome earns no successful stages. Report per-stage counts
against all scheduled objects and complete-scene success separately. Continuous
all-product tip/drop monitoring is required for a no-knock-over claim; final
upright poses alone are insufficient. Keep task criteria and abort limits fixed
while searching controller parameters, and disclose development seeds, search
budget, rejected candidates and validation failures.

Freeze a source checkout before a validation campaign and compare the Python
source fingerprints in every episode bundle. If child source fingerprints
differ, preserve the outcomes and report the mismatch as development evidence;
do not combine them into a frozen-policy success rate.

The benchmark baseline uses `--legacy-contact-control`; the selected script
enables checked alternate withdrawal paths and wrist/board clearance feedback.
The baseline also uses `--no-thumb-preshape`. The selected script opens the
native left thumb MCP to 0.4 rad before descent and retains that reference
through carry and withdrawal. `--no-thumb-preshape` isolates this change while
keeping the contact controller enabled. Check the intended thumb posture in
empty/loaded IK and coupled opening plans; validate carry, placement and retreat,
not just initial lift. Ordinary joint-path IK samples the future finger posture;
it does not certify the transient opening sweep or predict contact forces.
Use `--no-fixture-avoidance` to isolate the release-path change. The guarded
approach prototype requires explicit `--guarded-approach` and is not promoted by
a lift alone: verify carry retention and full-scene outcomes before changing its
default. All variants share the combined robot–active-product force guard.
Keep original unknown crash outcomes even if a later diagnostic retry succeeds.

Bundles and campaigns record Python-source fingerprints and the source revision
when available. For an archived checkout, record its commit explicitly alongside
the campaign. Compare settled integration-state hashes as well as scene/model
inputs before claiming a matched experiment. Historical reports must name their
historical checkout; the current benchmark command follows current defaults.

Work on a feature branch. Describe the concrete behavior, reproduction command,
tests actually run, and unresolved acceptance gates. Include a small rendered
preview for scene changes and numeric/trace evidence for control changes. Use a
draft PR for a control prototype that has not demonstrated the full task. Keep
generated downloads, full videos, credentials and machine-specific configuration
out of commits. Do not merge until reviewed.

Use **squash merge** when a PR is approved and merging is authorized. For a PR
stack, merge parents first. Rebase each dependent branch onto the parent's new
squash commit, dropping the old parent commits, then update its PR base and
rerun required checks. Check that its diff contains only the remaining child
changes. Review or stack maintenance alone does not authorize merging.

Potential Emet fixes belong in `home_robot_issues.md` first. Upstream drafts must
use upstream-only reproductions without naming this repository or task.

## Paper changes

Build with `./paper/build.sh` and check final references and layout. Keep verified
literature findings, observed diagnostics and proposed methods/experiments
explicitly separate. Cite primary sources and pin reviewed versions. Do not
commit generated TeX/PDF files or expose private test seeds in the draft. The
paper toolchain is optional and independent of the simulator installation.
