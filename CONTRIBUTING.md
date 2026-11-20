# Contributing

## Development environment

Use Python 3.11 and uv. The default install includes the Emet dependency stack:

```bash
./install.sh
source .venv/bin/activate
uv run pre-commit install --install-hooks
abstract-stack fetch-mabel
```

For native macOS development, follow [the Mac simulator setup](docs/macos.md):
`./scripts/install-macos.sh` installs the lightweight environment and hooks, and
`./scripts/run-macos.sh sim` launches the native viewer. Use `uv run --no-sync`
for the checks below so the full Emet stack is not synchronized. Run physics
tests with `MUJOCO_GL=cgl ./scripts/test.sh`.

Keep `GIT_LFS_SKIP_SMUDGE=1` set when updating Git dependencies; unavailable
upstream documentation media must not prevent a simulation-only checkout.
Never run `git reset --hard` or clean an existing upstream checkout to fix assets.

For the optional rollout throughput tools, install the pinned `rollout` group in
the normal project environment with `uv sync --locked --group rollout --inexact`.
It keeps the native MuJoCo version fixed and adds MuJoCo Warp; GPU execution
requires CUDA, while native forecasts remain usable without the optional group.
See [backend evaluation](docs/backend_evaluation.md) for supervised benchmarks.
The fast C++/GPU paths do not themselves provide force admission. Only fully
guarded shortlist members may be selected as forecasts, and those still require
fresh-state task/geometry admission before live execution. Report screened,
guarded and unchecked counts alongside timing; a restored state earns no fresh
manipulation credit. Keep compilation/cache and execution timings separate.

Demonstration collection and the neutral episode format use the main simulator
environment. The optional official LeRobot export worker has its own pinned
project and lock in `tools/lfd_export/`: LeRobot 0.7 requires Python 3.12 and
NumPy 2, conflicting with the simulator's NumPy 1.26.4 contract. Install it with:

```bash
UV_CACHE_DIR=outputs/cache/uv \
  UV_PROJECT_ENVIRONMENT="$PWD/outputs/envs/lfd-export" \
  uv sync --project tools/lfd_export --locked --python 3.12
```

Keep both locks current when changing their respective dependencies. The export
worker uses CPU Torch and imports the NumPy-only neutral reader, not MuJoCo.
See [collection and export](docs/learning_from_demonstration.md) for commands.
Capture observations before the policy decision and nominal actions before
every compensation path. Review videos and post-step diagnostic traces cannot
substitute for causally aligned raw training observations. Keep privileged
teacher context out of student features and preserve unknown/failed outcomes.

For the opt-in grasp teacher and skill-entry checkpoints, follow the
[skill curriculum](docs/skills.md). Evaluate isolated skills and complete task
sequences separately, retaining blocked and unreached scheduled cases. Freeze
the teacher before drawing private held-out seeds. A checkpoint retry earns no
fresh episode credit, and unknown historical nominal references require explicit
new-reference initialization. Entry snapshots alone are not BC demonstrations.

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

For new launchers, use [shared source bundles](docs/experiment_sources.md): record
Git HEAD and scoped dirty metadata, capture exact selected source bytes once,
and reference that content ID from each run. Keep runnable extractions in the
shared cache rather than copying Python trees into every output directory.
Portable checkpoints embed a compressed archive. Do not rewrite historical
manifests or discard failed experiment evidence during this migration.

The benchmark baseline uses `--force-profile legacy --legacy-contact-control`;
the selected script
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

For a long single-scene development run, `scripts/run_stocking_case.py`
accepts an optional `--idle-seconds` budget alongside `--wall-seconds`.
The worker writes `physics-progress.json` on phase changes and once per
simulated second of native motion. An idle stop records the last heartbeat
and earns zero terminal stage credit. Choose an idle budget that allows
the planned search and recording work for the scene.

`--force-profile selected` uses the current default grasp/push force setpoints
(4.5 N / 1.875 N). `--force-profile legacy` selects 6 N / 2 N without changing
contact-control or hand mode flags. Explicit `--control-config` values override
either profile. Campaigns and terminal summaries record the profile and effective
settings. `configs/controllers/low_force.json` remains a fixed historical snapshot,
even when it matches current defaults. Named force profiles do not reconstruct
historical source code; reproduce old algorithms at their recorded commits.

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

Before rewriting a stack, preserve source revisions cited by experiment reports
under a durable archive ref. The original reviewed stack is available on
`archive/pre-squash-stack-20261117`, and the independent Mac setup history is on
`archive/pre-squash-macos-20261117`. Historical commands still require their
recorded source commit and settings; rebasing must not rewrite measured results.

Potential Emet fixes belong in `home_robot_issues.md` first. Upstream drafts must
use upstream-only reproductions without naming this repository or task.

## Paper changes

Build with `./paper/build.sh` and check final references and layout. Keep verified
literature findings, observed diagnostics and proposed methods/experiments
explicitly separate. Cite primary sources and pin reviewed versions. Do not
commit generated TeX/PDF files or expose private test seeds in the draft. The
paper toolchain is optional and independent of the simulator installation.
