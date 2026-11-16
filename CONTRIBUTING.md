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
- Seed resets; record actual dependency/model revisions. Save traces and summaries
  under `outputs/`, with commands that reproduce the result. Document failed or
  partial runs as carefully as successful ones.

## Pull requests

Work on a feature branch. Describe the concrete behavior, reproduction command,
tests actually run, and unresolved acceptance gates. Include a small rendered
preview for scene changes and numeric/trace evidence for control changes. Use a
draft PR for a control prototype that has not demonstrated the full task. Keep
generated downloads, full videos, credentials and machine-specific configuration
out of commits. Do not merge until reviewed.

Potential Emet fixes belong in `home_robot_issues.md` first. Upstream drafts must
use upstream-only reproductions without naming this repository or task.
