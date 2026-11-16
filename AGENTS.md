# Repository rules

- Read `CONTRIBUTING.md` before changing code. Keep scene generation deterministic
  and keep procedural scene assets, control code, and evaluation separate.
- Preserve upstream MABEL geometry, inertias, actuator limits and collision masks.
  Model changes require an explicit configuration and a documented reason.
- Drive interactions through actuator targets and MuJoCo stepping. Never change
  live object/robot poses, apply hidden attachment constraints, disable contacts,
  or use external object forces to claim a physical grasp or stocking success.
  Reset/fixture construction and scratch-data IK are separate from live physics.
- Measure contact forces with MuJoCo, retain failure evidence, and distinguish
  commanded motion, measured tracking, grasp, lift, release, and contact success.
  Rendered previews and scripted waypoints do not prove manipulation success.
- Record the scene configuration, seed, model revision, controller settings and
  simulation timing with interaction artifacts. Keep full recordings in ignored
  `outputs/`; commit only small, reproducible documentation previews.
- Run pre-commit, relevant physics tests, and package checks before opening a PR.
  Do not skip hooks or weaken success criteria to make a demo pass. Report skips
  and failures accurately. Mark a PR draft when required acceptance remains open.
- Keep dependencies explicit and update `uv.lock` when changing them. CI must test
  the lightweight simulator as well as the real MABEL model when assets exist.
- Track upstream concerns in `home_robot_issues.md`. Do not file upstream issues
  or messages unless requested; keep any upstream reproductions free of this
  project's name, repository, task description, and other downstream context.
- Do not change upstream checkouts, publish generated assets, or merge a PR as
  part of routine implementation. Use a branch and describe final behavior,
  validation, and remaining limitations in the PR.
