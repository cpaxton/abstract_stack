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
- Include non-grasping arm links when guarding robot contact with the active
  product. Apply final motion bounds after every feedback/projection correction.
  Coupled hand plans must check the proposed finger posture as well as the arm.
- Prefilter IK before motion. Report position error in metres separately from
  orientation error and measured execution error. Search bounded orientation
  alternatives and check endpoint/path collisions. A closest candidate may be
  used only for bounded staging followed by replanning; it cannot pass a grasp
  gate. A failed local search does not prove global unreachability.
- Keep nominal references separate from applied compensated servo targets.
  Finger preshapes must appear in both native commands and predicted geometry;
  evaluate the full pick/carry/place/retreat sequence before promoting one.
  Controller changes need native-physics tracking evidence and explicit limit
  diagnostics. Learned-controller integration must validate model, action and
  observation contracts; never reinterpret another robot's policy weights.
- Record the scene configuration, seed, model revision, controller settings and
  simulation timing with interaction artifacts. Keep full recordings in ignored
  `outputs/`; commit only small, reproducible documentation previews.
- Pin the controller source revision and effective mode flags for reported
  experiments. Model/scene hashes alone do not identify a policy. Keep a fresh
  frozen-policy suite separate from public regression cases and development probes.
- Evaluate scripted changes on fixed scene suites. Report all scheduled objects
  for pick, shelf support, final positioning and retreat; retain unreached stages
  and process failures. Monitor every product at each physics step so a transient
  knock-over cannot disappear from the report when an object settles upright.
  Tune compliance on separate development seeds with bounded compute. Never tune
  success tolerances, force abort limits or collision acceptance to improve scores.
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
