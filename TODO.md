# Ten-carton contact-rich stocking

Agreed scope: ten rigid cartons start on the supply table; zero prefills; one
uninterrupted native-MABEL episode. Start with 10 mm total unused row width, then
5 mm. World-state access is allowed. Delay modeling and RL are deferred.

## 1. Receding-horizon task and motion planning — in progress

- [x] Review SS-Replan: reuse the remaining discrete plan, re-estimate continuous
  arguments, and relax reuse when new observations invalidate the plan.
- [x] Implement bounded deterministic tree search over remaining object order,
  with explicit budgets and deferred next-grasp checks. Execute only the next
  checked action; replan from measured state. This is not an MCTS claim and does
  not yet certify loaded transfer or the complete action sequence.
- [x] Reuse the remaining object order after a fresh next-grasp check; discard
  state-dependent pose/path certificates after motion. Preserve completed work
  and log why a tail was abandoned. Full-route tail admission remains below.
- [ ] Expand from whole-object actions to pick, loaded transfer, insert, compact,
  unload, retreat and bounded recovery. Replan at primitive boundaries and jams.
- [x] Add opt-in scratch forecasts of the empty approach, nominal closed grasp
  and loaded corridor; recheck actual loaded segments before execution. Preserve
  sampled Cartesian joint waypoints when direct joint interpolation fails.
- [ ] Extend admission to release/retreat and uncertainty margins. Nominal grasp
  forecasts do not guarantee actual contact/retention or the same IK branch.
  Candidate exhaustion is a local failure, not proof of global infeasibility.
- [ ] Add alternate staging poses, grasp heights, packing direction and temporary
  placements. Register against measured walls/neighbors rather than nominal gaps.
- [ ] Measure planning latency and compare greedy, tail-reuse and tree-search
  variants on matched scenes. Consider MCTS only with meaningful sampled contact
  outcomes and a measured advantage over this bounded-search baseline.

## 2. Contact skills and common compliant controller — in progress

- [x] Implement and run exposed-side native-physics probes, separate from full
  pick-task credit. Two development seeds achieved about 9.1 mm of a requested
  10 mm displacement, with unloading/retreat verified. Earlier failures retained.
- [ ] Combine middle-gap insertion with a sideways push toward a neighbor/side.
  Rear-row tangential-packing probes have not succeeded.
- [x] Add opt-in bounded finger load compensation, with native torque-sign tests,
  separate nominal/applied references and coupled applied-finger geometry checks.
- [ ] Validate native swerve-base repositioning as a checked action for contact
  poses near arm workspace limits; reset-only base fixtures do not prove motion.
- [ ] Use bounded wrench feedback through native position-servo reference offsets;
  retain gravity/load feedforward. Specify world/tool frames and wrench signs.
- [ ] Share normal-force, tangential/yaw compliance, anti-windup, reference bounds,
  force release and retreat between insertion and lateral compaction.
- [ ] Detect jams by force plus lack of object progress; unload, back off and try
  a bounded alternative. Force aborts remain terminal until safe recovery exists.
- [ ] Measure contact duration, object displacement, residual gaps, force peaks,
  recovery attempts and all-product stability; no success from commanded motion.

## 3. Robustness — after initial contact skills

- [ ] Separate observation adapter from clean evaluator state. Start with seeded
  object/shelf pose biases (2–10 mm, 1–5 degrees) and actuator reference error.
- [ ] Keep privileged collision checks explicitly labeled; do not silently use
  clean object geometry to repair the noisy planning/controller observations.
- [ ] Evaluate clean, perception-only, actuation-only and combined conditions.
- [ ] Later: sensor/control delay, correlated noise and parameter uncertainty.

## 4. Speed — after first complete ten-carton baseline

- [ ] Log simulated phase time, planning wall time and rendering wall time
  separately. Use event-driven completion in place of unnecessary fixed holds.
- [ ] Shorten checked free-space routes; use faster bounded trajectories away
  from contact and brake before contact. Tune speed and compliance together.
- [ ] Aim for 2x lower median simulated completion time at matched complete-shelf
  success relative to the first working ten-carton baseline; no such baseline yet.

## 5. Acceptance and evidence

- [ ] Development curriculum: isolated side push; middle insertion plus side push;
  several consecutive stocks; then ten cartons with zero prefills.
- [ ] Freeze source/config before ten-scene validation, separate from development.
  Count every scheduled carton for pick, shelf placement, final pose and retreat.
- [ ] Preserve process failures and unsuccessful attempts. Keep 25 N guards,
  0.2 mm individual/total dense-overlap limits and continuous tip/drop monitoring.
- [ ] Produce a report and synchronized third/first-person MP4s of full episodes.
  Do not infer a success rate from selected videos or prefilled fixtures.

Existing foundation: PRs 2–6 merged; PR 7 draft and PR 8 unmerged. Historical
full-ten attempt stopped at its first loaded transfer; ten-carton success remains
open. New work is isolated on `feat/contact-rich-packing` based on `037c61e`.

Design and references: [receding-horizon plan](docs/receding_horizon_stocking.md).
Initial evidence: [planning probes](docs/reports/replanning_development.md) and
[sideways contact probes](docs/reports/lateral_contact_development.md).
