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
- [x] Add an opt-in supported partial-insertion boundary: unload, clear the hand,
  replan native base position, then plan the remaining push from measured state.
  Implementation and guard checks exist; physical completion remains unvalidated.
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
- [x] Implement checked native swerve-base repositioning. Three isolated native
  translations passed, including ten cartons on the table. Integrated empty-hand
  repositioning reaches a previously rejected contact approach; a complete
  base-plus-insertion sequence remains unvalidated. Negative-Y motion currently
  rejects idle-right-hand/shelf clearance; add a checked empty-arm stow action.
- [x] Diagnose and correct incompatible online IK followed by independent joint
  rate clipping. Optimize inside the joint-rate box, report returned nominal FK,
  and include the actual measured start in retained Cartesian paths. Native
  contact tracking is now below 1 mm in development runs, with task failures
  retained separately from tracking results.
- [ ] Use bounded wrench feedback through native position-servo reference offsets;
  retain gravity/load feedforward. Specify world/tool frames and wrench signs.
- [x] Prototype contact-wrench admittance with bounded orientation alternatives,
  C++ screening, guarded native forecasts and measured-state replanning every
  100 ms. A retained-state continuation advances 9.75 mm to measured backstop
  contact with 2.33 N peak robot/product load; full placement/release is open.
  This is zero new task credit and slower than real time; see the
  [development report](docs/reports/compliant_lookahead_development.md).
- [x] Distinguish first back contact from sustained, supported, squared contact
  at every physics step. Add an opt-in yaw-priority forecast objective and
  bounded contact-lever feedback using measured contact position.
- [ ] Close the retained insertion's squaring gate: development follow-ups
  reduce yaw from 4.43 to 2.75 degrees, still above the unchanged 0.15-degree
  criterion. No release/retreat or fresh-stock credit; see the
  [sustained-contact report](docs/reports/sustained_contact_development.md).
- [x] Prototype measured wrist-load compensation for grasp/carry; reserve nominal
  wrist and lift headroom without changing native ranges, gains or force limits.
  Test native torque sign and applied geometry before actuation. Not promoted.
- [x] Add geometry-derived elevated inventory transit and sampled rigid-payload
  clearance checks. These nominal forecasts do not guarantee live clearance.
- [x] Specify/test a narrow named-neighbor contact contract with coupled finger
  checks and measured shelf support. Not integrated or physically validated.
- [ ] Share normal-force, tangential/yaw compliance, anti-windup, reference bounds,
  force release and retreat between insertion and lateral compaction.
- [ ] Detect jams by force plus lack of object progress; unload, back off and try
  a bounded alternative. Force aborts remain terminal until safe recovery exists.
- [ ] Validate the opt-in released/supported contact-reacquisition mode end to
  end. Its native tests preserve grasped/carry, support-loss, force and motion
  guards. New finger-abduction load compensation exposed downward-load/tilt
  failure; gate forward motion while unloading vertical force before promotion.
- [ ] Measure contact duration, object displacement, residual gaps, force peaks,
  recovery attempts and all-product stability; no success from commanded motion.

## 3. Robustness — after initial contact skills

- [x] Add compiled native checkpoints with complete integration state, explicit
  nominal/applied/controller history, corruption/runtime checks and strict
  state/force replay witnesses. Restores are separate zero-credit continuations.
- [x] Build five replay-validated contact focus states and a fixed held/compliant
  prefix comparison. All ten cases complete; no case passes squaring dwell.
  Retain native-import failures separately; see [checkpoints](docs/sim_checkpoints.md).

- [ ] Extend the measured CPU/Warp rollout primitives to diverse scripted
  contact continuations and live replanning. Audit MABEL model,
  actuator and force contracts before allowing GPU forecasts to guide execution.
  Consider mjlab for shared batching/instrumentation and Newton for later
  material/contact experiments; no backend migration or RL is validated.
  See [backend evaluation](docs/backend_evaluation.md).
- [x] Implement supervised CPU/GPU throughput probes in the main optional locked
  rollout group, plus per-step guarded native forecasts and C++ screening with
  a guarded shortlist. Initial 32-world held-target probes favor CPU. A native
  retained-state contact-search continuation exists; no fresh-episode stocking
  speedup is claimed.
- [x] Build a stationary bimanual/shared-lift WBC foundation with nominal/applied
  geometry admission and native unloaded tracking evidence. Mobile/loaded/contact
  integration remains open; see [whole-body control](docs/whole_body_control.md).

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

## 5. Demonstrations, teacher/student and sensing — in progress

- [x] Add a causal control-rate recorder: pre-action observation, nominal teacher
  reference, applied servo target and next observation. Keep privileged teacher
  state separate and preserve failure/unknown outcomes.
- [x] Audit legacy traces; refuse BC export when pre-action proprioception or
  nominal actions are absent. Diagnostic controls are not nominal teacher labels.
- [x] Export/reload a native ten-transition pipeline fixture with official
  LeRobot 0.7.0 locally. Preserve masks, action units/modes and outcome labels;
  no upload or successful stocking demonstration is claimed.
- [x] Add scratch-derived wrist reaction F/T, contact-only wrist wrenches and
  coarse native-geometry tactile force proxies with explicit frames/provenance.
- [ ] Wire the recorder into stocking and VR before all compensation paths;
  collect raw synchronized camera observations and verified task demonstrations.
- [ ] Add calibrated sensor noise/bias and hardware observation adapters.
- [ ] Train/evaluate a student with declared sensor inputs; add teacher queries
  on student-induced states after the teacher primitives are reliable. No RL yet.
- [x] Collect a small parallel pilot: three fresh controller/sensing fixtures,
  60 transitions and 126 raw camera images, with native guards and a verified
  real LeRobot multi-episode roundtrip. Training eligibility remains false;
  see the [pilot report](docs/reports/lfd_collection_pilot.md).

See [learning from demonstrations](docs/learning_from_demonstration.md) and
[simulated sensors](docs/sim_sensors.md). LeRobot's NumPy 2/Python 3.12 export
worker is independently locked; collection remains in the main simulator env.

## 6. Acceptance and evidence

- [ ] Close native process stability gate: primitive-only named-view accesses
  reproduce an intermittent signal. Retained model/data wrappers passed one
  227-test suite and exact replay, but a later frozen suite still segfaulted;
  preserve every unknown outcome. The mitigation is not a sufficient fix.
- [x] Freeze package, runner and control configs before each supervised probe;
  record wall budgets and process failures. Invalid/missing outcomes earn zero.
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
Tracking diagnosis: [sent-reference error](docs/reports/mobile_contact_reference_diagnostic.md).
Carry/insertion development: [payload and split insertion](docs/reports/payload_insertion_development.md).
Base evidence: [native wheel actions](docs/reports/native_base_motion_development.md).
