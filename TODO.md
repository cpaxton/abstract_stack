# Ten-carton contact-rich stocking

Agreed scope: ten rigid cartons start on the supply table; zero prefills; one
uninterrupted native-MABEL episode. Start with 10 mm total unused row width, then
5 mm. World-state access is allowed. Delay modeling and RL are deferred.

## Current focus: fixed shelf and perfect state

Active development ladder (`feat/sim-checkpoints`, 2026-11-18): preserve the
successful spacious three-carton baseline, integrate released-carton selected-
finger insertion, then run fresh tight three/four/ten-carton episodes. Use
`scripts/run_stocking_case.py` with the presets under `configs/scenes/` and
`configs/controllers/adjacent_insertion.json`. Freeze source per run; retain
all failures under `outputs/adjacent-insertion-v1/`. These are development
trials, not a generalization success rate.

The current ten-carton development suite uses the 20 mm row to advance the
sequence. Keep the 10 mm whole-row benchmark for acceptance. Reuse the saved
fifth approach and ninth front-square boundaries from the
[development curriculum](configs/curricula/development.json). Compare any
candidate on other lineages and preserve exact terminal failures when admission
requires rewinding to an earlier boundary.

- [x] Finish ten checkpoint-assisted withdrawals and verify the packed hold in
  the 20 mm development row with zero prefills and zero tip/drop incidents.
- [x] Complete the 20 mm packed-first row from reset on seed 5. All ten
  withdrawals and the packed dwell passed at 1799.11 simulated seconds in
  the original 1800 s budget.
- [ ] Reproduce a packed row from reset on both 20 mm development seeds
  (5 and 7), including all ten picks, placements and withdrawals. Report
  strict positioning at the end of each run and after any later alignment
  pass. Then freeze the controller for three untouched starts at 10 mm
  before 5 mm.
- [x] Run the successful frozen policy on a three-case public pose suite with
  the same ten-carton geometry and 20 mm budget. Retain all grasp and side-push
  failures and leave the paired held-out split sealed.
- [ ] Validate recovery from the seed 7 shallow landing through final retreat.
  Exercise the same re-registration and turn recovery on independent pose
  variants before sweeping object size, shelf height/depth or materials.

- [x] Share the existing dense selected-finger skill with ordinary stocking;
  opt in with `--ungrasped-insertion`. Recheck the pusher after native base moves.
- [x] Choose neighbor-clear front staging during carried transfer, hold the arm
  during grasp unloading, and ramp opening from applied native finger controls.
- [x] Integrate guarded short native forecasts for backstop squaring and measured
  contact-wrench feedback through release/retreat. Saved-state continuations
  remain zero fresh-task credit.
- [x] Complete the fresh tight three-carton sequence: all stages 3/3, zero
  prefills/incidents, 208.198 simulated seconds (`tight-three-14`). This has
  spacious walls and 3 mm target inter-carton gaps; it is not the whole-row test.
- [x] Verify nine retreats without restarting between objects in the 20 mm
  development row. Seed 5 completed ten picks, ten shelf-supported
  placements, five strict final positions and nine retreats. See the
  [fresh results](docs/reports/whole_shelf_development.md#dense-recovery-from-an-empty-shelf).
- [x] Finish the tenth retreat from seed 5's saved unloaded stop using a
  screened chassis move. Verify the complete packed hold and retain the
  exact arm-corridor failure in the retreat curriculum.
- [x] Recover the ten-carton edge-slot transfer from a saved native state:
  interior front handoff, measured sidewall bracing, outer-finger insertion,
  backstop contact and full shelf exit. Final yaw -0.07586 degrees; zero
  tip/drop incidents; zero fresh-task credit (`ten-edge-transfer-11`).
- [x] Preserve intermediate front-registration checkpoints and exact tool/
  finger-controller state, so insertion can be retried without replaying grasp
  and transfer. Empty navigation checks the supply workspace and keeps long
  base segments inside their motion budget.
- [x] Test two native fingertip pads through insertion, backstop squaring,
  unloading and retreat from saved four-/ten-carton skill entries. These are
  zero-credit continuations, not fresh episodes. Check partial thumb extension
  against both walls and the planned departure before opening.
- [x] Close lateral registration on the retained interior insertion: bounded
  fingertip load compensation and midpoint integral feedback reduce the latest
  12.03 mm release error to 0.179 mm (`ten-finger-load-insert-02`). Final yaw
  -0.0265 degrees, complete withdrawal, no incidents; zero fresh-task credit.
  Repeat in a fresh episode and retain the wall-side regression: the single-pad
  wall skill must rank squaring references, while paired pads can rank lateral
  registration. Do not relax the 25 N guards or task geometry criteria.
- [x] Verify hysteretic base stopping on the retained inventory route: all five
  waypoints pass, final position error 1.33 mm inside the original 2 mm bound.
  This continuation earns no fresh-task credit.
- [x] Add measured wall-first order and neighbor contact references, with an
  explicit reserve for all remaining carton widths; retain canonical task goals.
- [x] Recover the first wall-slot release/side-push/insert/square/exit sequence
  from its saved state (`ten-wall-first-opening-01`, yaw -0.02650 degrees,
  no incidents, zero fresh credit). Verify all ten consecutively next.
- [x] Attribute passive drift by exact native replay and retain numerical-contact
  ablations separately. The active wall-first run keeps the original model.
- [ ] Complete ten cartons, zero prefills, 10 mm **total** row free space. Both
  sidewalls participate in planning and native summed-force guards. Then 5 mm.
- [ ] Freeze the successful controller, validate untouched scenes, and generate
  exact actuator-replay third-/first-person videos and a stage-by-stage report.
- [ ] Resolve intermittent native crashes before declaring validation complete.
  Retain the unknown four-carton episode and native-test crash logs; a later
  successful retry does not replace either outcome.

Keep one fixed single-level shelf, carton geometry and reset seed while closing
the manipulation loop. The randomized suite remains available for later testing;
do not tune on its private held-out cases. Use a predictive planner for short
object/contact references and a faster compliant controller for execution; see
the [two-layer contract](docs/receding_horizon_stocking.md#fixed-shelf-two-layer-controller).

- [x] Re-run the saved single-shelf contact finish with the current source:
  squared support, unloading and measured retreat pass. Generate synchronized
  third-/first-person videos and retain exact native replay checks.
- [ ] Emit a common reference packet with measured-state identity, object goal,
  tool/contact target, nominal posture, force target, motion limits, expiry and
  predicted object poses. Reject stale references before commanding motion.
- [x] Admit the second pickup after earlier motions disturb the remaining
  inventory. A fresh fixed `pick_place` seed-0 attempt completes carton 1, then
  aborts carton 0's approach at 25.17 N despite accepted kinematic IK/path checks.
  Forecast intended hand contact and replan the grasp from measured object pose;
  preserve the 25 N guard. See [video evidence](docs/reports/fixed_shelf_videos.md).
  Object-aligned entry and bounded centering offsets clear the top collision.
  The frozen teacher passes 8 development front pickups, 3 isolated held-out
  front pickups and a saved-state rotated-carton recovery. These are scoped
  skills, not arbitrary-order or full-sequence robustness.
- [ ] Close the next carton's lateral registration and squaring together, then
  verify release/retreat. Retain the current 3.228-degree/50-mm failure.
- [x] Connect those primitives in one fresh fixed-shelf episode, initially two
  cartons, before increasing toward ten. Both pass every stage in 90.7 simulated
  seconds, with zero prefills or tip/drop incidents. Causal LfD remains open.
- [ ] Measure prediction versus execution error and planner/controller timing
  separately; freeze this baseline before introducing scene or sensor variation.

## Skill curriculum and current evidence

- [x] Create seeded native reset environments for grasp/lift, intermediate
  insertion, forward push to contact, and lateral row compaction. Track
  supported staging separately from strict final position. Keep row pressure,
  measured opening, and post-unload alignment in the skill evidence.
- [x] Verify contact-mediated row compaction and unloaded opening on two
  public two-carton resets. Strict post-compaction alignment is still open.
- [ ] Add checkpoint-backed environments for loaded transfer, front release,
  and withdrawal. Exercise learned or scripted policies on identical resets
  before comparing fresh full-shelf runs across geometries.
- [ ] Validate row compaction in both directions and with rear clearance.
  Select the push direction and contact face from measured geometry, then
  interleave verified compaction in fresh stocking plans.

- [x] Separate callable grasp/lift and transfer/placement; record five skill
  entry distributions and full native checkpoints with explicit nominal/applied
  references and retained observer history. See [skill contracts](docs/skills.md).
- [x] Add isolated grasp/lift/0.5-second-hold tests, independent held-out seeds,
  strict native sequence replay and a passed-held-out-first recovery tool.
- [x] Align the lifted carton to the shelf frame instead of preserving table
  yaw. Generate an exact actuator-replay video of two consecutive complete stocks.
- [x] Extend to three cartons. The supply layout requires a 0.42 m table; the
  first pickup passes, but loaded staging hits native shoulder limits. A checked
  loaded base relocation (planned on a scene occupancy grid with `emet` A*) and a
  checked Cartesian pregrasp route now complete all three: pick/place/final
  position/retreat 3/3, zero prefills or tip/drop incidents, 8.06 N peak
  robot/product load. See [navigation and three-carton](docs/reports/navigation_three_carton.md).
- [ ] Extend to four cartons, then begin dense packing with the same base-navigation
  and grasp/transfer admission. The 2D occupancy grid already re-marks randomized
  product positions as obstacles.
- [x] Add a separate fixed 60 mm rear-clearance layout and labeled prefill preview.
- [ ] Implement pose-based depth registration and final rear-gap checks for that
  layout; the current backstop controller does not solve it.
- [ ] Collect causal skill demonstrations and cover additional task-induced
  starts before training. Held-out sequence testing remains 3/6 scheduled picks:
  all first pickups pass, and transfer prevents the three second pickups.

Reports, source IDs and videos: [grasp and longer execution](docs/reports/grasp_skill_development.md).

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
- [x] Add dynamic-neighbor reset fixtures, fixed left/right held comparisons and
  checkpoint continuations. A supported front-face recovery passed the unchanged
  0.15-degree/0.12-second backstop gate in one fixture (2.748 to 0.143 degrees;
  2.655 N robot/product peak). Zero fresh task credit; the original task above
  stays open. See [squaring evidence](docs/reports/environmental_squaring_development.md).
- [ ] Establish measured left/right neighbor bracing; current carton-pair forces
  are zero. Test both yaw signs, unloading and retreat, then consecutive stocks.
- [x] Verify supported unloading/retreat in the saved squaring fixture. A tighter
  0.05-degree controller target preserved the unchanged 0.15-degree task gate
  after release. Actual hand clearance was 39.924 mm back / 20.011 mm up, with
  no tip/drop incidents. Native verified replay now covers 20.166 seconds; see
  [longer sequence and videos](docs/reports/squaring_withdrawal_development.md).
- [x] Connect the cleared withdrawal checkpoint to checked shelf clearance,
  grasp-preshape transition and next physical table pickup. Normal environment
  stepping retains per-physics-step product/stage observers. A 153.386-second
  verified replay now includes carry, shelf support, insertion and unfinished
  backstop recovery; see [continuous development](docs/reports/continuous_stocking_development.md).
- [ ] Finish the second carton's unchanged final-position/retreat gates. Current
  recovery reaches 3.228 degrees yaw, with lateral drift; diagnose compensated
  contact-corridor rejection and add measured lateral registration before
  claiming consecutive complete stocks.
- [x] Replace repeated Python trees in new continuation/contact/video runs with
  shared immutable source bundles, Git/dirty metadata and pinned checkpoint refs.
  Portable/legacy checkpoints remain supported; historical evidence is retained.
- [x] Prototype measured wrist-load compensation for grasp/carry; reserve nominal
  wrist and lift headroom without changing native ranges, gains or force limits.
  Test native torque sign and applied geometry before actuation. Not promoted.
- [x] Add geometry-derived elevated inventory transit and sampled rigid-payload
  clearance checks. These nominal forecasts do not guarantee live clearance.
- [x] Specify/test a narrow named-neighbor contact contract with coupled finger
  checks and measured shelf support. Not integrated or physically validated.
- [ ] Share normal-force, tangential/yaw compliance, anti-windup, reference bounds,
  force release and retreat between insertion and lateral compaction.
- [x] Prototype paired nominal/applied reference projection without widening
  bounds. Matched 1 N contact dwell: existing 0/2, projected 1/2 saved states;
  separate 2 N development probes pass both in 0.856–1.046 s. Not fresh stocks.
  See [compliance projection evidence](docs/reports/compliance_reference_projection.md).
- [x] Prototype refreshed gravity/contact-torque compensation for unloading.
  Matched release: frozen 0/2, refreshed 2/2 saved states; combined square/release/
  retreat passes 2/2 in 7.704–7.926 s. Preserve failures and narrow final yaw margin;
  see [release evidence](docs/reports/compliance_unload_feedback.md).
- [ ] Validate load-aware withdrawal on new skill-entry states and bounded
  actuation/observation errors, improve yaw margin, then integrate fresh full
  sequences. No default promotion or held-out robustness claim from checkpoints.
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
- [ ] Profile candidate generation, IK, collision queries, and guarded forecasts
  separately. Batch the hot queries through the existing native backend and
  measure decision latency on saved opening and withdrawal failures. Evaluate
  a Rust extension if map integration or swept checks remain a bottleneck.
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
- [ ] Prototype a local voxel-map plugin around the shelf opening. Fuse calibrated,
  timestamped depth into observed free, occupied, and unobserved space in the shelf
  frame. Provide clearance queries for swept finger, palm, wrist, and arm geometry
  through opening and withdrawal. Use tracked carton and shelf geometry to name
  intended contacts, and carry pose uncertainty into the clearance checks. Start
  with simulated depth and evaluate false clears, unnecessary rejections, and
  query latency against MuJoCo on saved failure scenes with occlusion and pose error.
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

- [x] Add a deterministic geometry DR curriculum with separate private held-out
  seeds, 2–10 objects, 1–3 shelf levels, dense/carton/mixed-house profiles and
  all-object evaluation ledgers. Four native construction/hold checks passed;
  the first development teacher case failed before pick. This is not robustness
  proof; see [DR suite and evaluation](docs/domain_randomization.md).
- [x] Carry the raw-camera LfD pilot forward and collect two DR sensing fixtures
  (40 transitions/84 RGB). Official LeRobot export/reload passed; they remain
  training-ineligible and no student has been trained.
- [ ] Hook causal collection into actual manipulation before all compensation,
  collect verified development demonstrations, then evaluate a frozen teacher
  on the untouched held-out suite. Add supported multi-level/bottle adapters;
  keep unsupported domains and unrun objects explicit in the report.

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

The historical full-ten attempt stopped at its first loaded transfer. Current
continuation work is on `feat/sim-checkpoints`; complete ten-carton success
remains open. See [adjacent insertion development](docs/reports/adjacent_insertion_development.md)
for the newer three-carton success and retained four/ten-carton failures.

Design and references: [receding-horizon plan](docs/receding_horizon_stocking.md).
Initial evidence: [planning probes](docs/reports/replanning_development.md) and
[sideways contact probes](docs/reports/lateral_contact_development.md).
Tracking diagnosis: [sent-reference error](docs/reports/mobile_contact_reference_diagnostic.md).
Carry/insertion development: [payload and split insertion](docs/reports/payload_insertion_development.md).
Base evidence: [native wheel actions](docs/reports/native_base_motion_development.md).
