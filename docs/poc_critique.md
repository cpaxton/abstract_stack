# Proof-of-concept critique

Reviewed source: `c74837384559479ca1b7522e446801e313dd026a`, November 17, 2026.
This is an assessment of technical evidence and experiment design, not a report
that the proposed feasibility gates have passed. See [the plan](technical_feasibility.md).

## Assessment

This is a useful manipulation testbed with unusually explicit evidence handling.
The strongest result is that native finger preshaping and force-aware scripted
execution can improve a complete grasp/carry/place/withdrawal sequence without
changing the task thresholds. Keep the physical stage checks and reproducibility
work. The main remaining risk is spending effort optimizing an easy oracle-state
task while the intended dense, recoverable, human-paced workflow stays untested.

## Findings in priority order

### 1. The successful distribution does not test the motivating constraint

`random_scenes.random_shelf_config` varies shelf height over 10 mm and front
position over 10 mm, with 35-55 mm nominal inter-product target gaps. It expands
shelf width when needed. The latest measured controller stocks two rigid cartons
using the left arm, with ample lateral space. More seeds in this family cannot
establish tight insertion, mixed products or general scene transfer.

Create a separately versioned 6-10 product task and hold its row width fixed at
the declared occupancy. Include order choices and prior-placement disturbance.
Retain the current family for regression rather than replacing its criteria.

### 2. Controller inputs are stronger than the proposed physical sensors

`StockingScript._sample` reads exact product pose, support and per-contact loads;
`_tick` uses actual object rotation and native geometry clearance. `env.observe`
also exports exact product poses. This is appropriate for teacher/debug control,
but does not yet validate tactile/vision-based feedback or the custom sensor bet.

Declare oracle and deployable observation contracts, then add bandwidth, latency,
noise, occlusion and estimation ablations. Keep the independent evaluator privileged.
Do not replace physically available sensing with unlabelled simulator ground truth.

### 3. Guarded aborts are not graceful recovery

`StockingScript.run` catches execution failures, records a result and stops. This
preserves evidence but does not demonstrate retreat from contact, regrasp, object
uprighting, human takeover or resumption. The report correctly limits no-tip/drop
claims to observed prefixes.

Add explicit recovery states with bounded retries and continued task accounting.
Evaluate controlled failures independently of naturally occurring successes.
Measure the post-abort state and safe continuation rather than assuming a hold
command establishes them.

### 4. Human-speed and responsive execution remain open

`ScriptConfig` defaults include an eight-second descent and 0.025 m/s pushing;
successful two-carton episodes take roughly 87-88 simulated seconds. These are
not matched human comparisons. Sampled IK and collision checking also contribute
wall-time latency that a simulation-time metric can hide.

Separate compute latency from tracking lag and motion duration. Publish phase
timing and a success/force-versus-speed curve before using human-speed language.
Do not obtain speed by loosening contact limits or suppressing checking.

### 5. Three control paths need one tested contract

The handwritten solver uses `MabelController`; the Emet adapter has its own
stationary-control path; the upstream MABEL VR/mjlab stack is not installed by
fetching model assets. The Emet adapter explicitly documents direct base-velocity
navigation rather than native wheel-driven swerve control. A shared robot model
does not establish equivalent teleop, policy, replay and mobile execution.

Converge on nominal/action/state/timing contracts and demonstrate authority
handoff and replay. For mobile-feasibility results, drive actual wheel actuators
and retain slip, steering and base tracking evidence. Keep pose/velocity-assisted
navigation results separately labelled.

### 6. Geometry success and strict process success are different interfaces

`env.info()['success']` reports stable geometric placement; `StageObserver` and
campaign summaries add stage and incident evidence. A caller using only the
environment flag could report success after transient tip/recovery or before a
verified withdrawal. Those may be legitimate distinct task outcomes, but the
names need an explicit contract.

Expose geometric completion, clean unassisted completion, recovered completion
and assisted completion separately. Future learning should not accidentally
optimize a weaker terminal flag than the benchmark being advertised.

## Scope discipline

Do not rewrite the entire script before proving the next task. Introduce a common
step/observation interface and extract grasp, insert, release and recovery modules
as experiments require them. The current approximately 1,400-line interaction
module couples phase logic, oracle sensing, planning, feedback and artifact
handling; that will otherwise make observation and recovery ablations difficult.

Preserve the tested baseline while adding dense tasks, recovery and VR. Household
renders, extra policy wrappers and a paper draft are useful only insofar as they
help run or communicate those experiments. No claim of novelty is established
by this code review; a literature comparison would be a separate investigation.

## Evidence used

- [Current compliance report](reports/compliance_followup.md)
- [Experiment index](experiments.md)
- [Controller and phase logic](../src/abstract_stack/interactions.py)
- [Random task distribution](../src/abstract_stack/random_scenes.py)
- [Environment evaluation](../src/abstract_stack/env.py)
- [Stage observer](../src/abstract_stack/evaluation.py)
- [Tracking controller](../src/abstract_stack/mabel_controller.py)
- [Emet adapter](../src/abstract_stack/robots/mabel.py)

Review findings above are static source observations and interpretations of
committed reports. They are not new measured manipulation outcomes.
