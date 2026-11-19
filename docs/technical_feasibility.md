# Technical feasibility plan

Status: proposed acceptance plan, November 17, 2026. Threshold's objective is
general real-world mobile manipulation. Continuous shelf stocking is the first
integration experiment, not a commitment to independent grocery as a market.
This document adds no claimed experimental results or implementation guarantees.

## What exists, and what it proves

Reviewed baseline: `c74837384559479ca1b7522e446801e313dd026a`.

The [compliance study](reports/compliance_followup.md) reports 40/40 two-carton
scenes across four ten-scene suites, with all 80 cartons passing pick, support,
final-position and retreat gates. Its selected settings reduce the mean of
episode force peaks relative to the paired earlier policy. Three suites were
already known; the newly frozen-policy suite has ten scenes. Do not describe
40/40 as an independent broad generalization estimate.

Native actuator execution, fixed acceptance criteria, paired settled-state
hashes, retained unknown outcomes and per-physics-step monitoring are valuable
foundations. They establish a narrow scripted simulation capability. They do
not establish dense packing, recovery, bottles, physical hardware reliability,
usable VR, learned policies, coordinated bimanual action or mobile task control.

The current program uses exact product poses and simulator contact information.
It completes successful two-carton scenes in roughly 87-88 simulated seconds.
There is no matched human baseline establishing human pace. See the
[proof-of-concept critique](poc_critique.md) for code-level evidence and priorities.

## Hypotheses to test

1. Contact-aware control completes constrained interactions that a strong
   conventional baseline cannot complete as reliably at the same pace.
2. Purpose-designed fingers/grippers improve that tradeoff through mechanics
   and informative sensing. A custom hand must earn its complexity.
3. Matched UMI/teleop tools reduce teaching effort and transfer useful contact
   behavior to the robot. VR demonstrations and robot-free UMI are distinct.
4. Detection, recovery and operator takeover preserve whole-episode progress.
5. These capabilities survive physical base motion, coordination, imperfect
   observations and eventual real hardware.

No single demo answers all five. Advance task difficulty without silently
changing the observation privilege, hand, controller and success criteria together.

## Experiment contract

Before each campaign, freeze task geometry, tolerances, materials, observation
access, controller/source/model revisions, timing, development budget, evaluation
distribution and failure-injection protocol. Use development, public regression
and genuinely untouched evaluation cases separately. Once inspected, an evaluation
suite becomes regression data for subsequent algorithm selection.

Report all scheduled episodes, full completion, assisted completion, per-stage
counts, incidents, unknown processes, simulation time, wall time and assistance
duration. Keep failed prefixes and post-abort behavior distinct. Never silently
replace a crashed run with a successful retry. Seed numbers alone are not a
distribution shift: vary geometry, physical parameters and task structure explicitly.

Every gate below is a proposed engineering target, not a demonstrated rate or
commercial reliability requirement. Freeze numerical thresholds before collecting
acceptance data; justify any revision and create a new benchmark version.

## First stocking experiment

Start with F0 by replaying the pinned spacious two-carton baseline on the
public regression scenes. Record each scheduled carton's pick, shelf support,
final positioning and retreat, along with all-product incidents and process
outcomes. Match the recorded scene, controller, model and settled-state inputs.
Keep any mismatch or failed process in the record.

Then run the first F1 development comparison on matched dense scenes with
the same MABEL hand, observation access, timing budget and task criteria.
Compare the existing scripted controller with the contact-aware candidate.
Start with the one-insertion curriculum to check force, support and withdrawal
accounting, then schedule a whole row from the table in one continuous episode.
Curriculum prefills earn no manipulation credit. This comparison can guide
controller development. A later frozen evaluation suite is required to assess
the proposed F1 completion target.

## Feasibility gates

| Gate | Experiment and proposed evidence | Dependency |
| --- | --- | --- |
| F0: reproducible baseline | Reproduce current presets and full stage outcomes at pinned revisions; archive controls, states, configuration and source hashes. Investigate mismatches before using this as a reference. | First |
| F1: continuous dense contact | Stock 6-10 mixed-size rigid cartons without resetting; test comfortable rows, then 10 mm and 5 mm total unused lateral width, counting wall gaps. Include randomized initial placements and insertion order. Provisional gate: at least 90% complete shelves on 100 frozen episodes per density, with uncertainty and every failure reported. | F0 |
| F2: recovery | Inject a blocked insertion, tipped product and failed grasp at declared phases; continue the same episode. At least 30 scheduled trials per failure class; provisional target at least 80% autonomous recovery within a predeclared class-specific time budget. Human rescue is a separate result. | F1 scaffold; can develop in parallel |
| F3: usable teleop and replay | At least three operators after a fixed training period; compare repeated tasks, takeover, clutch/recenter and retry behavior. Provisional target: at least 90% assisted completion across 30 episodes/operator. Measure command-to-motion, observation-to-display and data loss, plus perceived usability. Replay at original timing and quantify state divergence. | F0; parallel with F1 |
| F4: human-paced execution | Sweep trajectory speed with fixed contact limits; compare matched tasks and preparation against human performance. Provisional gate: at least 90% completion on 100 frozen episodes at human median task duration, with tail times, failures and intervention time included. Report tracking error and force versus speed. | F1 and F3 |
| F5: observation realism | Replace exact state/contact with explicitly available observations or estimates. Test noise, delay, dropout and occlusion separately before combining them. Compare to the privileged upper bound at matched pace; provisional allowable loss is at most 10 percentage points over 100 paired episodes. | F1/F4 benchmark |
| F6: learning and tooling | Compare demonstrations, robot-native corrections and matched UMI data at fixed collection labor and training budget. Hold out layouts, objects and sessions. Show learning curves and recovery outcomes; no numerical superiority threshold until baseline variance is measured. | F3 and observation contract |
| F7: bimanual mobile execution | Demonstrate genuine two-arm cooperation, then repeated shelf segments requiring physical swerve positioning and re-approach. Provisional gate: at least 90% whole-episode completion on 100 frozen episodes with declared coordination/navigation requirements. | F1/F2/F5 |
| F8: physical validation | One instrumented arm/tool pair, then full robot: characterize tracking, latency, loads and contact against independent measurements. Repeat the selected simulation interactions and reconcile failures before any field reliability claim. | Parallel hardware work; needed for physical feasibility |

The 90% gates are research progress thresholds. Commercial readiness will require
workflow-specific reliability, intervention and cost targets. Passing simulation
gates cannot establish physical material behavior or deployment economics.

## Dense task and contact physics

Keep current spacious pilots as regression tests. Add separately versioned dense
presets. Do not widen a shelf automatically to make a generated row fit: reject
invalid specifications and record the exclusion. Define density by usable shelf
width minus summed projected product widths, not just nominal target spacing.
Check wall/neighbor clearance and complete finger withdrawal, including previously
placed objects. Finite sampled IK checks are not continuous collision certificates.

Start with rigid cartons/bottles in pure simulation. Audit collision proxies,
friction, contact stiffness/damping, timestep and solver sensitivity. Check load
and impulse behavior against simple reference fixtures and convergence runs.
Separate numerical penetration from intended compliance. Add deformable packaging
only as a separately validated material experiment; rigid success says nothing
about carton crushing or elastic recovery.

## Controller, observation and data boundaries

Expose a common command path for teleop, replay and policies: timestamped nominal
references, hand state, base/lift commands, validity windows and explicit authority
handoff. Log nominal, compensated/applied and measured states independently.
Keep local contact regulation separate from slower planning and policy inference.
An observation interface must declare what is measured, estimated or privileged.

The current simulator's per-contact normals, object identities and exact world
poses are useful teacher/oracle inputs. They are not equivalent to wrist F/T or
tactile observations. Define sensor location, frames, bandwidth, latency, noise
and saturation before claiming deployable force feedback. Keep privileged labels
for evaluation even after removing them from the controller's observation.

Simulation time measures robot motion; wall time also exposes IK/inference and
rendering overhead. A physically correct trajectory taking minutes to plan cannot
establish responsive teleop. Human pace means matched end-to-end work, not merely
playing a video at 1x. Motion and force limits remain fixed during speed sweeps.

Record synchronized sensor streams, operator commands, applied controls, states,
calibration, frame transforms, mode switches, interventions and failure/recovery
labels. Demonstrate deterministic control replay before training where supported;
otherwise bound and report divergence. MP4 previews alone are not training data.

## Hand and matched collection experiment

The [gripper design tracker](gripper_design_tracker.md) records candidate designs,
asset availability and evidence gaps, with GET and Dream Gripper as initial
mechanical baselines.

Begin by reproducing the existing hand baseline. Then compare a conventional
parallel gripper and a three-finger prototype with declared geometry, mass,
actuation, sensing and controller differences. Include narrow placement, pushing,
release/withdrawal, slip and recovery. Factor mechanics, sensing, controller and
data source through matched ablations; do not attribute combined gains to one part.

Develop one physical finger/tool pair alongside simulation. Validate contact
sensing, actuator bandwidth, calibration, thermal limits and durability early.
A simulated finger or an ORCA preshape improvement does not prove the proposed
custom gripper. Matched tooling should improve learning per operator-hour and
reliability at pace; jaw-opening coverage alone is not the thesis.

## First work packages

| Work package | Suggested owner, subject to agreement | Reviewable deliverable |
| --- | --- | --- |
| Baseline replay and task contract | Chris | Pinned baseline manifest, stage outcomes and failure taxonomy |
| VR-to-simulator control and recording | Anuj | Short multi-object recording/replay with latency traces, clutch and takeover |
| Dense rows and controlled failures | Chris | Separate 10 mm/5 mm presets, blocked/tipped/failed-grasp injection protocol |
| Shared observation/action interface | Together | Versioned contracts and explicit oracle/sensor modes |
| Finger and UMI prototype | Owner to confirm | One instrumented contact-interface pair plus conventional baseline |

At the first integration review, require one reproducible native-physics episode,
one teleop recording/replay and one retained failure diagnosis. These are work
packages, not calendar or staffing commitments. Avoid adding more house scenes,
robot platforms or policy families before the next unproven gate is executable.

## Fundraising evidence boundary

Current evidence can support: native-physics scripted grasp/contact/release and
reproducible controller iteration on a narrow MABEL task. The next compelling
proof is dense continuous stocking plus recovery and usable teleop, followed by
matched speed/observation/tooling ablations and physical validation.

Track engineer-hours, operator-hours, compute and hardware spend per gate. An
inexpensive robot BOM does not establish inexpensive integration. Do not infer
customer savings, a commercial hand advantage or general mobile autonomy from
the present two-carton success rate. Customer discovery can run in parallel.
