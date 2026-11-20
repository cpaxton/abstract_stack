# Gripper design tracker

Reviewed November 17, 2026. This is a living reference and experiment register,
not a selected hardware BOM. Three fingers, independent actuation, compliant
pads, rigid nails and tactile feedback are separate design choices. Compare
their contribution to complete interactions rather than jaw fit alone.

## Reference register

| Design | Architecture and available assets | Relevance | Status |
| --- | --- | --- | --- |
| GET / Grasp EveryThing | 1-DoF parallel actuation; three tapered fingers, two against one; rigid nails, optional optical tactile sensing; robot and UMI CAD | Low-action-dimensionality three-contact baseline; matched collection geometry | Paper and repository reviewed; not built or tested here |
| Dream Gripper | DK1-compatible printed backbone, TPU insert and grip tape; regular/Jumbo CAD and STL | Simple rigid-tip/compliant-pad fabrication baseline | Repository reviewed; not built or tested here |
| Conventional parallel jaw | Specify an actual actuator and finger/pad geometry | Matched baseline for all comparisons | Hardware selection open |
| Existing MABEL ORCA hands | Current native model and scripted preshape | Regression control; isolate robot/controller changes | See current reports; not our custom-hand result |
| Independent three-finger prototype | Separate finger commands, proposed sensing and rigid nail | Test whether independent contact placement earns added complexity | Proposed; no CAD or physical validation claimed |

## GET: paper and release

- [Paper, v1](https://arxiv.org/html/2505.09771v1), Burgess and Adelson:
  *Grasp EveryThing (GET): 1-DoF, 3-Fingered Gripper with Tactile Sensing for Robust Grasping*.
- [Reviewed repository](https://github.com/GelSight-lab/GraspEveryThing/tree/baface2271e71b45029b5c30f31ec01401b0232f).

The authors report teleoperated comparisons on 15 objects and three tasks;
these are not autonomous shelf-stocking or human-speed benchmarks. Normal-force
estimation has 1.34 N validation RMSE and 1.40-2.83 N errors on three unseen
objects. This is not demonstrated multi-axis force sensing.

The release includes ALOHA, Panda and UMI finger CAD, a force-probe design,
streaming/training code and weights. Simplified VHB/grip-tape fingers avoid
gel molding. The tactile version requires molded silicone, lighting and a
camera; the paper's rigid nail is not evidence of a metal nail. One actuation
DoF cannot independently place fingers or perform general in-hand reorientation.
Repository-level license terms were not found in the reviewed tree; clarify
before incorporating assets. The paper's license does not resolve CAD/code rights.

## Dream Gripper: release and evidence boundary

- [Reviewed repository](https://github.com/Dream-Machines-Robotics/Dream-Gripper/tree/dc5a96f0760428b8a7d1f112eac6ed7c005771cd).
- This is a separate design, not the repository accompanying the GET paper.

The README specifies a printed rigid backbone, TPU95A insert and grip tape.
The slim printed tip provides its nail-like contact; it is not a metal nail.
PLA-CF and PA-CF backbone variants, regular/Jumbo Fusion 360 sources and STL
files are present. Jumbo is described as reducing bending and improving grip
on wide flat objects. A separate Tentacle variant is also included.

The repository claims less than EUR 1 of printed material per gripper; this is
not a complete robot gripper, tooling, labor or sensing cost. Photos and author
descriptions support a fabrication reference, not a quantified success-rate,
payload, durability or closed-loop-force result. Visible deflection is a
potential observation, not an already calibrated force sensor. No controller,
tactile pipeline or MuJoCo model was found in the reviewed tree. MABEL/UMI
mounting, actuator compatibility and collision geometry need engineering.
No repository license file was found; resolve reuse terms before copying assets.

## Experiments these references motivate

Use the same actuator where feasible to compare flat pads, Dream-style compliant
inserts and GET-style three contacts. Separate unavoidable embodiment differences
from controller or data advantages. Then compare independent finger actuation
only when a task exposes a specific limitation of the one-DoF designs.

| Question | Proposed test | Required measurements |
| --- | --- | --- |
| Does contact geometry improve retention? | Matched-object grasp/carry with controlled torque disturbances | Slip, rotation, grip force, payload, task success and time |
| Does the nail help without creating withdrawal problems? | Thin-object pickup followed by crowded placement/release | Clearance, snagging, neighbor displacement, peak force and damage |
| Is passive compliance enough? | Same insert geometry across stiffness/force settings | Tracking, force, retention, wear and speed tradeoff |
| Is a sensed finger useful? | Mechanics fixed; controller with/without tactile input | Held-out success, latency, force error and interventions |
| Are independent fingers necessary? | Side pushing/stabilization while retaining the product | Complete interaction outcome versus one-DoF baseline |
| Does matched collection scale? | Robot/handheld tool pair; same operator labor budget | Calibration, demonstration quality, transfer and learning curves |

Begin with printable non-tactile mechanical variants and an instrumented test
fixture before committing to gel fabrication or custom electronics. Test actual
materials and pad adhesion. In simulation, declare rigid/compliant approximations;
do not assign a friction coefficient and call the physical design validated.

The independent-finger prototype remains an experiment. A stronger one-DoF
baseline could reduce the actuation needed for the first workflow without
changing the general mobile manipulation objective.

## Per-prototype record

For every variant, record: design/revision and license status; actuator and
transmission; commanded versus passive DoFs; mass and inertia; stroke and contact
geometry; nail material/thickness; pad material, hardness and adhesion; sensor
coverage, frame, bandwidth and latency; CAD/STL and collision proxies; robot/UMI
mounts; BOM and fabrication labor; calibration; test distribution and failures.
Distinguish author claims, files inspected, reproduced measurements and proposed
adaptations. Link outcomes to [feasibility gates](technical_feasibility.md).
