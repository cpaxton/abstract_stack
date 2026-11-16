# Review: Coding Agents for Generalized Task and Motion Planning Problems

Reviewed 2026-11-16: Merler, Li, Roy, Liang, Wang, Huang and Silver,
[arXiv:2609.30233v1](https://arxiv.org/html/2609.30233v1) (2026-09-24).
The project is Agentic GenTAMP; the paper calls its approach **AgenticGenPlan**.
Code: [tomsilver/robocode](https://github.com/tomsilver/robocode), README inspected
at `d3d42dd12e74c00383ceeee5e75f16bd8c4abc09`. Bibliography: `references.bib`.

## What the primary sources establish

The paper studies simulator-guided program synthesis followed by frozen,
LLM-free evaluation. Its main access setting hides environment source and uses
fully observed object states. Five runs per method/environment use a $20
synthesis budget and 100 held-out instances, with a 60-second evaluation limit.
The study covers 28 environments and 98,000 episodes. See §§II–IV.

For Dynamic3D Shelf, Table II reports mean success of 0.27 for the planner,
0.60 for Opus and 1.00 for Astra in the main setting. Figure 3 and §IV-B describe
an agent calibrating arm/grasp geometry by probing. These are benchmark results,
not MABEL results or evidence of house-scale continuous stocking. Dynamic 3D
remains challenging; §VI assumes privileged state and does not establish
robustness to perception uncertainty.

The [repository README](https://github.com/tomsilver/robocode/blob/d3d42dd12e74c00383ceeee5e75f16bd8c4abc09/README.md)
warns that paper Shelf is `dynamicshelf3d_generalized`; the similarly named
`shelf3d_generalized` is preliminary and kinematic. It also distinguishes strict
black-box flags from source-access defaults. It describes evaluation as new
instances from the synthesis distribution, not a separately held-out object-count
distribution. Exact archived settings matter when reproducing reported results.

## Our assessment and implications

The relevant idea is a development procedure, not a pretrained controller we can
drop into MABEL. We can borrow the cycle of targeted probes, policy revision and
freezing without importing the full benchmark stack as a runtime dependency.
Our current code reads the MuJoCo model and uses engineered IK/force routines.
We should call the first experiment source-visible skill-based synthesis and
measure how much scaffolding it supplies. A strict black-box arm would be a
separate future experiment, not a relabeling of the current one.

The Shelf calibration example motivates our own small identification tests, but
our observed sag has a different explanation: finite-gain servos under load.
We should first verify kinematic frames, then isolate actuator equilibrium and
tracking, rather than fit away load-dependent errors as constant geometry.
Measured feedback is required when object loads or contact modes change.

An agent should learn choices that matter across instances: product ordering,
grasp alternatives, which shelf side to approach, when to switch hands, and
whether to move an earlier placement. Repeating our current seed-0 waypoint
script is not a generalized policy. A good failure record should explain which
choice needs to change instead of reporting only a scalar IK objective.

## Concrete implementation sequence

1. Establish the conventional MABEL tracking baseline: bounded nominal
   references for both arms/wrists and the shared lift, robot-gravity feedforward,
   measured wrist damping, and predicted-versus-executed error logs.
2. Complete one real grasp/lift/release/contact-push cycle. Calibrate existing
   multi-joint hands and record retention/support evidence. Add carried-object
   swept-volume checks and monitored base support before tight insertion claims.
3. Factor a task-independent skill executor returning explicit failure reasons.
   Add grasp regeneration, alternate-hand selection, obstacle routes and guarded
   retreats; a staged near miss must never count as a grasp.
4. Wrap the simulator behind an episode/policy interface and freeze code,
   model, prompts, budget, controller and any learned weights. Keep test seeds
   outside the development process. Track policy compute separately from physics.
5. Compare hand-written, one-shot and iterative synthesis policies with identical
   physical controllers. Evaluate a compatible RL controller as an independent
   controller factor, including action/observation/model admission checks.
6. Expand from cartons to bottles, tight gaps, rearrangement, jams, tipped
   products, bimanual coordination and the house. Use separate in-distribution
   and shifted-distribution suites and preserve all unsuccessful episodes.

## Questions our experiments must answer

| Question | Required comparison or evidence |
| --- | --- |
| Does synthesis help task reasoning? | Same controller/skills, paired instances, hand-written versus synthesized task programs. |
| Does RL improve contact execution? | Same frozen task program, controller swap, retention/force/impulse and tracking outcomes. |
| Is recovery useful? | Declared failure fixtures; successful within-episode recovery and complete-shelf outcomes, with retry time included. |
| Does behavior generalize? | Private test instances after freezing; count/geometry/friction shifts reported separately. |
| Is a fast program actually faster? | Separate synthesis cost, policy compute, physics time and completion time; include failures/timeouts. |

The acceptance milestone is a physically verified stocking cycle, then a frozen
policy that completes unseen shelves. A paper scaffold and a successful IK call
are development tools, not substitutes for those outcomes.
