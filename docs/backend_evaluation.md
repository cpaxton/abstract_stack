# GPU rollout and contact-backend evaluation

Research checked 2026-11-17. No new backend is installed or validated here.
The current environment has an RTX 4090 (24 GB); `mjlab`, `mujoco_warp`, `warp`
and `newton` are absent from the project's Python environment. Do not replace
the native reference simulator or add unpinned dependencies as part of this note.

## Recommended order

| Option | Useful experiment | Admission work |
| --- | --- | --- |
| Native `mujoco.rollout` | CPU batch of short scripted continuations | Same model/state/control contract; measure batch latency |
| MuJoCo Warp directly | GPU batch of contact continuations and controller candidates | Import existing model, preserve actuator/contact contracts, validate force export and capacity |
| mjlab | Reusable batched scenes, observations, contact instrumentation and later learning | Check native actuator configuration and reset isolation; no trained policy is required for scripted actions |
| Newton | Alternative collision/contact models and future deformable cartons | Explicit robot/model conversion and solver/material/coupling validation |

MuJoCo's [GPU guide](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
documents model/data transfer, independent batched worlds, CUDA graph capture,
capacity/overflow checks and differences from CPU MuJoCo. It also identifies
native `mujoco.rollout` as a CPU throughput option. GPU batching is a candidate
for faster *search*, not evidence that a single Python-driven episode becomes
faster or that the current native crash has been fixed. Device/host transfers,
compilation and our Python IK loop need separate timing.

[mjlab](https://github.com/mujocolab/mjlab) combines the Isaac Lab manager API
with MuJoCo Warp. Its [architecture](https://github.com/mujocolab/mjlab/blob/main/docs/source/architecture_overview.rst)
separates simulation from the learning managers. We can investigate scripted
batched controllers without beginning RL. It is not an automatic force-control
or joint-limit solution. Verify the exact selected release and actuator behavior;
do not substitute an ideal PD actuator for MABEL's native servos unnoticed.

[Newton's MuJoCo solver](https://newton-physics.github.io/newton/stable/solvers/mujoco.html)
also uses MuJoCo Warp. Its optional
[collision pipeline](https://newton-physics.github.io/newton/stable/concepts/collisions.html)
supports SDF/hydroelastic experiments, while the experimental
[VBD solver](https://newton-physics.github.io/newton/stable/solvers/vbd.html)
supports rigid/cloth/soft-body interactions. Hydroelastic contact does not by
itself establish cardboard bending, creasing or crushing. Material calibration
and two-way robot/material coupling remain separate experiments; see
[the cardboard investigation](cardboard_simulation.md).

## First bounded prototype

1. Freeze a small set of measured states from retained failures. Keep reset-only
   replay/forecast results separate from fresh continuous-episode credit.
2. Run native CPU batch continuations first. Try 32 candidates and a short
   declared horizon; record the exact commands, state, timing and process result.
3. Evaluate proposed finger, tilt, arm posture and checked base/staging choices
   before contact. Score measured object progress, predicted joint headroom,
   contact retention, slip, force peaks and instability. Reject any candidate
   violating the existing guards. Kinematic feasibility alone is insufficient.
4. Compare an isolated, pinned MuJoCo Warp environment on the same imported
   rigid model. Benchmark compilation, host transfer, batch simulation and
   decision latency separately. Increase the batch only within a declared budget.
5. Execute only the first short checked segment in the native environment,
   observe actual state/forces, then replan. Add mjlab's scene/manager layer when
   it saves concrete infrastructure work. A CEM/MPPI-style local candidate search
   is a proposal, not an implemented or benchmarked MPC claim.

Before GPU candidates guide execution, compare static support, lift/wrist
tracking, bounded grasp/release, a short push and an adjacent-carton case. Audit
all model arrays and actuator mappings, gains, limits, collision masks, masses,
inertias and integration timing. Validate world/tool wrench frames, torque signs,
summation across contact points and per-world indices. Capacity overflow, missing
forces, unstable states and process failures invalidate a candidate; retain them.

CPU/GPU trajectories are not expected to be bitwise identical. Predeclare
diagnostic comparison metrics and retain backend differences without changing
the task's success tolerances or force guards. Continue independent native
process-crash investigation; changing engines does not establish its cause.
