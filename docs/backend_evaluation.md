# GPU rollout and contact-backend evaluation

Research and first throughput probes checked 2026-11-17. The main project now
has an optional locked `rollout` dependency group: MuJoCo 3.17.0, MuJoCo Warp
3.17.0 and NumPy 1.26.4, with Warp 1.19.0 resolved in `uv.lock`. The native
MuJoCo version and Emet Git revision are unchanged. mjlab and Newton remain
uninstalled. The workstation has an RTX 4090 (24 GB).

```bash
uv sync --locked --group rollout --inexact
```

This installs into the normal `.venv`; the GPU dependency is optional for
lightweight/macOS users. The initial isolated installation is retained only
under ignored `outputs/` and is not the benchmark environment.

## Implemented throughput probe

```bash
flock --no-fork /tmp/abstract-stack-render.lock \
  .venv/bin/python scripts/benchmark_rollout_backends.py \
  --episode outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode \
  --output outputs/my-fresh-rollout-probe \
  --worlds 32 --steps 250 --threads 4 --wall-seconds 60
```

Use a fresh directory and an existing complete episode bundle. The supervisor
copies sources, lockfile, exported XML and terminal state, records hashes and
versions, and runs each backend in a separate process group with a wall budget.
CUDA access may require execution outside this workstation's agent filesystem
sandbox; a failed CUDA initialization is retained and never becomes a CPU GPU
benchmark. All forecasts earn **zero fresh manipulation credit**.

`contact_rollout.evaluate_batch` owns one native data object per candidate and
checks forces, summed carton-pair loads, non-grasping robot links, all-product
tip/drop events, released-object support and contact loss at every physics step.
Native targets and rates are checked before stepping. Model arrays and the input
state remain unchanged. It is a supported-push forecast primitive, not a grasp,
carry or general whole-body policy.

`evaluate_screened_batch` ranks up to 32 native C++ trajectories, then replays a
bounded shortlist with the complete per-step guard. Unchecked candidates remain
ineligible. Final ranking uses guarded progress, and no candidate is admitted
for live execution by this function. The caller still needs task-specific IK,
coupled finger/path admission and a fresh-state execution contract.

The benchmark currently holds recorded native servo targets in identical cloned
states. It measures backend costs; it does not measure candidate diversity,
controller robustness or shelf success. C++ stepping and the Python
step-plus-forward reference reached identical terminal states in the initial
probes. Warp uses float32 and differs numerically. Its current fast path has no
per-step force admission and cannot select a live action. A first GPU state-helper
compilation crash is preserved; fresh native-to-GPU data construction then passed.
See [the measured report](reports/rollout_backend_development.md).

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
