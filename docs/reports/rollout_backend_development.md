# CPU and GPU rollout development

The first measured improvement is a **2.6x reduction in shortlist evaluation
latency**: 0.808 s versus 2.123 s for checking every candidate. This reduces
the number of guarded continuations from 32 to four; it does not speed up the
same amount of force-checking work. Unchecked candidates cannot be selected.
These are held-target forecasts, not improved shelf-stock success or episode
completion time. Ten-carton stocking and native process stability remain open.

## Fixed inputs and scope

The main probe restores the failed terminal integration state of
`mobile-table-ten-v17/episode`: one carton is partly inserted and all ten
products remain represented. Each of 32 independent worlds holds the same
recorded native actuator targets for 250 steps at 2 ms, a 0.5 s horizon.
There are no fresh picks, placements or task credits. The source episode began
with ten table cartons and zero prefills; restoring its terminal state does not
create a new zero-prefill episode.

MABEL revision is `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`. The normal project
environment uses the optional locked `rollout` group: MuJoCo 3.17.0, MuJoCo Warp
3.17.0, NumPy 1.26.4 and resolved Warp 1.19.0. GPU hardware is an RTX 4090
(24 GB); native batch/threaded paths use four workers. Solver options, native
gains, limits, model geometry and collision masks remain unchanged. Model
contract hashes are checked before/after the final benchmark workers.

## Measurements

| Path | Wall time for main probe | Evidence available |
| --- | --- | --- |
| Serial native step + forward | 3.836 s | Post-step states; no contact admission in this timing |
| Four-thread C++ native rollout | 0.528 s | All trajectory states; no per-step force admission |
| Four-thread native guarded rollout, all 32 | 2.123 s | Every physics step: forces/support/contact/all-product guards |
| C++ screen 32, guard best four | 0.808 s | Four complete guarded forecasts; other 28 remain ineligible |
| Warp GPU step + forward, 32 worlds | 9.008 s | Finite terminal state and latched overflow check; no force admission |

Native values above come from `final-guarded32`; the GPU value comes from the
immediately preceding `final32` probe. Both restore identical exported XML and
terminal-state inputs. Each has its own frozen script/module/lockfile hashes.
The later native guard explicitly includes backstop force; all backstop loads
in that held-state probe were zero. These development timings are single
measurements, not a statistical speed benchmark or a frozen-policy task suite.

The native C++ path and the step-plus-forward reference reached bitwise-identical
terminal physics states in these held-target probes. That does not establish
general equivalence for every model or command sequence. Guarded shortlist
selection always reruns from the original integration state and ranks by its
own measured progress, retaining rejected and unchecked candidates separately.
All paths change only actuator controls after independent reset construction.

Warp is slower at this model/batch size. Reducing GPU capacities from
256 contacts / 2048 constraints to 128 / 512 changed execution from 9.36 s to
about 9.04 s in the earlier comparison; no overflow occurred. Final sampled
solver counts were five to seven iterations, with 205–211 constraints per world.
Those terminal counts do not establish iteration costs throughout the horizon
or explain the GPU bottleneck. No solver tolerance, contact acceptance or force
guard was weakened. This result favors CPU batching for the next local search
prototype; it does not establish GPU performance at much larger batch sizes.

The successful GPU probe reused compiled kernels. Its reported setup/capture
time was about 0.75 s and is separate from execution. Cold compilation was not
free: an earlier process spent 61.92 s before a retained compilation signal.

## Numerical diagnostics and failures

In `final32`, native/GPU model-contract hashes match. After 0.5 s of held
commands, maximum hinge-position difference was 0.000317 rad, maximum slide
difference was 0.000000109 m, and the active carton's largest coordinate
difference was 0.0000199 m. Absolute simulation-clock difference was 0.000270 s.
These are terminal diagnostics, not force equivalence or contact-rich trajectory
validation. Raw states and `parity_diagnostic.json` remain with the bundle.

The sandboxed smoke process could not see CUDA and refused CPU fallback.
An explicitly authorized CUDA process then imported MABEL and compiled stepping,
but exited with signal 11 while compiling `mujoco_warp.set_state`; its outcome
is unknown. The next implementation uses a new independent `put_data` state
after a disposable compile forecast. Initial qpos/controls and unadvanced CUDA
capture time are checked. It passed the subsequent GPU probes, with no overflow,
but does not fix or explain the earlier state-helper failure. See local SIM-002
in [upstream concern tracking](../../home_robot_issues.md); nothing was sent.

All attempts, including failures, are retained in
`outputs/rollout-backends-v1/` and summarized in the
[machine-readable ledger](rollout_backend_development.json). The supervisor
declares a separate process-group wall budget for each backend and records
unknown outcomes rather than granting task credit.

## Validation and next integration

Ten lightweight native rollout tests passed, covering independent threaded
states, target/rate rejection, non-grasping links, summed carton-pair force,
backstop force, retained transient tip evidence, external-force rejection and
guarded-only shortlist selection. The separate stationary bimanual/shared-lift
prototype passed six focused native tests and tiny unloaded tracking probes;
see [whole-body control](../whole_body_control.md). Package build, lockfile check
and hooks are separate checks. The previous incomplete full-suite/push and
native crash records remain unresolved; no broad stability fix is claimed.

Next, replace identical hold candidates with bounded, coupled-geometry-admitted
contact continuations, evaluate ranking quality and first-segment execution,
and measure end-to-end replanning latency including candidate generation.
The current API does not admit live execution, provide a grasp/carry controller,
or certify final dense-row tolerances. GPU force reduction and calibrated
native/GPU contact comparisons remain prerequisites for GPU action selection.

Reproduction commands and environment setup are in
[backend evaluation](../backend_evaluation.md).
