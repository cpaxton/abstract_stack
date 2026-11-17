# Home Robot / Emet issue drafts

Local tracking only. No issues have been filed. Keep future reports standalone:
use upstream commands or minimal upstream-only reproductions, with no downstream
project names, task descriptions, repository links, or private context.

## HR-001 — Missing LFS media blocks Git-based package installation

- **Status:** observed 2026-11-15; remote object availability has not been rechecked.
- **Revision:** `ee3f1c9461c6e18654143a04461172bd175ffa62`.
- **Observed failure:** installing `emet` from the Git repository fails during
  `git reset --hard` because the LFS object for
  `docs/experiments/media/contact_20261013/contact_evidence.png` is unavailable.
  Object ID: `796d8354dc614ebd052215aea0f3b4fd27d8a541b5d824d29729206156617402`.
- **Reproduction:** in a fresh uv cache, install
  `emet @ git+https://github.com/cpaxton/home_robot_v3.git@ee3f1c9461c6e18654143a04461172bd175ffa62`
  with Git LFS smudging enabled. The failure happens before package dependencies
  can be built; documentation media is not required for importing the package.
- **Workaround:** set `GIT_LFS_SKIP_SMUDGE=1` on the installation command.
- **Proposed upstream fix:** restore the referenced LFS object and audit other
  published pointers; consider distributing Python wheels so a package install
  does not depend on fetching documentation media.

## HR-002 — Missing observations bypass publisher pacing

- **Status:** source confirmed and runtime reproduced 2026-11-15.
- **Reviewed revision:** `6c2ce9d841e7de36cd518d3a71f48960e97ad377`.
- **Location:** `src/emet/core/server.py`, `spin_send`, `spin_send_state`, and
  `spin_send_servo`.
- **Problem:** each loop immediately executes `continue` when its observation
  method returns `None`. The normal pacing sleep is below that branch and never
  runs. Disabling cameras causes the full/servo publishers to spin continuously,
  consuming CPU and slowing the physics/control threads.
- **Reproduction:** run an existing MuJoCo server with an empty camera list, or
  make its image/servo observation methods return `None`. Compare CPU usage and
  physics progression with those unused publishers disabled. Setting the
  publication-rate environment variables alone does not fix the `None` path.
- **Workaround:** do not start unused image/servo publication loops in a
  state-only server.
- **Proposed upstream fix:** apply a bounded idle sleep on all `None` paths and
  offer explicit per-channel enable/disable options. Add a regression that
  counts retries over a bounded interval without requiring a GL context.

## HR-003 — Simulation-server construction downloads speech assets

- **Status:** source confirmed and runtime reproduced 2026-11-15.
- **Reviewed revision:** `6c2ce9d841e7de36cd518d3a71f48960e97ad377`.
- **Location:** `BaseZmqServer.__init__` in `src/emet/core/server.py`, and
  `PiperTextToSpeech.__init__` in `src/emet/audio/text_to_speech.py`.
- **Problem:** when the speech module imports successfully, the base server
  constructs Piper unconditionally. The `text_to_speech_engine` argument does
  not control this branch. Constructing a simulation server in a fresh working
  directory downloads the Piper archive and voice files into `piper_tts/`, even
  when no speech operation was requested.
- **Reproduction:** instantiate an existing MuJoCo ZMQ server in a fresh working
  directory with speech imports available. Observe downloads during construction,
  before the server handles any speech command. Passing a disabled speech-engine
  value does not change the unconditional constructor branch.
- **Workaround:** suppress the speech constructor branch for simulation servers
  that do not need speech.
- **Proposed upstream fix:** honor an explicit disabled engine and initialize
  speech lazily on the first speech operation. A constructor regression should
  assert that a speech-disabled server makes no downloads or speech directories.

## SIM-001 — Intermittent native MuJoCo signal during MABEL control

- **Import-only failures during checkpoint work:** separate capture and replay
  children segfaulted while importing the native `_specs` module, before any
  checkpoint model was loaded. Logs remain in
  `outputs/checkpoints-v1/carton8-stalled/` and
  `outputs/compliance-focus-v2-fresh-verification/yawed_back_contact/`.
  A bounded GDB capture and other ordinary MJB replay processes completed, but
  do not replace these unknown outcomes or identify their cause. MJB avoids XML
  recompilation on restore; it is not a fix for native import/process stability.
  Nothing has been filed or sent upstream.
- **Checkpoint publication check:** at `2660b35`, the complete required hook
  passed 59 tests before a signal 11 in `MjModel.from_xml_string` during an
  existing released-contact fixture construction. Retain
  `outputs/pr-publication-checkpoints-v1/`; Git exited 1 and neither branch was
  published. No terminal test XML exists. Passing replay witnesses and focused
  tests do not replace this separate unknown full-suite outcome.

- **Additional construction abort, 2026-11-17:** a required publication hook at
  `3a0dcab` collected 310 tests, passed seven and aborted with signal 6 while
  compiling a fresh MABEL model in `MjModel.from_xml_string`. The Python frame
  is environment construction in a base-control fixture; the new dwell/search
  control had not executed. No native C backtrace or corrupting operation is
  identified. The Git process exited 1, the hook 134, and no publication occurred.
  Preserve `outputs/pr-publication-squaring-v1/` separately from the earlier
  successful 300-test publication and any allocator diagnostic. This need not
  share a cause with the previously observed signals. Nothing was filed upstream.
- **Debug allocator follow-up:** the full required hook at `d886e1a`, with
  `PYTHONMALLOC=debug`, also aborted with signal 6 during fresh model compilation,
  this time in a released-contact fixture. It produced no terminal test XML;
  Git exited 1 and no publication occurred. Preserve
  `outputs/pr-publication-squaring-v2-debug/` with the original failure. Changing
  the allocator did not establish a workaround or isolate the native cause.

- **Status:** at least two SIGSEGVs observed 2026-11-16: an isolated headless
  process during concurrent simulation runs, and a later serial pre-push test
  process. Separate serial full-suite runs completed. The native cause remains
  unknown; this is not an established Emet defect.
- **Environment:** MuJoCo 3.17.0, Python 3.11.12, NumPy 1.26.4; upstream MABEL
  revision `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`.
- **Scope:** direct MuJoCo stepping and scratch-data kinematics/collision queries;
  no ZMQ server or publisher. The first process had no renderer; the later test
  process had executed rendering tests earlier, with no active rollout renderer.
- **Evidence boundary:** the original process has exit signal 11 and no terminal
  summary. Keep that outcome unknown; do not replace it with the successful retry.
  The first attempt did not have a native stack trace. The later process has a
  Python faulthandler traceback, with its current frame at a named body-matrix
  access during a physics observer. This location does not establish a binding
  defect or the source of memory corruption. Its initial model/state and an
  explicit traceback excerpt are retained under ignored
  `outputs/native-crashes/dense-prepush-abd1297/`. No terminal summary exists,
  and the failed test receives zero successful stages.
- **Next diagnostic:** enable `PYTHONFAULTHANDLER=1`, retain the model revision,
  initial integration state and controls, and isolate a standalone upstream-only
  model/state sequence before reporting a library bug. Do not include downstream
  names, repository paths, task details or private artifacts in a future report.
- **Additional observation, 2026-11-17:** a frozen-source headless regression
  child exited with signal 11 during a contact-force observer. Faulthandler's
  current Python frame is the hand-contact accumulator at `contact_forces`, not
  a native C backtrace; this does not identify the corrupting operation. The
  process has no terminal summary and retains zero successful stages. Model,
  initial state, controls/source hashes and the full log are retained locally at
  `outputs/contact-rich-replanning-v1/standard-regression/benchmark/tuned/seed-2/`.
  The other nine selected-controller episodes completed; no retry replaces the
  unknown outcome. The native cause remains open and no upstream issue was filed.
- **Further direct-simulator observations:** separate frozen development
  processes exited with signal 11 at named model-body access and at an observer
  displacement calculation. Both were headless; their initial state, source
  hashes and faulthandler logs remain under ignored local artifacts. A later
  diagnostic with matching controller source returned an ordinary guarded
  failure. That does not replace either unknown outcome or identify a cause.
  No upstream issue or message was sent.
- **Primitive-only reproduction:** a twenty-body model with named model/data
  array reads, Python collection and no physics steps also segfaulted. In three
  matched pairs, transient wrappers failed twice while retained model wrappers
  completed all three. Independent GDB captures stop in `memmove` without an
  identified corrupting operation. A local wrapper-retention mitigation is under
  validation; this does not prove all earlier signals share its cause. The
  standalone `scripts/diagnose_mujoco_views.py` has no robot/controller assets or
  downstream imports. Keep fresh-process failures, return codes and logs. See
  [the diagnostic report](docs/reports/native_view_crash_diagnostic.md). No
  library-level fix is established and nothing was filed upstream.
- **Mitigation limit:** model-wrapper retention alone later failed a full native
  suite at a data-array observer access. Preserve that failure separately. The
  next candidate retains data wrappers as well; this is still a local lifetime
  hypothesis and requires its own native validation.
- **Combined-wrapper validation:** one frozen suite completed 227 tests, while
  an earlier process exceeded its 600 s budget and a later frozen working-source
  suite exited with native signal 11 after 135 s. Both model and data wrappers
  were retained in these candidates. This mitigation is not sufficient; the
  passing process does not replace the failed or unknown outcomes. The last
  printed test label in the latter process does not identify the corrupting
  operation. Exact source fingerprints and process logs remain local. Nothing
  has been sent upstream.
- **Later recording process:** a frozen two-view development run exited with
  signal 11 after 548.93 s wall time. Its Python frame was in IK during a pusher
  planning call; that frame is not a native corruption diagnosis. There is no
  terminal episode summary, so it earns zero successful stages. Raw progress,
  incomplete recordings, frozen sources and stack remain under ignored local
  artifacts. An earlier whole-body focused-test process also failed during
  environment construction before the new controller was created; its later
  passing focused tests do not replace that unknown outcome.

## SIM-002 — Warp CUDA compilation signal in state helper

- **Status:** observed 2026-11-17 during a bounded backend admission probe;
  not established as an Emet defect and no upstream report has been sent.
- **Environment:** CPython 3.11.12, NumPy 1.26.4, MuJoCo 3.17.0,
  MuJoCo Warp 3.17.0, Warp 1.19.0, RTX 4090.
- **Observation:** model import and the first step/forward compilation returned;
  the process subsequently exited with signal 11 while compiling the
  `mujoco_warp.set_state` helper. The Python stack ends in Warp `build_cuda`;
  it does not identify the underlying native cause. The original return code,
  source/lock hashes and log remain at
  `outputs/rollout-backends-v1/cuda-admission/` with zero task credit.
- **Bounded alternative:** initialize a fresh independent device data object
  from native data after the disposable compilation forecast; verify initial
  qpos/controls and that CUDA capture did not advance time. This path completed
  subsequent throughput probes without model changes. It does not establish
  that the state-helper defect or earlier native process faults are fixed.
- **Next diagnostic:** a minimal upstream-only state-helper compilation example
  without robot assets, downstream imports, paths or task context. Do not file
  or send this draft without authorization.
