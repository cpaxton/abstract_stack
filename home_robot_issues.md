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
