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
