# Native named-view crash investigation

The full native suite at `be7d930` terminated with signal 11 during the ordinary
two-carton regression. No full-suite pass is claimed for that process. Two
isolated GDB runs also stopped at signal 11. Their native traces stop in
`memmove`; the stack does not identify the original corrupting operation. The
core dump, logs and original failed artifacts are retained, including a lossless
compressed core, under `outputs/native-crashes/contact-controller-be7d930/`.

The diagnostic then removed all robot assets, controller code, contact queries,
rendering and physics steps. A twenty-body primitive model repeatedly reads
named model/body views and copies position/matrix arrays, with periodic Python
collection. This primitive-only process also segfaulted. A version with scratch
data copies failed; another without copies failed. Consequently, neither MABEL
assets nor contact control is necessary for at least this observed crash.
This does not prove that every historical native signal has the same cause.

## Wrapper-lifetime comparison

The original isolation source is preserved as `named_access_isolation.py` with
its SHA-256 in `isolation-comparison.json`. Six fresh processes used identical
source and alternating modes; each requested 100,000 iterations over 20 bodies.

| Mode | Native signals / processes | Completed processes |
| --- | --- | --- |
| Repeated temporary model/body view access | 2 / 3 | 1 / 3 |
| Retain model body wrappers, same repeated access | 0 / 3 | 3 / 3 |

Separate model-only and data-only lookup probes completed, as did an indexed
array probe. Holding model wrappers is a useful local mitigation hypothesis;
the sample does not establish a universal fix. Allocation layout affects this
intermittent reproduction. The cleaner standalone diagnostic may complete in
transient mode as well; absence of a crash is not a proof of safety.

`ShelfStockingEnv` now retains commonly used native model-view wrappers for its
lifetime. The implementation only stores Python references; it changes no native
model array, qpos, qvel, contact, gain, limit or controller target. A focused test
checks view identity through collection and unchanged model/state arrays. This
test checks the mitigation's contract, not deterministic reproduction of the
library defect. Native rollout/replay checks are required separately.

The mitigation replayed all **79,930** recorded controls from mobile-table v10
and reproduced its entire terminal MuJoCo integration state **bit for bit**
(maximum error zero). This is a replay with zero fresh-task credit. The cleaner
standalone-script campaign completed all six processes: two transient, three
retained and one indexed. Those passing transient runs reinforce that the
reproduction is sensitive to allocation layout; they do not replace earlier
failures. Raw results are in `retained-v10-replay.json` and
`standalone-results.json` beside the original isolation comparison.

The **model-only** candidate at `283f166` subsequently failed its full native
suite with signal 11 during a data-array observer access. It is therefore not a
complete mitigation. That log is preserved as
`full-suite-retained-283f166.log`. The next candidate retains both model and data
view wrappers, with the same no-model/state-mutation contract. Its native
validation must be assessed separately; no passing retry replaces this failure.

## Standalone reproduction

### Subsequent retained-model-and-data validation

These are separate processes, not replacements for earlier failures. Frozen
revision `498059a` retains both model and data wrappers.

| Frozen source / attempt | Outcome |
| --- | --- |
| `498059a`, initial process | Interrupted; no terminal test result, unknown |
| `498059a`, r2 | Checkout unavailable before tests; orchestration failure |
| `498059a`, r3 | Exceeded declared 600 s wall budget; no completed suite |
| `498059a`, r4 | 227 passed in 199.20 s; process exited normally |
| `working-controller-v1` source snapshot | Native signal 11 after 135.23 s; no completed suite |
| `split-insertion-v2` source snapshot | 250 passed in 205.03 s; process exited normally |
| `236a308`, required pre-push hook | Stopped after over eight minutes without suite completion; push failed |

The final printed test in the working snapshot was
`test_adaptive_solver_verifies_a_physical_lift`. No terminal native traceback
identifies the original corrupting operation, so that test label is not a causal
diagnosis. Its exact source manifest is retained in
`outputs/checkouts/working-controller-v1/snapshot_sources.json`; process results
and logs sit beside the earlier crash artifacts. The r3 timeout also does not
identify a native signal or a specific cause. The passing r4 suite establishes a
pass for that revision and process only. Retaining both sets of wrappers is
**not a sufficient fix**, and native process stability remains an open gate.
The later 250-test pass has its own source manifest and process record; it
neither replaces the signaled process nor identifies the native cause.
The subsequent required hook collected 251 tests and was still CPU-active in
the `tests/test_mabel.py` module when its identified pytest child was stopped
with SIGTERM. This cutoff was imposed during diagnosis, not declared at process
startup. The hook exited 143 and Git exited 141 after its remote connection had
also closed. No tests were skipped to publish the branch. An attempted GDB
attachment was unavailable (`ptrace` refused); no native frame or crash signal
was obtained, so this does not identify a shared cause with earlier SIGSEGVs.
The cutoff record and failed attachment log are retained beside the other
native artifacts. Revision `236a308` remains a local commit pending checks.

The diagnostic script imports only MuJoCo, NumPy and the standard library. It
contains no robot/task assets or downstream imports, so it can be copied into an
independent environment for an upstream-only investigation. No issue has been
filed and no message has been sent.

```bash
PYTHONFAULTHANDLER=1 python scripts/diagnose_mujoco_views.py \
  --mode transient --iterations 100000 --output /tmp/new-transient-probe
PYTHONFAULTHANDLER=1 python scripts/diagnose_mujoco_views.py \
  --mode retained --iterations 100000 --output /tmp/new-retained-probe
PYTHONFAULTHANDLER=1 python scripts/diagnose_mujoco_views.py \
  --mode indexed --copy-every 1000 --output /tmp/new-indexed-probe
```

Each directory must be fresh. Preserve return codes and progress files even
without `result.json`; do not replace a native failure with a passing retry.
The recorded environment is MuJoCo 3.17.0, CPython 3.11.12 and NumPy 1.26.4.
`PYTHONMALLOC=debug` completed one previously failing targeted test, but that is a
different allocator configuration, not evidence that the default native fault
has been repaired. The library-level cause and broad stability gate remain open.

### Withdrawal force-guard follow-up

A reported full-suite run at `8b72b50` stopped with SIGSEGV while
`test_withdrawal_does_not_bypass_force_guard` was the active test. The
same revision also completed a clean local pre-push physics run. The test
label alone does not identify the corrupting operation.

A focused follow-up ran the original fixture and an indexed version 120
times each in separate fresh Python processes. Both completed normally.
The indexed fixture uses the monitor's existing active-body ID and resolves
the other body ID explicitly. The same native stepping, measured load and
force-guard assertions apply. These runs establish that the assertions
still pass, with no evidence that this cleanup repairs the intermittent
native crash. The original fixture source and both logs are retained under
`outputs/native-crashes/withdrawal-force-guard-8b72b50/`.

A clean committed-source export of `8b72b50` plus the indexed fixture then
ran the complete local suite with native MABEL assets and EGL. All 488 tests
passed in 259.78 s. This is one passing process under the default allocator.
It does not close the intermittent crash gate or replace the reported signal.

The three standalone view probes (transient, retained, indexed) also
completed 100,000 iterations each. The independent fresh
`tight-ten-20mm-fresh-01` stocking rollout subsequently segfaulted at
`insertion_skill.py`'s native contact iterator during base navigation.
Its process record, traceback and four verified retreat checkpoints remain
in the original run directory. A focused production change reads the
native contact-geom column in order and keeps the existing per-pair and
sidewall force aggregation. The two native summed-force guard fixtures
and the wider 41-test targeted set passed. The recovered execution and the
failed fresh run retain separate outcome records.

The indexed continuation passed the earlier failure point and reached
contact refinement for carton five. It stopped normally on the bounded
release-goal search and saved a terminal failure fixture. One continuation
through the interval supports further testing of the indexed path. It does
not establish the native fault's cause or demonstrate full process stability.
