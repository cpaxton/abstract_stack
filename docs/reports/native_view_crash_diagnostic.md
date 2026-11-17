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

## Standalone reproduction

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
