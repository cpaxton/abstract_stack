# Simulation checkpoints and last-millimeter tests

Use a native checkpoint to repeat a difficult insertion state without repeating
the pickup. Checkpoints contain the **compiled model and complete integration
state**, not just poses. Restores create new model/data objects; they never
teleport objects in a running manipulation episode. Every restored trial is a
continuation with zero fresh task credit. The original failed episode stays
failed.

## Capture, inspect and verify

Import an existing episode's exported scene and terminal state once:

```bash
abstract-stack checkpoint capture \
  --episode outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode \
  --output outputs/my-checkpoint --witness-steps 50 --wall-seconds 30
abstract-stack checkpoint inspect outputs/my-checkpoint
abstract-stack checkpoint verify outputs/my-checkpoint \
  --output outputs/my-checkpoint-verification --wall-seconds 30
```

Each output directory must be fresh. Capture/verify use frozen-source child
processes with declared wall budgets and retained logs/outcomes. The actual
bundle is `my-checkpoint/checkpoint/`; both that path and its wrapper work with
inspection/loading. Inspection verifies hashes without importing MuJoCo.
An interrupted, signaled or missing-result process is not a verified restore.

MJB embeds the compiled meshes/textures, geometry, inertias, transmissions,
limits, contacts and solver options. Loading it bypasses XML compilation and
does not require the original asset paths. The original MJCF remains an audit
artifact; there is no silent XML fallback or cross-version conversion. Current
loading requires matching MuJoCo, NumPy, float width and byte order. Other
platform/CPU differences remain visible in strict replay verification.

Importing a historical XML/state pair cannot prove the original live model's
binary identity: those older episodes did not save MJB. The import records this
limitation. New live checkpoint captures preserve the actual compiled model.
Archived actuator replay reaching the exact original terminal state provides
additional evidence for the imported development scenes.

## What is preserved

- MuJoCo's `mjSTATE_INTEGRATION`: time, qpos/qvel, actuator activation/history,
  warm starts, applied controls/forces, equality activation, mocap, userdata and
  plugin state supported by that runtime.
- Scene configuration, seeds/model revision supplied by the scene manifest,
  controller settings, explicitly known nominal references, applied targets and
  caller-owned phase/observer memory. Omitted nominal labels remain unknown.
- Python/NumPy global RNG snapshots and explicitly supplied generator states.
  Loading returns these records without changing process-global RNGs; a
  stochastic controller must explicitly restore its own generators.
- Copied Python sources, source fingerprints, model-array/solver-option hashes,
  runtime versions and lineage. The native compilation signature is not a
  physical parameter and resets on MJB load; native arrays/options and full MJB
  hashes identify the model instead.

The `save_checkpoint()` API takes an existing model/data pair and explicit
`scene`, `controller` and `context` dictionaries. It does not reset or step the
caller's live world. It records a short private-copy replay witness. Verification
rebuilds a new model/data and compares integration state, body transforms,
accelerations, constraint forces and all native contact records bit for bit at
every witness step. A replay match proves reproducibility of those controls,
not safe control admission, a grasp or task success.

Arbitrary Python controllers are not pickled or automatically resumed. Supply
their memory explicitly. The supported finger-push probe restores known arm/lift
references, held applied finger controls, orientation/bounds, yaw integral,
continuous product-monitor history and partial contact dwell. Incident history,
contact-loss timers and original reference bounds must not reset at a checkpoint.

## Focus bank

The fixed development recipe is [compliance_focus.json](../configs/compliance_focus.json).
It selects five states from three archived actuator trajectories:

| State | Back-plane gap | Yaw | Back force | Focus |
| --- | --- | --- | --- | --- |
| `stalled_before_backstop` | 9.715 mm | 3.654 deg | 0 N | Break stiction and advance without downward loading |
| `approaching_backstop` | 0.021 mm | 4.854 deg | 0 N | Brake and acquire measured contact |
| `first_back_contact` | -0.052 mm | 4.708 deg | 0.708 N | Establish support and begin squaring |
| `yawed_back_contact` | -0.136 mm | 4.426 deg | 0.991 N | Improve contact lever without dragging the carton |
| `bound_limited_squaring` | -0.402 mm | 2.748 deg | 2.083 N | Detect bounds/stall, unload and re-approach |

The depth gap is the projected carton rear envelope relative to the backstop
face. Negative values record native penalty-contact penetration. It is not the
whole-row lateral free-space budget or a relaxed task tolerance. The existing
0.15-degree squaring and 0.12-second dwell criteria remain unchanged.

Each frame has a self-contained checkpoint, metrics JSON and native third-person,
first-person and close views. A separately labeled wide overhead debug view
shows yaw/contact geometry. It changes the render camera/projection only and
is not a student observation. Rendering must leave integration state unchanged.

Build the gallery from an imported checkpoint and retained source trajectories:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/build_compliance_focus.py \
  --checkpoint outputs/my-checkpoint --output outputs/my-focus-bank
```

Before emitting frames, the builder replays the recorded native applied controls
and requires the terminal integration state to match the source bit for bit.
All-product/contact guards run every 2 ms. Capture/render failures remain in the
bank report; an incomplete bank cannot enter the comparison runner.

Run a small fixed comparison or a single last-millimeter experiment:

```bash
.venv/bin/python scripts/evaluate_compliance_focus.py \
  --bank outputs/my-focus-bank --output outputs/my-focus-comparison --replans 2
.venv/bin/python scripts/probe_compliant_lookahead.py \
  --checkpoint outputs/my-focus-bank/frames/approaching_backstop/checkpoint \
  --output outputs/my-last-mm --candidates 8 --shortlist 4 \
  --execute-prefix 0.1 --replans 1 --save-checkpoint
abstract-stack checkpoint verify outputs/my-last-mm/terminal-checkpoint \
  --output outputs/my-last-mm-verification
```

For squaring experiments, add `--require-dwell --square-feedback`. The saved
`terminal-checkpoint` can be the next input. A physics-boundary checkpoint also
records the interrupted control-phase step; the current probe starts a fresh
bounded decision, rather than silently pretending to resume every arbitrary
controller's cadence.

The comparison runs held targets and compliant prefixes on every bank state,
preserves unknown processes and reports all scheduled cases. Force peaks carry
the inherited monitor history; they are cumulative, not prefix-only peaks.
These are public development states, not a fresh frozen validation suite or a
new stocking success rate. See the [measured report](reports/checkpoint_focus_development.md).
