# Domain randomization and robustness evaluation

The first version, `stocking_geometry_dr_v1`, randomizes explicit procedural
scene configuration. It changes object counts/types, global carton dimensions,
shelf dimensions and levels, supply layout/jitter, and studio/house context.
It never edits a live model or MABEL geometry, inertias, limits or collision masks.
Geometric validity, native reset/hold validation and manipulation success are
separate results.

| Family | Objects | Shelf levels | Current teacher adapter |
| --- | --- | --- | --- |
| Carton bench | 2–6 cartons | 1 | Generic carton script; robustness unvalidated |
| Dense row | 2–10 cartons | 1 | Receding dense script; robustness unvalidated |
| Multiple shelves | 2–10 cartons | 2–3 | Not admitted by the current evaluation adapter |
| Small house | 2–10 mixed cartons/bottles | 2–3 | Not admitted by the current evaluation adapter |

Dense rows retain **5 or 10 mm total whole-row free width**, including both
walls. Other families have explicit larger gaps and are not dense-packing
benchmarks. Carton dimensions vary per scene; bottles retain the existing fixed
shape. Materials, masses, sensor noise/delay and native camera calibration are
not randomized in v1. Those need separate explicit configurations and evidence.

![Randomized carton bench](media/domain_randomization/bench.png)
![Mixed objects and multiple shelves in a house](media/domain_randomization/house.png)

## Suites and reproducibility

```bash
.venv/bin/python scripts/domain_randomization.py generate \
  --output outputs/my-dr-suite --development 8 --heldout 12 \
  --development-seed 2026
```

Each case has a loadable `scene.json` and a hashed `context.json`, including
scene/reset seeds, family/factors and scheduled object names. These are
privileged provenance, not student observations. Layout generation validates
shelf/table fit and rejects invalid requests; it makes no reachability claim.
All objects begin on the supply table, with zero prefills.

Development seeds are reproducible. Held-out entropy is independent and private
by default, saved only in ignored outputs. Keep that split untouched during
tuning. Each split starts with an all-object evaluation ledger marked `not_run`;
no generated scene starts with successful stages or training eligibility.

The current suite is `outputs/domain-randomization-v1/`: 8 development cases and
12 private held-out cases. Held-out task evaluation and collection have not run.

### Pose-only carton rows

Seeds in the fixed ten-carton preset control independent supply X/Y jitter
within ±2 mm of each nominal position.
Generate more position variation while preserving carton dimensions, table pose,
shelf height, object count and the whole-row free-space budget.

```bash
.venv/bin/python scripts/domain_randomization.py generate \
  --pose-base configs/scenes/tight_ten_cartons_20mm.json \
  --output outputs/ten-carton-pose-suite --development 8 --heldout 12 \
  --development-seed 2026
```

Each scene samples shelf X translations from −12 to +4 mm and Y translations
from −15 to +15 mm. Each case samples a per-axis jitter limit from 2 to 6 mm,
then uses the recorded reset seed to independently offset each supply carton
within that limit along each axis. The profile uses the base shelf and carton
orientations. The scene and context record shelf and table world positions
and the reset seed.
Use `scripts/run_stocking_case.py` with the case's scene, reset seed and a frozen
stocking profile for manipulation trials. The generic DR teacher evaluation
has no adapter for this suite.

The first ten-carton pose pilot reused the source and controller settings from
the successful fixed seed 5 rollout. It changed shelf X/Y and carton spawn
jitter while keeping the 20 mm total row budget. All three public development
scenes passed native initial clearance and a guarded hold. The stocking results
were as follows.

| Case | Shelf X / Y shift | Spawn jitter bound | Picks | Shelf placements | Strict positions | Withdrawals | Terminal stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 000 | +3.56 / +3.44 mm | 5.70 mm | 0 | 0 | 0 | 0 | Coupled grasp closure rejected before actuation |
| 001 | −8.46 / −2.99 mm | 5.93 mm | 1 | 1 | 0 | 0 | Side registration timed out |
| 002 | +3.46 / +13.63 mm | 5.78 mm | 0 | 0 | 0 | 0 | Pregrasp exceeded the 25 N guard |

There were zero packed rows across the three fresh episodes. Each has a
terminal report covering all ten scheduled cartons. The source reference
and controller inputs are under
`outputs/runs/domain-randomization/ten-carton-pose-v1-baseline/`. The combined
[stage report](../outputs/runs/domain-randomization/ten-carton-pose-v1-baseline/report.json)
checks source identity, controller settings and all scheduled cartons against
the frozen baseline. The development curriculum retains the three exact
stops and their earlier entry states. The held-out
split remains sealed. Start follow-up experiments with independent shelf
translation and supply jitter before combining these factors with changes to
carton size, shelf height/depth, mass or friction.

The same reset seed as case 000 also failed on the fixed scene, at the grasp
closure check. Applying only the sampled shelf translation or only the larger
jitter led to pregrasp force stops instead. Each comparison used the same
frozen controller. This places grasp clearance at the start of the pose
generalization work, including the original 2 mm jitter profile. See
`outputs/runs/domain-randomization/ten-carton-pose-v1-ablation/`.

Pass `--policy-source-ref` to `scripts/run_stocking_case.py` to reuse an
authenticated source bundle. Use the archived controller settings and mode
flags from the successful run when comparing against its policy.

## Native scene validation

```bash
MUJOCO_GL=egl MABEL_ROOT=third_party/open_mabel \
  flock /tmp/abstract-stack-render.lock .venv/bin/python \
  scripts/domain_randomization.py validate --suite outputs/my-dr-suite \
  --output outputs/my-dr-native-check --limit 4 --preview
```

The four development profiles compiled with native MABEL, passed initial
collision clearance and completed 0.2-second guarded holds. They contained
3 cartons/1 level, 10 dense-row cartons/1 level, 10 cartons/multiple levels,
and 10 mixed objects/2 levels in a house. All product/force guards ran at each
2 ms step with no incidents. Full compiled checkpoints and replay witnesses
are retained. This verifies scene construction and holding, **not stocking**.

See `outputs/domain-randomization-native-v1/report.json`; every scheduled case,
including a future rejection/crash, remains in its denominator.

## Frozen teacher evaluation

```bash
MABEL_ROOT=third_party/open_mabel flock /tmp/abstract-stack-render.lock \
  .venv/bin/python scripts/evaluate_domain_suite.py \
  --suite outputs/my-dr-suite --split development \
  --output outputs/my-dr-teacher-development --limit 1 --case-wall-seconds 180
```

The evaluator freezes one source bundle and mode profile, checks every case hash,
and supervises fresh native episodes independently. It reports pick, shelf
support, final positioning and retreat for **all scheduled objects**, clean
complete-scene success, process failures, unsupported domains and cases not run.
No episode can silently drop a hard object. Unsupported multi-shelf/mixed
teacher domains stay explicit zero-stage outcomes when scheduled; they are not
removed from a broad robustness denominator. This is an adapter limitation,
not evidence of physical unreachability.

The initial development baseline attempted one three-carton case and stopped
at `approach: contact outside bounded grasp region`. No pick or later stage was
verified. Seven development cases remain unrun. The current stack therefore
has **not demonstrated randomized task robustness**. The result is retained at
`outputs/domain-randomization-teacher-dev-v1/report.json`.

Held-out evaluation requires an explicit `--policy-source-ref` from an already
frozen development policy. Do not tune after seeing held-out results; create a
new private evaluation suite for a revised policy. Compare success/time/contact
and recovery results by family, object count, shelf count and density before
aggregating. Keep force limits and task tolerances fixed.

### Geometry sweep around the three-carton baseline

The full public development split at `76e11bb` scheduled 48 objects across
eight scenes. Four supported scenes ran, with 19 scheduled objects, and four
multi-level or mixed scenes remained unsupported. The bench teacher verified
one pick. No placement, final position, retreat, or complete scene passed.
Both dense cases stopped at the root planning budget before attempting a pick.
See `outputs/domain-randomization-teacher-dev-76e11bb/report.json`.

A separate sweep kept the supply table, reset seed, force profile, and
controller flags fixed around the previously successful three-carton layout.
Its first pass exposed a Python import scoped to the released-insertion branch
that the grasped transfer also needed. The fixed sweep used `--grasp-skill
--base-reposition` and the same scene files. All five processes returned stage
reports. No product was tipped or dropped during any run.

| Scene change | Pick | Shelf support | Final position | Retreat | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline, 800 mm shelf height | 3 | 3 | 3 | 3 | Complete, 150.58 s simulated |
| Shelf 40 mm lower | 0 | 0 | 0 | 0 | Empty-hand route rejected before pregrasp |
| Shelf 40 mm higher | 3 | 3 | 3 | 3 | Complete, 146.84 s simulated |
| Cartons 80 × 60 × 220 mm | 1 | 1 | 0 | 0 | Fixture contact reached 26.19 N during push |
| Shelf 50 mm deeper | 1 | 1 | 0 | 0 | Push feedback path rejected before actuation |
| Total scheduled | 8/15 | 8/15 | 6/15 | 6/15 | 2/5 clean scenes |

Inputs, stage ledgers, and the frozen source digest for the matched sweep are
under `outputs/stocking-geometry-sweep-v2/`. The first-pass process failures
remain under `outputs/stocking-geometry-sweep-v1/`. These are public development
cases and the private held-out split remains untouched.

The follow-up changed each carton dimension on its own, with the same 180 g
mass and friction. At 80 mm depth the script stocked all three cartons in
144.88 s. At 220 mm height it stocked all three in 145.68 s. At 60 mm width it
picked and placed the first carton, then the planned feedback correction
predicted 2.02 mm of penetration into the shelf board and stopped before
actuation. The width-only stop had no knock-overs. This isolates sensitivity
to width under the tested posture. The combined-size case had a different
force-guard failure. See `outputs/stocking-carton-size-sweep-v1/`.

The deeper shelf's final correction had a 0.63 mm predicted position error
and was still rejected because the lift base and body mount would intersect.
Use the measured shelf geometry to choose an empty-hand route for lower
shelves and a different base stance or push posture for deeper shelves.
Evaluate those cases before combining size, height, depth, and material changes.

## Learning/data collection

The [LfD pipeline](learning_from_demonstration.md) now accepts explicit DR scene
and context files. Two bounded development fixtures collected 40 transitions,
42 snapshots and 84 raw first-/third-person images. Official LeRobot 0.7 export
and reload preserved pixels, nominal/applied actions, clocks, sensors, masks
and outcome labels. Sensor/proprioception semantics must match across episodes;
shape equality alone cannot justify combining different robots or taxel orders.

These are training-ineligible sensing fixtures. No successful manipulation
dataset, student training, upload or held-out collection has occurred. Diagnostic
teacher rollouts from the evaluator are not causal BC records. The next task is
to attach the causal recorder before every compensation path in the actual
stocking loop, then collect verified skill/task demonstrations on development
scenes. Teacher privileged context must remain outside student features.

Reports: `outputs/lfd-dr-pilot-v1/lfd_report.md` and
`outputs/lfd-dr-pilot-v1/sdk-roundtrip-v2/verified/audit.json`.
The original cache-related reload failure remains archived; the successful
retry used a writable local HF cache and the same exported dataset.

Integration checks: 169 focused controller, native-physics, checkpoint/source,
DR and LfD tests passed together. Pre-commit and offline wheel/sdist builds
passed. Logs are in `outputs/final-dr-lfd-validation-v1/`. These checks do not
replace a complete frozen-policy manipulation campaign.
