# Contact-wrench lookahead development

Measured-wrench feedback and short native forecasts moved a stalled carton to
backstop contact in a restored-state continuation. This does not establish a
completed placement, a successful fresh episode, or a ten-carton success rate.
The held-target comparator barely moved. All probes have zero fresh task credit.

## Method and contracts

The source is the retained terminal state of
`outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode/`, a failed
ten-carton episode. The active product is `carton_8`; its first insertion had
stalled short of the backstop. Restoring this checkpoint is fixture setup.
Every subsequent interaction uses native actuator targets and 2 ms MuJoCo
steps. No geometry, inertias, gains, native limits, contact masks, object poses,
attachment constraints or external forces are changed during interaction.

`PushAdmittance` converts the measured contact wrench **on the hand**, in world
axes, into bounded normal/tangential/vertical velocity and yaw alternatives.
The shelf insertion direction is -X. Positive hand Fz indicates downward loading
of the supported carton: forward advance is reduced and the hand unloads upward.
Native wrist reaction F/T, which includes gravity/inertia, is a separate signal;
see [simulated sensor conventions](../sim_sensors.md).

The experiment uses the source controller's 2.5 N normal target, unchanged 25 N
abort limits, and unchanged task tolerances. Each decision schedules eight
proposals, including a held-applied-target baseline. Endpoint/path IK is
prefiltered, joint rates/ranges are bounded, and applied arm/finger geometry is
checked together. C++ forecasts screen 0.4 s trajectories; four shortlisted
candidates receive per-physics-step native force/support/contact/tip guards.
Unchecked candidates cannot be selected. Future wrench is frozen within each
forecast; executing a 100 ms prefix is followed by a new measured-state decision.
This is bounded receding-horizon search, not MCTS or a learned policy.

Prefix execution additionally verifies the exact captured integration state and
monitors every product at every physics step. The physical tool corridor remains
the original bounded corridor. The continuation stops at the first measured
backstop normal force of at least 0.4 N. It does not claim the existing contact
dwell, final pose, release or retreat gates have passed.

Nominal physical references and applied servo attractors remain distinct. In
this loaded source state, FK of the compensated servo targets lies 51.56 mm
above FK of the nominal reference. Comparing that virtual attractor against the
physical tool corridor incorrectly rejected every moving proposal. The corrected
proposal check bounds applied-reference increments in the same-sized envelope
translated by the captured initial compensation. Coupled applied geometry and
the original physical corridor remain checked. This changes reference-frame
bookkeeping; no contact guard or task tolerance is relaxed.

## Retained outcomes

All runs below completed their supervised process with return code 0. The table
reports continuation motion, not pick/place success. Peak load is the combined
hand-plus-other-robot contact normal load on the active carton.

| Probe | Physics time | Carton advance toward back | Peak robot/product | Backstop normal | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `probe-v1` | forecast only, 0.4 s | -0.00133 mm | 1.35 N | 0 N | Seven moving proposals rejected by incorrect absolute applied-reference corridor; held baseline only |
| `probe-v2-diagnostic` | forecast only, 0.4 s | -0.00133 mm | 1.35 N | 0 N | One moving proposal rejected; retained diagnostics identify virtual-reference offset |
| `probe-v3` | forecast only, 0.4 s | 2.23 mm selected | 1.63 N selected | 0 N | Eight generated, four guarded; selected movement remains a forecast |
| `held-baseline` | 2.0 s | -0.00485 mm | 1.35 N | 0 N | Same source state, held targets, 20 decisions |
| `continued-v1` | 2.0 s | 9.48 mm | 1.64 N | 0 N | Eight-proposal search, 20 executed prefixes; no back contact yet |
| `continued-v2` | 2.1 s | 9.75 mm | 2.33 N | 0.708 N | 21 of 40 scheduled decisions executed; first back contact stops continuation |
| `continued-v3` | 2.1 s | 9.75 mm | 2.33 N | 0.708 N | Same motion with measured execution tracking added |

The selected forecast's precise peak is retained in the JSON ledger; table
values are rounded. All actual continuations recorded no force abort, support
loss, tip/drop incident, non-grasping-arm contact, robot/fixture contact or
carton-pair load. Non-active products moved at numerical roundoff scale. These
claims apply to the monitored continuation, not earlier parts of its source
episode. The original failed episode remains failed.

For `continued-v3`, 1,050 native steps advanced 2.1 s. Peak measured-base nominal
FK execution error was 2.70 mm, reported separately from each candidate's IK
position/orientation residuals. Parent wall time was 27.34 s; the held baseline
took 5.73 s. Generation averages roughly 0.8 s per moving decision. This search
is **slower than real time** and does not establish the desired stocking speedup.
These are single-state development probes, not a fresh validation suite.

## Provenance and reproduction

The [JSON ledger](compliant_lookahead_development.json) records all seven
scheduled probes, process outcomes, exact settings, source hashes and selected
forecast/continuation measurements. Full immutable bundles remain under
`outputs/compliant-lookahead-v1/`, including copied package and runner sources,
input scene/manifest/checkpoint, source SHA256 values, candidate controls,
continuation controls, tracking samples and terminal state.

Base checkout revision was `75fc28e4ebb646b062cd977bba78e9e42bc36ab9` plus
uncommitted copied sources, identified by their per-file hashes. Native model
revision was `d0181f04025c06e1790b7ab41fe44d602cc2ee5b`, MuJoCo 3.17.0 and NumPy
1.26.4. The initial integration-state SHA256 is
`d2acf8ae96cd6ffc25f67a88649729511c8f53f52028c6b8940e7d41ffd481b2`
for both the comparator and moving continuations. Policy source hashes match;
runner/sensor validation and reporting developed between earlier runs. This is
a development comparison, not a frozen-policy evaluation campaign.

`continued-v1` incorrectly retained the summary flag
`source_state_unchanged=true` after executing physics. Raw artifacts are retained
unchanged; that flag describes forecast generation only and is wrong for the
executed continuation. Later runners distinguish unchanged forecast generation,
unchanged checkpoint file, and changed continuation state. Older
`live_execution_admitted=false` fields refer to fresh-task admission and are
ambiguous for a continuation; current output names fresh-task and continuation
admission separately. Neither grants task credit.

Run with the retained source episode available and a fresh output directory:

```bash
.venv/bin/python scripts/probe_compliant_lookahead.py \
  --episode outputs/contact-rich-replanning-v1/mobile-table-ten-v17/episode \
  --output outputs/my-fresh-compliant-continuation \
  --candidates 8 --shortlist 4 --horizon 0.4 \
  --execute-prefix 0.1 --replans 40 \
  --generation-seconds 10 --wall-seconds 90
```

Use `--candidates 1 --shortlist 1 --replans 20` for the held-target comparator,
or omit `--execute-prefix` for forecast-only inspection. Historical results
require their bundled sources; the current command captures current sources.
Interrupted, signaled or missing-summary processes remain unknown with zero
credit. The existing intermittent native-process fault remains unresolved.

## Next gates

Verify sustained supported back contact and final positioning, then unload,
release and retreat without displacing any carton. Integrate the decision loop
into a fresh pick/carry/insert sequence before a whole-shelf campaign. Profile
proposal generation and guarded validation separately; retain force/geometry
admission while reducing planning latency. Add causal demonstration recording
before all compensation paths, then collect successful and failed whole episodes
with raw synchronized cameras for the [LfD pipeline](../learning_from_demonstration.md).
