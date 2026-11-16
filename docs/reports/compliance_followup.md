# Compliant-control follow-up

The selected controller completes **40/40 two-carton scenes across four separate
ten-scene suites**, with every measured stage verified for all **80 scheduled
cartons**. It uses a 0.4 rad thumb MCP preshape, **4.5 N balanced-grip target**
and **1.875 N push-force target**. These are now the script defaults.

| Suite | Earlier preshape policy, 6 N / 2 N | Selected, 4.5 N / 1.875 N | Selected pick / place / final / retreat |
|---|---:|---:|---|
| Public regression, 0–9 | 10/10 | **10/10** | **20 / 20 / 20 / 20**, each out of 20 |
| Earlier suite, 200–209 | 10/10 | **10/10** | **20 / 20 / 20 / 20**, each out of 20 |
| Earlier suite, 300–309 | 10/10 | **10/10** | **20 / 20 / 20 / 20**, each out of 20 |
| Fresh force-validation suite, 400–409 | 8/10 | **10/10** | **20 / 20 / 20 / 20**, each out of 20 |

All forty reset/model input hashes and actual settled integration-state hashes
match within the pairs. There are zero unknown process outcomes, interventions
or observed tip/drop incidents in these validation cohorts. The first three
earlier-policy cohorts pass every stage. In 400–409, the earlier policy records
20/20 picks and shelf placements, then 18/20 final positions and retreats.
Both failures remain in the evidence. Each fresh ten-scene 10/10 estimate has
a Wilson 95% interval of approximately 72.2–100%; the four suites are not a
single independent generalization sample.

Across the forty matched cases, the mean of each episode's highest monitored
force channel decreases from **10.159 to 8.032 N (20.9%)**. This is a mean of
peaks, not an average force or impulse. Selected maximum hand-normal force is
**8.624 N**. Maximum selected final per-axis XY target error is **3.82 mm**,
against the fixed 35 mm tolerance. Complete-scene mean simulation times are
87.99, 87.97, 87.03 and 87.45 seconds for the four selected cohorts. The lower
forces do not establish faster execution; the first thirty paired successes
take slightly longer on average.

## What changed and what was tested

The [thumb-preshape study](thumb_preshape_followup.md) first removed the top-edge
pickup obstruction. Its separate seven-configuration development search on
100/101 chose the lower grip/push targets without changing task criteria,
collision acceptance, the 25 N force guard, native robot parameters or feedback
rates. The complete search record includes thirteen verified episodes and one
unknown final orchestration outcome; the unknown candidate was rejected and
the selected candidate's two development scenes are both verified.

The targets were frozen before generating the new 400–409 comparison. After
10/10 on that suite, the same settings passed all thirty earlier scenes without
regressing a verified success. The short-carton development cases 112/119 also
complete under the selected settings. Promoting the two defaults reproduces
the explicit preset's complete scene-112 force samples and all actuator
controls bitwise.

The 6 N policy's scenes 402 and 409 stop during second-carton insertion because
the requested continuation intersects native lift/body geometry. Its locally
safe prefix is not accepted as a grasp or insertion goal. The lower-force
policy changes the measured grasp and subsequent pose trajectory and completes
both scenes. This is observed coupled-system behavior, not proof that reducing
squeeze alone solves all IK problems. No collision allowance or pose tolerance
was relaxed to accept these paths.

This remains a stationary, left-arm, privileged-world-state task with **ample
lateral shelf space**. It does not demonstrate dense packing, deliberate jam
recovery, bottles, navigation or coordinated bimanual manipulation. A separate
dense-row task will measure **5–10 mm total unused lateral shelf width**, rather
than borrowing this benchmark's success rate.

## Reproduction

Physical experiments use source revision
`95107d78c725d5f254a51a2ee7a91ef541b4e61c` with explicitly recorded overrides.
The selected preset is [low_force.json](../../configs/controllers/low_force.json).
The subsequent default change is numerically equivalent to that preset; the
report records the actual experimental revision and settings. Native MABEL is
`d0181f04025c06e1790b7ab41fe44d602cc2ee5b`, MuJoCo is 3.17.0 and NumPy is
1.26.4. Physics/control periods are 2/20 ms.

At the frozen experimental checkout:

```bash
abstract-stack batch --seeds 400 401 402 403 404 405 406 407 408 409 \
  --episode-timeout 300 --output-dir outputs/firm
abstract-stack batch --seeds 400 401 402 403 404 405 406 407 408 409 \
  --control-config configs/controllers/low_force.json \
  --episode-timeout 300 --output-dir outputs/low-force
```

The preset was frozen after the source commit; copy its two recorded values
into a JSON file when reproducing from that detached checkout. On the current
checkout, `solve`/`batch` already use the selected targets. Explicitly pass
`{"grasp_force_target": 6.0, "push_force_target": 2.0}` via `--control-config`
to reproduce the earlier force settings on the current implementation.
Each cohort ran as two isolated five-scene serial workers with 300 seconds of
wall budget per episode. Every scheduled case remains in the denominator.

The [machine-readable evidence](compliance_followup.json) retains eighty paired
run rows, effective settings, source fingerprints, all-object stage/retreat
evidence, actual poses, target errors and the complete finalized search record.
Full states, controls and MP4s remain in ignored `outputs/`.

```bash
.venv/bin/python scripts/summarize_stocking_campaigns.py \
  --campaign firm_public=outputs/thumb-regression-v1/campaign.json \
  --campaign low_public=outputs/thumb-low-force-regression-v1/campaign.json \
  --campaign firm_known=outputs/thumb-known-v1/campaign.json \
  --campaign low_known=outputs/thumb-low-force-known-v1/campaign.json \
  --campaign firm_first_fresh=outputs/thumb-fresh-v1/campaign.json \
  --campaign low_first_fresh=outputs/thumb-low-force-first-fresh-v1/campaign.json \
  --campaign firm_second_fresh=outputs/thumb-default-force-fresh-v1/campaign.json \
  --campaign low_second_fresh=outputs/thumb-low-force-fresh-v1/campaign.json \
  --pair firm_public=low_public --pair firm_known=low_known \
  --pair firm_first_fresh=low_first_fresh --pair firm_second_fresh=low_second_fresh \
  --search outputs/thumb-compliance-search-v1/search-finalized.json \
  --output docs/reports/compliance_followup.json
```
