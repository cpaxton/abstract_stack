# Explicit adjacent-carton contact proposal

This is an isolated planning/monitoring prototype, **not a successful compaction
skill**. `contact_set.py` is not connected to the front-tangential executor.
Native geometry, collisions, actuator gains, existing collision acceptance,
25 N abort limits and the final individual/summed 0.2 mm overlap gates are
unchanged. No new physical controller campaign has run for this proposal.

## What the v16 rejection means

A separate scratch reconstruction of the saved terminal state has the exact
original model XML hash. It reproduces the rejected `carton_8_geom` versus
`left_middle_ip` contact at **2.068336 mm** penetration. The contact is at carton
8's lower front corner on the side toward active carton 7. Its local position is
`(+0.033332, -0.022908, -0.089201)` m against half dimensions
`(0.034079, 0.023623, 0.116132)` m. Its normal points into the carton and toward
+Y, so this surface could serve a deliberately planned compaction contact.

The measured finger does not contact that neighbor. The proposed abduction
reference is **+0.229087 rad**, versus **-0.002433 rad** measured. Applying that
full reference at the measured arm pose forecasts **0.633415 mm** penetration;
combining it with the proposed nominal arm pose produces **2.068336 mm**. The
nominal finger forecast has no neighbor contact. Thus low measured forces do not
certify the proposed loaded/unloading sweep, and naming the neighbor cannot
legitimize this command. The original 2 mm collision gate correctly rejected it.
The final projected row overlap also exceeded its independent limit, at
0.340 mm. Those original failed outcomes remain unchanged.

Evidence and the scratch-only diagnostic script are retained under
`outputs/contact-rich-replanning-v1/tangential-v16-contact-diagnostic-v1/`.
`policy_report.json` shows that the new stricter contract also rejects v16.
This reconstruction performs no MuJoCo stepping and earns no manipulation credit.

## Narrow interface

- `plan_adjacent_contact(env, active_name, direction, finger="Middle")` names
  exactly the next compatible carton in **measured** Y order at the same task
  level. Candidate measured X/Z envelopes must overlap the active carton's,
  the candidate bottom must lie within 3 mm of the expected shelf board, and
  actual contact normals on that board must sum to more than 0.2 N. Task-level
  labels and `initially_stocked` do not establish row occupancy. The active
  identity remains explicit. Changed eligible-neighbor identity or lost row
  support invalidates the plan with a terminal interaction failure.
- `check_coupled_contact_set(hand, plan, prediction_q, nominal, applied)` checks
  99 scratch poses: measured-to-nominal, measured-to-applied, and the complete
  nominal-to-applied unloading interval at nine arm fractions. It retains every
  existing IK collision error, with only the original active-object permission.
  It grants no whole-row permission and never filters out an existing rejection.
  Neighbor contacts must additionally involve the selected distal finger only,
  have at most **0.5 mm** forecast penetration, lie within 3 mm of the declared
  lower front/side corner, and push inward/toward the requested side with limited
  vertical normal. This is sampled geometry, not a continuous collision or force
  certificate.
- `NamedContactFingerLoadCompensator(..., plan=plan)` aggregates native holding
  effort over exactly the active carton and its named neighbor. It keeps
  per-carton torque diagnostics, native bounds, nominal/applied separation,
  offset caps and reference rates. It does not silently compensate a third
  carton. The old single-target compensator remains unchanged.
- `ContactSetMonitor(env, plan).step(env)` adds per-physics-step checks without
  resetting observers or touching controls. It sums all contact-point normals
  for each carton pair and each robot/carton pair before its 25 N guard. It
  tracks every product, applies a 15 degree tilt limit and bounded physical
  support loss/travel to both named cartons, and retains tip/drop rejection for
  all products. Neighbor hand contact must meet the same corner contract.
  Skipping interior physics steps is rejected. The original global,
  other-product, fixture, pair and final-fit guards still belong to the caller.

The final 0.2 mm individual and total row-overlap checks, all original placement
conditions, requested displacement, force release and verified retreat remain
independent acceptance gates. The helper never reports stocking success.

## Validation and next gate

Focused tests cover the two directional corner windows, unsafe penetration,
wrong finger/surface/normal/height, measured neighbor changes, native forecast
isolation and native bounds, physics-step monitoring, and two-carton torque
aggregation against MuJoCo generalized constraint effort. A third contact fixture
checks that undeclared carton load is excluded rather than silently absorbed.
A zero-prefill reset regression places a same-level table carton between two
physically supported shelf cartons in Y. It excludes that table carton in both
push directions, then excludes an X/Z-compatible carton hovering 2 mm above the
board without actual support. Shelf occupants are recognized without any
`initially_stocked` labels. The focused contact-set and existing finger-load
suites pass **15 tests** after this admission fix.
Synthetic overlap fixtures are numerical contract tests, not physical task
successes. Native MABEL fixtures exercise the real tool and model contracts.

Review this helper before opt-in integration in `FrontTangentialStockingScript`.
The first experiment must retain the v16 failure, reject its original unsafe
command, plan a bounded alternative with safe loaded **and unloaded** geometry,
and report all failures. An explicit contact set does not solve the reference
geometry or final row alignment by itself. No broad tuning or native collision
changes are proposed here.
