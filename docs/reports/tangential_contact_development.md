# Front-face tangential packing investigation

The exposed-side prototype can shift a carton at the shelf front. A carton
already inside a row with millimetre-scale gaps cannot generally admit the hand
beside it. `FrontTangentialStockingScript` is a separate opt-in experiment: retain
front-face contact, apply normal preload, then sweep laterally to compact the
active carton and possibly its neighbors. **No packed-row success is established.**

Fixtures explicitly reset the active carton to its rear-row target, with nine
other cartons already present. Task evaluation remains active, but automatic
termination is disabled for the diagnostic. No reset earns pick or complete-task
credit. Direct packing success independently requires the requested lateral
shift, unloaded retreat, all original placement conditions, no product incidents,
and both original final overlap limits of 0.2 mm. Model geometry, native gains,
collision masks, friction, force caps and grading tolerances are unchanged.

The native fingertip sliding friction coefficient is 1.0; carton/backstop/board
coefficients are 0.8. An ideal Coulomb estimate suggests a single carton pinned
against the backstop needs more than 7 N normal preload to overcome the additional
backstop friction plus floor friction; moving a two-carton group needs roughly
14 N. This is a diagnostic estimate, not a dynamics certificate or tuned result.

## Retained frozen development probes

Each `outputs/contact-rich-replanning-v1/tangential-probe-v*/` contains a copied
full package and runner in `source_snapshot/`, SHA-256 manifest, declared 300 s
process budget, log and either terminal evidence or a process-failure record.
These are development attempts with changing policies and fixtures, not a frozen
success-rate campaign. The parent history is recorded as `037c61e` plus actual
copied source hashes.

- **v1:** baseline base XY `(-0.04, 0)`, 8 N preload and 8 N tangential setpoint.
  A globally accepted pressure pose still had about 5 mm IK residual at arm
  limits. Physical preload crossed the unchanged 25 N guard at 26.16 N. Lateral
  drift was -0.064 mm; no tip/drop incidents; final envelope overlap was zero.
- **v2:** require 1 mm contact-reference accuracy and reduce normal admittance.
  The bounded posture search rejected all candidates before pressure actuation.
- **v3:** explicit reset-only base XY `(-0.08, 0)` was rejected by native fixture
  clearance: the chassis intersected a supply-table leg. The process record and
  traceback are retained; no manipulation stages receive credit.
- **v4:** explicit base XY `(-0.055, 0)` passed reset clearance. Thirty seconds of
  preload produced only 0.114 mm lateral displacement for a 1.5 mm request and
  timed out. Peak hand normal was 4.84 N, fixture load zero, final overlap zero.
- **v5:** add measured finger-load compensation with coupled future-geometry
  checks. Middle MCP measured -0.20021 rad against nominal -0.2 rad; PIP measured
  +0.00059 rad against nominal zero. Applied offsets were +0.207/+0.081 rad;
  peak offset 0.228 rad, no offset saturation. Hand-normal peak reached 8.01 N.
  Actual tool Y still missed its reference by about 8 mm near arm limits, and
  the 1.5 mm lateral request timed out (-0.262 mm displacement). Overlap zero.
- **v6:** explicit base XY `(-0.08, +0.08)` cleared the table leg, but the old
  world-frame staging waypoint failed its bounded reachability check. Failed.
- **v7:** stage relative to that base and enforce the 1 mm online IK gate.
  Preload remained below the lateral-enable threshold; the 30 s skill timed out.
  Hand-normal peak 4.87 N, fixture load zero, final overlap zero. No incidents.
- **v8:** raise normal admittance gain while retaining its 1 mm/s contact speed
  cap. The commanded tangential sweep reached its 21.5 mm reference bound while
  the active carton moved only **+0.046 mm** for a **1.5 mm** request. Failed.
  Peak hand/back/fixture normals were **5.114 / 4.289 / 0 N**; final overlap zero;
  no incidents. Actual tool tracking error grew to **12.67 mm**, mostly vertical,
  while the existing wrist feedforward reached its 0.3 Nm cap (443 clipped
  updates). Finger nominal-angle tracking remained within about 0.5 mrad with
  peak reference offset 0.222 rad and no finger-offset saturation.

These results establish useful finger-load tracking evidence, but do not show
that this front-face frictional strategy can maintain sufficient traction and
pose accuracy to compact the row. A supported grasped shift, another contact
surface, improved wrist compensation or a second arm are next alternatives to
investigate; none is claimed to solve the task. Base-position variants above are
explicit different reset fixtures, not live base actions or matched episodes.

## Finger-load compensation

`FingerLoadCompensator` aggregates measured carton/finger contact wrenches about
native MCP/PIP axes, adds native bias effort, and converts holding effort into
bounded position-servo reference offsets. Nominal references remain separate
from applied targets; the native actuator and joint bounds are applied last.
The caller checks coupled arm/finger geometry through the proposed applied
postures at every tick, including force changes and unloading. Compensation does
not license treating nominal finger geometry as a transient safety certificate.
Diagnostics record measured angles, contact effort, nominal/applied references,
rate/offset clipping and peak offset. No actuator gains are modified.

Tests compare the measured contact torque against MuJoCo's generalized constraint
force in a separate scalar-joint contact fixture, and check signs, reference/rate
bounds and rejection before commands. Native tangential episodes remain the
physical acceptance gate; improved finger-angle tracking does not prove packing.

## Reproduce a new declared diagnostic

Use a fresh output directory; this command reproduces the latest v8 setup from
current source, not historical source unless run from its retained snapshot:

```bash
PYTHONPATH=src MABEL_ROOT=/path/to/open_mabel \
  python scripts/probe_lateral_contact.py --mode front-tangential \
  --output outputs/new-tangential-probe --objects 10 --seed 0 \
  --direction 1 --distance 0.0015 --contact-height 0.018 \
  --force-target 8 --normal-force 8 --push-speed 0.003 --timeout 30 \
  --base-x -0.08 --base-y 0.08
```

The side-push and front-tangential controllers are opt-in; baseline stocking is
unchanged by these modules. Six finger-compensation tests and ten lateral tests
pass with native MABEL assets. Ruff formatting and lint pass. No direct packing
success, full-ten stocking success or general success rate is inferred.
