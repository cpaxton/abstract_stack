# Cardboard deformation: options and first native-physics probe

Start robot integration with an **explicit articulated-panel carton variant**,
while keeping rigid cartons as the regression baseline. Native elastic shells
are feasible, but are a separate integration project: the installed MuJoCo
3.17.0 rejects flex elasticity with the current MABEL `implicitfast` integrator.
The panel coupon below uses `discrete`/CG, has no robot, and is **not calibrated
cardboard or a stocking success**.

The dense task targets **5–10 mm total unused lateral shelf width**, rather than
that clearance at every object pair. Cardboard compliance may help insertion,
but numerical overlap must not substitute for physical bending or crushing.

## Modeling choices

| Variant | What changes physically | Incremental integration and limits |
|---|---|---|
| Existing rigid cartons | A rigid cuboid with the assigned mass/inertia | Keep unchanged for baseline comparisons; no local panel bending or crush. |
| Contact-only compliance | Regularized contact between otherwise rigid shapes | Small implementation change, but no deformed carton shape, folds or damage. Contact penetration must remain a separately reported numerical error. |
| Articulated rigid panels | Thin rigid faces with native hinge joints, torsional stiffness and damping at folds | A practical next prototype compatible with `implicitfast`. Represents fold compliance, while faces remain rigid. Linear elastic hinges do not establish plastic creasing, delamination or calibrated buckling. |
| Native 2D flex shell | Contacting triangle surfaces with membrane and bending elasticity | Captures local face deformation. Requires an explicit integrator/model variant and new controller/contact/evaluation validation; present coupon only. |
| Detailed paperboard FEM | Orthotropic layers, nonlinear plasticity and cohesive interfaces | Appropriate reference for material calibration and crease/crush studies, with substantially greater modeling and computation cost. Not installed here. |

For the articulated variant, model the actual outer dimensions and hollow
geometry, with panel thickness, seam/fold locations, and mass/inertia explicit.
Model contents separately when they carry most of a package's weight. Any seam
constraints should be part of the object definition, never a grasp attachment.
Before choosing a constitutive law, measure fold torque versus angle on loading
and unloading; do not infer a plastic crease from a damped spring's transient
deflection. Native joint stiffness/damping support is documented in the
[versioned MJCF joint reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#body-joint).

Paperboard has direction-dependent and nonlinear behavior. Mäkelä and Östlund's
[orthotropic elastic–plastic model](https://doi.org/10.1016/S0020-7683(03)00318-4)
uses directional material tests; Xia, Boyce and Parks' [three-dimensional
model](https://doi.org/10.1016/S0020-7683(02)00238-X) also addresses anisotropic
plastic response. Experiments on [creasing and folding three
paperboards](https://doi.org/10.1016/j.mechmat.2013.09.016) show why fold/through-
thickness behavior needs its own characterization. A homogeneous isotropic
elastic sheet is therefore an approximation, not a material identification.

## Verified engine/version boundaries

The [MuJoCo 3.17.0 flex reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#flex-elasticity)
provides `young`, `poisson`, `thickness`, damping and `elastic2d` bending/stretching
modes. The [official plate example at tag
3.17.0](https://github.com/google-deepmind/mujoco/blob/3.17.0/model/flex/plate.xml)
is a useful starting point. These are elastic capabilities; material-specific
anisotropy, permanent creases and damage are not established by this interface.

Actual compiler probes on installed **MuJoCo 3.17.0** produced:

| Probe | Result |
|---|---|
| This 2D elastic panel with `discrete`/CG | Compiles and runs |
| Same panel with `implicitfast` | Rejected: flex elasticity requires `discrete` instead of `implicit`/`implicitfast` |
| Add `yield=".001"` to `elasticity` | Rejected: unrecognized `yield` attribute |

These results are saved in `outputs/cardboard-feature-probes/summary.json`.
The unversioned latest documentation describes newer plasticity features; those
must not be treated as available in the installed version. An engine upgrade
would need its own model admission and regression work.

Keep ordinary frictional flex contact for shelf experiments. In this version,
`passive="true"` flex contact against static geometry omits friction, as stated
in the [pinned contact reference](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#flex-contact).
It is not an interchangeable numerical fix for a shelf-sliding experiment.

Newton's [VBD backend](https://newton-physics.github.io/newton/1.7.0/solvers/vbd.html)
offers another route for coupled cloth/soft-body/rigid experiments, but its
documented API is experimental. Changing backends would require robot import,
actuator, contact-force and two-way coupling validation. It is not necessary
for the immediate articulated-panel experiment and was not installed or tested
here.

## Measured elastic-panel coupon

Run the isolated prototype:

```bash
.venv/bin/python scripts/cardboard_panel_probe.py \
  --output-dir outputs/cardboard-panel-example
```

It defines a 120×120 mm, 6 g panel with its perimeter pinned to the world fixture.
A 24 mm diameter, 20 g spherical platen moves only through a bounded native
position actuator: 0.5 s settle, 1 s descent, 0.5 s hold, 1 s retract and 1 s
settle. It commands 20 mm travel; actual travel differs under load and is
recorded separately. No live pose writes, grasp constraints, disabled contacts
or external body/generalized forces are used. Material defaults—**E=300 MPa,
ν=0.25, thickness=0.5 mm**—are deliberately uncalibrated. Native contact radius is
half the thickness, with the same explicit contact settings on sheet and platen.

The six final runs below use the same script hash, MuJoCo 3.17.0 and NumPy 1.26.4.
All ran the full four seconds with finite states, monotonic time and zero MuJoCo
warnings. Force is the measured **sum of platen/contact normal magnitudes**;
the trace also records the distinct net upward contact force and actuator force.
Deflection is center displacement from its gravity-settled state.

| Variant | Additional deflection (mm) | Peak normal sum (N) | Peak penetration (mm) | Hold penetration (mm) | Fixture checks |
|---|---:|---:|---:|---:|---|
| 9×9, 0.5 ms, default | 5.579 | 4.912 | 0.19997 | 0.005 | Pass |
| Half timestep: 0.25 ms | 5.580 | 4.915 | 0.142 | 0.005 | Pass |
| Refined mesh: 13×13 | 5.661 | 4.724 | 0.172 | 0.010 | Pass |
| Current robot period: 2 ms | 5.958 | 6.499 | **1.540** | **1.535** | **Fail: penetration** |
| Softer material: E=100 MPa | 6.375 | 2.747 | **0.307** | 0.003 | **Fail: penetration** |
| Contact time constant 4 ms, versus 1 ms default | 5.560 | 4.863 | 0.138 | 0.047 | Pass |

“Hold penetration” is the maximum during simulation times 1.8–2.0 s. Peaks cover
the entire episode, including contact onset. Passing means contact-driven
deflection >1 mm, rebound within 0.1 mm, maximum penetration <0.2 mm, complete
finite stepping and no warnings. It does **not** mean calibrated or converged
cardboard. In particular, every variant reached the configured 100-iteration CG
cap at some step; the default did so on 53/8000 steps, the 2 ms case on 310/2000.
Mesh refinement changes peak force by about 3.8%. Further solver/mesh/contact
convergence work is needed before interpreting transient force peaks physically.
The failed cases remain in their denominators and artifacts.

The default gravity sag is 0.500 mm; its residual center displacement after
unloading is approximately 0.85 nm in this numerical model. That return is
elastic, not evidence of real cardboard's residual crease behavior. A repeated
default run reproduces the physics trace bitwise.

Each `outputs/cardboard-panel-v2-*` directory contains `scene.xml`, the model and
script hashes, initial/terminal integration states, full-rate NPZ/CSV force and
displacement traces, initial/peak/final panel vertices, warnings and summary.
Sensitivity commands, all requiring fresh directories, are:

```bash
.venv/bin/python scripts/cardboard_panel_probe.py --timestep .00025 --output-dir outputs/panel-half-dt
.venv/bin/python scripts/cardboard_panel_probe.py --grid 13 --output-dir outputs/panel-grid13
.venv/bin/python scripts/cardboard_panel_probe.py --timestep .002 --output-dir outputs/panel-2ms
.venv/bin/python scripts/cardboard_panel_probe.py --young 1e8 --output-dir outputs/panel-softer
.venv/bin/python scripts/cardboard_panel_probe.py --contact-timeconst .004 --output-dir outputs/panel-contact
```

## Before deformable stocking counts as a result

1. Calibrate the chosen carton: mass, outside dimensions and contents; bending
   along machine/cross directions; loading/unloading indentation; fold torque
   and residual set; carton compression and shelf/product friction. Fit curves
   over the force and displacement range used during manipulation, then validate
   on different coupons/cartons.
2. Keep an explicit rigid/articulated/flex variant in scene manifests. Revalidate
   MABEL tracking, force guards and stock-task physics after any integrator or
   timestep change. The isolated coupon does not validate robot `discrete` use.
3. Extend contact attribution before adding flexes: current robot helpers use
   `contact.geom1/geom2`, while native flex contacts can use `geom=-1` and separate
   flex/element IDs. Likewise, shelf support, grasp evidence and bounds must use
   the actual deformable object, not accidental indexing into the last geom.
4. Evaluate actual exterior geometry, local deflection, penetration, support,
   permanent damage and contents retention. A nominal root pose can stay upright
   while a side panel collapses. Define whether creasing/crushing is allowed;
   do not count a damaged or interpenetrating carton as successful packing by
   default.
5. Budget **total** 5–10 mm slack using actual outside envelopes, including wall
   thickness/contact radius. With ten cartons, an unaccounted 0.5 mm width per
   carton consumes all 5 mm. Report per-contact numerical overlap and its
   cumulative effect separately from physical material deformation. The coupon's
   0.2 mm gate is not an acceptance threshold for a whole dense shelf.

The next bounded experiment should compare rigid and articulated-panel cartons
under the same insertion controller and outside dimensions, then add a calibrated
elastic-panel carton only after the engine/contact/evaluation gates above pass.
