# Sideways contact skill: first development probes

The ten-carton milestone remains **zero prefilled cartons and one uninterrupted
stocking episode**. The new opt-in `LateralStockingScript.push_laterally()` is an
isolated exposed-side skill. Two development probes now demonstrate a supported
sideways push with unloading and retreat. It has not yet been composed with
middle-row insertion or demonstrated packed-row compaction.

The primitive searches native selected-finger orientations, checks approach and
push kinematics in independent scratch data, and uses the existing compensated
position servos. Its outer force loop advances toward a 1.5 N lateral setpoint,
unloads above that setpoint, and bounds travel, speed and height. Every corrected
reference receives another IK/collision check before native actuation. Success
also requires shelf support and displacement to survive unloading and retreat.
The existing 25 N abort guards and all-product per-step tip/drop monitoring remain
active. The default stocking controller is unchanged.

The existing supported-slide predictor only forecasts motion along X. This
prototype deliberately does not use it to certify lateral carton motion; it
checks robot geometry while retaining measured, stationary object geometry in
planning. Contact dynamics must be validated in execution. A full lateral object
forecast and uncertainty margins remain work to do.

## Observed development evidence

Artifacts are retained in
`outputs/contact-rich-replanning-v1/lateral-probe-v1` through `lateral-probe-v12`,
with the original first-five aggregate `lateral-summary.json` preserved. These are changing-source development
probes, not a frozen-policy success-rate campaign.

- Probes 1–2 used a three-carton shelf, two prefilled cartons and a middle carton
  explicitly reset to the shelf front. Both 90-degree side approaches were
  rejected before actuation.
- Probe 3 added diagonal approach orientations. An accurately reached endpoint
  still collided with fixtures or had a colliding sampled path.
- Probe 4 used the ten-carton shelf with nine prefilled cartons. The best endpoint
  had 0.00000037 m position error and 0.00066 degree orientation error, and passed
  endpoint collision checks. The direct sampled approach still collided.
- Probe 5 added the existing checked high staging motion. Planning accepted an
  approach/push route with approximately 0.00000077 m endpoint position error.
  Actual motion stopped at 11.142 simulated seconds when measured arm/fixture
  contact reached 32.79 N, crossing the unchanged 25 N guard. There was no hand
  contact with the active carton, no lateral carton movement and no recorded
  product tip/drop incident. No force-limited run counts as success.

- Probe 6 added orientation above the carton, Cartesian descent and a 12 mm
  predicted board-clearance margin. Approach tracking error was 0.307 mm and no
  fixture contact occurred. A 28 mm contact height caused rocking: the carton
  moved 31.2 mm laterally and crossed the 15 degree control tilt guard. Failed.
- Probes 7–8 lowered nominal contact height to 18 mm (measured contacts roughly
  21–23 mm above the board). The carton remained upright, moving 0.62 mm and
  2.17 mm before the 8 s displacement timeout. Failed.
- Probe 9 used the 1.875 N force target, 10 mm/s maximum reference speed, a 20 s
  budget and distance-based braking. It reached 9.125 mm for a 10 mm request but
  the original joint withdrawal rejected a fingertip/carton path contact. Failed.
- Probes 10–11 changed unloading to an explicit Cartesian normal withdrawal,
  allowing contact between the selected hand and active carton while preserving
  every other collision/force guard. Both passed the isolated skill on seeds 0
  and 1: final lateral shifts **9.174 mm / 9.120 mm**, final errors
  **0.826 mm / 0.880 mm**, X drift **0.444 mm / 0.351 mm**, peak hand normals
  **1.489 N / 1.494 N**, no fixture forces and no product incidents. Unloading and
  retreat passed. Total fixture episodes took **36.22 s / 36.18 s** simulated time.
- Probe 12 tested the opposite push direction with the same settings; its bounded
  approach search was rejected. Bidirectional skill success remains unverified.

All reset fixtures earn **zero pick or complete-stocking credit**. Probes 10–11
keep the active carton at the shelf front; its shifted Y envelope overlaps the
projected envelopes of rear-row cartons even though their actual X ranges do
not intersect. They therefore **fail full-row grading**, correctly. They are not
insertion-then-packing successes. The whole-row 0.2 mm overlap criteria remain
unchanged. Start/terminal package hashes matched within each successful probe,
but development source was not copied before those runs. Matching files have
been recovered separately under `lateral-v10-matching-source`; four changed
modules are explicitly missing. This is not a complete frozen-policy snapshot.

A rejected bounded search does not establish global unreachability. Subsequent
front-face tangential probes use complete immutable package snapshots; see the
[tangential investigation](tangential_contact_development.md).

## Reproduce a new diagnostic

From the feature worktree, using a fresh output directory:

```bash
PYTHONPATH=src MABEL_ROOT=/path/to/open_mabel \
  python scripts/probe_lateral_contact.py \
  --output outputs/new-lateral-probe --objects 10 --direction 1 --distance 0.01 \
  --contact-height 0.018 --force-target 1.875 --push-speed 0.01 --timeout 20
```

The command saves scene/source metadata, start/terminal states, native control
and force traces, candidate rejection reasons and per-product incidents. It
never overwrites an existing run. Probe 5 predates the final documentation and
metadata additions; use its recorded Python hashes to identify its exact source.

Validation: ten lateral-skill tests passed with real MABEL assets, including
force-response reversal/bounds and unchanged live state during scratch planning.
The side-pusher subclass explicitly retains the original pusher’s 0.5 rad
wrist reference-offset budget without modifying native gains. Success requires
symmetric lateral tolerance, bounded X drift and upright support after retreat.
Requests must exceed the displacement tolerance, and measured hand force in the
push direction is required before displacement can pass its dwell gate.
Ruff format/lint passed. These checks do not establish manipulation success.
