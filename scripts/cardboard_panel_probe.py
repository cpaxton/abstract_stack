#!/usr/bin/env python3
"""Actuator-driven elastic panel coupon; not calibrated cardboard or a robot task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def panel_xml(*, grid=9, young=3e8, thickness=0.0005, timestep=0.0005, contact_timeconst=0.001):
    """A 120 mm square, 6 g sheet with its perimeter pinned to the world fixture."""
    pins = " ".join(
        str(i * grid + j)
        for i in range(grid)
        for j in range(grid)
        if i in (0, grid - 1) or j in (0, grid - 1)
    )
    spacing = 0.12 / (grid - 1)
    return f"""<mujoco model="uncalibrated elastic cardboard-panel coupon">
  <option timestep="{timestep}" integrator="discrete" solver="CG"
          iterations="100" tolerance="1e-8" gravity="0 0 -9.81">
    <flag autoreset="disable"/>
  </option>
  <size memory="64M"/>
  <default>
    <geom solref="{contact_timeconst} 1" solimp=".999 .9999 .0001"
          friction=".5 .005 .0001"/>
  </default>
  <worldbody>
    <flexcomp name="panel" type="grid" dim="2" count="{grid} {grid} 1"
              spacing="{spacing} {spacing} {spacing}" pos="0 0 .1"
              radius="{thickness / 2}" mass=".006" rgba=".65 .42 .2 1">
      <contact condim="3" selfcollide="auto" friction=".5 .005 .0001"
               solref="{contact_timeconst} 1" solimp=".999 .9999 .0001"/>
      <elasticity young="{young}" poisson=".25" thickness="{thickness}"
                  elastic2d="both" damping=".001"/>
      <pin id="{pins}"/>
    </flexcomp>
    <body name="platen" pos="0 0 .124">
      <joint name="slide" type="slide" axis="0 0 -1" range="0 .025" damping="1"/>
      <geom name="tip" type="sphere" size=".012" mass=".02" rgba=".5 .6 .7 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="press" joint="slide" kp="2000" ctrlrange="0 .02"
              forcerange="-10 10"/>
  </actuator>
</mujoco>
"""


def command(t):
    """Settle 0.5 s, press 1 s, hold 0.5 s, retract 1 s, settle 1 s."""
    if t < 0.5:
        return 0.0
    if t < 1.5:
        return 0.02 * (t - 0.5)
    if t < 2.0:
        return 0.02
    return 0.02 * max(0.0, 3.0 - t)


def run(args):
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    settings = {
        key: getattr(args, key)
        for key in ("grid", "young", "thickness", "timestep", "contact_timeconst")
    }
    xml = panel_xml(**settings)
    (directory / "scene.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.flexvert_xpos.copy()
    baseline_z = initial[args.grid**2 // 2, 2]
    edges = model.flex_edge
    rest_lengths = np.linalg.norm(initial[edges[:, 0]] - initial[edges[:, 1]], axis=1)
    center = args.grid**2 // 2
    tip = model.geom("tip").id
    slide = int(model.joint("slide").qposadr[0])
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)
    np.savez_compressed(directory / "initial_state.npz", state=state, state_spec=int(state_spec))
    rows, peak_vertices = [], initial.copy()
    peak_deflection, failure = 0.0, None
    started = time.monotonic()
    columns = (
        "time_s",
        "commanded_travel_m",
        "measured_travel_m",
        "center_z_m",
        "center_below_settled_m",
        "platen_normal_sum_N",
        "platen_upward_contact_N",
        "actuator_force_N",
        "maximum_penetration_m",
        "maximum_edge_strain",
        "platen_contacts",
        "solver_iterations",
    )
    try:
        for _ in range(int(np.ceil(4.0 / args.timestep))):
            previous_time = float(data.time)
            data.ctrl[0] = command(previous_time)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if data.time <= previous_time or not (
                np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            ):
                raise RuntimeError("Non-finite state or physics time reset")
            normal, upward, penetration, contacts = 0.0, 0.0, 0.0, 0
            for index, contact in enumerate(data.contact):
                if tip not in contact.geom:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force)
                world = contact.frame.reshape(3, 3).T @ force[:3]
                normal += float(force[0])
                upward += float(world[2] * (1 if contact.geom[1] == tip else -1))
                penetration = max(penetration, -float(contact.dist))
                contacts += 1
            vertices = data.flexvert_xpos
            lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            strain = float(np.max(np.abs(lengths / rest_lengths - 1)))
            z = float(vertices[center, 2])
            if data.time <= 0.5 + args.timestep / 2:
                baseline_z = z
            if initial[center, 2] - z > peak_deflection:
                peak_deflection = initial[center, 2] - z
                peak_vertices = vertices.copy()
            rows.append(
                [
                    float(data.time),
                    float(data.ctrl[0]),
                    float(data.qpos[slide]),
                    z,
                    0.0,
                    normal,
                    upward,
                    float(data.actuator_force[0]),
                    penetration,
                    strain,
                    contacts,
                    int(data.solver_niter[0]),
                ]
            )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failure = str(error)
    samples = np.asarray(rows, dtype=float).reshape(-1, len(columns))
    samples[:, 4] = baseline_z - samples[:, 3]
    np.savez_compressed(directory / "trace.npz", samples=samples, columns=columns)
    with (directory / "trace.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(samples)
    np.savez_compressed(
        directory / "panel_shapes.npz",
        initial=initial,
        peak=peak_vertices,
        terminal=data.flexvert_xpos.copy(),
        edges=edges,
    )
    mujoco.mj_getState(model, data, state, state_spec)
    np.savez_compressed(directory / "terminal_state.npz", state=state, state_spec=int(state_spec))
    warnings = {
        mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number
    }
    metrics = (
        {
            "maximum_additional_center_deflection_m": float(samples[:, 4].max()),
            "settled_gravity_sag_m": float(initial[center, 2] - baseline_z),
            "residual_center_displacement_m": float(samples[-1, 4]),
            "maximum_contact_normal_sum_N": float(samples[:, 5].max()),
            "maximum_upward_contact_N": float(samples[:, 6].max()),
            "maximum_penetration_m": float(samples[:, 8].max()),
            "maximum_edge_strain": float(samples[:, 9].max()),
            "maximum_solver_iterations": int(samples[:, 11].max()),
            "wall_seconds": time.monotonic() - started,
        }
        if len(samples)
        else {}
    )
    passed = (
        bool(len(samples))
        and failure is None
        and not warnings
        and len(samples) * args.timestep >= 4.0
        and metrics["maximum_additional_center_deflection_m"] > 0.001
        and abs(metrics["residual_center_displacement_m"]) < 0.0001
        and metrics["maximum_penetration_m"] < 0.0002
    )
    summary = {
        "scope": "Uncalibrated isotropic elastic panel coupon; no robot or stocking claim",
        "calibrated_cardboard": False,
        "fixture_checks_passed": bool(passed),
        "failure": failure,
        "warnings": warnings,
        "parameters": settings,
        "fixture_check_definition": {
            "physical_contact_and_reversible_deflection": "Additional center deflection >1mm",
            "rebound": "Residual center displacement <0.1mm from gravity-settled state",
            "contact_accuracy": "Maximum contact penetration <0.2mm",
            "numerical_health": "4s finite state, monotonic clock, zero MuJoCo warnings",
            "limitations": "These checks do not establish constitutive accuracy or convergence",
        },
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "model_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "integrator": "discrete",
        "solver": "CG",
        "sim_seconds": float(data.time),
        "external_generalized_force_nonzero": bool(np.any(data.qfrc_applied)),
        "external_body_force_nonzero": bool(np.any(data.xfrc_applied)),
        "metrics": metrics,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=9)
    parser.add_argument("--young", type=float, default=3e8)
    parser.add_argument("--thickness", type=float, default=0.0005)
    parser.add_argument("--timestep", type=float, default=0.0005)
    parser.add_argument("--contact-timeconst", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cardboard-panel-probe"))
    args = parser.parse_args()
    if not 5 <= args.grid <= 25 or args.grid % 2 != 1:
        parser.error("--grid must be odd, between 5 and 25")
    for name, low, high in (
        ("young", 1e6, 1e10),
        ("thickness", 0.0001, 0.005),
        ("timestep", 1e-5, 0.002),
        ("contact_timeconst", 0.0001, 0.02),
    ):
        if not np.isfinite(getattr(args, name)) or not low <= getattr(args, name) <= high:
            parser.error(f"--{name} must be finite within [{low}, {high}]")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
