#!/usr/bin/env python3
"""Isolated native finger characterization; proposals are not engagement approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop import retarget_fingers as module
from abstract_stack.teleop import retarget_fingers_fixtures as fixture_module
from abstract_stack.teleop.retarget_fingers import FingerRetargeter
from abstract_stack.teleop.retarget_fingers_fixtures import synthetic_hand
from abstract_stack.teleop.sim import MujocoTeleopAdapter


def curls(t):
    if t < 0.4:
        return 0.0, 0.0
    if t < 1.4:
        return t - 0.4, 0.0
    if t < 2.4:
        return 1.0, t - 1.4
    if t < 3.0:
        return 1.0, 1.0
    return max(0.0, 4 - t), max(0.0, 4 - t)


def finger_contacts(model, data):
    maximum, rows = 0.0, []
    for index, contact in enumerate(data.contact):
        bodies = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
        if not any(
            any(f"_{finger}_" in name for finger in ("thumb", "index", "middle", "ring", "pinky"))
            for name in bodies
        ):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise RuntimeError("Nonfinite native contact force")
        maximum = max(maximum, float(force[0]))
        rows.append(
            {"bodies": bodies, "normal_force_n": float(force[0]), "distance_m": float(contact.dist)}
        )
    return maximum, rows


def run_case(directory, name, front_x, render):
    config = ShelfConfig(product_count=0, front_x=front_x, terminate_on_success=False)
    with MujocoTeleopAdapter(config, seed=0) as sim:
        model, data = sim._env.model, sim._env.data
        mapper = FingerRetargeter(model)
        state = sim.read_state()
        mapper.reset(dict(zip(state.actuator_names, state.applied_targets, strict=True)))
        ids = [state.actuator_names.index(n) for n in mapper.names]
        before = {
            key: getattr(model, key).copy()
            for key in (
                "body_mass",
                "body_inertia",
                "geom_contype",
                "geom_conaffinity",
                "jnt_range",
                "actuator_ctrlrange",
                "actuator_gainprm",
                "actuator_gear",
                "eq_data",
            )
        }
        report = {
            "case": name,
            "provenance": sim.describe(),
            "mapper": mapper.describe(),
            "contact_sampling": "final native substep at each50Hz controltick; not allsubsteppeak",
            "contact_sample_count": 0,
            "maximum_observed_normal_force_n": 0.0,
        }
        errors = []
        closed = None
        renderer = None
        trace = directory / f"{name}.jsonl"
        try:
            with trace.open("w") as stream:
                for step in range(210):
                    left, right = curls(step * sim.control_dt)
                    proposal = mapper.propose(
                        synthetic_hand("Left", left), synthetic_hand("Right", right), sim.control_dt
                    )
                    if not proposal.valid:
                        raise RuntimeError(proposal.reason)
                    command = state.applied_targets.copy()
                    command[ids] = [proposal.targets[n] for n in mapper.names]
                    sim.apply_actuator_targets(command)
                    state = sim.step()
                    peak, contacts = finger_contacts(model, data)
                    report["contact_sample_count"] += len(contacts)
                    report["maximum_observed_normal_force_n"] = max(
                        report["maximum_observed_normal_force_n"], peak
                    )
                    measured = state.actuator_lengths[ids]
                    error = measured - command[ids]
                    errors.append(np.abs(error))
                    if 2.8 <= state.sim_time_s < 2.8 + sim.control_dt:
                        closed = {
                            "nominal_rad": proposal.nominal,
                            "applied_rad": proposal.targets,
                            "measured_rad": dict(
                                zip(mapper.names, map(float, measured), strict=True)
                            ),
                            "contacts": contacts,
                        }
                        if render:
                            renderer = mujoco.Renderer(model, width=480, height=360)
                            camera = mujoco.MjvCamera()
                            camera.lookat[:] = data.xpos[model.body("left_palm").id]
                            camera.distance = 0.32
                            camera.azimuth = 145
                            camera.elevation = -15
                            renderer.update_scene(data, camera)
                            Image.fromarray(renderer.render()).save(
                                directory / f"{name}-closed.png"
                            )
                    stream.write(
                        json.dumps(
                            {
                                "sim_time_s": state.sim_time_s,
                                "input_curl": [left, right],
                                "nominal_rad": proposal.nominal,
                                "applied_rad": proposal.targets,
                                "measured_rad": dict(
                                    zip(mapper.names, map(float, measured), strict=True)
                                ),
                                "native_actuator_force_nm": dict(
                                    zip(
                                        mapper.names,
                                        map(float, state.actuator_forces[ids]),
                                        strict=True,
                                    )
                                ),
                                "contacts": contacts,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
        finally:
            if renderer is not None:
                renderer.close()
        for key, value in before.items():
            if not np.array_equal(value, getattr(model, key)):
                raise RuntimeError(f"Native model changed: {key}")
        errors = np.array(errors)
        report.update(
            {
                "trace": trace.name,
                "native_model_fields_preserved": True,
                "closed_sample": closed,
                "duration_sim_s": state.sim_time_s,
                "absolute_tracking_error_rad_p50": dict(
                    zip(mapper.names, np.percentile(errors, 50, axis=0).tolist(), strict=True)
                ),
                "absolute_tracking_error_rad_max": dict(
                    zip(mapper.names, np.max(errors, axis=0).tolist(), strict=True)
                ),
                "scope": (
                    "isolated ctrl-only synthetic characterization; "
                    "no collision/engagement gate, no grasp claim"
                ),
            }
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/quest-retarget-fingers")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_revision_at_start": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty_at_start": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(module.__file__), Path(fixture_module.__file__))
        },
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }
    results = [
        run_case(directory, "clear-space", -1.4, args.render),
        run_case(directory, "default-shelf-contact", -0.62, args.render),
    ]
    (directory / "summary.json").write_text(
        json.dumps({**provenance, "cases": results}, indent=2, allow_nan=False) + "\n"
    )
    for result in results:
        closed = result["closed_sample"]
        print(
            json.dumps(
                {
                    "case": result["case"],
                    "sampled_contact_peak_n": result["maximum_observed_normal_force_n"],
                    "left_index_closed_applied_rad": closed["applied_rad"]["Left Index MCP"],
                    "left_index_closed_measured_rad": closed["measured_rad"]["Left Index MCP"],
                }
            )
        )


if __name__ == "__main__":
    main()
