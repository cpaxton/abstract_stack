#!/usr/bin/env python3
"""Replay recorded native controls from a checkpoint and attribute product contact."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.checkpoints import native_state
from abstract_stack.native_views import retain_data_views, retain_model_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noslip-iterations", type=int, choices=(0, 1, 3, 10, 30, 100))
    parser.add_argument("--impratio", type=float, choices=(1, 10, 100))
    parser.add_argument("--procedural-contact-timeconstant", type=float, choices=(0.005, 0.02))
    parser.add_argument("--procedural-contact-impedance", type=float, choices=(0.9, 0.99))
    args = parser.parse_args()
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    model_path = args.checkpoint / "model.mjb"
    model = mujoco.MjModel.from_binary_path(str(model_path))
    views = retain_model_views(model)
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    procedural = [
        g
        for g in range(model.ngeom)
        if model.geom(g).name.startswith(("carton_", "shelf_", "supply_"))
    ]
    original_solref = {model.geom(g).name: model.geom_solref[g].tolist() for g in procedural}
    if args.procedural_contact_timeconstant is not None:
        model.geom_solref[procedural, 0] = args.procedural_contact_timeconstant
    if args.procedural_contact_impedance is not None:
        model.geom_solimp[procedural, 0] = args.procedural_contact_impedance
        model.geom_solimp[procedural, 1] = 0.999
    original_noslip = int(model.opt.noslip_iterations)
    original_impratio = float(model.opt.impratio)
    if args.impratio is not None:
        model.opt.impratio = args.impratio
    if args.noslip_iterations is not None:
        model.opt.noslip_iterations = args.noslip_iterations
    with np.load(args.checkpoint / "state.npz") as archive:
        mujoco.mj_setState(model, data, archive["integration"], int(archive["state_spec"]))
    mujoco.mj_forward(model, data)
    active = model.body(args.product).id
    summary = json.loads((args.episode / "summary.json").read_text())
    events = summary["events"]
    groups, samples = {}, []
    start = float(data.time)
    next_sample = start
    force = np.empty(6)
    with np.load(args.episode / "trace.npz") as archive:
        controls, trace = archive["controls"], archive["samples"]
        begin = int(np.searchsorted(trace[:, 0], start + 1e-8))
        for control, row in zip(controls[begin:], trace[begin:], strict=True):
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isclose(data.time, row[0], rtol=0, atol=1e-8):
                raise ValueError("Archived clock differs from native replay")
            normals = {}
            for index, contact in enumerate(data.contact):
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if active not in bodies:
                    continue
                other = int(bodies[1] if bodies[0] == active else bodies[0])
                name = model.body(other).name
                mujoco.mj_contactForce(model, data, index, force)
                normals[name] = normals.get(name, 0.0) + abs(float(force[0]))
            for name, normal in normals.items():
                if normal < 0.01:
                    continue
                group = groups.setdefault(
                    name,
                    dict(first_s=float(data.time), last_s=None, peak_normal_N=0.0, impulse_Ns=0.0),
                )
                group["last_s"] = float(data.time)
                group["peak_normal_N"] = max(group["peak_normal_N"], normal)
                group["impulse_Ns"] += normal * model.opt.timestep
            if data.time >= next_sample:
                rotation = data.xmat[active].reshape(3, 3)
                event = next((e for e in reversed(events) if e["sim_time"] <= data.time), {})
                samples.append(
                    dict(
                        time_s=float(data.time),
                        phase=event.get("phase"),
                        active_product=event.get("product"),
                        position_m=data.xpos[active].tolist(),
                        yaw_deg=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                        contact_normals_N=normals,
                    )
                )
                next_sample += 1.0
    with np.load(args.episode / "rollout_terminal.npz") as archive:
        exact = bool(np.array_equal(native_state(model, data), archive["state"]))
    result = dict(
        kind="recorded_product_contact_diagnostic",
        product=args.product,
        checkpoint=str(args.checkpoint.resolve()),
        episode=str(args.episode.resolve()),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        versions=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        input_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.checkpoint / "manifest.json",
                args.checkpoint / "state.npz",
                args.episode / "summary.json",
                args.episode / "trace.npz",
                args.episode / "rollout_terminal.npz",
            )
        },
        script_sha256=script_digest,
        full_terminal_integration_bit_equal=exact,
        fresh_task_credit=0,
        counterfactual_physics=any(
            x is not None
            for x in (
                args.noslip_iterations,
                args.impratio,
                args.procedural_contact_timeconstant,
                args.procedural_contact_impedance,
            )
        ),
        original_procedural_solref=original_solref,
        procedural_contact_impedance=args.procedural_contact_impedance,
        procedural_contact_timeconstant_s=args.procedural_contact_timeconstant,
        original_impratio=original_impratio,
        effective_impratio=float(model.opt.impratio),
        original_noslip_iterations=original_noslip,
        effective_noslip_iterations=int(model.opt.noslip_iterations),
        contacts=groups,
        samples=samples,
    )
    (args.output / "diagnosis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(dict(exact=exact, contacts=groups)))
    assert views and data_views
    if not exact and not result["counterfactual_physics"]:
        raise ValueError("Native replay terminal state differs; attribution is unverified")


if __name__ == "__main__":
    main()
