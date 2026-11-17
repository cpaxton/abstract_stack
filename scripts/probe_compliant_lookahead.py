#!/usr/bin/env python3
"""Supervised retained-state compliant push forecasts, never fresh task credit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    import mujoco
    import numpy as np

    from abstract_stack.compliant_lookahead import PushAdmittance, generate_push_candidates
    from abstract_stack.contact_finish import BackstopDwell
    from abstract_stack.contact_rollout import (
        ContactMonitor,
        RolloutScene,
        evaluate_screened_batch,
        integration_state,
    )
    from abstract_stack.interactions import ScriptConfig
    from abstract_stack.load_aware_pusher import LoadAwareFingerPusher
    from abstract_stack.native_views import retain_data_views, retain_model_views
    from abstract_stack.sim_sensors import ContactSensorSuite

    started = time.monotonic()
    source = args.output / "inputs"
    original = json.loads((source / "summary.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
    data = mujoco.MjData(model)
    views = retain_model_views(model), retain_data_views(model, data)
    with np.load(source / "rollout_terminal.npz") as state:
        if int(state["state_spec"]) != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("Expected complete native integration state")
        mujoco.mj_setState(model, data, state["state"], int(state["state_spec"]))
    mujoco.mj_forward(model, data)
    initial = integration_state(model, data)
    active = original["events"][-1]["product"]
    product = next(p for p in manifest["products"] if p["name"] == active)
    choices = next(
        e["selected"] for e in reversed(original["events"]) if e["phase"] == "pusher_plan"
    )
    hand = LoadAwareFingerPusher(
        SimpleNamespace(model=model, data=data, dt=0.02),
        ScriptConfig(**original["controller"]),
        **choices,
    )
    nominal = np.array([original["last_nominal_targets"][n] for n in [*hand.names, "Lift"]])
    hand.command = nominal.copy()
    suite = ContactSensorSuite.mabel(model)
    reading = suite.sample(data)
    rotation = data.xmat[hand.palm].reshape(3, 3)
    wrench = np.r_[
        rotation @ reading["contact_wrench"][0, :3], rotation @ reading["contact_wrench"][0, 3:]
    ]
    origin, _ = hand.ik.forward(nominal)
    level = manifest["config"]["levels"][product["level"]]
    lower = np.array([origin[0] - 0.03, product["target"][1] - 0.02, level + 0.023])
    upper = np.array([origin[0] + 0.03, product["target"][1] + 0.02, level + 0.053])
    configuration = PushAdmittance(normal_target_N=original["controller"]["push_force_target"])
    candidates, generation, rates = generate_push_candidates(
        hand,
        active,
        nominal=nominal,
        wrench_on_hand_world=wrench,
        lower_xyz=lower,
        upper_xyz=upper,
        horizon_s=args.horizon,
        max_candidates=args.candidates,
        wall_seconds=args.generation_seconds,
        admittance=configuration,
    )
    scene = RolloutScene(
        products=tuple(p["name"] for p in manifest["products"]),
        active=active,
        robot_root="mabel_chassis",
        hand_root="left_palm",
        support_geoms=("supply_table", f"shelf_board_{product['level']}"),
        active_half_size=product["half_size"],
        support_height=level,
        back_geom="shelf_back",
        base_root="mabel_chassis",
    )
    evaluated = evaluate_screened_batch(
        model,
        initial,
        candidates,
        scene,
        rates=rates,
        workers=4,
        shortlist=min(args.shortlist, len(candidates)),
    )
    if not np.array_equal(initial, integration_state(model, data)):
        raise ValueError("Forecast generation changed the source native state")
    np.savez_compressed(
        args.output / "candidates.npz",
        controls=np.stack([c["controls"] for c in candidates]),
        initial_state=initial,
        rates=rates,
    )
    selected = next(
        (r for r in evaluated["candidates"] if r["name"] == evaluated["selected_name"]), None
    )
    report = dict(
        kind="retained_state_teacher_contact_forecast",
        fresh_task_credit=0,
        task_success=False,
        live_task_execution_admitted=False,
        model_revision=manifest["robot_source"]["revision"],
        active=active,
        policy=asdict(configuration),
        wrench_on_hand_world_N_Nm=wrench.tolist(),
        sensor_metadata=suite.metadata(),
        generation=generation,
        evaluation=evaluated,
        candidate_metadata=[c["metadata"] for c in candidates],
        selected=selected,
        worker_wall_seconds=time.monotonic() - started,
        source_state_unchanged=True,
    )
    if args.execute_prefix > 0:
        prefix_steps = int(round(args.execute_prefix / model.opt.timestep))
        control_steps = int(round(hand.env.dt / model.opt.timestep))
        if (
            prefix_steps % control_steps
            or not np.isclose(
                prefix_steps * model.opt.timestep, args.execute_prefix, rtol=0, atol=1e-9
            )
            or not 1 <= prefix_steps <= len(candidates[0]["controls"])
        ):
            raise ValueError("Prefix must contain whole control intervals within the horizon")
        monitor = ContactMonitor(model, data, scene)
        yaw_limit = next(
            (
                event["requested_backstop_yaw_tolerance_deg"]
                for event in reversed(original["events"])
                if event.get("phase") == "finger_push"
                and "requested_backstop_yaw_tolerance_deg" in event
            ),
            None,
        )
        if args.require_dwell and yaw_limit is None:
            raise ValueError("Require the retained source's explicit backstop yaw criterion")
        dwell = (
            BackstopDwell(
                float(model.opt.timestep),
                hand.config.contact_threshold,
                hand.config.contact_dwell,
                hand.config.payload_upright_degrees,
                yaw_limit,
            )
            if args.require_dwell
            else None
        )
        records = []
        executed = []
        tracking_samples = []
        actor_failure = None
        active_body = model.body(active).id
        start_x = float(data.xpos[active_body, 0])
        yaw_integral_lever = 0.0
        for cycle in range(args.replans):
            lateral_goal = lateral_measured = None
            objective = "advance"
            if cycle:
                expected = integration_state(model, data)
                reading = suite.sample(data)
                rotation = data.xmat[hand.palm].reshape(3, 3)
                wrench = np.r_[
                    rotation @ reading["contact_wrench"][0, :3],
                    rotation @ reading["contact_wrench"][0, 3:],
                ]
                if (
                    args.square_feedback
                    and monitor.last.get("back_N", 0) >= hand.config.contact_threshold
                ):
                    object_rotation = data.xmat[active_body].reshape(3, 3)
                    yaw = float(np.arctan2(object_rotation[1, 0], object_rotation[0, 0]))
                    velocity = np.empty(6)
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_BODY, active_body, velocity, 0
                    )
                    # Reuse the dense controller's bounded yaw lever and
                    # anti-windup gains; task/force tolerances stay unchanged.
                    yaw_integral_lever -= 0.03 * yaw * args.execute_prefix
                    lever = float(
                        np.clip(
                            -hand.config.insertion_yaw_lever_gain * yaw
                            + yaw_integral_lever
                            - 0.01 * velocity[2],
                            -0.014,
                            0.014,
                        )
                    )
                    yaw_integral_lever = (
                        lever + hand.config.insertion_yaw_lever_gain * yaw + 0.01 * velocity[2]
                    )
                    lateral_goal = float(
                        np.clip(data.xpos[active_body, 1] + lever, lower[1], upper[1])
                    )
                    contact_points = []
                    for index, contact in enumerate(data.contact):
                        b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                        if contact.efc_address < 0 or not (
                            b1 == active_body
                            and b2 in suite.subtrees[0]
                            or b2 == active_body
                            and b1 in suite.subtrees[0]
                        ):
                            continue
                        load = np.empty(6)
                        mujoco.mj_contactForce(model, data, index, load)
                        if load[0] > 0:
                            contact_points.append((float(contact.pos[1]), float(load[0])))
                    lateral_measured = (
                        float(
                            np.average(
                                [point[0] for point in contact_points],
                                weights=[point[1] for point in contact_points],
                            )
                        )
                        if contact_points
                        else float(hand.position()[1])
                    )
                    objective = "square_backstop"
                candidates, generation, rates = generate_push_candidates(
                    hand,
                    active,
                    nominal=nominal,
                    wrench_on_hand_world=wrench,
                    lower_xyz=lower,
                    upper_xyz=upper,
                    horizon_s=args.horizon,
                    max_candidates=args.candidates,
                    wall_seconds=args.generation_seconds,
                    admittance=configuration,
                    lateral_goal_m=lateral_goal,
                    lateral_measurement_m=lateral_measured,
                )
                evaluated = evaluate_screened_batch(
                    model,
                    expected,
                    candidates,
                    scene,
                    rates=rates,
                    workers=4,
                    shortlist=min(args.shortlist, len(candidates)),
                    objective=objective,
                )
            else:
                expected = initial
            selected_name = evaluated["selected_name"]
            record = dict(
                cycle=cycle,
                generation=generation,
                evaluation=evaluated,
                candidate_metadata=[c["metadata"] for c in candidates],
                yaw_integral_lever_m=yaw_integral_lever,
                privileged_lateral_goal_m=lateral_goal,
                privileged_lateral_measurement_m=lateral_measured,
            )
            records.append(record)
            if selected_name is None:
                actor_failure = "no_guarded_continuation"
                break
            chosen = next(c for c in candidates if c["name"] == selected_name)
            # All candidate endpoint/path, nominal/applied/finger and motion
            # gates were computed from this exact unchanged captured state.
            if not np.array_equal(expected, integration_state(model, data)):
                actor_failure = "stale_prefix_admission"
                break
            actor_failure = monitor.sample(data, elapsed=0)
            if actor_failure:
                break
            if dwell is not None and dwell.last_time_s is None:
                dwell.update(data.time, monitor.last, data.xmat[active_body].reshape(3, 3))
            for step_index, control in enumerate(chosen["controls"][:prefix_steps]):
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                executed.append(control.copy())
                if (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or np.any(data.warning.number)
                    or abs(data.time - float(initial[0]) - len(executed) * model.opt.timestep)
                    > 1e-7
                ):
                    actor_failure = "native_state_or_warning"
                    break
                actor_failure = monitor.sample(data, elapsed=model.opt.timestep)
                if actor_failure:
                    break
                trace = chosen["metadata"].get("nominal_references")
                if trace:
                    tick = step_index // control_steps
                    previous = nominal if tick == 0 else np.asarray(trace[tick - 1])
                    fraction = (step_index % control_steps + 1) / control_steps
                    reference = previous + (np.asarray(trace[tick]) - previous) * fraction
                    predicted, _ = hand.ik.forward(reference, data.qpos.copy())
                    tracking_samples.append(
                        [float(data.time), float(np.linalg.norm(hand.position() - predicted))]
                    )
                if np.any(hand.position() < lower) or np.any(hand.position() > upper):
                    actor_failure = "measured_tool_left_corridor"
                    break
                if dwell is not None and dwell.update(
                    data.time, monitor.last, data.xmat[active_body].reshape(3, 3)
                ):
                    break
            record.update(
                executed_steps=min(prefix_steps, len(executed) - cycle * prefix_steps),
                after_x_m=float(data.xpos[active_body, 0]),
                last_sample=monitor.last.copy(),
            )
            contact_finished = (
                dwell.verified
                if dwell is not None
                else monitor.last.get("back_N", 0) >= hand.config.contact_threshold
            )
            if actor_failure or contact_finished:
                break
            if chosen["metadata"].get("nominal_references"):
                nominal = np.array(
                    chosen["metadata"]["nominal_references"][prefix_steps // control_steps - 1]
                )
                hand.command = nominal.copy()
        if dwell is not None and not dwell.verified and actor_failure is None:
            actor_failure = "backstop_dwell_decision_budget_exhausted"
        report["continuation"] = dict(
            kind="retained_checkpoint_continuation_not_new_episode",
            replans=records,
            scheduled_replans=args.replans,
            executed_physics_steps=len(executed),
            continuation_execution_admitted=bool(executed),
            simulated_seconds=len(executed) * model.opt.timestep,
            progress_minus_x_m=start_x - float(data.xpos[active_body, 0]),
            failure=actor_failure,
            measured_back_contact_N=monitor.last.get("back_N", 0),
            backstop_dwell=None if dwell is None else dwell.report(),
            active_rotation_world=data.xmat[active_body].reshape(3, 3).tolist(),
            active_position_world_m=data.xpos[active_body].tolist(),
            peak_forces_N=monitor.peaks,
            incidents=monitor.incidents,
            maximum_product_displacement_m=monitor.max_displacement,
            measured_base_nominal_fk_tracking_peak_m=(
                max(r[1] for r in tracking_samples) if tracking_samples else None
            ),
            tracking_scope="actual tool versus nominal FK at measured base; separate from IK error",
            full_stocking_success=False,
            fresh_task_credit=0,
        )
        np.savez_compressed(
            args.output / "continuation.npz",
            controls=np.asarray(executed),
            terminal_state=integration_state(model, data),
            tracking_samples=np.asarray(tracking_samples),
        )
        report["worker_wall_seconds"] = time.monotonic() - started
        report["forecast_generation_source_state_unchanged"] = True
        report["source_state_unchanged"] = bool(
            np.array_equal(initial, integration_state(model, data))
        )
        report["source_checkpoint_file_unchanged"] = True
    write(args.output / "summary.json", report)
    print(
        json.dumps(
            dict(
                generated=len(candidates),
                selected=evaluated["selected_name"],
                selected_progress_m=None if selected is None else selected["progress_minus_x_m"],
                wall_seconds=report["worker_wall_seconds"],
                fresh_task_credit=0,
            )
        )
    )
    assert views  # Retain wrapper lifetimes until all native work has ended.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=0.4)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--generation-seconds", type=float, default=10)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--execute-prefix", type=float, default=0)
    parser.add_argument("--require-dwell", action="store_true")
    parser.add_argument("--square-feedback", action="store_true")
    parser.add_argument("--replans", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not (
        1 <= args.candidates <= 16
        and 1 <= args.shortlist <= args.candidates
        and 0 < args.wall_seconds <= 180
        and 0 < args.generation_seconds <= 30
        and 0 <= args.execute_prefix <= args.horizon
        and 1 <= args.replans <= 80
        and (not args.require_dwell or args.execute_prefix > 0)
        and (not args.square_feedback or args.require_dwell)
    ):
        parser.error("Keep candidate, shortlist and compute budgets bounded")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.episode is None:
        parser.error("--episode is required")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    for src, name in (
        (args.episode / "summary.json", "summary.json"),
        (args.episode / "initial-scene/manifest.json", "manifest.json"),
        (args.episode / "initial-scene/scene.xml", "scene.xml"),
        (args.episode / "rollout_terminal.npz", "rollout_terminal.npz"),
    ):
        shutil.copy2(src, inputs / name)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    shutil.copytree(
        root / "src/abstract_stack",
        snapshot / "src/abstract_stack",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for src, name in (
        (Path(__file__), "probe.py"),
        (root / "scripts/run_frozen_replanning.py", "supervisor.py"),
        (root / "uv.lock", "uv.lock"),
    ):
        shutil.copy2(src, snapshot / name)
    write(
        args.output / "manifest.json",
        dict(
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            versions={n: importlib.metadata.version(n) for n in ("mujoco", "numpy")},
            settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            sha256={
                str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for d in (inputs, snapshot)
                for p in d.rglob("*")
                if p.is_file()
            },
        ),
    )
    command = [
        sys.executable,
        str(snapshot / "probe.py"),
        "--worker",
        "--output",
        str(args.output),
        "--horizon",
        str(args.horizon),
        "--candidates",
        str(args.candidates),
        "--shortlist",
        str(args.shortlist),
        "--generation-seconds",
        str(args.generation_seconds),
        "--execute-prefix",
        str(args.execute_prefix),
        "--replans",
        str(args.replans),
    ]
    if args.require_dwell:
        command.append("--require-dwell")
    if args.square_feedback:
        command.append("--square-feedback")
    write(
        args.output / "process.json",
        dict(returncode=None, verification_status="unknown", fresh_task_credit=0),
    )
    supervisor = runpy.run_path(str(snapshot / "supervisor.py"))["supervise"]
    result = supervisor(
        command,
        cwd=root,
        environment=dict(os.environ, PYTHONPATH=str(snapshot / "src"), PYTHONFAULTHANDLER="1"),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    result.update(
        verification_status="completed_forecast"
        if result["returncode"] == 0 and (args.output / "summary.json").exists()
        else "unknown",
        command=command,
        fresh_task_credit=0,
        task_success=False,
    )
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["verification_status"] == "completed_forecast" else 2


if __name__ == "__main__":
    raise SystemExit(main())
