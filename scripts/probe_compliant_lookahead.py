#!/usr/bin/env python3
"""Supervised retained-state compliant push forecasts, never fresh task credit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
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

    from abstract_stack.checkpoints import load_checkpoint, save_checkpoint
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
    from abstract_stack.squaring_fixture import neighbor_reset
    from abstract_stack.withdrawal import supported_withdrawal

    started = time.monotonic()
    source = args.output / "inputs"
    checkpoint = source / "checkpoint"
    if args.checkpoint and (source / "checkpoint_ref.json").exists():
        reference = json.loads((source / "checkpoint_ref.json").read_text())
        checkpoint = Path(reference["path"])
        if (
            hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()
            != reference["sha256"]
        ):
            raise ValueError("Checkpoint reference changed")
    restored = load_checkpoint(checkpoint) if args.checkpoint else None
    if restored is not None:
        if restored.controller.get("kind") != "retained_finger_push":
            raise ValueError("Checkpoint lacks the explicit finger-push controller contract")
        original, manifest = restored.context["source_summary"], restored.scene
        model, data = restored.model, restored.data
    else:
        original = json.loads((source / "summary.json").read_text())
        manifest = json.loads((source / "manifest.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
        data = mujoco.MjData(model)
        with np.load(source / "rollout_terminal.npz") as state:
            if int(state["state_spec"]) != int(mujoco.mjtState.mjSTATE_INTEGRATION):
                raise ValueError("Expected complete native integration state")
            mujoco.mj_setState(model, data, state["state"], int(state["state_spec"]))
        mujoco.mj_forward(model, data)
    active = original["events"][-1]["product"]
    reset_created = bool(args.reset_neighbor)
    fixture = manifest.get("reset_fixture")
    if args.reset_neighbor:
        if restored is None:
            raise ValueError("Neighbor fixtures require a verified compiled checkpoint")
        data, manifest, fixture = neighbor_reset(model, data, manifest, active, args.reset_neighbor)
    views = retain_model_views(model), retain_data_views(model, data)
    initial = integration_state(model, data)
    product = next(p for p in manifest["products"] if p["name"] == active)
    choices = (
        restored.controller["pusher_settings"]
        if restored
        else next(
            e["selected"] for e in reversed(original["events"]) if e["phase"] == "pusher_plan"
        )
    )
    hand = LoadAwareFingerPusher(
        SimpleNamespace(model=model, data=data, dt=0.02),
        ScriptConfig(**original["controller"]),
        **choices,
    )
    nominal_targets = (
        restored.controller["nominal_subset"] if restored else original["last_nominal_targets"]
    )
    nominal = np.array([nominal_targets[n] for n in [*hand.names, "Lift"]])
    hand.command = nominal.copy()
    wrist_range = restored.controller.get("planning_wrist_range") if restored else None
    if args.wrist_freedom_deg:
        index = hand.ik.names.index("Left Wrist")
        radius = np.radians(args.wrist_freedom_deg)
        wrist_range = [
            float(max(hand.ik.original_lower[index], nominal[index] - radius)),
            float(min(hand.ik.original_upper[index], nominal[index] + radius)),
        ]
    if wrist_range is not None:
        # Release only this explicit planning posture. Native bounds and all
        # proposed arm/finger, wrist/board and path checks remain in force.
        hand.ik.posture_constraints.pop(hand.ik.names.index("Left Wrist"))
        hand.ik.constrain_reference_range("Left Wrist", *wrist_range)
    if restored and "tool_offset_m" in restored.controller:
        offset = np.asarray(restored.controller["tool_offset_m"], float)
        if (
            offset.shape != (3,)
            or not np.isfinite(offset).all()
            or not np.allclose(offset, hand.OFFSET, rtol=0, atol=1e-9)
        ):
            raise ValueError("Checkpoint tool calibration does not match native finger geometry")
        hand.OFFSET = offset.copy()
        hand.ik.offset = offset.copy()
    if restored and "active_rotation_world" in restored.controller:
        hand.active_rotation = np.asarray(restored.controller["active_rotation_world"], float)
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
    if restored and "motion_bounds" in restored.controller:
        lower = np.asarray(restored.controller["motion_bounds"]["lower_m"], float)
        upper = np.asarray(restored.controller["motion_bounds"]["upper_m"], float)
    bounds_reason = "captured insertion motion corridor"
    if args.front_face_sweep:
        # A separate, opt-in supported recovery workspace. Restrict the lateral
        # sweep to the measured carton Y envelope with 6 mm edge margins. Every nominal
        # and compensated proposal still needs the original geometry/path gates;
        # this box alone never certifies collision clearance or manipulation.
        active_body = model.body(active).id
        extent = np.abs(data.xmat[active_body].reshape(3, 3)) @ product["half_size"]
        lower[1] = data.xpos[active_body, 1] - extent[1] + 0.006
        upper[1] = data.xpos[active_body, 1] + extent[1] - 0.006
        bounds_reason = "supported front-face recovery; measured carton Y span minus 6 mm edges"
    policy_settings = dict(restored.controller.get("push_admittance", {})) if restored else {}
    policy_settings.setdefault("normal_target_N", original["controller"]["push_force_target"])
    if args.normal_target is not None:
        policy_settings["normal_target_N"] = args.normal_target
    configuration = PushAdmittance(**policy_settings)
    if reset_created:
        save_checkpoint(
            args.output / "fixture-checkpoint",
            model,
            data,
            scene=manifest,
            controller=dict(
                restored.controller,
                planning_wrist_range=wrist_range,
                push_admittance=asdict(configuration),
                motion_bounds=dict(lower_m=lower.tolist(), upper_m=upper.tolist()),
                mode_flags=dict(sweep_side=args.sweep_side, front_face_sweep=args.front_face_sweep),
            ),
            context=dict(
                source_summary=original,
                fixture=fixture,
                parent_checkpoint=restored.manifest["hashes"]["state.npz"],
            ),
        )
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
        restored_checkpoint=None if restored is None else str(restored.directory),
        reset_fixture=fixture,
        reset_created=reset_created,
        planning_wrist_range=wrist_range,
        motion_bounds=dict(lower_m=lower.tolist(), upper_m=upper.tolist(), reason=bounds_reason),
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
        if restored and not reset_created and restored.context.get("contact_monitor"):
            monitor.restore_state(restored.context["contact_monitor"], data)
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
        if (
            dwell is not None
            and restored
            and not reset_created
            and restored.context.get("backstop_dwell")
        ):
            dwell.restore_state(restored.context["backstop_dwell"], float(data.time))
        executed = []
        tracking_samples = []
        actor_failure = None
        active_body = model.body(active).id
        start_x = float(data.xpos[active_body, 0])
        start_position = data.xpos[active_body].copy()
        contact_trace = []

        def reserve_reached():
            rotation = data.xmat[active_body].reshape(3, 3)
            yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
            return args.squaring_reserve_deg is None or abs(yaw) <= args.squaring_reserve_deg

        yaw_integral_lever = (
            float(restored.controller.get("yaw_integral_lever_m", 0)) if restored else 0.0
        )
        for cycle in range(0 if args.withdraw_only else args.replans):
            lateral_goal = lateral_measured = None
            objective = "advance"
            if cycle or restored is not None and (args.square_feedback or args.sweep_side):
                expected = integration_state(model, data)
                reading = suite.sample(data)
                rotation = data.xmat[hand.palm].reshape(3, 3)
                wrench = np.r_[
                    rotation @ reading["contact_wrench"][0, :3],
                    rotation @ reading["contact_wrench"][0, 3:],
                ]
                if args.sweep_side:
                    sign = -1 if args.sweep_side == "left" else 1
                    lateral_goal = float(np.clip(origin[1] + sign * 0.012, lower[1], upper[1]))
                    lateral_measured = float(hand.position()[1])
                    objective = (
                        "square_backstop"
                        if monitor.last.get("carton_pair_N", 0) >= 0.4
                        else f"lateral_{args.sweep_side}"
                    )
                elif (
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
            latest_nominal = nominal.copy()
            for step_index, control in enumerate(chosen["controls"][:prefix_steps]):
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                executed.append(control.copy())
                trace = chosen["metadata"].get("nominal_references")
                if trace:
                    tick = step_index // control_steps
                    previous = nominal if tick == 0 else np.asarray(trace[tick - 1])
                    fraction = (step_index % control_steps + 1) / control_steps
                    latest_nominal = previous + (np.asarray(trace[tick]) - previous) * fraction
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
                active_rotation = data.xmat[active_body].reshape(3, 3)
                contact_trace.append(
                    [
                        float(data.time),
                        *data.xpos[active_body].tolist(),
                        float(np.degrees(np.arctan2(active_rotation[1, 0], active_rotation[0, 0]))),
                        monitor.last.get("hand_N", 0),
                        monitor.last.get("back_N", 0),
                        monitor.last.get("carton_pair_N", 0),
                        monitor.last.get("support_N", 0),
                    ]
                )
                if actor_failure:
                    break
                if trace:
                    predicted, _ = hand.ik.forward(latest_nominal, data.qpos.copy())
                    tracking_samples.append(
                        [float(data.time), float(np.linalg.norm(hand.position() - predicted))]
                    )
                if np.any(hand.position() < lower) or np.any(hand.position() > upper):
                    actor_failure = "measured_tool_left_corridor"
                    break
                if (
                    dwell is not None
                    and dwell.update(data.time, monitor.last, data.xmat[active_body].reshape(3, 3))
                    and reserve_reached()
                ):
                    break
            record.update(
                executed_steps=min(prefix_steps, len(executed) - cycle * prefix_steps),
                after_x_m=float(data.xpos[active_body, 0]),
                last_sample=monitor.last.copy(),
            )
            nominal = latest_nominal.copy()
            hand.command = nominal.copy()
            contact_finished = (
                dwell.verified and reserve_reached()
                if dwell is not None
                else monitor.last.get("back_N", 0) >= hand.config.contact_threshold
            )
            if actor_failure or contact_finished:
                break
        if dwell is not None and not dwell.verified and actor_failure is None:
            actor_failure = "backstop_dwell_decision_budget_exhausted"
        if (
            args.squaring_reserve_deg is not None
            and not reserve_reached()
            and actor_failure is None
        ):
            actor_failure = "controller_squaring_reserve_budget_exhausted"
        withdrawal = None
        if args.withdraw and actor_failure is None and dwell is not None and dwell.verified:
            withdrawal, withdrawal_controls, withdrawal_trace = supported_withdrawal(
                hand, nominal, monitor, dwell
            )
            executed.extend(withdrawal_controls)
            contact_trace.extend(withdrawal_trace.tolist())
            nominal = np.asarray(withdrawal["terminal_nominal"])
            actor_failure = withdrawal["failure"]
        report["continuation"] = dict(
            kind="declared_reset_fixture_probe"
            if reset_created
            else "retained_checkpoint_continuation_not_new_episode",
            replans=records,
            scheduled_replans=args.replans,
            executed_physics_steps=len(executed),
            continuation_execution_admitted=bool(executed),
            simulated_seconds=len(executed) * model.opt.timestep,
            progress_minus_x_m=start_x - float(data.xpos[active_body, 0]),
            displacement_xyz_m=(data.xpos[active_body] - start_position).tolist(),
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
            terminal_nominal_subset=dict(zip([*hand.names, "Lift"], nominal.tolist(), strict=True)),
            withdrawal=withdrawal,
        )
        np.savez_compressed(
            args.output / "continuation.npz",
            controls=np.asarray(executed),
            terminal_state=integration_state(model, data),
            tracking_samples=np.asarray(tracking_samples),
            contact_trace=np.asarray(contact_trace),
            contact_trace_columns=np.array(
                [
                    "time_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "yaw_deg",
                    "hand_N",
                    "back_N",
                    "max_carton_pair_N",
                    "support_N",
                ]
            ),
        )
        report["worker_wall_seconds"] = time.monotonic() - started
        report["forecast_generation_source_state_unchanged"] = True
        report["source_state_unchanged"] = bool(
            np.array_equal(initial, integration_state(model, data))
        )
        report["source_checkpoint_file_unchanged"] = True
        if args.save_checkpoint:
            write(args.output / "summary.json", report)
            checkpoint_path = args.output / "terminal-checkpoint"
            controller_state = dict(
                kind="retained_finger_withdrawal" if withdrawal else "retained_finger_push",
                settings=original["controller"],
                pusher_settings=choices,
                nominal_subset=report["continuation"]["terminal_nominal_subset"],
                active_rotation_world=hand.active_rotation.tolist(),
                tool_offset_m=hand.OFFSET.tolist(),
                motion_bounds=dict(lower_m=lower.tolist(), upper_m=upper.tolist()),
                yaw_integral_lever_m=yaw_integral_lever,
                control_dt_s=hand.env.dt,
                applied_controls=data.ctrl.tolist(),
                mode_flags=dict(
                    require_dwell=args.require_dwell,
                    square_feedback=args.square_feedback,
                    sweep_side=args.sweep_side,
                    front_face_sweep=args.front_face_sweep,
                    withdraw=args.withdraw,
                    squaring_reserve_deg=args.squaring_reserve_deg,
                ),
                planning_wrist_range=wrist_range,
                push_admittance=asdict(configuration),
            )
            xml_path = (restored.directory if restored else source) / "scene.xml"
            try:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    data,
                    scene=manifest,
                    controller=controller_state,
                    context=dict(
                        source_summary=original,
                        source_continuation=str(args.output),
                        parent_checkpoint=None
                        if restored is None
                        else restored.manifest["hashes"]["state.npz"],
                        original_outcome=original.get("failure"),
                        fixture=fixture,
                        contact_monitor=monitor.state_dict(),
                        backstop_dwell=None if dwell is None or withdrawal else dwell.report(),
                        completed_backstop_dwell=dwell.report() if withdrawal else None,
                        control_phase_step=len(executed) % control_steps,
                    ),
                    scene_xml=xml_path.read_text() if xml_path.exists() else None,
                    validation_controls=np.tile(data.ctrl, (20, 1)),
                )
                report["checkpoint"] = str(checkpoint_path)
            except (OSError, ValueError, RuntimeError) as error:
                report["checkpoint_error"] = str(error)
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
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--episode", type=Path)
    inputs.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=0.4)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--generation-seconds", type=float, default=10)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--execute-prefix", type=float, default=0)
    parser.add_argument("--require-dwell", action="store_true")
    parser.add_argument("--square-feedback", action="store_true")
    parser.add_argument("--reset-neighbor", choices=("left", "right"))
    parser.add_argument("--sweep-side", choices=("left", "right"))
    parser.add_argument("--normal-target", type=float)
    parser.add_argument("--wrist-freedom-deg", type=float, default=0)
    parser.add_argument("--front-face-sweep", action="store_true")
    parser.add_argument("--withdraw", action="store_true")
    parser.add_argument("--withdraw-only", action="store_true")
    parser.add_argument("--squaring-reserve-deg", type=float)
    parser.add_argument("--replans", type=int, default=1)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.withdraw = args.withdraw or args.withdraw_only
    if not (
        1 <= args.candidates <= 16
        and 1 <= args.shortlist <= args.candidates
        and 0 < args.wall_seconds <= 180
        and 0 < args.generation_seconds <= 30
        and 0 <= args.execute_prefix <= args.horizon
        and 1 <= args.replans <= 80
        and (not args.require_dwell or args.execute_prefix > 0)
        and (not args.square_feedback or args.require_dwell)
        and (not args.save_checkpoint or args.execute_prefix > 0)
        and (not args.reset_neighbor or args.checkpoint)
        and (not args.sweep_side or args.require_dwell)
        and (args.normal_target is None or 0 < args.normal_target < 25)
        and 0 <= args.wrist_freedom_deg <= 20
        and (
            not args.front_face_sweep
            or (args.sweep_side or args.square_feedback)
            and args.require_dwell
        )
        and (not args.withdraw or args.require_dwell)
        and (not args.withdraw_only or args.checkpoint)
        and (
            args.squaring_reserve_deg is None
            or (
                0 < args.squaring_reserve_deg <= 0.15
                and args.require_dwell
                and not args.withdraw_only
            )
        )
    ):
        parser.error("Keep candidate, shortlist and compute budgets bounded")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.episode is None and args.checkpoint is None:
        parser.error("--episode or --checkpoint is required")
    root = Path(__file__).resolve().parents[1]
    from abstract_stack.source_store import freeze_source

    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    if args.checkpoint:
        from abstract_stack.checkpoint_manifest import inspect_checkpoint

        checkpoint_directory, _ = inspect_checkpoint(args.checkpoint)
        write(
            inputs / "checkpoint_ref.json",
            dict(
                path=str(checkpoint_directory.resolve()),
                sha256=hashlib.sha256(
                    (checkpoint_directory / "manifest.json").read_bytes()
                ).hexdigest(),
            ),
        )
    else:
        for src, name in (
            (args.episode / "summary.json", "summary.json"),
            (args.episode / "initial-scene/manifest.json", "manifest.json"),
            (args.episode / "initial-scene/scene.xml", "scene.xml"),
            (args.episode / "rollout_terminal.npz", "rollout_terminal.npz"),
        ):
            shutil.copy2(src, inputs / name)
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/probe_compliant_lookahead.py",
            "scripts/run_frozen_replanning.py",
            "uv.lock",
            "pyproject.toml",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    snapshot = frozen.runtime_root
    write(
        args.output / "manifest.json",
        dict(
            source_revision=frozen.git["revision"],
            source_content_id=frozen.content_id,
            versions={n: importlib.metadata.version(n) for n in ("mujoco", "numpy")},
            settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            sha256={
                "source_ref.json": hashlib.sha256(
                    (args.output / "source_ref.json").read_bytes()
                ).hexdigest(),
                **{
                    str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in inputs.rglob("*")
                    if p.is_file()
                },
            },
        ),
    )
    command = [
        sys.executable,
        str(snapshot / "scripts/probe_compliant_lookahead.py"),
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
    if args.front_face_sweep:
        command.append("--front-face-sweep")
    if args.withdraw:
        command.append("--withdraw")
    if args.withdraw_only:
        command.append("--withdraw-only")
    if args.squaring_reserve_deg is not None:
        command += ["--squaring-reserve-deg", str(args.squaring_reserve_deg)]
    for flag, value in (
        ("--reset-neighbor", args.reset_neighbor),
        ("--sweep-side", args.sweep_side),
        ("--normal-target", args.normal_target),
        ("--wrist-freedom-deg", args.wrist_freedom_deg),
    ):
        if value is not None:
            command += [flag, str(value)]
    if args.checkpoint:
        command += ["--checkpoint", str(inputs / "checkpoint")]
    if args.save_checkpoint:
        command.append("--save-checkpoint")
    write(
        args.output / "process.json",
        dict(returncode=None, verification_status="unknown", fresh_task_credit=0),
    )
    supervisor = runpy.run_path(str(snapshot / "scripts/run_frozen_replanning.py"))["supervise"]
    result = supervisor(
        command,
        cwd=root,
        environment=dict(
            os.environ,
            PYTHONPATH=str(snapshot / "src"),
            PYTHONFAULTHANDLER="1",
            PYTHONDONTWRITEBYTECODE="1",
            ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
            ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
        ),
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
