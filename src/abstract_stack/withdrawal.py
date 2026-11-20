"""Guarded actuator-only withdrawal from a verified supported square contact."""

from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.ik import rotation_error
from abstract_stack.withdrawal_feedback import WithdrawalTracker, compensation_translation


def supported_withdrawal(
    hand,
    nominal,
    monitor,
    dwell,
    *,
    clear_distance_m=0.04,
    clear_height_m=0.02,
    clear_speed_m_s=0.01,
    resume_withdrawal=False,
    on_physics_step=None,
    feedback=None,
    contact_neighbors=(),
    release_yaw_limit_deg=0.15,
    coarse_loaded_entry=False,
    unload_only=False,
):
    """Unload toward 6 mm, clear 40 mm/+20 mm, measure released squared support.

    Keep native finger commands. Default to frozen servo offsets; explicit
    feedback refreshes robot gravity, wrist damping and measured contact torque.
    Replan every 20 ms, check nominal/applied geometry after final projection,
    and guard every 2 ms physics step. Return failures with partial controls;
    never write poses or claim a pick/fresh stocking episode.
    """
    model, data = hand.model, hand.data
    neighbor_geoms = {f"{name}_geom" for name in contact_neighbors}
    neighbor_origins = {name: data.body(name).xpos.copy() for name in contact_neighbors}

    def admit_neighbor_contact(error):
        # This exception is for distal fingertips brushing named, stocked
        # cartons during exit. Arm, shelf, self and unlisted contacts still
        # use the normal collision checks.
        first, second = error["geoms"]
        return error["distance_m"] >= -0.003 and any(
            carton in neighbor_geoms and finger.startswith("left_") and "_ip:" in finger
            for carton, finger in ((first, second), (second, first))
        )

    admission = {"admit_contact": admit_neighbor_contact} if neighbor_geoms else {}
    if (
        not np.isfinite([clear_distance_m, clear_height_m, clear_speed_m_s]).all()
        or not 0.04 <= clear_distance_m <= 0.25
        or not 0.02 <= clear_height_m <= 0.08
        or not 0 < clear_speed_m_s <= 0.025
        or not 0.15 <= release_yaw_limit_deg <= 1.0
    ):
        raise ValueError("Keep clearance goals and physical motion bounded")
    if unload_only and resume_withdrawal:
        raise ValueError("An unload forecast requires a loaded entry")
    nominal = np.array(nominal, float, copy=True)
    if resume_withdrawal:
        if monitor.phase != "withdrawal" or monitor.withdrawal_started_s is None:
            raise ValueError("Extended clearance needs a previously admitted withdrawal phase")
        failure = monitor.sample(data, elapsed=0)
        if failure or monitor.last["robot_product_N"] > 0.2 or monitor.last["support_N"] <= 0.2:
            raise ValueError("Extended clearance requires fresh unloaded surface support")
    elif coarse_loaded_entry:
        monitor.begin_coarse_loaded_withdrawal(data)
    else:
        monitor.begin_withdrawal(data, dwell)
    interval = int(round(hand.env.dt / model.opt.timestep))
    if interval < 1 or not np.isclose(interval * model.opt.timestep, hand.env.dt):
        raise ValueError("Withdrawal needs whole native control intervals")
    aids = np.r_[hand.aids, model.actuator("Lift").id]
    rates = np.r_[np.full(len(hand.names), hand.config.max_joint_speed), 0.08]
    applied = data.ctrl.copy()
    offsets = applied[aids] - nominal
    tracker = (
        WithdrawalTracker(hand, nominal, monitor.scene.active, feedback)
        if feedback is not None
        else None
    )
    finger_tracker = (
        getattr(hand, "finger_load_compensation", None) if tracker is not None else None
    )
    selected_finger_ids = (
        []
        if finger_tracker is None
        else [state["actuator"] for state in finger_tracker.joints.values()]
    )
    hand.command = nominal.copy()
    origin, rotation = hand.ik.forward(nominal)
    plane_bias = 0.0
    hand.active_rotation = rotation.copy()
    measured_origin = hand.position().copy()
    applied_origin, _ = hand.ik.forward(applied[aids])
    tolerance = hand.config.ik_position_tolerance
    lower = origin - tolerance
    upper = origin + [clear_distance_m + tolerance, tolerance, clear_height_m + tolerance]
    applied_lower, applied_upper = lower + applied_origin - origin, upper + applied_origin - origin
    measured_lower, measured_upper = (
        lower + measured_origin - origin,
        upper + measured_origin - origin,
    )
    finger_targets = {}
    for aid in range(model.nu):
        if aid in aids or model.actuator_trntype[aid] != 0:
            continue
        joint = int(model.actuator_trnid[aid, 0])
        if joint >= 0 and model.joint(joint).name in hand.finger_targets(0):
            finger_targets[model.joint(joint).name] = float(applied[aid])
    controls, traces, plans = [], [], []
    failure = None
    applied_motion_bound_only = False
    source_time = float(data.time)
    active = monitor.active
    active_origin = data.xpos[active].copy()
    source_nominal = nominal.copy()
    release_held = 0.0
    strict_release_held = 0.0
    unloaded_held = 0.0
    max_tracking = 0.0
    joint_limits, rate_limits = set(), set()
    native_projection_count = rate_projection_count = 0
    max_ik_position = max_ik_orientation = max_applied_step = 0.0
    feedback_trace = []

    def record_step():
        nonlocal failure, max_tracking, release_held, unloaded_held, strict_release_held
        failure = monitor.sample(data, elapsed=model.opt.timestep)
        if failure is None and any(
            np.linalg.norm(data.body(name).xpos - origin) > 0.002
            for name, origin in neighbor_origins.items()
        ):
            failure = "withdrawal_stocked_neighbor_displacement"
        if on_physics_step is not None:
            on_physics_step(hand.env)
        r = data.xmat[active].reshape(3, 3)
        yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
        traces.append(
            [
                float(data.time),
                *data.xpos[active].tolist(),
                yaw,
                monitor.last.get("hand_N", 0),
                monitor.last.get("back_N", 0),
                monitor.last.get("carton_pair_N", 0),
                monitor.last.get("support_N", 0),
            ]
        )
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.any(data.warning.number)
        ):
            failure = "native_state_or_warning"
        if failure is None and (
            np.any(hand.position() < measured_lower) or np.any(hand.position() > measured_upper)
        ):
            failure = "withdrawal_measured_tool_left_corridor"
        point, _ = hand.ik.forward(hand.command, data.qpos.copy())
        max_tracking = max(max_tracking, float(np.linalg.norm(hand.position() - point)))
        joint = int(model.body_jntadr[active])
        address = int(model.jnt_dofadr[joint])
        velocity = data.qvel[address : address + 6]
        released = (
            monitor.last.get("robot_product_N", 0) <= 0.2
            and monitor.last.get("support_N", 0) > 0.2
            and abs(yaw) <= release_yaw_limit_deg
            and r[2, 2] >= np.cos(np.radians(5))
            and np.linalg.norm(velocity[:3]) < 0.04
            and np.linalg.norm(velocity[3:]) < 0.2
        )
        release_held = release_held + model.opt.timestep if released else 0.0
        strict_release_held = (
            strict_release_held + model.opt.timestep if released and abs(yaw) <= 0.15 else 0.0
        )
        if tracker is not None and tracker.config.yaw_feedback:
            unloaded_held = (
                unloaded_held + model.opt.timestep
                if released
                and monitor.last.get("robot_product_N", 0) <= tracker.config.unload_force_goal_N
                else 0.0
            )

    unload_seconds = (
        tracker.config.unload_seconds
        if tracker is not None and tracker.config.yaw_feedback
        else 3.0
    )
    unload_speed = (
        tracker.config.unload_speed_m_s
        if tracker is not None and tracker.config.yaw_feedback
        else 0.004
    )
    stages = (
        []
        if resume_withdrawal
        else [("unload", np.array([0.006, 0, 0]), unload_speed, unload_seconds)]
    )
    displacement = np.array([clear_distance_m, 0, clear_height_m])
    straight_first = (
        tracker is not None
        and tracker.config.yaw_feedback
        and tracker.config.straight_retreat_first
    )
    if not unload_only:
        stages.append(
            (
                "retreat",
                np.array([clear_distance_m, 0, 0]) if straight_first else displacement,
                clear_speed_m_s,
                max(7.0, 1.5 * float(np.linalg.norm(displacement)) / clear_speed_m_s),
            )
        )
    if straight_first and not unload_only:
        stages.append(
            (
                "retreat_lift",
                displacement,
                clear_speed_m_s,
                max(4.0, 1.5 * clear_height_m / clear_speed_m_s),
            )
        )
    for stage, displacement, speed, seconds in stages:
        if tracker is not None:
            # Reserve part of the unchanged applied Cartesian speed bound for
            # refreshed load compensation: nominal FK and compensated FK have
            # different Jacobians even when their joint offsets are held.
            speed = min(
                speed, tracker.config.nominal_servo_speed_fraction * tracker.config.servo_speed_m_s
            )
        goal = measured_origin + displacement
        if stage.startswith("retreat") and tracker is not None and tracker.config.yaw_feedback:
            # Keep the admitted lateral unload offset until the hand is clear;
            # do not sweep back sideways across a barely unloaded carton.
            goal[1] = hand.position()[1]
        reached = False
        for tick in range(int(seconds / hand.env.dt)):
            hand.command = nominal.copy()
            point, _ = hand.ik.forward(nominal)
            # Close the loop on actual tool motion. Frozen contact feedforward
            # becomes inaccurate as load disappears; nominal FK alone cannot
            # establish that the physical hand reached its clearance target.
            remaining = goal - hand.position()
            unload_ready = (
                unloaded_held + 1e-12 >= tracker.config.unload_dwell_s
                if tracker is not None and tracker.config.yaw_feedback
                else release_held + 1e-12 >= 0.12
            )
            if stage == "unload" and unload_ready:
                # The unload phase ends on measured released support, not on
                # an arbitrary extra translation after pressure is already gone.
                reached = True
                break
            force_unload = stage == "unload" and tracker is not None and tracker.config.yaw_feedback
            if np.linalg.norm(remaining) <= 0.0005 and not force_unload:
                reached = True
                break
            if stage == "unload" and tracker is not None and tracker.config.yaw_feedback:
                # Yaw control owns lateral motion during unloading. The old
                # lateral origin must not consume the normal separation speed.
                remaining[1] = 0.0
                # Release is a force goal. Never reverse into the carton when
                # the arbitrary 6 mm waypoint is reached before pressure clears.
                remaining[0] = max(
                    remaining[0],
                    0.005
                    * max(
                        0.0,
                        monitor.last.get("robot_product_N", 0) - tracker.config.unload_force_goal_N,
                    ),
                )
            delta = remaining * min(1, speed * hand.env.dt / max(np.linalg.norm(remaining), 1e-12))
            yaw_feedback = None
            if stage == "unload" and tracker is not None and tracker.config.yaw_feedback:
                contact = tracker.contact_center()
                r = data.xmat[active].reshape(3, 3)
                yaw = float(np.arctan2(r[1, 0], r[0, 0]))
                yaw_error = yaw - np.radians(tracker.config.yaw_reference_deg)
                velocity = np.empty(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, active, velocity, 0)
                desired_lever = float(
                    np.clip(
                        -tracker.config.yaw_lever_gain_m_rad * yaw_error
                        - tracker.config.yaw_lever_damping_m_s_rad * velocity[2],
                        -0.006,
                        0.006,
                    )
                )
                measured_lever = (
                    None if contact is None else float(contact[1] - data.xpos[active, 1])
                )
                contact_wrench = tracker.contact_wrench_on_product()
                desired_moment = float(
                    np.clip(
                        -tracker.config.yaw_moment_stiffness_Nm_rad * yaw_error
                        - tracker.config.yaw_moment_damping_Nms_rad * velocity[2],
                        -0.01,
                        0.01,
                    )
                )
                delta[1] = (
                    0.0
                    if contact is None
                    or monitor.last.get("robot_product_N", 0) <= tracker.config.unload_force_goal_N
                    else float(
                        np.clip(
                            (
                                2 * desired_lever
                                + tracker.config.yaw_moment_admittance_m_Nms
                                * (desired_moment - contact_wrench[5])
                            )
                            * hand.env.dt,
                            -speed * hand.env.dt,
                            speed * hand.env.dt,
                        )
                    )
                )
                plane_yaw = None
                if tracker.config.paired_plane_feedback:
                    if not hasattr(hand, "contact_plane_yaw"):
                        raise ValueError("Paired withdrawal needs measured native pad references")
                    plane_yaw = hand.contact_plane_yaw()
                    if monitor.last.get("robot_product_N", 0) > tracker.config.unload_force_goal_N:
                        plane_goal = np.radians(tracker.config.yaw_reference_deg) - 2 * yaw_error
                        plane_bias = float(
                            np.clip(
                                plane_bias
                                - np.clip(2 * (plane_yaw - plane_goal), -0.3, 0.3) * hand.env.dt,
                                -np.radians(3),
                                np.radians(3),
                            )
                        )
                        c, s = np.cos(plane_bias), np.sin(plane_bias)
                        hand.active_rotation = (
                            np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation
                        )
                    delta[1] = 0.0
                delta *= min(1.0, speed * hand.env.dt / max(np.linalg.norm(delta), 1e-12))
                yaw_feedback = dict(
                    yaw_deg=float(np.degrees(yaw)),
                    yaw_error_deg=float(np.degrees(yaw_error)),
                    yaw_reference_deg=tracker.config.yaw_reference_deg,
                    desired_contact_lever_y_m=desired_lever,
                    measured_contact_lever_y_m=measured_lever,
                    nominal_step_m=delta.tolist(),
                    contact_wrench_on_product_world_N_Nm=contact_wrench.tolist(),
                    desired_yaw_moment_Nm=desired_moment,
                    paired_plane_yaw_deg=None
                    if plane_yaw is None
                    else float(np.degrees(plane_yaw)),
                    paired_plane_bias_deg=float(np.degrees(plane_bias)),
                )
            target = point + delta
            proposal = hand.predict(
                target,
                allowed_object=monitor.scene.active,
                alternatives=False,
                carrying=False,
                sliding=False,
                path_mode="cartesian",
            )
            if not proposal.feasible:
                plans.append(dict(stage=stage, tick=tick, prediction=proposal.report()))
                failure = "withdrawal_endpoint_or_path_prefilter"
                break
            linear_bound = speed * hand.env.dt + 1e-6
            angular_bound = 0.3 * hand.env.dt + 1e-6
            motion_options = (
                dict(max_translation_step_m=linear_bound, max_rotation_step_rad=angular_bound)
                if tracker is not None and tracker.config.yaw_feedback
                else {}
            )
            prediction = bounded_contact_prediction(
                hand, proposal, target, hand.env.dt, rates, **motion_options
            )
            diagnostics = prediction.bounded_feedback or {}
            joint_limits.update(diagnostics.get("active_joint_limits", []))
            rate_limits.update(diagnostics.get("active_rate_limits", []))
            max_ik_position = max(max_ik_position, prediction.position_error_m)
            max_ik_orientation = max(
                max_ik_orientation, float(np.degrees(prediction.orientation_error_rad))
            )
            old_point, old_rotation = hand.ik.forward(nominal, proposal.reference_qpos)
            linear_bound = speed * hand.env.dt + 1e-6
            angular_bound = 0.3 * hand.env.dt + 1e-6
            if (
                not prediction.feasible
                or np.any(prediction.position < lower)
                or np.any(prediction.position > upper)
                or np.linalg.norm(prediction.position - old_point) > linear_bound
                or np.linalg.norm(rotation_error(old_rotation, prediction.rotation)) > angular_bound
            ):
                plans.append(dict(stage=stage, tick=tick, prediction=prediction.report()))
                failure = "withdrawal_bounded_nominal_geometry_or_motion"
                break
            wanted = applied.copy()
            if tracker is not None:
                # Recompute offsets from the last admitted pair, never from a
                # rejected tracker proposal. Preserve the nominal prediction.
                offsets = applied[aids] - nominal
                refreshed = tracker.propose(prediction.q)
                wanted[selected_finger_ids] = refreshed[selected_finger_ids]
            wanted[aids] = prediction.q + offsets
            requested = wanted.copy()
            limited = model.actuator_ctrllimited != 0
            wanted[limited] = np.clip(
                wanted[limited],
                model.actuator_ctrlrange[limited, 0],
                model.actuator_ctrlrange[limited, 1],
            )
            servo_bound = (
                tracker.config.servo_speed_m_s * hand.env.dt + 1e-6
                if tracker is not None
                else linear_bound
            )
            alphas = [2.0**-i for i in range(11)] + [0.0] if tracker is not None else [0.0]
            for alpha in alphas:
                candidate = wanted.copy()
                if tracker is not None:
                    candidate[aids] += alpha * (refreshed[aids] - wanted[aids])
                    candidate[selected_finger_ids] = applied[selected_finger_ids] + alpha * (
                        wanted[selected_finger_ids] - applied[selected_finger_ids]
                    )
                candidate_requested = candidate.copy()
                candidate[limited] = np.clip(
                    candidate[limited],
                    model.actuator_ctrlrange[limited, 0],
                    model.actuator_ctrlrange[limited, 1],
                )
                bounded = applied.copy()
                bounded[selected_finger_ids] = candidate[selected_finger_ids]
                bounded[aids] += np.clip(
                    candidate[aids] - applied[aids], -rates * hand.env.dt, rates * hand.env.dt
                )
                applied_point, applied_rotation = hand.ik.forward(
                    bounded[aids], proposal.reference_qpos
                )
                previous_point, previous_rotation = hand.ik.forward(
                    applied[aids], proposal.reference_qpos
                )
                if tracker is not None:
                    shift = compensation_translation(
                        hand.ik,
                        source_nominal,
                        prediction.q,
                        bounded[aids],
                        proposal.reference_qpos,
                    )
                    applied_lower, applied_upper = lower + shift, upper + shift
                if (
                    np.all(applied_point >= applied_lower)
                    and np.all(applied_point <= applied_upper)
                    and np.linalg.norm(applied_point - previous_point) <= servo_bound
                    and np.linalg.norm(rotation_error(previous_rotation, applied_rotation))
                    <= angular_bound
                ):
                    collision = hand.check_hand_step(
                        SimpleNamespace(q=bounded[aids]),
                        0,
                        monitor.scene.active,
                        targets={
                            name: float(bounded[model.actuator(name).id]) for name in finger_targets
                        },
                        **admission,
                    )
                    if not collision:
                        break
            native_projection_count += int(
                np.any(wanted != requested) or np.any(candidate != candidate_requested)
            )
            rate_projection_count += int(np.any(bounded != candidate))
            applied_point, applied_rotation = hand.ik.forward(
                bounded[aids], proposal.reference_qpos
            )
            previous_point, previous_rotation = hand.ik.forward(
                applied[aids], proposal.reference_qpos
            )
            max_applied_step = max(
                max_applied_step, float(np.linalg.norm(applied_point - previous_point))
            )
            collision = hand.check_hand_step(
                SimpleNamespace(q=bounded[aids]),
                0,
                monitor.scene.active,
                targets={name: float(bounded[model.actuator(name).id]) for name in finger_targets},
                **admission,
            )
            outside_applied_corridor = np.any(applied_point < applied_lower) or np.any(
                applied_point > applied_upper
            )
            translation_step = float(np.linalg.norm(applied_point - previous_point))
            rotation_step = float(
                np.linalg.norm(rotation_error(previous_rotation, applied_rotation))
            )
            exceeds_motion_bound = translation_step > servo_bound or rotation_step > angular_bound
            if collision or outside_applied_corridor or exceeds_motion_bound:
                applied_motion_bound_only = not collision and not outside_applied_corridor
                plans.append(
                    dict(
                        stage=stage,
                        tick=tick,
                        collision=collision,
                        applied_point_m=applied_point.tolist(),
                        applied_lower_m=applied_lower.tolist(),
                        applied_upper_m=applied_upper.tolist(),
                        applied_translation_step_m=translation_step,
                        maximum_translation_step_m=servo_bound,
                        applied_rotation_step_rad=rotation_step,
                        maximum_rotation_step_rad=angular_bound,
                    )
                )
                failure = "withdrawal_final_applied_geometry_or_motion"
                break
            if tracker is not None:
                feedback_trace.append(
                    dict(
                        time_s=float(data.time),
                        stage=stage,
                        compensation_fraction=alpha,
                        nominal_tool_step_m=float(np.linalg.norm(prediction.position - old_point)),
                        applied_virtual_tool_step_m=float(
                            np.linalg.norm(applied_point - previous_point)
                        ),
                        applied_virtual_lower_m=applied_lower.tolist(),
                        applied_virtual_upper_m=applied_upper.tolist(),
                        admitted_nominal_endpoint=dict(
                            zip((*hand.names, "Lift"), prediction.q.tolist(), strict=True)
                        ),
                        admitted_servo_endpoint=bounded[aids].tolist(),
                        unload_yaw_feedback=yaw_feedback,
                        **tracker.last_report,
                    )
                )
            previous_nominal = nominal.copy()
            for part in range(1, interval + 1):
                control = applied + (bounded - applied) * (part / interval)
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                controls.append(control.copy())
                nominal = previous_nominal + (prediction.q - previous_nominal) * (part / interval)
                hand.command = nominal.copy()
                record_step()
                if failure:
                    break
            applied = data.ctrl.copy()
            if tracker is not None:
                tracker.acknowledge(nominal, applied)
                feedback_trace[-1].update(
                    executed_nominal=tracker.last_report["accepted_nominal"],
                    executed_servo_targets=tracker.last_report["accepted_servo_targets"],
                )
            if failure:
                break
        plans.append(
            dict(
                stage=stage,
                stage_complete=reached,
                completion_mode="measured_released_squared_support"
                if stage == "unload"
                else "measured_tool_goal",
                measured_goal_reached=bool(np.linalg.norm(goal - hand.position()) <= 0.0005),
                measured_goal_m=goal.tolist(),
                measured_tool_m=hand.position().tolist(),
                forces=monitor.last.copy(),
            )
        )
        if failure:
            break
        if not reached:
            failure = f"withdrawal_{stage}_decision_budget_exhausted"
            break
        if stage == "unload" and monitor.last.get("robot_product_N", 0) > 0.2:
            failure = "withdrawal_pressure_not_unloaded"
            break
    if not failure and not unload_only:
        for _ in range(int(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            controls.append(data.ctrl.copy())
            record_step()
            if failure:
                break
    measured_delta = hand.position() - measured_origin
    terminal_nominal_point, _ = hand.ik.forward(nominal, data.qpos.copy())
    unloaded = (
        failure is None
        and release_held + 1e-12 >= 0.12
        and np.max(np.abs(data.xpos[active] - active_origin)) <= 0.005
    )
    verified = (
        unloaded
        and not unload_only
        and measured_delta[0] >= clear_distance_m - 0.005
        and measured_delta[2] >= clear_height_m - 0.005
    )
    if not (unloaded if unload_only else verified) and failure is None:
        failure = "withdrawal_released_squared_support_or_clearance_not_verified"
    return (
        dict(
            kind="supported_withdrawal_continuation",
            resumed_withdrawal=resume_withdrawal,
            coarse_loaded_entry=coarse_loaded_entry,
            settings=dict(
                clear_distance_m=clear_distance_m,
                clear_height_m=clear_height_m,
                clear_speed_m_s=clear_speed_m_s,
                feedback=None if tracker is None else vars(tracker.config),
                applied_envelope_frame=(
                    "entry_nominal_with_admitted_compensation_translation"
                    if tracker is not None
                    else "fixed_entry_compensation_translation"
                ),
            ),
            failure=failure,
            applied_motion_bound_only=bool(applied_motion_bound_only),
            unload_only=unload_only,
            unloaded_support_verified=bool(unloaded),
            release_retreat_verified=bool(verified),
            fresh_task_credit=0,
            task_success=False,
            simulated_seconds=float(data.time) - source_time,
            plans=plans,
            release_held_seconds=release_held,
            release_yaw_limit_deg=release_yaw_limit_deg,
            strict_release_yaw_verified=strict_release_held + 1e-12 >= 0.12,
            measured_tool_displacement_m=measured_delta.tolist(),
            measured_base_nominal_fk_tracking_peak_m=max_tracking,
            measured_base_nominal_fk_tracking_terminal_m=float(
                np.linalg.norm(hand.position() - terminal_nominal_point)
            ),
            initial_nominal=source_nominal.tolist(),
            terminal_nominal=nominal.tolist(),
            active_displacement_m=(data.xpos[active] - active_origin).tolist(),
            last_forces=monitor.last.copy(),
            peak_forces_N=monitor.peaks.copy(),
            incidents=monitor.incidents.copy(),
            limit_diagnostics=dict(
                active_joint_limits=sorted(joint_limits),
                active_rate_limits=sorted(rate_limits),
                native_range_projection_ticks=native_projection_count,
                rate_projection_ticks=rate_projection_count,
                planning_reference_ranges=hand.ik.reference_range_report(),
                maximum_nominal_ik_position_error_m=max_ik_position,
                maximum_nominal_ik_orientation_error_deg=max_ik_orientation,
                maximum_applied_virtual_tool_step_m=max_applied_step,
            ),
            motion_bounds=dict(
                nominal_lower_m=lower.tolist(),
                nominal_upper_m=upper.tolist(),
                measured_lower_m=measured_lower.tolist(),
                measured_upper_m=measured_upper.tolist(),
            ),
            feedback_trace=feedback_trace,
        ),
        np.asarray(controls).reshape(-1, model.nu),
        np.asarray(traces).reshape(-1, 9),
    )
