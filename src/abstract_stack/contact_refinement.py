"""Short guarded native forecasts for a live supported-carton release goal."""

import mujoco
import numpy as np

from abstract_stack.compliant_lookahead import PushAdmittance, generate_push_candidates
from abstract_stack.contact_rollout import evaluate_screened_batch, integration_state
from abstract_stack.sim_sensors import ContactSensorSuite


def refine_supported_contact(script, monitor, dwell, *, yaw_target_deg=0.0, normal_target_N=None):
    """Execute at most twenty checked 200 ms prefixes toward the release goal."""
    from abstract_stack.contact_execution import sync_tracking_state
    from abstract_stack.interactions import InteractionFailure

    env, hand, config = script.env, script.controller, script.config
    model, data = env.model, env.data
    if (
        not np.isfinite(yaw_target_deg)
        or abs(yaw_target_deg) + config.pre_release_yaw_goal_deg > 0.15
    ):
        raise ValueError("Release reference must remain inside the unchanged .15 degree gate")
    if (
        normal_target_N is not None
        and not 0.3 <= normal_target_N <= config.insertion_back_force_target
    ):
        raise ValueError("Release preload must stay between .3 N and the insertion target")
    script._stage("contact_refine")
    event = script.events[-1]
    event.update(
        maximum_replans=20,
        horizon_s=0.2,
        controller_yaw_goal_deg=config.pre_release_yaw_goal_deg,
        controller_yaw_target_deg=float(yaw_target_deg),
        unchanged_contact_yaw_criterion_deg=0.15,
        decisions=[],
    )
    level = env.config.levels[script.product.level]
    origin = hand.ik.forward(hand.command)[0]
    lower = np.array([origin[0] - 0.03, script.product.target[1] - 0.02, level + 0.023])
    upper = np.array([origin[0] + 0.03, script.product.target[1] + 0.02, level + 0.053])
    # Nominal IK and measured tracking use distinct, fixed corridors. Native
    # compensation can put the measured tool outside the nominal reference box
    # already at entry; do not confuse that offset with unbounded motion.
    measured_shift = hand.position() - origin
    measured_lower, measured_upper = lower + measured_shift, upper + measured_shift
    event.update(
        fixed_tool_lower_world_m=lower.tolist(),
        fixed_tool_upper_world_m=upper.tolist(),
        fixed_measured_tool_lower_world_m=measured_lower.tolist(),
        fixed_measured_tool_upper_world_m=measured_upper.tolist(),
        initial_measured_minus_nominal_m=measured_shift.tolist(),
        normal_target_N=config.insertion_back_force_target,
        yaw_lever_gain_m_rad=config.insertion_yaw_lever_gain,
        native_timestep_s=float(model.opt.timestep),
        control_interval_s=env.dt,
    )
    sensor = ContactSensorSuite.mabel(model)
    integral = 0.0
    interval = int(round(env.dt / model.opt.timestep))
    force_goal = config.insertion_back_force_target if normal_target_N is None else normal_target_N
    event["release_preload_target_N"] = normal_target_N
    event["normal_control"] = (
        "backstop target plus bounded measured tangential/support reaction"
        if normal_target_N is None
        else "explicit total hand preload target"
    )
    paired = bool(getattr(hand, "fingers", ()))
    registration_required = config.paired_finger_insertion and env.config.row_free_space is not None
    row_count = sum(p.level == script.product.level for p in env.products)
    lateral_tolerance = (
        min(0.0005, env.config.row_free_space / (row_count + 1) / 4)
        if registration_required
        else None
    )
    lateral_reference = float(getattr(script, "lateral_goal_y", script.product.target[1]))
    event.update(
        controller_lateral_goal_world_y_m=lateral_reference,
        controller_lateral_goal_tolerance_m=lateral_tolerance,
    )
    initial_error = lateral_reference - float(data.xpos[monitor.active, 1])
    lateral_bias = float(
        np.clip(hand.position()[1] - data.xpos[monitor.active, 1] - initial_error, -0.006, 0.006)
    )
    plane = hand.active_rotation @ hand.ROTATION.T
    plane_bias = float(np.clip(np.arctan2(plane[1, 0], plane[0, 0]), -np.radians(3), np.radians(3)))
    event["lateral_feedback_reference"] = (
        "paired tool midpoint" if paired else "single-finger center of pressure"
    )
    for cycle in range(20):
        expected = integration_state(model, data)
        reading = sensor.sample(data)
        rotation = data.xmat[hand.palm].reshape(3, 3)
        wrench = np.r_[
            rotation @ reading["contact_wrench"][0, :3], rotation @ reading["contact_wrench"][0, 3:]
        ]
        reaction = float(np.clip(wrench[0] - monitor.last.get("back_N", 0), 0, 3))
        hand_force_goal = force_goal + reaction if normal_target_N is None else force_goal
        if hand_force_goal >= config.max_force:
            raise InteractionFailure("contact_refine: compensated setpoint exceeds force guard")
        policy = PushAdmittance(normal_target_N=hand_force_goal)
        body = data.xmat[monitor.active].reshape(3, 3)
        yaw = float(np.arctan2(body[1, 0], body[0, 0]))
        error = yaw - np.radians(yaw_target_deg)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, monitor.active, velocity, 0)
        integral -= 0.03 * error * 0.2
        lever = float(
            np.clip(
                -config.insertion_yaw_lever_gain * error + integral - 0.01 * velocity[2],
                -0.014,
                0.014,
            )
        )
        integral = lever + config.insertion_yaw_lever_gain * error + 0.01 * velocity[2]
        goal = float(np.clip(data.xpos[monitor.active, 1] + lever, lower[1], upper[1]))
        contact = script.last_forces.get("hand_contact_position_world_m")
        measured = float(contact[1] if contact is not None else hand.position()[1])
        if paired:
            integral = 0.0
            center = float(data.xpos[monitor.active, 1])
            error_y = lateral_reference - center
            lateral_bias = float(np.clip(lateral_bias + 2 * error_y * 0.2, -0.006, 0.006))
            goal = float(np.clip(center + error_y + lateral_bias, center - 0.006, center + 0.006))
            goal = float(np.clip(goal, lower[1], upper[1]))
            measured = float(hand.position()[1])
            # A two-pad pusher controls yaw with its contact plane, not by
            # moving its midpoint beyond the carton face. Integrate measured
            # carton yaw with an explicit three-degree reference budget.
            plane_goal = np.radians(yaw_target_deg) - 2 * error
            plane_error = hand.contact_plane_yaw() - plane_goal
            plane_bias = float(
                np.clip(plane_bias - 2 * plane_error * 0.2, -np.radians(3), np.radians(3))
            )
            c, s = np.cos(plane_bias), np.sin(plane_bias)
            hand.active_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ hand.ROTATION
        nominal = hand.command.copy()
        finger_goals = None
        finger_report = None
        finger_tracker = getattr(hand, "finger_load_compensation", None)
        if finger_tracker is not None:
            finger_controls, finger_report = finger_tracker.propose(data.ctrl, script.product.name)
            finger_goals = {
                name: float(finger_controls[state["actuator"]])
                for name, state in finger_tracker.joints.items()
            }
        candidates, generation, rates = generate_push_candidates(
            hand,
            script.product.name,
            nominal=nominal,
            wrench_on_hand_world=wrench,
            lower_xyz=lower,
            upper_xyz=upper,
            horizon_s=0.2,
            max_candidates=10,
            wall_seconds=10,
            admittance=policy,
            lateral_goal_m=goal,
            lateral_measurement_m=measured,
            project_reference_bounds=True,
            finger_control_targets=finger_goals,
        )
        if normal_target_N is not None and monitor.last.get("hand_N", 0) > 1.3 * force_goal:
            candidates = [c for c in candidates if c["name"] != "held_native_targets"]
            generation["held_baseline_filtered_for_preload_taper"] = True
        if not candidates:
            raise InteractionFailure("contact_refine: no checked preload-taper candidate")
        lateral_error = lateral_reference - float(data.xpos[monitor.active, 1])
        objective = "square_backstop"
        if monitor.last.get("back_N", 0) < config.contact_threshold:
            objective = "advance"
        elif paired and registration_required and abs(lateral_error) > lateral_tolerance:
            objective = "lateral_left" if lateral_error < 0 else "lateral_right"
        evaluated = evaluate_screened_batch(
            model,
            expected,
            candidates,
            monitor.scene,
            rates=rates,
            workers=4,
            shortlist=min(4, len(candidates)),
            objective=objective,
            yaw_target_rad=float(np.radians(yaw_target_deg)),
        )
        event["decisions"].append(
            dict(
                cycle=cycle,
                generation=generation,
                evaluation=evaluated,
                yaw_integral_lever_m=integral,
                measured_support_reaction_estimate_N=reaction,
                effective_hand_normal_target_N=hand_force_goal,
                selected_finger_tracking=finger_report,
                measured_lateral_error_m=lateral_error,
                lateral_integral_reference_bias_m=lateral_bias,
                paired_plane_reference_bias_deg=float(np.degrees(plane_bias)) if paired else None,
            )
        )
        selected = evaluated["selected_name"]
        if selected is None or not np.array_equal(expected, integration_state(model, data)):
            raise InteractionFailure("contact_refine: no fresh guarded candidate")
        chosen = next(c for c in candidates if c["name"] == selected)
        event["decisions"][-1]["executed_candidate_metadata"] = chosen["metadata"]
        trace = chosen["metadata"].get("nominal_references")
        for index, control in enumerate(chosen["controls"]):
            if trace:
                tick = index // interval
                previous = nominal if tick == 0 else np.asarray(trace[tick - 1])
                fraction = (index % interval + 1) / interval
                hand.command = previous + (np.asarray(trace[tick]) - previous) * fraction
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            env.update_task_state(model.opt.timestep)
            sync_tracking_state(script)
            script._sample(env)
            failure = monitor.sample(data, elapsed=model.opt.timestep)
            if failure:
                raise InteractionFailure(f"contact_refine: {failure}")
            if np.any(hand.position() < measured_lower) or np.any(hand.position() > measured_upper):
                raise InteractionFailure("contact_refine: measured tool left fixed corridor")
            dwell.update(data.time, monitor.last, data.xmat[monitor.active].reshape(3, 3))
            body = data.xmat[monitor.active].reshape(3, 3)
            yaw = float(np.degrees(np.arctan2(body[1, 0], body[0, 0])))
            if data.time >= env.config.time_limit:
                raise InteractionFailure("contact_refine: episode time limit")
            force_ready = (
                normal_target_N is None or monitor.last.get("hand_N", 0) <= 1.3 * force_goal
            )
            if (
                dwell.verified
                and force_ready
                and abs(yaw - yaw_target_deg) <= config.pre_release_yaw_goal_deg
                and (
                    not registration_required
                    or abs(lateral_reference - data.xpos[monitor.active, 1]) <= lateral_tolerance
                )
            ):
                event.update(native_dwell=dwell.report(), verified_pre_release_goal=True)
                return
    event["native_dwell"] = dwell.report()
    if (
        dwell.verified
        and (
            not registration_required
            or abs(lateral_reference - data.xpos[monitor.active, 1])
            <= max(0.001, 4 * lateral_tolerance)
        )
        and (normal_target_N is None or monitor.last.get("hand_N", 0) <= 1.3 * force_goal)
    ):
        # The tighter controller reference is not the contact success gate.
        # A bounded square staging result may proceed only to a complete
        # native release forecast, followed by fresh live release admission.
        event.update(
            verified_pre_release_goal=False,
            decision="bounded_square_staging_then_full_release_replan",
            fresh_task_credit=0,
        )
        return
    raise InteractionFailure("contact_refine: bounded release-goal search exhausted")
