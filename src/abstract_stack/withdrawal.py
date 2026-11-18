"""Guarded actuator-only withdrawal from a verified supported square contact."""

from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.ik import rotation_error


def supported_withdrawal(hand, nominal, monitor, dwell):
    """Unload toward 6 mm, clear 40 mm/+20 mm, measure released squared support.

    Keep native finger commands and initial applied-minus-nominal servo offsets.
    Replan every 20 ms, check nominal/applied geometry after final projection,
    and guard every 2 ms physics step. Return failures with partial controls;
    never write poses or claim a pick/fresh stocking episode.
    """
    model, data = hand.model, hand.data
    nominal = np.array(nominal, float, copy=True)
    monitor.begin_withdrawal(data, dwell)
    interval = int(round(hand.env.dt / model.opt.timestep))
    if interval < 1 or not np.isclose(interval * model.opt.timestep, hand.env.dt):
        raise ValueError("Withdrawal needs whole native control intervals")
    aids = np.r_[hand.aids, model.actuator("Lift").id]
    rates = np.r_[np.full(len(hand.names), hand.config.max_joint_speed), 0.08]
    applied = data.ctrl.copy()
    offsets = applied[aids] - nominal
    hand.command = nominal.copy()
    origin, rotation = hand.ik.forward(nominal)
    hand.active_rotation = rotation.copy()
    measured_origin = hand.position().copy()
    applied_origin, _ = hand.ik.forward(applied[aids])
    tolerance = hand.config.ik_position_tolerance
    lower = origin - tolerance
    upper = origin + [0.04 + tolerance, tolerance, 0.02 + tolerance]
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
    source_time = float(data.time)
    active = monitor.active
    active_origin = data.xpos[active].copy()
    source_nominal = nominal.copy()
    release_held = 0.0
    max_tracking = 0.0
    joint_limits, rate_limits = set(), set()
    native_projection_count = rate_projection_count = 0
    max_ik_position = max_ik_orientation = max_applied_step = 0.0

    def record_step():
        nonlocal failure, max_tracking, release_held
        failure = monitor.sample(data, elapsed=model.opt.timestep)
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
        if np.any(hand.position() < measured_lower) or np.any(hand.position() > measured_upper):
            failure = "withdrawal_measured_tool_left_corridor"
        point, _ = hand.ik.forward(hand.command, data.qpos.copy())
        max_tracking = max(max_tracking, float(np.linalg.norm(hand.position() - point)))
        joint = int(model.body_jntadr[active])
        address = int(model.jnt_dofadr[joint])
        velocity = data.qvel[address : address + 6]
        released = (
            monitor.last.get("robot_product_N", 0) <= 0.2
            and monitor.last.get("support_N", 0) > 0.2
            and abs(yaw) <= 0.15
            and r[2, 2] >= np.cos(np.radians(5))
            and np.linalg.norm(velocity[:3]) < 0.04
            and np.linalg.norm(velocity[3:]) < 0.2
        )
        release_held = release_held + model.opt.timestep if released else 0.0

    for stage, displacement, speed, seconds in (
        ("unload", np.array([0.006, 0, 0]), 0.004, 3.0),
        ("retreat", np.array([0.04, 0, 0.02]), 0.01, 7.0),
    ):
        goal = measured_origin + displacement
        reached = False
        for tick in range(int(seconds / hand.env.dt)):
            hand.command = nominal.copy()
            point, _ = hand.ik.forward(nominal)
            # Close the loop on actual tool motion. Frozen contact feedforward
            # becomes inaccurate as load disappears; nominal FK alone cannot
            # establish that the physical hand reached its clearance target.
            remaining = goal - hand.position()
            if stage == "unload" and release_held + 1e-12 >= 0.12:
                # The unload phase ends on measured released support, not on
                # an arbitrary extra translation after pressure is already gone.
                reached = True
                break
            if np.linalg.norm(remaining) <= 0.0005:
                reached = True
                break
            target = point + remaining * min(1, speed * hand.env.dt / np.linalg.norm(remaining))
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
            prediction = bounded_contact_prediction(hand, proposal, target, hand.env.dt, rates)
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
            wanted[aids] = prediction.q + offsets
            requested = wanted.copy()
            limited = model.actuator_ctrllimited != 0
            wanted[limited] = np.clip(
                wanted[limited],
                model.actuator_ctrlrange[limited, 0],
                model.actuator_ctrlrange[limited, 1],
            )
            bounded = applied.copy()
            bounded[aids] += np.clip(
                wanted[aids] - applied[aids], -rates * hand.env.dt, rates * hand.env.dt
            )
            native_projection_count += int(np.any(wanted != requested))
            rate_projection_count += int(np.any(bounded != wanted))
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
                SimpleNamespace(q=bounded[aids]), 0, monitor.scene.active, targets=finger_targets
            )
            if (
                collision
                or np.any(applied_point < applied_lower)
                or np.any(applied_point > applied_upper)
                or np.linalg.norm(applied_point - previous_point) > linear_bound
                or np.linalg.norm(rotation_error(previous_rotation, applied_rotation))
                > angular_bound
            ):
                plans.append(
                    dict(
                        stage=stage,
                        tick=tick,
                        collision=collision,
                        applied_point_m=applied_point.tolist(),
                    )
                )
                failure = "withdrawal_final_applied_geometry_or_motion"
                break
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
    if not failure:
        for _ in range(int(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            controls.append(data.ctrl.copy())
            record_step()
            if failure:
                break
    measured_delta = hand.position() - measured_origin
    verified = (
        failure is None
        and release_held + 1e-12 >= 0.12
        and measured_delta[0] >= 0.035
        and measured_delta[2] >= 0.015
        and np.max(np.abs(data.xpos[active] - active_origin)) <= 0.005
    )
    if not verified and failure is None:
        failure = "withdrawal_released_squared_support_or_clearance_not_verified"
    return (
        dict(
            kind="supported_withdrawal_continuation",
            failure=failure,
            release_retreat_verified=bool(verified),
            fresh_task_credit=0,
            task_success=False,
            simulated_seconds=float(data.time) - source_time,
            plans=plans,
            release_held_seconds=release_held,
            measured_tool_displacement_m=measured_delta.tolist(),
            measured_base_nominal_fk_tracking_peak_m=max_tracking,
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
        ),
        np.asarray(controls).reshape(-1, model.nu),
        np.asarray(traces).reshape(-1, 9),
    )
