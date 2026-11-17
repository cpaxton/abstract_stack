"""Bounded contact-feedback proposals for independent native push forecasts.

This is a privileged teacher-side primitive. A frozen measured wrench shapes
open-loop future proposals; forces are checked again by the rollout evaluator.
No forecast is a grasp, a fresh episode, or authorization to execute live.
"""

import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.ik import interpolate_rotation, rotation_error


@dataclass(frozen=True)
class PushAdmittance:
    normal_target_N: float = 2.5
    normal_gain_m_Ns: float = 0.008
    tangent_gain_m_Ns: float = 0.01
    vertical_gain_m_Ns: float = 0.012
    forward_speed_m_s: float = 0.02
    reverse_speed_m_s: float = 0.01
    lateral_speed_m_s: float = 0.01
    vertical_speed_m_s: float = 0.005
    downward_unload_N: float = 0.6
    yaw_gain_rad_Nms: float = 0.4
    angular_speed_rad_s: float = 0.3

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Admittance settings must be finite and positive")
        if self.normal_target_N >= 25:
            raise ValueError("Normal target must remain below the unchanged force guard")

    def velocity(self, wrench_on_hand_world, *, lateral_bias=0.0, normal_scale=1.0):
        """World velocity; +X reaction resists the desired -X insertion."""
        force = np.asarray(wrench_on_hand_world, float)
        if force.shape != (6,) or not np.isfinite(force).all():
            raise ValueError("Expected finite world wrench ON the hand, force then torque")
        if (
            not np.isfinite(lateral_bias)
            or not np.isfinite(normal_scale)
            or not 0 < normal_scale <= 1
        ):
            raise ValueError("Invalid bounded admittance alternative")
        vx = -self.normal_gain_m_Ns * (self.normal_target_N - force[0]) * normal_scale
        if vx < 0:
            # Positive hand Fz means downward load ON the supported carton.
            vx *= float(np.clip(1 - force[2] / self.downward_unload_N, 0, 1))
        return np.array(
            [
                np.clip(vx, -self.forward_speed_m_s, self.reverse_speed_m_s),
                np.clip(
                    self.tangent_gain_m_Ns * force[1] + lateral_bias,
                    -self.lateral_speed_m_s,
                    self.lateral_speed_m_s,
                ),
                np.clip(
                    self.vertical_gain_m_Ns * force[2],
                    -self.vertical_speed_m_s,
                    self.vertical_speed_m_s,
                ),
            ]
        )


def generate_push_candidates(
    hand,
    active,
    *,
    nominal,
    wrench_on_hand_world,
    lower_xyz,
    upper_xyz,
    horizon_s=0.4,
    max_candidates=16,
    wall_seconds=10.0,
    admittance=None,
    lateral_goal_m=None,
    lateral_measurement_m=None,
):
    """Generate checked arm/lift trajectories with held native finger commands.

    Caller supplies an independent planning hand and known uncompensated nominal
    references. Applied servo offsets are frozen from the initial native state,
    not mislabeled as nominal actions. A later guarded physics forecast must
    test actual retention/forces. The original hand command/orientation is
    restored and native model/data arrays are never edited here.
    """
    config = admittance or PushAdmittance()
    nominal = np.array(nominal, float, copy=True)
    lower_xyz, upper_xyz = np.asarray(lower_xyz, float), np.asarray(upper_xyz, float)
    model, data = hand.model, hand.data
    if (
        nominal.shape != hand.command.shape
        or not np.isfinite(nominal).all()
        or np.any(nominal < hand.ik.lower)
        or np.any(nominal > hand.ik.upper)
        or lower_xyz.shape != (3,)
        or upper_xyz.shape != (3,)
        or not np.isfinite(lower_xyz).all()
        or not np.isfinite(upper_xyz).all()
        or np.any(lower_xyz >= upper_xyz)
        or not np.isfinite(horizon_s)
        or not hand.env.dt <= horizon_s <= 1
        or type(max_candidates) is not int
        or not 1 <= max_candidates <= 16
        or not np.isfinite(wall_seconds)
        or not 0 < wall_seconds <= 30
        or lateral_goal_m is not None
        and (not np.isfinite(lateral_goal_m) or not lower_xyz[1] <= lateral_goal_m <= upper_xyz[1])
        or lateral_measurement_m is not None
        and (lateral_goal_m is None or not np.isfinite(lateral_measurement_m))
    ):
        raise ValueError("Invalid bounded push forecast configuration")
    # Validate wrench even when only the held-target baseline is requested.
    config.velocity(wrench_on_hand_world)
    steps = int(round(horizon_s / model.opt.timestep))
    interval = int(round(hand.env.dt / model.opt.timestep))
    if steps < 1 or interval < 1 or not np.isclose(interval * model.opt.timestep, hand.env.dt):
        raise ValueError("Control interval must contain an integer number of physics steps")
    if steps % interval or not np.isclose(steps * model.opt.timestep, horizon_s, rtol=0, atol=1e-9):
        raise ValueError("Horizon must contain complete control intervals")
    aids = np.r_[hand.aids, model.actuator("Lift").id]
    base = data.ctrl.copy()
    rates = np.zeros(model.nu)
    nominal_rates = np.r_[np.full(len(hand.names), hand.config.max_joint_speed), 0.08]
    rates[aids] = nominal_rates
    hand_bodies = {hand.palm}
    for body in range(hand.palm + 1, model.nbody):
        if int(model.body_parentid[body]) in hand_bodies:
            hand_bodies.add(body)
    finger_targets = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != 0 or aid in aids:
            continue
        joint = int(model.actuator_trnid[aid, 0])
        if joint >= 0 and int(model.jnt_bodyid[joint]) in hand_bodies:
            finger_targets[model.joint(joint).name] = float(base[aid])
    old_command, old_rotation = hand.command.copy(), hand.active_rotation.copy()
    old_prediction = hand.last_prediction
    old_carry, old_slide = hand.ik.carry, hand.ik.slide
    started = time.monotonic()
    origin, rotation = hand.ik.forward(nominal)
    applied_origin, _ = hand.ik.forward(base[aids])
    # Compensated servo targets are virtual attractors, not desired physical
    # joint poses. Bound their increments in an envelope translated by the
    # captured initial compensation; retain the physical nominal corridor.
    applied_lower = lower_xyz + applied_origin - origin
    applied_upper = upper_xyz + applied_origin - origin
    baseline = dict(
        name="held_native_targets",
        controls=np.tile(base, (steps, 1)),
        metadata=dict(kind="held_applied_targets_baseline", fresh_task_credit=0),
    )
    candidates = [baseline]
    rejected = []
    alternatives = [
        (1, 0.004, 0),
        (0.5, 0.004, 0),
        (1, 0, 0),
        (0.5, 0, 0),
        (1, -0.004, 0),
        (0.5, -0.004, 0),
        (1, 0.004, -5),
        (1, 0.004, 5),
        (0.5, 0.004, -5),
        (0.5, 0.004, 5),
        (1, 0, -5),
        (1, 0, 5),
        (0.5, 0, -5),
        (0.5, 0, 5),
        (1, -0.004, 5),
    ][: max_candidates - 1]
    try:
        for index, (normal_scale, lateral_bias, tilt_deg) in enumerate(alternatives, 1):
            if time.monotonic() - started >= wall_seconds:
                rejected.extend(
                    dict(name=f"proposal_{i}", failure="generation_wall_budget")
                    for i in range(index, len(alternatives) + 1)
                )
                break
            hand.command = nominal.copy()
            velocity = config.velocity(
                wrench_on_hand_world, lateral_bias=lateral_bias, normal_scale=normal_scale
            )
            if lateral_goal_m is not None:
                velocity[1] = np.clip(
                    2
                    * (
                        lateral_goal_m
                        - (origin[1] if lateral_measurement_m is None else lateral_measurement_m)
                    )
                    + config.tangent_gain_m_Ns * np.asarray(wrench_on_hand_world)[1]
                    + lateral_bias,
                    -config.lateral_speed_m_s,
                    config.lateral_speed_m_s,
                )
            angle = np.radians(tilt_deg)
            turn = np.array(
                [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
            )
            desired_rotation = turn @ old_rotation
            yaw = (
                float(
                    np.clip(
                        config.yaw_gain_rad_Nms * np.asarray(wrench_on_hand_world)[5],
                        -config.angular_speed_rad_s,
                        config.angular_speed_rad_s,
                    )
                )
                * horizon_s
            )
            yaw_rotation = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            desired_rotation = yaw_rotation @ desired_rotation
            angle = float(np.linalg.norm(rotation_error(rotation, desired_rotation)))
            if angle > config.angular_speed_rad_s * horizon_s:
                desired_rotation = interpolate_rotation(
                    rotation, desired_rotation, config.angular_speed_rad_s * horizon_s / angle
                )
            goal = np.clip(origin + velocity * horizon_s, lower_xyz, upper_xyz)
            hand.active_rotation = desired_rotation
            endpoint = hand.predict(
                goal,
                allowed_object=active,
                alternatives=False,
                carrying=False,
                sliding=True,
                path_mode="cartesian",
            )
            report = dict(
                name=f"proposal_{index}",
                normal_scale=normal_scale,
                lateral_bias_m_s=lateral_bias,
                tilt_delta_deg=tilt_deg,
                velocity_world_m_s=velocity.tolist(),
                privileged_lateral_goal_m=lateral_goal_m,
                privileged_lateral_measurement_m=lateral_measurement_m,
                endpoint=endpoint.report(),
                compensation="frozen initial applied-minus-nominal offsets",
            )
            if not endpoint.feasible:
                rejected.append(dict(report, failure="endpoint_or_path_prefilter"))
                continue
            applied = base.copy()
            controls = []
            nominal_trace = []
            failure = None
            for tick in range(steps // interval):
                if time.monotonic() - started >= wall_seconds:
                    failure = "generation_wall_budget"
                    break
                fraction = (tick + 1) * interval / steps
                target = np.clip(origin + velocity * (tick + 1) * hand.env.dt, lower_xyz, upper_xyz)
                proposal_rotation = interpolate_rotation(rotation, desired_rotation, fraction)
                # The physical baseline/reference and contact-mode cache remain
                # unchanged throughout this independent open-loop forecast.
                proposal = replace(endpoint, requested_rotation=proposal_rotation)
                prediction = bounded_contact_prediction(
                    hand, proposal, target, hand.env.dt, nominal_rates
                )
                previous_point, previous_rotation = hand.ik.forward(
                    hand.command, endpoint.reference_qpos
                )
                linear_bound = (
                    np.linalg.norm(
                        [
                            config.forward_speed_m_s,
                            config.lateral_speed_m_s,
                            config.vertical_speed_m_s,
                        ]
                    )
                    * hand.env.dt
                )
                if (
                    not prediction.feasible
                    or np.any(prediction.position < lower_xyz)
                    or np.any(prediction.position > upper_xyz)
                    or np.linalg.norm(prediction.position - previous_point) > linear_bound + 1e-6
                    or np.linalg.norm(rotation_error(previous_rotation, prediction.rotation))
                    > config.angular_speed_rad_s * hand.env.dt + 1e-6
                ):
                    failure = "bounded_nominal_geometry_or_motion"
                    report["nominal_rejection"] = dict(
                        control_tick=tick,
                        prediction=prediction.report(),
                        lower_m=lower_xyz.tolist(),
                        upper_m=upper_xyz.tolist(),
                        linear_step_m=float(np.linalg.norm(prediction.position - previous_point)),
                        max_linear_step_m=float(linear_bound),
                        angular_step_rad=float(
                            np.linalg.norm(rotation_error(previous_rotation, prediction.rotation))
                        ),
                        max_angular_step_rad=float(config.angular_speed_rad_s * hand.env.dt),
                    )
                    break
                wanted = base.copy()
                wanted[aids] = base[aids] + prediction.q - nominal
                limited = model.actuator_ctrllimited != 0
                wanted[limited] = np.clip(
                    wanted[limited],
                    model.actuator_ctrlrange[limited, 0],
                    model.actuator_ctrlrange[limited, 1],
                )
                # Apply final native rate/range projection, then check exactly
                # that compensated arm AND the held applied finger posture.
                bounded = applied + np.clip(
                    wanted - applied, -rates * hand.env.dt, rates * hand.env.dt
                )
                old_point, old_applied_rotation = hand.ik.forward(
                    applied[aids], endpoint.reference_qpos
                )
                applied_point, applied_rotation = hand.ik.forward(
                    bounded[aids], endpoint.reference_qpos
                )
                if (
                    np.any(applied_point < applied_lower)
                    or np.any(applied_point > applied_upper)
                    or np.linalg.norm(applied_point - old_point) > linear_bound + 1e-6
                    or np.linalg.norm(rotation_error(old_applied_rotation, applied_rotation))
                    > config.angular_speed_rad_s * hand.env.dt + 1e-6
                ):
                    failure = "projected_applied_reference_outside_corridor"
                    report["applied_rejection"] = dict(
                        position_m=applied_point.tolist(),
                        lower_m=applied_lower.tolist(),
                        upper_m=applied_upper.tolist(),
                        linear_step_m=float(np.linalg.norm(applied_point - old_point)),
                        max_linear_step_m=float(linear_bound),
                        angular_step_rad=float(
                            np.linalg.norm(rotation_error(old_applied_rotation, applied_rotation))
                        ),
                        max_angular_step_rad=float(config.angular_speed_rad_s * hand.env.dt),
                    )
                    break
                collision = hand.check_hand_step(
                    SimpleNamespace(q=bounded[aids]), 0, active, targets=finger_targets
                )
                if collision:
                    failure = "coupled_applied_arm_finger_geometry"
                    report["collision"] = collision
                    break
                for part in range(1, interval + 1):
                    controls.append(applied + (bounded - applied) * (part / interval))
                applied = bounded
                hand.command = prediction.q.copy()
                nominal_trace.append(prediction.q.tolist())
            if failure:
                rejected.append(dict(report, failure=failure))
            else:
                report.update(
                    nominal_references=nominal_trace,
                    nominal_names=[*hand.names, "Lift"],
                    fresh_task_credit=0,
                )
                candidates.append(
                    dict(name=report["name"], controls=np.asarray(controls), metadata=report)
                )
    finally:
        hand.command, hand.active_rotation, hand.last_prediction = (
            old_command,
            old_rotation,
            old_prediction,
        )
        hand.ik.carry, hand.ik.slide = old_carry, old_slide
    return (
        candidates,
        dict(
            kind="privileged_open_loop_contact_forecasts",
            nominal_origin_m=origin.tolist(),
            applied_virtual_origin_m=applied_origin.tolist(),
            applied_envelope_translation_m=(applied_origin - origin).tolist(),
            generation_seconds=time.monotonic() - started,
            scheduled_candidates=1 + len(alternatives),
            generated=len(candidates),
            rejected=rejected,
            fresh_task_credit=0,
        ),
        rates,
    )
