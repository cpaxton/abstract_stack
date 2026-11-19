"""Experimental sideways carton registration through native position servos.

This skill requires an exposed side face. It is not an insertion certificate:
first insert/reposition so the selected finger can reach the side, then replan
from the measured row after the push. No object pose is assigned by this module.
"""

import time
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from abstract_stack.dense_interactions import DenseStockingScript, FingerPusher
from abstract_stack.interactions import InteractionFailure
from abstract_stack.mabel_controller import MabelController

FRONT_SIDE_BASE_OFFSET_M = 0.093
PRESS_BASE_FOOTPRINT_RADIUS_M = 0.25


@dataclass(frozen=True)
class LateralPushConfig:
    speed_m_s: float = 0.01
    force_target_N: float = 1.5
    distance_limit_m: float = 0.025
    timeout_s: float = 8.0
    position_tolerance_m: float = 0.001
    contact_dwell_s: float = 0.12
    height_m: float = 0.028

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Lateral push settings must be finite and positive")
        if self.distance_limit_m > 0.05 or self.force_target_N >= 25:
            raise ValueError("Lateral push exceeds prototype distance/force bounds")

    def velocity(self, force_in_push_direction_N):
        """Admittance: advance unloaded, stop at setpoint, unload excessive force."""
        if not np.isfinite(force_in_push_direction_N):
            raise ValueError("Measured force must be finite")
        return self.speed_m_s * float(
            np.clip((self.force_target_N - force_in_push_direction_N) / self.force_target_N, -1, 1)
        )


class SideFingerPusher(FingerPusher):
    """Rotate the native selected-finger posture so its tip points along +/-Y."""

    def finger_targets(self, grip):
        targets = super().finger_targets(grip)
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            if finger != self.finger or getattr(self, "park_all_fingers", False):
                for joint in ("MCP", "PIP"):
                    targets[f"Left {finger} {joint}"] = self.folded_flexion
        return targets

    def __init__(self, env, config, *, direction=-1, yaw_degrees=90, folded_flexion=1.0, **kwargs):
        if not np.isfinite(folded_flexion) or not 1 <= folded_flexion <= 1.5:
            raise ValueError("Folded side-pusher references must stay within 1–1.5 rad")
        self.folded_flexion = float(folded_flexion)
        if direction not in (-1, 1):
            raise ValueError("Lateral direction must be -1 or +1")
        if not np.isfinite(yaw_degrees) or not 30 <= yaw_degrees <= 90:
            raise ValueError("Side approach yaw must be within 30–90 degrees")
        super().__init__(env, config, **kwargs)
        # Parent fingertip extends toward -X. Rz(direction * -pi/2)
        # maps that extension onto direction * Y, preserving native geometry.
        angle = -direction * np.radians(yaw_degrees)
        yaw = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        self.ROTATION = yaw @ self.ROTATION
        self.active_rotation = self.ROTATION.copy()
        self.direction = direction
        if config.ungrasped_insertion:
            index = self.ik.names.index("Left Wrist")
            self.ik.constrain_reference_range(
                "Left Wrist", self.ik.original_lower[index], self.ik.original_upper[index] - 0.35
            )
        self.last_target = self.position().copy()


class LateralContact:
    """Checked side contact for the native stocking controllers."""

    def _change_hand(self, kind, **kwargs):
        super()._change_hand(kind, **kwargs)
        if isinstance(self.controller, SideFingerPusher):
            # Preserve the existing pusher wrist reference-offset budget when
            # subclassing its geometry. Native gain and force limits stay fixed.
            previous = self.tracker
            self.tracker = MabelController(
                self.env.model,
                self.env.data,
                replace(previous.config, wrist_offset_limit=0.5),
            )
            self.tracker.nominal = previous.nominal.copy()
            self.tracker.applied = previous.applied.copy()

    def _cartesian_move(self, name, end, *, speed_m_s=0.025, allowed_object=None):
        """Execute a straight tool-space route with per-tick native-geometry checks."""
        from abstract_stack.grasping import timed_joint_path

        self._stage(name)
        hand = self.controller
        start = hand.position().copy()
        end = np.asarray(end)
        duration = max(1.0, float(np.linalg.norm(end - start)) / speed_m_s * 1.875)
        event = self.events[-1]
        event.update(
            path_mode="cartesian",
            requested_target_m=end.tolist(),
            planned_duration_s=duration,
            minimum_board_clearance_m=0.012,
        )
        prediction = hand.predict(
            end,
            alternatives=False,
            carrying=False,
            allowed_object=allowed_object,
            path_mode="cartesian",
        )
        event["ik_prediction"] = prediction.report()
        if not prediction.feasible:
            raise InteractionFailure(f"{name}: Cartesian route rejected before actuation")
        if prediction.joint_path is None:
            raise InteractionFailure(f"{name}: missing checked Cartesian waypoints")
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        path, knots, minimum = timed_joint_path(prediction.joint_path, rates)
        duration = max(duration, minimum * 1.05)
        event.update(planned_duration_s=duration, retained_waypoints=len(path))
        for k in range(max(1, int(np.ceil(duration / self.env.dt)))):
            t = min(1.0, (k + 1) * self.env.dt / duration)
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            q = np.array([np.interp(blend, knots, path[:, i]) for i in range(path.shape[1])])
            collisions = hand.check_hand_step(replace(prediction, q=q), 0, allowed_object)
            if collisions:
                event["coupled_rejection"] = collisions
                raise InteractionFailure(
                    f"{name}: coupled Cartesian step rejected before actuation"
                )
            action = hand.joint_targets(q, 0)
            hand.last_target = hand.ik.forward(hand.command, prediction.reference_qpos)[0]
            self._apply_tick(action)
        error = float(np.linalg.norm(hand.position() - end))
        event.update(end_time=float(self.env.data.time), measured_tracking_error_m=error)
        if error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: measured Cartesian tracking error {error:.3f}m")

    def push_laterally(
        self,
        *,
        direction,
        distance_m,
        settings=None,
        yaw_candidates=(45, 30, 60, 90),
        brace_wall=False,
        near_front_corner=False,
        stage_for_squaring=False,
        row_ready=None,
        front_contact_fraction=0.65,
        base_goal_xy=None,
    ):
        settings = settings or LateralPushConfig()
        if stage_for_squaring and (not near_front_corner or brace_wall):
            raise ValueError("Side staging requires the front-corner tool and a separate square")
        if row_ready is not None and (not callable(row_ready) or brace_wall or stage_for_squaring):
            raise ValueError("Row presses use their own measured stop before guarded unloading")
        if not np.isfinite(front_contact_fraction) or not 0 <= front_contact_fraction <= 0.65:
            raise ValueError(
                "Side contact must stay between face center and the checked front patch"
            )
        if base_goal_xy is not None:
            base_goal_xy = np.asarray(base_goal_xy, dtype=float)
            if base_goal_xy.shape != (2,) or not np.isfinite(base_goal_xy).all():
                raise ValueError("Lateral workspace goal must be a finite XY pair")
            if not self.config.ungrasped_insertion or not self.config.base_reposition:
                raise ValueError("An explicit lateral workspace goal requires base motion")
        if type(brace_wall) is not bool:
            raise ValueError("Wall bracing must be explicit")
        if not yaw_candidates or any(
            not np.isfinite(yaw) or not 30 <= yaw <= 90 for yaw in yaw_candidates
        ):
            raise ValueError("Side approach yaw candidates must stay within 30–90 degrees")
        if direction not in (-1, 1) or not np.isfinite(distance_m):
            raise ValueError("Expected +/-1 direction and finite distance")
        if not 0 < distance_m <= settings.distance_limit_m:
            raise ValueError("Requested lateral displacement is outside the bounded skill")
        if distance_m <= settings.position_tolerance_m:
            raise ValueError("Requested displacement must exceed its verification tolerance")
        if settings.force_target_N >= self.config.max_force:
            raise ValueError("Lateral force target must be below the unchanged abort guard")
        env = self.env
        body = env.data.body(self.product.name)
        start_body = body.xpos.copy()
        start_rotation = body.xmat.reshape(3, 3).copy()
        extent = np.abs(start_rotation) @ np.asarray(self.product.half_size)
        height = env.config.levels[self.product.level] + settings.height_m
        # Stay on the exposed side face and leave room for the selected pad
        # before the front edge. Stage the higher way-point outside the shelf.
        contact_x = start_body[0] + (
            front_contact_fraction * self.product.half_size[0] if near_front_corner else 0
        )
        precontact = np.array([contact_x, start_body[1] - direction * (extent[1] + 0.015), height])
        raised = precontact + ([0.09, 0, 0.10] if near_front_corner else [0, 0, 0.14])
        # Use the existing checked high-clearance staging motion before
        # changing palm orientation; a direct joint interpolation from the
        # parked hand can sweep through the table even with a reachable endpoint.
        staging = [-0.30, -0.20, 1.35]
        if near_front_corner:
            tool = self.controller.position().copy()
            if tool[0] < env.config.front_x + 0.04:
                tool[0] = env.config.front_x + 0.08
                self._cartesian_move("lateral_clear_entry", tool, allowed_object=self.product.name)
            self._restore_grasp_hand()
            base = env.data.body("mabel_chassis").xpos
            staging = [base[0] - 0.26, base[1] - 0.20, 1.35]
        self._move("lateral_stage", staging, carrying=False)
        self._stage("lateral_plan")
        candidates = []
        self.events[-1]["candidates"] = candidates
        selected = None
        # Native model is shared read-only; every candidate owns independent MjData.
        started = time.monotonic()
        if self.config.ungrasped_insertion and self.config.base_reposition:
            base = env.data.body("mabel_chassis").xpos[:2].copy()
            dy = float(
                precontact[1] + (FRONT_SIDE_BASE_OFFSET_M if near_front_corner else 0.12) - base[1]
            )
            dx = getattr(self, "side_base_dx", 0.0)
            workspace = (
                {} if base_goal_xy is None else dict(footprint_radius=PRESS_BASE_FOOTPRINT_RADIUS_M)
            )
            displacement = np.array([dx, dy]) if base_goal_xy is None else base_goal_xy - base
            if abs(displacement[1]) > 0.03 or displacement[0] != 0:
                if np.linalg.norm(displacement) > 0.25:
                    raise InteractionFailure("lateral_plan: workspace shift exceeds bounded search")
                self._move_base(
                    displacement,
                    reason="lateral_contact_workspace",
                    **workspace,
                )
                self._stage("lateral_plan")
                self.events[-1]["candidates"] = candidates
        wrists = (0.2, 0.0) if self.config.ungrasped_insertion else (0.2, 0.4, 0.0, 0.6)
        tilts = (0, 15, 30, 40) if near_front_corner else (20, 30, 40)
        raised_paths = ("joint",) if base_goal_xy is None else ("joint", "cartesian")
        for yaw, tilt, wrist, raised_path in product(yaw_candidates, tilts, wrists, raised_paths):
            if time.monotonic() - started > 45:
                break
            kwargs = dict(
                direction=direction,
                yaw_degrees=yaw,
                finger="Middle",
                tilt_degrees=tilt,
                wrist_posture=wrist,
                folded_flexion=1.5 if self.config.ungrasped_insertion else 1.0,
            )
            hand_config = (
                replace(self.config, ik_position_tolerance=0.002)
                if self.config.ungrasped_insertion
                else self.config
            )
            hand = SideFingerPusher(env.clone_for_planning(), hand_config, **kwargs)
            hand.command = self.controller.command.copy()
            candidate = {
                **kwargs,
                "waypoints_m": [raised.tolist(), precontact.tolist()],
                "raised_path": raised_path,
            }
            candidates.append(candidate)
            prediction = hand.predict(
                raised, alternatives=False, carrying=False, path_mode=raised_path
            )
            candidate["raised_prediction"] = prediction.report()
            if not prediction.feasible:
                continue
            self._planning_pose(hand, prediction)
            prediction = hand.predict(
                precontact, alternatives=False, carrying=False, path_mode="cartesian"
            )
            candidate["prediction"] = prediction.report()
            if prediction.feasible:
                self._planning_pose(hand, prediction)
                end = precontact + [0, direction * (distance_m + 0.02), 0]
                forward = hand.predict(
                    end,
                    alternatives=False,
                    carrying=False,
                    allowed_object=self.product.name,
                    path_mode="cartesian",
                )
                candidate["push_prediction"] = forward.report()
                if forward.feasible:
                    margins = self._clearance_margin(hand, forward, precontact, end)
                    candidate["clearance_m"] = margins
                    if margins["support"] < 0.012 or margins["sides"] < 0.004:
                        candidate["rejected"] = "tracking_clearance_margin"
                        continue
                    selected = kwargs
                    break
        self.events[-1]["planning_wall_seconds"] = time.monotonic() - started
        if selected is None:
            raise InteractionFailure("lateral_plan: no checked exposed-side approach; replan")
        self.events[-1]["selected"] = selected
        self._change_hand(SideFingerPusher, **selected)
        if self.config.ungrasped_insertion:
            self.controller.config = replace(self.config, ik_position_tolerance=0.002)
        self.events[-1]["controller_ik_position_tolerance_m"] = (
            self.controller.config.ik_position_tolerance
        )
        if raised_path == "cartesian":
            self._cartesian_move("lateral_orient_high", raised, speed_m_s=0.08)
        else:
            self._move(
                "lateral_orient_high",
                raised,
                carrying=False,
                orientation_alternatives=False,
            )
        self._cartesian_move("lateral_approach", precontact)
        self._stage("lateral_push")
        event = self.events[-1]
        target = self.controller.last_target.copy()
        origin = target.copy()
        desired_y = start_body[1] + direction * distance_m
        half_height = self.product.half_size[2]
        bottom_center_y = desired_y - half_height * start_rotation[1, 2]
        if near_front_corner:
            # The center can move several millimetres while a tall carton
            # rocks in place. Brake from the center of the supported face.
            event["desired_bottom_center_y_m"] = float(bottom_center_y)
        event.update(
            start_body_m=start_body.tolist(),
            desired_body_y_m=float(desired_y),
            requested_displacement_m=distance_m,
            direction=direction,
            contact_mode="exposed-side supported push, no rigid attachment",
            forecast="robot/finger kinematics only; no lateral object dynamics certificate",
            settings=vars(settings),
            row_capacity_stop=row_ready is not None,
            samples=[],
            yaw_controller_goal_deg=0.15 if self.config.paired_finger_insertion else None,
            x_reference_bound_m=0.015,
            brace_wall=brace_wall,
        )
        held = 0.0
        had_push_contact = False
        for _ in range(int(settings.timeout_s / env.dt)):
            force = np.asarray(self.last_forces.get("hand_force_world_N", [0, 0, 0]))
            body = env.data.body(self.product.name)
            if body.xmat.reshape(3, 3)[2, 2] < np.cos(np.radians(15)):
                raise InteractionFailure("lateral_push: carton tilt; unload/replan required")
            if self.observer is not None and self.observer.incidents:
                raise InteractionFailure("lateral_push: product tip/drop incident; replan")
            remaining = direction * (desired_y - body.xpos[1])
            if near_front_corner:
                supported_center_y = body.xpos[1] - half_height * body.xmat[5]
                bottom_error = bottom_center_y - supported_center_y
                remaining = direction * bottom_error
                event["measured_bottom_center_y_m"] = float(supported_center_y)
            velocity = settings.velocity(direction * force[1])
            # Stop forward pressure at the measured displacement; excessive
            # contact force still commands bounded unloading rather than windup.
            if remaining <= settings.position_tolerance_m and not brace_wall:
                velocity = min(velocity, 0.0)
            elif velocity > 0 and not brace_wall:
                velocity = min(velocity, 1.5 * remaining)
            target[1] += direction * velocity * env.dt
            wall_normal = self.carton_wall_forces.get(env.model.body(self.product.name).id, 0.0)
            if brace_wall and wall_normal >= self.config.contact_threshold:
                # Let the wall supply the opposing squaring contact; unload
                # measured hand yaw moment instead of dragging a free box.
                moment = self.last_forces.get("hand_moment_about_body_origin_Nm", [0, 0, 0])[2]
                vx = -direction * 0.10 * moment + 0.10 * (start_body[0] - body.xpos[0])
                target[0] += np.clip(vx * env.dt, -0.003 * env.dt, 0.003 * env.dt)
                target[0] = np.clip(target[0], origin[0] - 0.015, origin[0] + 0.015)
                event.setdefault("wall_moment_feedback_samples", []).append(
                    [float(env.data.time), wall_normal, moment, float(target[0])]
                )
            if self.config.paired_finger_insertion and not brace_wall:
                matrix = body.xmat.reshape(3, 3)
                yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
                lever = float(
                    np.clip(-direction * self.config.insertion_yaw_lever_gain * yaw, -0.006, 0.006)
                )
                contact = self.last_forces.get("hand_contact_position_world_m")
                measured_x = contact[0] if contact is not None else self.controller.position()[0]
                center_x = body.xpos[0] + (contact_x - start_body[0])
                vx = (
                    2 * (center_x + lever - measured_x)
                    + (start_body[0] - body.xpos[0])
                    - 0.004 * force[0]
                )
                target[0] += np.clip(vx * env.dt, -0.005 * env.dt, 0.005 * env.dt)
                target[0] = np.clip(target[0], origin[0] - 0.015, origin[0] + 0.015)
            target[2] += np.clip(-0.012 * force[2] * env.dt, -0.005 * env.dt, 0.005 * env.dt)
            # Project the reference into the unchanged corridor before IK.
            # Admittance at a height limit must saturate, not command an
            # out-of-corridor step or discard a still-supported wall approach.
            if self.config.ungrasped_insertion:
                requested = target.copy()
                lower = np.array(
                    [origin[0] - 0.015, origin[1] - distance_m - 0.025, height - 0.005]
                )
                upper = np.array(
                    [origin[0] + 0.015, origin[1] + distance_m + 0.025, height + 0.025]
                )
                target = np.clip(target, lower + 1e-6, upper - 1e-6)
                if not np.array_equal(requested, target):
                    event["reference_saturation_updates"] = (
                        event.get("reference_saturation_updates", 0) + 1
                    )
            # Final bounds follow every force/height correction.
            if (
                abs(target[1] - origin[1]) > distance_m + 0.025
                or abs(target[0] - origin[0]) > 0.015 + 1e-12
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("lateral_push: final corrected reference bound")
            prediction = self._contact_prediction(
                target,
                allowed_object=self.product.name,
                alternatives=False,
                carrying=False,
            )
            if not prediction.feasible:
                event["feedback_rejection"] = prediction.report()
                raise InteractionFailure("lateral_push: checked feedback path rejected; replan")
            if self.config.ungrasped_insertion:
                target = prediction.position.copy()
                if (
                    abs(target[1] - origin[1]) > distance_m + 0.025
                    or abs(target[0] - origin[0]) > 0.015 + 1e-12
                    or not height - 0.005 <= target[2] <= height + 0.025
                ):
                    raise InteractionFailure("lateral_push: projected reference outside bounds")
            self.controller.last_target = target.copy()
            self._apply_tick(self.controller.joint_targets(prediction.q, 0))
            measured_force = self.last_forces.get("hand_force_world_N", [0, 0, 0])
            had_push_contact |= direction * measured_force[1] > 0.2
            contact = self.last_forces.get("hand_contact_position_world_m")
            if contact is not None:
                event.setdefault("contact_positions_m", []).append([float(env.data.time), *contact])
            event["samples"].append(
                [
                    float(env.data.time),
                    *body.xpos,
                    *self.last_forces.get("hand_force_world_N", [0, 0, 0]),
                    *target,
                ]
            )
            upright = body.xmat.reshape(3, 3)[2, 2] >= np.cos(
                np.radians(self.config.payload_upright_degrees)
            )
            verified = (
                abs(desired_y - body.xpos[1]) <= settings.position_tolerance_m
                and had_push_contact
                and (
                    not near_front_corner
                    or abs(bottom_center_y - (body.xpos[1] - half_height * body.xmat[5]))
                    <= settings.position_tolerance_m
                )
                and abs(body.xpos[0] - start_body[0]) <= 0.005
                and self.last_forces["shelf"] > 0.2
                and upright
            )
            if brace_wall:
                event["last_carton_wall_normal_N"] = wall_normal
                verified = verified and wall_normal >= self.config.contact_threshold
                if remaining < -0.002:
                    raise InteractionFailure("lateral_push: wall registration displacement guard")
            if self.config.paired_finger_insertion:
                matrix = body.xmat.reshape(3, 3)
                yaw_limit = 5.0 if stage_for_squaring else 0.15
                verified = verified and abs(np.arctan2(matrix[1, 0], matrix[0, 0])) <= np.radians(
                    yaw_limit
                )
            if row_ready is not None:
                verified = (
                    had_push_contact
                    and self.last_forces["shelf"] > 0.2
                    and upright
                    and abs(body.xpos[0] - start_body[0]) <= 0.005
                    and row_ready()
                )
            held = held + env.dt if verified else 0.0
            if held >= settings.contact_dwell_s:
                event.update(
                    displacement_before_retreat_verified=True,
                    separate_square_required=stage_for_squaring,
                    loaded_tilt_deg=float(np.degrees(np.arccos(np.clip(body.xmat[8], -1.0, 1.0)))),
                    measured_displacement_m=float(direction * (body.xpos[1] - start_body[1])),
                )
                break
        else:
            raise InteractionFailure("lateral_push: displacement/support dwell timeout; replan")
        # Withdraw along the contact normal before lifting; success requires
        # the measured displacement to survive physical unloading and retreat.
        unload_distance = 0.036 if row_ready is not None else 0.012 if near_front_corner else 0.03
        unload = [0, -direction * unload_distance, 0]
        self._cartesian_move("lateral_unload", target + unload, allowed_object=self.product.name)
        if near_front_corner:
            event["unloaded_position_world_m"] = env.data.body(self.product.name).xpos.tolist()
            event["unloaded_hand_normal_N"] = self.last_forces["hand"]
            if self.last_forces["hand"] > 0.2:
                raise InteractionFailure("lateral_unload: hand still loaded after side clearance")
            self._cartesian_move("lateral_front_exit", self.controller.last_target + [0.06, 0, 0])
        self._cartesian_move(
            "lateral_clear",
            self.controller.last_target + [0, 0, 0.04],
        )
        final = env.data.body(self.product.name)
        error = abs(float(final.xpos[1] - desired_y))
        if (
            (row_ready is None and error > settings.position_tolerance_m)
            or self.last_forces["hand"] > 0.2
            or self.last_forces["shelf"] <= 0.2
            or abs(final.xpos[0] - start_body[0]) > 0.005
            or final.xmat.reshape(3, 3)[2, 2]
            < np.cos(np.radians(self.config.payload_upright_degrees))
        ):
            raise InteractionFailure("lateral_retreat: displacement or hand unloading failed")
        if self.observer is not None and self.observer.incidents:
            raise InteractionFailure("lateral_retreat: product incident recorded")
        event.update(
            lateral_staging_verified=stage_for_squaring,
            verified_lateral_displacement=not stage_for_squaring and row_ready is None,
            retreat_verified=True,
            final_lateral_error_m=error,
        )
        return event


class LateralStockingScript(LateralContact, DenseStockingScript):
    """Dense stocking scene with the opt-in lateral contact primitive."""
