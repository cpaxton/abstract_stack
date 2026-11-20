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

    def __init__(self, env, config, *, direction=-1, yaw_degrees=90, **kwargs):
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
        self.last_target = self.position().copy()


class LateralStockingScript(DenseStockingScript):
    """Opt-in primitive; baseline DenseStockingScript behavior is unchanged."""

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
        for k in range(max(1, int(np.ceil(duration / self.env.dt)))):
            t = min(1.0, (k + 1) * self.env.dt / duration)
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            target = start + blend * (end - start)
            prediction = hand.predict(
                target, alternatives=False, carrying=False, allowed_object=allowed_object
            )
            if not prediction.feasible:
                event["feedback_rejection"] = prediction.report()
                raise InteractionFailure(f"{name}: Cartesian feedback route rejected")
            hand.last_target = target.copy()
            self._apply_tick(hand.joint_targets(prediction.q, 0))
        error = float(np.linalg.norm(hand.position() - end))
        event.update(end_time=float(self.env.data.time), measured_tracking_error_m=error)
        if error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: measured Cartesian tracking error {error:.3f}m")

    def push_laterally(self, *, direction, distance_m, settings=None):
        settings = settings or LateralPushConfig()
        if direction not in (-1, 1) or not np.isfinite(distance_m):
            raise ValueError("Expected +/-1 direction and finite distance")
        if not 0 < distance_m <= settings.distance_limit_m:
            raise ValueError("Requested lateral displacement is outside the bounded skill")
        if distance_m <= settings.position_tolerance_m:
            raise ValueError("Requested displacement must exceed its verification tolerance")
        if settings.force_target_N >= self.config.max_force:
            raise ValueError("Lateral force target must be below the unchanged abort guard")
        env = self.env
        start_body = env.data.body(self.product.name).xpos.copy()
        extent = np.abs(env.data.body(self.product.name).xmat.reshape(3, 3)) @ np.asarray(
            self.product.half_size
        )
        height = env.config.levels[self.product.level] + settings.height_m
        precontact = np.array(
            [start_body[0], start_body[1] - direction * (extent[1] + 0.015), height]
        )
        # Use the existing checked high-clearance staging motion before
        # changing palm orientation; a direct joint interpolation from the
        # parked hand can sweep through the table even with a reachable endpoint.
        self._move("lateral_stage", [-0.30, -0.20, 1.35], carrying=False)
        self._stage("lateral_plan")
        candidates = []
        self.events[-1]["candidates"] = candidates
        selected = None
        # Native model is shared read-only; every candidate owns independent MjData.
        started = time.monotonic()
        for yaw, tilt, wrist in product((45, 30, 60, 90), (20, 30, 40), (0.2, 0.4, 0.0, 0.6)):
            if time.monotonic() - started > 45:
                break
            kwargs = dict(
                direction=direction,
                yaw_degrees=yaw,
                finger="Middle",
                tilt_degrees=tilt,
                wrist_posture=wrist,
            )
            hand = SideFingerPusher(env.clone_for_planning(), self.config, **kwargs)
            hand.command = self.controller.command.copy()
            raised = precontact + [0, 0, 0.14]
            candidate = {**kwargs, "waypoints_m": [raised.tolist(), precontact.tolist()]}
            candidates.append(candidate)
            prediction = hand.predict(raised, alternatives=False, carrying=False)
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
        self._change_hand(SideFingerPusher, **selected)
        self._move(
            "lateral_orient_high",
            precontact + [0, 0, 0.14],
            carrying=False,
            orientation_alternatives=False,
        )
        self._cartesian_move("lateral_approach", precontact)
        self._stage("lateral_push")
        event = self.events[-1]
        target = self.controller.last_target.copy()
        origin = target.copy()
        desired_y = start_body[1] + direction * distance_m
        event.update(
            start_body_m=start_body.tolist(),
            desired_body_y_m=float(desired_y),
            requested_displacement_m=distance_m,
            direction=direction,
            contact_mode="exposed-side supported push, no rigid attachment",
            forecast="robot/finger kinematics only; no lateral object dynamics certificate",
            settings=vars(settings),
            samples=[],
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
            velocity = settings.velocity(direction * force[1])
            # Stop forward pressure at the measured displacement; excessive
            # contact force still commands bounded unloading rather than windup.
            if remaining <= settings.position_tolerance_m:
                velocity = min(velocity, 0.0)
            elif velocity > 0:
                velocity = min(velocity, 1.5 * remaining)
            target[1] += direction * velocity * env.dt
            target[2] += np.clip(-0.012 * force[2] * env.dt, -0.005 * env.dt, 0.005 * env.dt)
            # Final bounds follow every force/height correction.
            if (
                abs(target[1] - origin[1]) > distance_m + 0.025
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("lateral_push: final corrected reference bound")
            prediction = self.controller.predict(
                target,
                allowed_object=self.product.name,
                alternatives=False,
                carrying=False,
            )
            if not prediction.feasible:
                event["feedback_rejection"] = prediction.report()
                raise InteractionFailure("lateral_push: checked feedback path rejected; replan")
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
                and abs(body.xpos[0] - start_body[0]) <= 0.005
                and self.last_forces["shelf"] > 0.2
                and upright
            )
            held = held + env.dt if verified else 0.0
            if held >= settings.contact_dwell_s:
                event.update(
                    displacement_before_retreat_verified=True,
                    measured_displacement_m=float(direction * (body.xpos[1] - start_body[1])),
                )
                break
        else:
            raise InteractionFailure("lateral_push: displacement/support dwell timeout; replan")
        # Withdraw along the contact normal before lifting; success requires
        # the measured displacement to survive physical unloading and retreat.
        self._cartesian_move(
            "lateral_unload",
            target - np.array([0, direction * 0.03, 0]),
            allowed_object=self.product.name,
        )
        self._cartesian_move(
            "lateral_clear",
            self.controller.last_target + [0, 0, 0.04],
        )
        final = env.data.body(self.product.name)
        error = abs(float(final.xpos[1] - desired_y))
        if (
            error > settings.position_tolerance_m
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
            verified_lateral_displacement=True, retreat_verified=True, final_lateral_error_m=error
        )
        return event
