"""Experimental frictional row compaction from the exposed front face.

Normal preload against the backstop adds friction. A successful isolated skill
must retain original row-overlap criteria after unloading; reset fixtures earn
no table-to-shelf credit.
"""

import time
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.dense_interactions import FingerPusher
from abstract_stack.finger_load_control import FingerLoadCompensator
from abstract_stack.interactions import InteractionFailure
from abstract_stack.lateral_contact import LateralPushConfig, LateralStockingScript
from abstract_stack.load_aware_pusher import LoadAwareFingerPusher
from abstract_stack.mabel_controller import MabelController
from abstract_stack.packing import row_geometry

# Keep the prototype import name available; new users share the generic tool.
TangentialFingerPusher = LoadAwareFingerPusher


class FrontTangentialPress:
    def _change_hand(self, kind, **kwargs):
        super()._change_hand(kind, **kwargs)
        self.finger_load = None
        if isinstance(self.controller, FingerPusher):
            previous = self.tracker
            self.tracker = MabelController(
                self.env.model,
                self.env.data,
                replace(previous.config, wrist_offset_limit=0.5),
            )
            self.tracker.nominal = previous.nominal.copy()
            self.tracker.applied = previous.applied.copy()
            finger = self.controller.finger
            joints = [f"Left {finger} MCP", f"Left {finger} PIP"]
            if isinstance(self.controller, TangentialFingerPusher):
                joints.insert(0, f"Left {finger} Abd")
            self.finger_load = FingerLoadCompensator(
                self.env.model,
                self.env.data,
                joints,
                product_name=self.product.name,
            )

    def _apply_tick(self, action):
        compensation = getattr(self, "finger_load", None)
        if compensation is not None and isinstance(self.controller, FingerPusher):
            applied = compensation.targets(action, dt=self.env.dt)
            future = self.controller.finger_targets(0)
            future.update({name: applied[name] for name in compensation.names})
            collisions = self.controller.check_hand_step(
                SimpleNamespace(q=self.controller.command),
                0,
                self.product.name,
                targets=future,
            )
            if collisions:
                self.events[-1]["finger_load_collision_rejection"] = collisions
                raise InteractionFailure("finger_load: coupled applied-reference geometry rejected")
            action = applied
        return super()._apply_tick(action)

    def episode_evidence(self):
        evidence = super().episode_evidence()
        compensation = getattr(self, "finger_load", None)
        if compensation is not None:
            evidence["finger_load_compensation"] = dict(
                joints=compensation.last_report,
                peak_offset_rad=compensation.peak_offset,
                clipped_updates=compensation.clipped_updates,
                maximum_offset_rad=0.5,
                maximum_offset_rate_rad_s=1.0,
                future_geometry="coupled arm and applied-finger references checked each tick",
            )
        return evidence

    def _cartesian_move(self, name, end, *, speed_m_s=0.025, allowed_object=None):
        """Execute the complete retained route, including any initial prefix."""
        self._stage(name)
        hand = self.controller
        end = np.asarray(end, dtype=float)
        start = hand.position().copy()
        prediction = hand.predict(
            end,
            allowed_object=allowed_object,
            alternatives=False,
            carrying=False,
            path_mode="cartesian",
        )
        event = self.events[-1]
        event["ik_prediction"] = prediction.report()
        if not prediction.feasible or prediction.joint_path is None:
            raise InteractionFailure(f"{name}: retained Cartesian route rejected")
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        path = prediction.joint_path
        segment_times = np.max(np.abs(np.diff(path, axis=0)) / rates, axis=1)
        knots = np.r_[0.0, np.cumsum(segment_times)]
        duration = max(
            1.0, float(np.linalg.norm(end - start)) / speed_m_s * 1.875, 1.875 * knots[-1] * 1.05
        )
        if duration > self.config.max_motion_seconds:
            raise InteractionFailure(f"{name}: retained Cartesian duration bound")
        knots = knots / knots[-1] if knots[-1] > 1e-12 else np.linspace(0, 1, len(path))
        ticks = max(1, int(np.ceil(duration / self.env.dt)))
        event.update(
            path_mode="retained_cartesian",
            planned_duration_s=duration,
            retained_waypoints=len(path),
        )
        for index in range(ticks):
            t = (index + 1) / ticks
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            q = np.array([np.interp(blend, knots, column) for column in path.T])
            action = hand.joint_targets(q, 0)
            hand.last_target, _ = hand.ik.forward(hand.command, prediction.reference_qpos)
            self._apply_tick(action)
        error = float(np.linalg.norm(hand.position() - end))
        event.update(measured_tracking_error_m=error, end_time=float(self.env.data.time))
        if error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: Cartesian tracking guard")

    def _bounded_tangential_prediction(self, target):
        hand = self.controller
        proposal = hand.predict(
            target,
            allowed_object=self.product.name,
            alternatives=False,
            carrying=False,
            check_path=False,
        )
        # This new contact skill requires sub-millimetre geometric tracking;
        # its task/force acceptance criteria remain independent and unchanged.
        proposal = replace(proposal, position_tolerance_m=0.001)
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        result = bounded_contact_prediction(hand, proposal, target, self.env.dt, rates)
        hand.last_prediction = result
        self.events[-1].setdefault("bounded_feedback_samples", []).append(
            {
                "sim_time": float(self.env.data.time),
                "feasible": bool(result.feasible),
                **result.bounded_feedback,
            }
        )
        return result

    def push_front_tangentially(
        self,
        *,
        direction,
        distance_m,
        normal_force_N=8.0,
        settings=None,
        force_guided=False,
        contact_offset_m=0.0,
        row_products=None,
        staged_approach=False,
    ):
        if not isinstance(force_guided, bool):
            raise ValueError("force_guided must be boolean")
        settings = settings or LateralPushConfig(
            speed_m_s=0.002,
            force_target_N=8.0,
            timeout_s=30.0,
            position_tolerance_m=0.0002,
            height_m=0.018,
        )
        if (
            direction not in (-1, 1)
            or not np.isfinite(distance_m)
            or not settings.position_tolerance_m < distance_m <= settings.distance_limit_m
        ):
            raise ValueError("Tangential direction/displacement outside bounded skill")
        if (
            not np.isfinite(normal_force_N)
            or not 0 < normal_force_N < self.config.max_force
            or settings.force_target_N >= self.config.max_force
        ):
            raise ValueError("Tangential force targets must be below unchanged abort guard")
        env = self.env
        row_products = set(row_products) if row_products is not None else None
        if row_products is not None and (
            self.product.name not in row_products
            or not row_products <= self.observer.completed
            or self.observer.incidents
        ):
            raise ValueError("Partial-row press requires incident-free stocked products")
        start = env.data.body(self.product.name).xpos.copy()
        extent = np.abs(env.data.body(self.product.name).xmat.reshape(3, 3)) @ np.asarray(
            self.product.half_size
        )
        if not np.isfinite(contact_offset_m) or abs(contact_offset_m) > max(0.0, extent[1] - 0.006):
            raise ValueError("Front contact offset exceeds the bounded active face")
        height = env.config.levels[self.product.level] + settings.height_m
        precontact = np.array([start[0] + extent[0] + 0.02, start[1] + contact_offset_m, height])
        high = precontact + [0.085 if staged_approach else 0, 0, 0.12]
        end = precontact + [-0.024, direction * distance_m, 0]
        # Stage relative to the measured base, rather than retaining a world
        # waypoint from a different reset fixture. No live base motion is applied.
        base = env.data.body("mabel_chassis").xpos
        staging = np.array([base[0] - 0.26, base[1] - 0.20, 1.35])
        self._move("tangential_stage", staging, carrying=False)
        self._stage("tangential_plan")
        attempts, selected = [], None
        event = self.events[-1]
        event["candidates"] = attempts
        started = time.monotonic()
        for tilt, wrist in product((10, 20, 30, 40), (0.2, 0.4, 0.0, 0.6)):
            if time.monotonic() - started > 45:
                break
            kwargs = dict(finger=self._pushing_finger(), tilt_degrees=tilt, wrist_posture=wrist)
            hand = TangentialFingerPusher(env.clone_for_planning(), self.config, **kwargs)
            hand.command = self.controller.command.copy()
            attempt = {**kwargs, "predictions": []}
            attempts.append(attempt)
            for point, path_mode, allowed in (
                (high, "joint", None),
                (precontact, "cartesian", None),
                (end, "cartesian", self.product.name),
            ):
                prediction = hand.predict(
                    point,
                    alternatives=False,
                    carrying=False,
                    allowed_object=allowed,
                    path_mode=path_mode,
                )
                attempt["predictions"].append(prediction.report())
                if not prediction.feasible or prediction.position_error_m > 0.001:
                    attempt["rejected"] = "requires_submillimetre_contact_reference"
                    break
                self._planning_pose(hand, prediction)
            else:
                margin = self._clearance_margin(hand, prediction, precontact, end)
                attempt["clearance_m"] = margin
                if margin["support"] >= 0.012 and margin["sides"] >= 0.004:
                    selected = kwargs
                    break
        event["planning_wall_seconds"] = time.monotonic() - started
        if selected is None:
            raise InteractionFailure("tangential_plan: bounded approach search rejected")
        event["selected"] = selected
        event["outside_shelf_orientation_stage"] = bool(staged_approach)
        self._change_hand(TangentialFingerPusher, **selected)
        if staged_approach:
            self._settle_pusher_preshape()
        self._move("tangential_orient_high", high, carrying=False, orientation_alternatives=False)
        self._cartesian_move("tangential_approach", precontact)
        self._stage("front_tangential_push")
        event = self.events[-1]
        target = self.controller.last_target.copy()
        origin = target.copy()
        desired_y = start[1] + direction * distance_m
        held, had_contact = 0.0, False
        event.update(
            normal_force_target_N=normal_force_N,
            front_contact_offset_m=float(contact_offset_m),
            lateral_control=(
                "force_guided_with_velocity_braking"
                if force_guided
                else "position_capped_force_admittance"
            ),
            settings=vars(settings),
            start_body_m=start.tolist(),
            desired_body_y_m=float(desired_y),
            samples=[],
            forecast="native robot geometry only; lateral object effects checked by physics",
        )
        for _ in range(int(settings.timeout_s / env.dt)):
            body = env.data.body(self.product.name)
            force = np.asarray(self.last_forces.get("hand_force_world_N", [0, 0, 0]))
            normal = max(0.0, -float(force[0]))
            recent = self.rows[-env.config.frame_skip :]
            peak_normal = max((row[7] + row[21] for row in recent), default=normal)
            feedback_normal = max(normal, peak_normal)
            contact_gap = target[0] - (body.xpos[0] + extent[0])
            if feedback_normal < 0.1 and contact_gap > 0.004:
                normal_speed = 0.004
            else:
                normal_speed = float(
                    np.clip(0.0002 * (normal_force_N - feedback_normal), -0.001, 0.001)
                )
            target[0] -= normal_speed * env.dt
            remaining = direction * (desired_y - body.xpos[1])
            if force_guided:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
                )
                projected_remaining = remaining - max(0.0, direction * velocity[4]) * 0.05
                lateral_force_goal = (
                    settings.force_target_N
                    if projected_remaining > settings.position_tolerance_m
                    else 0.0
                )
                lateral = (
                    settings.speed_m_s
                    * float(
                        np.clip(
                            (lateral_force_goal - direction * force[1]) / settings.force_target_N,
                            -1,
                            1,
                        )
                    )
                    if normal > 0.6 * normal_force_N
                    else 0.0
                )
            else:
                lateral = (
                    settings.velocity(direction * force[1]) if normal > 0.6 * normal_force_N else 0
                )
                if remaining <= settings.position_tolerance_m:
                    lateral = min(lateral, 0.0)
                elif lateral > 0:
                    lateral = min(lateral, 1.5 * remaining)
            target[1] += direction * lateral * env.dt
            target[2] += np.clip(-0.004 * force[2] * env.dt, -0.002 * env.dt, 0.002 * env.dt)
            if (
                not origin[0] - 0.04 <= target[0] <= origin[0] + 0.01
                or abs(target[1] - origin[1]) > distance_m + 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("front_tangential_push: corrected reference bound")
            prediction = self._bounded_tangential_prediction(target)
            if not prediction.feasible:
                event["feedback_rejection"] = prediction.report()
                raise InteractionFailure("front_tangential_push: bounded feedback IK rejected")
            target = prediction.position.copy()  # Nominal FK projection anti-windup.
            if (
                not origin[0] - 0.04 <= target[0] <= origin[0] + 0.01
                or abs(target[1] - origin[1]) > distance_m + 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("front_tangential_push: projected reference bound")
            self.controller.last_target = target.copy()
            self._apply_tick(self.controller.joint_targets(prediction.q, 0))
            measured = self.last_forces.get("hand_force_world_N", [0, 0, 0])
            had_contact |= direction * measured[1] > 0.2
            event["samples"].append([float(env.data.time), *body.xpos, *measured, *target])
            upright = body.xmat.reshape(3, 3)[2, 2] >= np.cos(np.radians(5))
            if not upright or (self.observer is not None and self.observer.incidents):
                raise InteractionFailure("front_tangential_push: upright/incident guard")
            verified = (
                abs(body.xpos[1] - desired_y) <= settings.position_tolerance_m
                and had_contact
                and self.last_forces["shelf"] > 0.2
            )
            held = held + env.dt if verified else 0.0
            if held >= settings.contact_dwell_s:
                break
        else:
            raise InteractionFailure("front_tangential_push: displacement/support timeout")
        self._cartesian_move(
            "tangential_unload", target + [0.04, 0, 0], allowed_object=self.product.name
        )
        self._cartesian_move("tangential_clear", self.controller.last_target + [0, 0, 0.04])
        for _ in range(int(0.5 / env.dt)):
            self._apply_tick(self.controller.joint_targets(self.controller.command, 0))
        final = env.data.body(self.product.name)
        geometry = row_geometry(env, product_names=row_products)
        placed = env._placement_conditions()
        if row_products is not None:
            placed = [
                ok
                for product, ok in zip(env.products, placed, strict=True)
                if product.name in row_products
            ]
        error = abs(float(final.xpos[1] - desired_y))
        event.update(final_lateral_error_m=error, final_row_geometry=geometry)
        if (
            error > settings.position_tolerance_m
            or not geometry["envelopes_fit"]
            or self.last_forces["hand"] > 0.2
            or self.last_forces["shelf"] <= 0.2
            or abs(final.xpos[0] - start[0]) > 0.005
            or not all(placed)
            or (self.observer is not None and self.observer.incidents)
        ):
            raise InteractionFailure("tangential_retreat: final displacement/row/support guard")
        event.update(verified_tangential_displacement=True, retreat_verified=True)
        return event


class FrontTangentialStockingScript(FrontTangentialPress, LateralStockingScript):
    """Dense reset-fixture runner for the standalone front press."""
