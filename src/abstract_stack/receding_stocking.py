"""Opt-in object-level replanning around the existing guarded dense controller."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.base_motion import (
    BaseContactGuard,
    BaseMotionConfig,
    NativeSwerve,
    screen_base_translation,
)
from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.dense_interactions import DenseStockingScript, FingerPusher
from abstract_stack.finger_load_control import FingerLoadCompensator
from abstract_stack.ik import PoseIK
from abstract_stack.interactions import CartesianHand, InteractionFailure
from abstract_stack.load_aware_pusher import LoadAwareFingerPusher
from abstract_stack.loaded_route import (
    admit_loaded_route,
    carried_product_clearance,
    object_tool_goal,
    plan_loaded_route,
)
from abstract_stack.replanning import (
    ActionCheck,
    CartonEstimate,
    RowObservation,
    RowReplanner,
)


def observe_row(env, pending, tool_position):
    """Privileged state adapter; later noisy observations replace this boundary."""
    cartons = []
    for product in env.products:
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        cartons.append(
            CartonEstimate(
                product.name,
                tuple(map(float, body.xpos)),
                tuple(map(float, extent)),
                tuple(product.target),
                tuple(product.half_size),
            )
        )
    return RowObservation(
        tuple(cartons), tuple(p.name for p in pending), tuple(map(float, tool_position))
    )


class RecedingStockingScript(DenseStockingScript):
    """Recheck discrete order after each completed carton; retain native guards.

    Optional admission forecasts the empty approach and nominal loaded corridor.
    Execution refines loaded segments from measured state. Primitive-level
    jam recovery, stochastic MCTS and full belief-space planning remain future
    work. Released-contact reacquisition is a separately bounded opt-in mode.
    """

    def __init__(
        self,
        *args,
        search_config=None,
        loaded_route=False,
        route_admission=False,
        table_front=False,
        base_reposition=False,
        finger_load_compensation=False,
        bounded_contact_ik=False,
        square_backstop=False,
        contact_recovery=False,
        grasp_wrist_posture=None,
        payload_load_compensation=False,
        payload_wrist_reserve=0.4,
        split_insertion=False,
        **kwargs,
    ):
        self.grasp_wrist_posture = grasp_wrist_posture
        if not isinstance(payload_load_compensation, bool) or not (
            np.isfinite(payload_wrist_reserve) and 0.15 <= payload_wrist_reserve <= 0.5
        ):
            raise ValueError("Payload compensation requires a boolean and 0.15–0.5 rad reserve")
        self.payload_load_compensation = payload_load_compensation
        self.payload_wrist_reserve = float(payload_wrist_reserve)
        self.loaded_product_clearance_m = 0.02 if payload_load_compensation else 0.0
        self.split_insertion = split_insertion
        self._split_push_phase = None
        self._pusher_base_backoff_origin = None
        super().__init__(*args, **kwargs)
        if not self.config.adaptive_grasps:
            raise ValueError("Receding order search requires adaptive grasps")
        self.replanner = RowReplanner(search_config)
        self.order_plans = []
        self.loaded_route_enabled = loaded_route
        self.table_front = table_front
        self.base_reposition_enabled = base_reposition
        self.base_repositions = []
        self._pusher_base_backoff_done = False
        self._base_guard = None
        self._base_translation_origin = None
        self.finger_load_compensation = finger_load_compensation
        self._finger_load = None
        self.peak_finger_load_offset = 0.0
        self.bounded_contact_ik = bounded_contact_ik
        self.square_backstop = square_backstop
        self.contact_recovery = contact_recovery
        if contact_recovery and not bounded_contact_ik:
            raise ValueError("Contact recovery requires bounded Cartesian feedback")
        self.PUSHER_CLASS = LoadAwareFingerPusher if finger_load_compensation else FingerPusher
        self._start_free_push()
        self._last_progress_heartbeat = float("-inf")
        if base_reposition and not loaded_route:
            raise ValueError("Base repositioning requires the loaded-route executor")
        if table_front and not loaded_route:
            raise ValueError("Table-front staging requires the loaded-route executor")
        if table_front:
            self.FRONT_SUPPORT = "table"
            self.CONTACT_PROTOCOL = "table_front_release_then_selected_finger_slide_to_shelf"
        self.PUSHER_APPROACH_PATH_MODE = "cartesian" if loaded_route else "joint"
        self.route_admission_enabled = route_admission
        if route_admission and not loaded_route:
            raise ValueError("Loaded-route admission requires the loaded-route executor")
        if split_insertion and not (
            loaded_route and base_reposition and bounded_contact_ik and payload_load_compensation
        ):
            raise ValueError(
                "Split insertion requires checked routes, base, bounded IK and payload mode"
            )

    def _front_center_x(self, extent):
        if self.table_front:
            return self.env.config.front_x + 0.06
        return super()._front_center_x(extent)

    def _new_cartesian_hand(self):
        hand = super()._new_cartesian_hand()
        self._configure_grasp_hand(hand)
        return hand

    def _configure_grasp_hand(self, hand):
        if self.payload_load_compensation:
            wrist = hand.ik.names.index("Left Wrist")
            hand.ik.constrain_reference_range(
                "Left Wrist",
                hand.ik.original_lower[wrist] + self.payload_wrist_reserve,
                hand.ik.original_upper[wrist] - self.payload_wrist_reserve,
            )
            lift = hand.ik.names.index("Lift")
            hand.ik.constrain_reference_range(
                "Lift",
                hand.ik.original_lower[lift],
                hand.ik.original_upper[lift] - self._tracking_config().lift_offset_limit,
            )
        if self.grasp_wrist_posture is not None:
            hand.ik.constrain_posture("Left Wrist", self.grasp_wrist_posture)

    def _tracking_config(self):
        config = super()._tracking_config()
        return (
            replace(config, wrist_offset_limit=self.payload_wrist_reserve)
            if self.payload_load_compensation
            else config
        )

    def _wrist_offset_limit(self, kind):
        if self.payload_load_compensation and not issubclass(kind, FingerPusher):
            return self.payload_wrist_reserve
        return super()._wrist_offset_limit(kind)

    def _load_feedforward(self):
        if self.payload_load_compensation and not isinstance(self.controller, FingerPusher):
            result = self._wrist_contact_load()
            if self.events:
                self.events[-1].setdefault("payload_wrist_load_samples", []).append(
                    [
                        float(self.env.data.time),
                        self.wrist_load_last_requested,
                        self.wrist_load,
                    ]
                )
            return result
        return super()._load_feedforward()

    def _check_tracked_targets(self, targets):
        super()._check_tracked_targets(targets)
        if not self.payload_load_compensation or isinstance(self.controller, FingerPusher):
            return
        model, hand = self.env.model, self.controller
        applied_q = np.array([targets[model.actuator(name).id] for name in [*hand.names, "Lift"]])
        fingers = {}
        for aid in range(model.nu):
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(model.actuator_trnid[aid, 0])
            if int(model.jnt_bodyid[jid]) in self.hand_bodies:
                fingers[model.actuator(aid).name] = float(targets[aid])
        intended_contact = self.active_grip is not None or self.phase in (
            "approach",
            "close",
            "place",
            "stage_align",
            "release_unload",
            "release_front",
        )
        errors = hand.check_hand_step(
            SimpleNamespace(q=applied_q),
            0,
            self.product.name if intended_contact else None,
            targets=fingers,
        )
        if errors:
            self.events[-1]["applied_tracking_geometry_rejection"] = errors
            raise InteractionFailure("payload_load: compensated arm/finger geometry rejected")

    def _change_hand(self, kind, **kwargs):
        if kind is FingerPusher:
            kind = self.PUSHER_CLASS
        super()._change_hand(kind, **kwargs)
        if kind is CartesianHand:
            self._configure_grasp_hand(self.controller)
        self._finger_load = None
        if self.finger_load_compensation and isinstance(self.controller, FingerPusher):
            finger = self.controller.finger
            self._finger_load = FingerLoadCompensator(
                self.env.model,
                self.env.data,
                [f"Left {finger} Abd", f"Left {finger} MCP", f"Left {finger} PIP"],
                product_name=self.product.name,
            )

    def _apply_tick(self, action):
        compensation = self._finger_load
        if compensation is not None and self.active_grip is None:
            applied = compensation.targets(action, dt=self.env.dt)
            future = {
                name: applied.get(name, value)
                for name, value in self.controller.finger_targets(0).items()
            }
            collisions = self.controller.check_hand_step(
                SimpleNamespace(q=self.controller.command), 0, self.product.name, targets=future
            )
            if collisions:
                self.events[-1]["finger_load_collision_rejection"] = collisions
                raise InteractionFailure("finger_load: coupled applied-reference geometry rejected")
            self.events[-1].setdefault("finger_load_compensation_samples", []).append(
                {"sim_time": float(self.env.data.time), "joints": compensation.last_report}
            )
            self.peak_finger_load_offset = max(
                self.peak_finger_load_offset, compensation.peak_offset
            )
            action = applied
        ended = super()._apply_tick(action)
        now = float(self.env.data.time)
        if now - self._last_progress_heartbeat >= 1.0:
            self._last_progress_heartbeat = now
            progress = {
                "phase": self.phase,
                "product": self.product.name,
                "sim_time": now,
                "completed_products": sorted(self.observer.completed) if self.observer else [],
                "active_product_xyz_m": self.env.data.body(self.product.name).xpos.tolist(),
                "hand_reference_error_m": float(
                    np.linalg.norm(self.controller.position() - self.controller.last_target)
                ),
                "contact_normal_N": {
                    name: self.last_forces.get(name, 0.0)
                    for name in ("hand", "table", "shelf", "back", "robot_product", "arm_fixture")
                },
            }
            temporary = self.directory / "physics-progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(self.directory / "physics-progress.json")
        return ended

    def _move(self, name, target, grip=0.0, seconds=None, **kwargs):
        if self.loaded_route_enabled and name == "staging_clear" and grip == 0:
            # The carton has been released onto its support. Search checked,
            # bounded orientation alternatives for empty withdrawal rather than
            # preserving a grasp rotation that can run into the torso.
            if self.active_grip is not None or self.carry_reference is not None:
                raise InteractionFailure("staging_clear: empty-hand withdrawal contract violated")
            kwargs["orientation_alternatives"] = True
        if self._base_translation_origin is not None and name in (
            "clear_high",
            "pusher_high_clear",
            "retract",
            "raise",
        ):
            target = np.asarray(target, dtype=float).copy()
            target[:2] += (
                self.env.data.body("mabel_chassis").xpos[:2] - self._base_translation_origin
            )
        if (
            self.loaded_route_enabled
            and name == "clear_high"
            and self.observer is not None
            and self.observer.completed
        ):
            self._stage("empty_hand_clear_plan")
            event = self.events[-1]
            event["candidates"] = []
            for height in (1.35, 1.25, 1.20):
                goal = np.asarray(target, dtype=float).copy()
                goal[2] = height
                prediction = self.controller.predict(
                    goal, alternatives=True, carrying=False, path_mode="cartesian"
                )
                event["candidates"].append(prediction.report())
                if prediction.feasible:
                    return self._checked_cartesian_move(
                        "clear_high", goal, prediction.requested_rotation, 0, carrying=False
                    )
            raise InteractionFailure("clear_high: bounded Cartesian clearance search rejected")
        if self.loaded_route_enabled and name == "pusher_approach":
            return self._checked_cartesian_move(
                name,
                target,
                self.controller.ROTATION,
                0,
                carrying=False,
            )
        if self.loaded_route_enabled and name == "align_shelf" and grip > 0:
            route = plan_loaded_route(
                self.env,
                self.controller,
                self.config,
                self.product,
                self.payload_rotation_goal,
                grip=grip,
                front_offset=0.06 if self.table_front else None,
                product_clearance_m=self.loaded_product_clearance_m,
            )
            self._stage("loaded_route_plan")
            self.events[-1].update(
                candidates=route.candidates,
                decision="refine_each_segment" if route.accepted else "hold_for_replan",
                forecast="measured rigid grasp; not a contact/tracking guarantee",
            )
            if not route.accepted:
                raise InteractionFailure("loaded_route: bounded corridor search rejected")
            for index, (goal, rotation) in enumerate(
                zip(route.object_waypoints, route.object_rotations, strict=True)
            ):
                self._cartesian_loaded_move(
                    f"loaded_route_{index}", goal, grip, object_rotation=np.asarray(rotation)
                )
            return False
        if self.loaded_route_enabled and name == "transfer" and grip > 0:
            self._stage("transfer")
            self.events[-1].update(decision="completed_by_checked_loaded_corridor")
            return False
        return super()._move(name, target, grip, seconds, **kwargs)

    def _cartesian_loaded_move(self, name, object_goal, grip, *, object_rotation=None):
        """Execute retained Cartesian-IK joint waypoints, never a joint shortcut."""
        hand = self.controller
        target, rotation = object_tool_goal(
            hand,
            self.product.name,
            object_goal,
            self.payload_rotation_goal if object_rotation is None else object_rotation,
        )
        return self._checked_cartesian_move(
            name, target, rotation, grip, carrying=True, object_goal=object_goal
        )

    def _pushing_finger_candidates(self):
        preferred = self._pushing_finger()
        if not self.payload_load_compensation:
            return (preferred,)
        # A non-edge carton can still bring folded fingers against a wall.
        # Search native fingers with the same coupled geometry and force gates.
        return tuple(dict.fromkeys((preferred, "Ring", "Index", "Middle")))

    def _plan_pusher(self, precontact, end):
        attempts = 4 if self.payload_load_compensation else 3
        for attempt in range(attempts):
            if self.loaded_route_enabled:
                self._move("pusher_high_clear", [-0.30, -0.20, 1.35])
            try:
                return super()._plan_pusher(precontact, end)
            except InteractionFailure as error:
                if (
                    not self.base_reposition_enabled
                    or attempt >= attempts - 1
                    or not str(error).startswith("pusher_plan:")
                ):
                    raise
                if (
                    self.payload_load_compensation
                    and not self._pusher_base_backoff_done
                    and self._split_push_phase != "deep"
                ):
                    # A tucked arm can collide with the torso at a near target.
                    # One checked empty-base backoff opens extension workspace;
                    # subsequent alternatives retain the existing lateral moves.
                    self._pusher_base_backoff_origin = float(
                        self.env.data.body("mabel_chassis").xpos[0]
                    )
                    self._move_base([0.05, 0.0], reason="empty_hand_pusher_extension_replan")
                    self._pusher_base_backoff_done = True
                    continue
                base_y = float(self.env.data.body("mabel_chassis").xpos[1])
                delta = float(np.clip(self.product.target[1] + 0.22 - base_y, -0.12, 0.12))
                if abs(delta) < 0.01:
                    raise
                self._move_base([0.0, delta], reason="empty_hand_pusher_workspace_replan")
        raise InteractionFailure("pusher_plan: bounded base alternatives exhausted")

    def _insert_from_front(self):
        if not self.split_insertion:
            return super()._insert_from_front()
        body = self.env.data.body(self.product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(self.product.half_size)
        stop_x = self.env.config.front_x - extent[0] - 0.005
        full_x = self.env.config.front_x - self.env.config.depth + extent[0] - 0.0005
        if not full_x + 0.02 < stop_x < body.xpos[0] - 0.01:
            return super()._insert_from_front()
        self._stage("split_insertion_plan")
        self.events[-1].update(partial_object_x_m=float(stop_x), final_object_x_m=float(full_x))
        self._split_push_phase = "front"
        super()._insert_from_front(partial_stop_x=float(stop_x))
        self._change_hand(CartesianHand)
        self._clear_split_push_hand()
        self.events[-1]["partial_state_before_base_handoff"] = self._partial_push_state(
            float(stop_x), unloaded=True
        )
        if not self.events[-1]["partial_state_before_base_handoff"]["accepted"]:
            raise InteractionFailure("partial_push: intermediate state changed before base handoff")
        if self._pusher_base_backoff_origin is not None:
            delta = self._pusher_base_backoff_origin - self.env.data.body("mabel_chassis").xpos[0]
            if abs(delta) > 0.002:
                self._move_base(
                    [float(delta), 0.0], reason="supported_partial_push_workspace_replan"
                )
            self._pusher_base_backoff_origin = None
            self._pusher_base_backoff_done = False
        self.events[-1]["partial_state_after_base_handoff"] = self._partial_push_state(
            float(stop_x), unloaded=True
        )
        if not self.events[-1]["partial_state_after_base_handoff"]["accepted"]:
            raise InteractionFailure("partial_push: intermediate state changed during base handoff")
        self._split_push_phase = "deep"
        super()._insert_from_front()
        self._split_push_phase = None

    def _clear_pusher(self, hand):
        if self._split_push_phase != "front":
            return super()._clear_pusher(hand)
        origin = hand.position().copy()
        self._stage("partial_clear_plan")
        self.events[-1]["candidates"] = []
        for delta in ([0.04, 0, 0.02], [0.02, 0, 0.01], [0.03, 0, 0]):
            goal = origin + delta
            prediction = hand.predict(
                goal, alternatives=False, carrying=False, path_mode="cartesian"
            )
            self.events[-1]["candidates"].append(prediction.report())
            if prediction.feasible:
                return self._checked_cartesian_move(
                    "partial_clear", goal, prediction.requested_rotation, 0, carrying=False
                )
        raise InteractionFailure("partial_clear: no checked empty-hand withdrawal")

    def _clear_split_push_hand(self):
        offset = (
            np.zeros(2)
            if self._base_translation_origin is None
            else self.env.data.body("mabel_chassis").xpos[:2] - self._base_translation_origin
        )
        self._stage("split_push_clear_plan")
        self.events[-1]["candidates"] = []
        for height in (1.25, 1.35, 1.20):
            goal = np.r_[np.array([-0.30, -0.20]) + offset, height]
            prediction = self.controller.predict(
                goal, alternatives=True, carrying=False, path_mode="cartesian"
            )
            self.events[-1]["candidates"].append(prediction.report())
            if prediction.feasible:
                return self._checked_cartesian_move(
                    "split_push_clear", goal, prediction.requested_rotation, 0, carrying=False
                )
        raise InteractionFailure("split_push_clear: bounded empty-hand route rejected")

    def _sample(self, env):
        free_push = (
            self.contact_recovery
            and isinstance(self.controller, FingerPusher)
            and self.phase in ("finger_push", "reacquire_push_contact")
            and self.active_grip is None
            and self.carry_reference is None
        )
        if free_push:
            # The object is released and surface-supported. Its hand contact
            # may be deliberately reacquired; the grasped-slide contract is
            # still used unchanged for all other contact modes.
            self.supported_slide = False
        super()._sample(env)
        if free_push:
            body = env.data.body(self.product.name)
            if body.xmat.reshape(3, 3)[2, 2] < np.cos(np.radians(15)):
                raise InteractionFailure("free_push: carton tilting; replan required")
            extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(self.product.half_size)
            gap = body.xpos[2] - extent[2] - env.config.levels[self.product.level]
            supported = (
                self.last_forces["table"] + self.last_forces["shelf"] > 0.2
                and -0.003 <= gap <= self.config.insertion_clearance
            )
            self._free_push_support_loss = (
                0.0 if supported else self._free_push_support_loss + env.config.timestep
            )
            if self._free_push_support_loss > 0.10:
                raise InteractionFailure("free_push: lost physical surface support")
            if self.last_forces["hand"] >= 0.2:
                self._free_push_contact_seen = True
                self._free_push_loss_seconds = 0.0
            elif self._free_push_contact_seen:
                self._free_push_loss_seconds += env.config.timestep
        if self._base_guard is not None:
            self._base_guard.step(env)

    def _start_free_push(self):
        self._free_push_contact_seen = False
        self._free_push_loss_seconds = 0.0
        self._free_push_support_loss = 0.0
        self._free_push_recoveries = 0

    def _reacquire_push_contact(self, reference, push_origin, height, push_event):
        if self._free_push_recoveries >= 3:
            raise InteractionFailure("free_push: bounded contact recovery count exhausted")
        self._free_push_recoveries += 1
        self._stage("reacquire_push_contact")
        event = self.events[-1]
        push_event.setdefault("contact_recoveries", []).append(event)
        target = np.asarray(reference, dtype=float).copy()
        initial_x, held = float(target[0]), 0.0
        hand = self.controller
        recovery_force = min(2.5, self.config.push_force_target)
        event.update(
            maximum_seconds=1.5,
            maximum_advance_m=0.015,
            forward_unload_downforce_N=self.FORWARD_UNLOAD_DOWNFORCE_N,
            vertical_control="same bounded vertical-force admittance as free insertion",
        )
        for _ in range(int(1.5 / self.env.dt)):
            body = self.env.data.body(self.product.name)
            target[1] += float(
                np.clip(body.xpos[1] - target[1], -0.01 * self.env.dt, 0.01 * self.env.dt)
            )
            force = self.last_forces.get("hand", 0.0)
            scale = float(np.clip((recovery_force - force) / recovery_force, -1, 1))
            if scale > 0:
                scale *= self._ungrasped_forward_fraction()
            target[0] -= 0.01 * self.env.dt * scale
            contact = self.last_forces.get("hand_contact_position_world_m")
            measured_height = contact[2] if contact is not None else hand.position()[2]
            force_z = self.last_forces.get("hand_force_world_N", [0, 0, 0])[2]
            vertical_speed = -0.012 * force_z + 0.2 * (height - measured_height)
            target[2] = float(
                np.clip(
                    target[2] + np.clip(vertical_speed, -0.015, 0.015) * self.env.dt,
                    height - 0.005 + 1e-6,
                    height + 0.025 - 1e-6,
                )
            )
            if (
                abs(target[0] - initial_x) > 0.015
                or not push_origin[0] - self.env.config.depth - 0.15
                <= target[0]
                <= push_origin[0] + 0.03
                or abs(target[1] - self.product.target[1]) > 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("free_push: reacquisition reference bound")
            prediction = self._contact_prediction(
                target,
                allowed_object=self.product.name,
                alternatives=False,
                carrying=False,
                sliding=True,
            )
            if not prediction.feasible:
                event["rejection"] = prediction.report()
                raise InteractionFailure("free_push: reacquisition IK/path rejected")
            target = prediction.position.copy()
            if (
                abs(target[0] - initial_x) > 0.015
                or not push_origin[0] - self.env.config.depth - 0.15
                <= target[0]
                <= push_origin[0] + 0.03
                or abs(target[1] - self.product.target[1]) > 0.02
                or not height - 0.005 <= target[2] <= height + 0.025
            ):
                raise InteractionFailure("free_push: projected reacquisition reference bound")
            hand.last_target = target.copy()
            action = hand.joint_targets(prediction.q, 0)
            if not np.allclose(hand.command, prediction.q, atol=1e-9, rtol=0):
                raise InteractionFailure("free_push: unexpected second nominal rate projection")
            self._apply_tick(action)
            tracked_nominal = np.array(
                [self.tracker.last_report["nominal"][name] for name in [*hand.names, "Lift"]]
            )
            if not np.allclose(tracked_nominal, hand.command, atol=1e-9, rtol=0):
                raise InteractionFailure("free_push: hand and tracker nominal references diverged")
            good = self.last_forces["hand"] >= self.config.contact_threshold and (
                self.last_forces["table"] + self.last_forces["shelf"] > 0.2
            )
            held = held + self.env.dt if good else 0.0
            if held >= 0.04:
                event.update(
                    verified_contact_reacquisition=True, end_time=float(self.env.data.time)
                )
                self._free_push_loss_seconds = 0.0
                self._stage("finger_push")
                return target
        raise InteractionFailure("free_push: contact reacquisition timeout")

    def _contact_prediction(self, target, **kwargs):
        proposal = super()._contact_prediction(target, **kwargs)
        if not self.bounded_contact_ik:
            return proposal
        hand = self.controller
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        result = bounded_contact_prediction(hand, proposal, target, dt=self.env.dt, rates=rates)
        hand.last_prediction = result
        hand.last_ik_error = float(
            np.linalg.norm([result.position_error_m, 0.15 * result.orientation_error_rad])
        )
        self.events[-1].setdefault("bounded_feedback_samples", []).append(
            {
                "sim_time": float(self.env.data.time),
                "feasible": bool(result.feasible),
                **result.bounded_feedback,
            }
        )
        return result

    def _move_base(self, displacement, *, reason):
        """Move with native wheels while the load is released; no pose writes."""
        if self.active_grip is not None or self.carry_reference is not None:
            raise InteractionFailure("base_reposition: loaded base motion is not admitted")
        env = self.env
        settings = BaseMotionConfig(fixture_force_limit_N=min(25.0, self.config.max_force))
        self._stage("base_reposition_plan")
        screen = screen_base_translation(env, displacement, config=settings)
        self.events[-1].update(reason=reason, path_screening=screen)
        if not screen["accepted"]:
            raise InteractionFailure("base_reposition: whole-robot path rejected")
        body = env.model.body("mabel_chassis").id
        start = env.data.xpos[body].copy()
        if self._base_translation_origin is None:
            self._base_translation_origin = start[:2].copy()
        goal = start[:2] + np.asarray(displacement)
        rotation = env.data.xmat[body].reshape(3, 3)
        reference_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        guard = BaseContactGuard(
            env,
            config=settings,
            reference_yaw=reference_yaw,
            max_product_displacement_m=0.002,
        )
        swerve = NativeSwerve(env.model, env.data, settings)
        self._stage("base_reposition")
        event = self.events[-1]
        self.base_repositions.append(event)
        event.update(reason=reason, start_xy_m=start[:2].tolist(), goal_xy_m=goal.tolist())
        self._base_guard = guard
        velocity, held = np.zeros(6), 0.0
        try:
            for _ in range(int(settings.motion_timeout_s / env.dt)):
                rotation = env.data.xmat[body].reshape(3, 3)
                yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                yaw_error = float(
                    np.arctan2(np.sin(reference_yaw - yaw), np.cos(reference_yaw - yaw))
                )
                error = goal - env.data.xpos[body, :2]
                mujoco.mj_objectVelocity(
                    env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
                )
                arrived = (
                    np.linalg.norm(error) <= settings.position_tolerance_m
                    and abs(yaw_error) <= settings.yaw_tolerance_rad
                )
                twist = np.r_[1.2 * error - 0.3 * velocity[3:5], 2 * yaw_error - 0.4 * velocity[2]]
                if arrived:
                    twist[:] = 0
                if self._apply_tick(swerve.targets(twist, dt=env.dt)):
                    raise InteractionFailure(
                        "base_reposition: task ended before stocking validation"
                    )
                mujoco.mj_objectVelocity(
                    env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
                )
                stopped = np.linalg.norm(velocity[3:5]) < 0.003 and abs(velocity[2]) < 0.01
                held = held + env.dt if arrived and stopped else 0.0
                if held >= 0.3:
                    break
            else:
                raise InteractionFailure("base_reposition: measured pose/stop dwell timeout")
            for _ in range(int(0.5 / env.dt)):
                self._apply_tick(swerve.targets([0, 0, 0], dt=env.dt))
            final_error = float(np.linalg.norm(goal - env.data.xpos[body, :2]))
            if final_error > settings.position_tolerance_m:
                raise InteractionFailure("base_reposition: final position guard")
            event.update(verified_base_reposition=True, final_position_error_m=final_error)
            self.controller.last_target = self.controller.position().copy()
        finally:
            event.update(end_time=float(env.data.time), guard=guard.report())
            self._base_guard = None

    def _checked_cartesian_move(self, name, target, rotation, grip, *, carrying, object_goal=None):
        self._stage(name)
        hand = self.controller
        hand.active_rotation = rotation
        prediction = hand.predict(
            target,
            allowed_object=self.product.name if carrying else None,
            alternatives=False,
            carrying=carrying,
            grip=grip,
            path_mode="cartesian",
        )
        event = self.events[-1]
        event.update(object_goal_m=object_goal, ik_prediction=prediction.report())
        if not prediction.feasible or prediction.joint_path is None:
            raise InteractionFailure(f"{name}: measured-state Cartesian route rejected")
        if carrying and self.loaded_product_clearance_m > 0:
            clearance = carried_product_clearance(
                hand, prediction, self.product, self.env.products, self.loaded_product_clearance_m
            )
            event["measured_state_product_clearance_check"] = clearance
            if not clearance["accepted"]:
                raise InteractionFailure(f"{name}: measured-state product clearance rejected")
        hand.active_rotation = prediction.requested_rotation.copy()
        path = prediction.joint_path
        rates = np.r_[np.full(len(hand.names), self.config.max_joint_speed), 0.08]
        segment_times = np.max(np.abs(np.diff(path, axis=0)) / rates, axis=1)
        knots = np.r_[0.0, np.cumsum(segment_times)]
        duration = max(self.config.approach_seconds, 1.875 * knots[-1] * 1.05)
        if duration > self.config.max_motion_seconds:
            raise InteractionFailure(f"{name}: Cartesian duration exceeds motion bound")
        knots = knots / knots[-1] if knots[-1] > 1e-12 else np.linspace(0, 1, len(path))
        ticks = max(1, int(np.ceil(duration / self.env.dt)))
        event.update(planned_duration_s=duration, decision="execute_retained_cartesian_path")
        for step in range(ticks):
            t = (step + 1) / ticks
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            q = np.array([np.interp(blend, knots, column) for column in path.T])
            action = hand.joint_targets(q, grip)
            hand.last_target, _ = hand.ik.forward(hand.command, prediction.reference_qpos)
            if self._apply_tick(action):
                raise InteractionFailure(f"{name}: task ended before insertion validation")
        error = float(np.linalg.norm(hand.position() - target))
        event.update(
            end_time=float(self.env.data.time),
            measured_tracking_error_m=error,
        )
        if object_goal is not None:
            event["measured_object_position_error_m"] = float(
                np.linalg.norm(self.env.data.body(self.product.name).xpos - object_goal)
            )
        if error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: measured tracking error {error:.3f} m")

    def _grasp_predictions(self, product, box):
        heights = (product.half_size[2] / 3, 0.0, product.half_size[2] * 2 / 3)
        for height in heights:
            target = box + [0, 0, height]
            yield (
                target,
                self.controller.predict(target, allowed_object=product.name, check_path=False),
            )
        if self.route_admission_enabled:
            for offset in ((0.012, 0), (0, -0.012), (0, 0.012), (-0.012, 0)):
                for height in heights:
                    target = box + [*offset, height]
                    yield (
                        target,
                        self.controller.predict(
                            target, allowed_object=product.name, check_path=False
                        ),
                    )
            for _, rotation in PoseIK.orientations(
                self.controller.ROTATION, self.config.ik_orientation_span_deg
            ):
                self.controller.active_rotation = rotation
                for height in heights:
                    target = box + [0, 0, height]
                    yield (
                        target,
                        self.controller.predict(
                            target,
                            allowed_object=product.name,
                            check_path=False,
                            alternatives=False,
                        ),
                    )

    def _select_grasp(self, pending):
        if isinstance(self.controller, FingerPusher):
            self._change_hand(CartesianHand)
        by_name = {p.name: p for p in pending}
        selected = {}
        overall_deadline = time.monotonic() + self.replanner.config.wall_seconds

        def check_first(name):
            deadline = min(
                overall_deadline, time.monotonic() + self.replanner.config.root_wall_seconds
            )
            self.product = product = by_name[name]
            box = self.env.data.body(name).xpos.copy()
            for target, prediction in self._grasp_predictions(product, box):
                if time.monotonic() >= deadline:
                    return ActionCheck(False, "grasp_root_planning_budget")
                self._stage("grasp_candidate")
                self.events[-1].update(
                    ik_prediction=prediction.report(),
                    decision="select_candidate" if prediction.feasible else "try_next_candidate",
                    admission_scope="grasp_endpoint_only_not_loaded_route",
                )
                if prediction.feasible:
                    if self.route_admission_enabled:
                        route_check = admit_loaded_route(
                            self.env,
                            self.controller,
                            self.config,
                            product,
                            target,
                            prediction,
                            front_offset=0.06 if self.table_front else None,
                            product_clearance_m=self.loaded_product_clearance_m,
                            allow_lower_clearance=bool(
                                self.observer is not None and self.observer.completed
                            ),
                            home_displacement_xy=(
                                None
                                if self._base_translation_origin is None
                                else self.env.data.body("mabel_chassis").xpos[:2]
                                - self._base_translation_origin
                            ),
                        )
                        self.events[-1]["loaded_route_admission"] = route_check
                        self.events[-1]["admission_scope"] = "grasp_plus_nominal_loaded_corridor"
                        if not route_check["accepted"]:
                            self.events[-1]["decision"] = "try_next_loaded_route_candidate"
                            continue
                    selected[name] = (target, prediction)
                    return ActionCheck(
                        True,
                        "fresh_grasp_and_loaded_route_accepted"
                        if self.route_admission_enabled
                        else "fresh_grasp_endpoint_accepted",
                    )
            return ActionCheck(
                False,
                "bounded_grasp_or_loaded_route_search_rejected"
                if self.route_admission_enabled
                else "bounded_grasp_height_search_rejected",
            )

        observation = observe_row(self.env, pending, self.controller.position())
        plan = self.replanner.plan(observation, check_first)
        report = plan.report()
        report["sim_time"] = float(self.env.data.time)
        self.order_plans.append(report)
        self._stage("replan_order")
        self.events[-1].update(report)
        if not plan.first_action_checked:
            raise InteractionFailure(f"Order replan stopped: {plan.stop_reason}; no pick commanded")
        self.product = by_name[plan.order[0]]
        return selected[self.product.name]

    def episode_evidence(self):
        return {
            **super().episode_evidence(),
            "replanning": {
                "state_access": "privileged_mujoco_world_state",
                "boundary": "after_each_completed_carton",
                "first_action_admission": (
                    "grasp_plus_nominal_loaded_corridor"
                    if self.route_admission_enabled
                    else "grasp_endpoint_only"
                ),
                "plans": self.order_plans,
                "primitive_recovery_implemented": self.contact_recovery,
                "primitive_recovery_scope": "released_supported_contact_reacquisition_only",
                "task_level_jam_recovery_implemented": False,
                "loaded_corridor_enabled": self.loaded_route_enabled,
                "grasp_wrist_posture_rad": self.grasp_wrist_posture,
                "payload_wrist_load_compensation_enabled": self.payload_load_compensation,
                "payload_wrist_reference_reserve_rad": (
                    self.payload_wrist_reserve if self.payload_load_compensation else None
                ),
                "payload_lift_reference_reserve_m": (
                    self._tracking_config().lift_offset_limit
                    if self.payload_load_compensation
                    else None
                ),
                "loaded_product_clearance_m": self.loaded_product_clearance_m,
                "split_supported_insertion_enabled": self.split_insertion,
                "pusher_finger_search": (
                    "preferred_then_remaining_native_fingers"
                    if self.payload_load_compensation
                    else "preferred_only"
                ),
                "partial_insertion_boundary": (
                    "settled_support_unload_clear_base_recheck_then_deep_push"
                    if self.split_insertion
                    else None
                ),
                "loaded_route_admission_enabled": self.route_admission_enabled,
                "staging_support": self.FRONT_SUPPORT,
                "base_reposition_enabled": self.base_reposition_enabled,
                "base_repositions": self.base_repositions,
                "finger_load_compensation_enabled": self.finger_load_compensation,
                "peak_finger_load_offset_rad": self.peak_finger_load_offset,
                "bounded_contact_ik_enabled": self.bounded_contact_ik,
                "contact_recovery_enabled": self.contact_recovery,
                "pusher_class": self.PUSHER_CLASS.__name__,
            },
        }
