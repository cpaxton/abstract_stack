"""Opt-in object-level replanning around the existing guarded dense controller."""

import time

import numpy as np

from abstract_stack.dense_interactions import DenseStockingScript, FingerPusher
from abstract_stack.ik import PoseIK
from abstract_stack.interactions import CartesianHand, InteractionFailure
from abstract_stack.loaded_route import admit_loaded_route, object_tool_goal, plan_loaded_route
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
    recovery, stochastic MCTS and full belief-space planning remain future work.
    """

    def __init__(
        self,
        *args,
        search_config=None,
        loaded_route=False,
        route_admission=False,
        table_front=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.config.adaptive_grasps:
            raise ValueError("Receding order search requires adaptive grasps")
        self.replanner = RowReplanner(search_config)
        self.order_plans = []
        self.loaded_route_enabled = loaded_route
        self.table_front = table_front
        if table_front and not loaded_route:
            raise ValueError("Table-front staging requires the loaded-route executor")
        if table_front:
            self.FRONT_SUPPORT = "table"
            self.CONTACT_PROTOCOL = "table_front_release_then_selected_finger_slide_to_shelf"
        self.PUSHER_APPROACH_PATH_MODE = "cartesian" if loaded_route else "joint"
        self.route_admission_enabled = route_admission
        if route_admission and not loaded_route:
            raise ValueError("Loaded-route admission requires the loaded-route executor")

    def _front_center_x(self, extent):
        if self.table_front:
            return self.env.config.front_x + 0.06
        return super()._front_center_x(extent)

    def _move(self, name, target, grip=0.0, seconds=None, **kwargs):
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
                goal = np.array([-0.30, -0.20, height])
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

    def _plan_pusher(self, precontact, end):
        if self.loaded_route_enabled:
            self._move("pusher_high_clear", [-0.30, -0.20, 1.35])
        return super()._plan_pusher(precontact, end)

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
                            allow_lower_clearance=bool(
                                self.observer is not None and self.observer.completed
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
                "primitive_recovery_implemented": False,
                "loaded_corridor_enabled": self.loaded_route_enabled,
                "loaded_route_admission_enabled": self.route_admission_enabled,
                "staging_support": self.FRONT_SUPPORT,
            },
        }
