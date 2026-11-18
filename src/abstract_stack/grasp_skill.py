"""An isolated actuator-driven grasp/lift/hold task using the same stocking skill."""

import copy
import hashlib
import json
import time

import mujoco
import numpy as np

from abstract_stack.grasping import object_aligned_rotation, open_hand_clearance
from abstract_stack.interactions import CartesianHand, InteractionFailure, StockingScript


def loaded_staging_reachable(env, controller, config, product, *, base_xy=(0.0, 0.0)):
    """Planning-only forecast of the loaded shelf staging reachability.

    Uses the **actual** measured carried state (grasp offset and world-aligned
    payload orientation), shifting both the base and the carried carton together
    to model a base relocation with the load retained. This mirrors
    ``StockingScript._transfer_and_place_front``'s bounded shelf-front staging
    search; live execution still rechecks every segment and retains all guards.
    """
    from abstract_stack.env import PlanningState

    body = env.data.body(product.name)
    offset = controller.position() - body.xpos
    palm_rotation = env.data.xmat[controller.palm].reshape(3, 3)
    active_rotation = np.eye(3) @ body.xmat.reshape(3, 3).T @ palm_rotation

    placement = np.asarray(product.target, dtype=float)
    placement[0] = env.config.front_x - product.half_size[0] - 0.006

    data = mujoco.MjData(env.model)
    mujoco.mj_copyData(data, env.model, env.data)
    shift = np.asarray(base_xy, dtype=float)
    base_adr = int(env.model.joint("base_root").qposadr[0])
    carton_adr = int(env.model.joint(f"{product.name}_joint").qposadr[0])
    data.qpos[base_adr : base_adr + 2] += shift
    data.qpos[carton_adr : carton_adr + 2] += shift
    mujoco.mj_forward(env.model, data)

    hand = CartesianHand(PlanningState(env.model, data, env.dt), config)
    hand.command = controller.command.copy()
    hand.active_rotation = active_rotation.copy()

    staging = placement + offset + [0, 0, config.lift_height]
    staging[0] = env.config.front_x + 0.20 + offset[0]
    staging[2] = max(staging[2], hand.position()[2])
    for front_clearance in (0.20, 0.16, 0.12, 0.08):
        proposed = staging.copy()
        proposed[0] = env.config.front_x + front_clearance + offset[0]
        prediction = hand.predict(
            proposed,
            allowed_object=product.name,
            alternatives=False,
            grip=config.grip,
        )
        if prediction.feasible and prediction.minimum_seconds * 1.05 <= config.max_motion_seconds:
            return True
    return False


def empty_reachable(
    env,
    controller,
    config,
    target,
    *,
    base_xy=(0.0, 0.0),
    alternatives=True,
    active_rotation=None,
    path_mode="joint",
):
    """Planning-only: is an empty-hand target reachable from a candidate base?

    Shifts only the base in scratch; the target is a world-frame waypoint (for
    example the pregrasp approach point). ``alternatives``, ``active_rotation``
    and ``path_mode`` mirror the caller's execution posture so the screen is no
    more lenient than the real move. This is a kinematic reachability screen,
    not a tracking or contact guarantee.
    """
    from abstract_stack.env import PlanningState

    data = mujoco.MjData(env.model)
    mujoco.mj_copyData(data, env.model, env.data)
    shift = np.asarray(base_xy, dtype=float)
    base_adr = int(env.model.joint("base_root").qposadr[0])
    data.qpos[base_adr : base_adr + 2] += shift
    mujoco.mj_forward(env.model, data)

    hand = CartesianHand(PlanningState(env.model, data, env.dt), config)
    hand.command = controller.command.copy()
    hand.abduction_targets = dict(controller.abduction_targets)
    hand.open_posture_overrides = dict(controller.open_posture_overrides)
    if active_rotation is not None:
        hand.active_rotation = np.asarray(active_rotation, dtype=float).copy()
    prediction = hand.predict(
        np.asarray(target, dtype=float),
        alternatives=alternatives,
        grip=0,
        check_path=False,
        path_mode=path_mode,
    )
    return prediction.feasible


def select_grasp(script, pending):
    """Search bounded object-frame grasp offsets with native open-hand clearance."""
    env, hand = script.env, script.controller
    base = env.data.body("mabel_chassis").xpos
    offsets = ((0, 0), (0.004, -0.004), (0.008, -0.004), (0.004, -0.008), (0.008, -0.008))
    for product in sorted(
        pending, key=lambda p: np.linalg.norm(env.data.body(p.name).xpos[:2] - base[:2])
    ):
        script.product = product
        body = env.data.body(product.name)
        rotation = body.xmat.reshape(3, 3).copy()
        try:
            grasp_rotation, yaw = object_aligned_rotation(rotation, hand.ROTATION)
        except ValueError as error:
            script._stage("grasp_candidate")
            script.events[-1].update(decision="unsupported_grasp_entry", reason=str(error))
            continue
        for height in (product.half_size[2] / 3, 0.0, product.half_size[2] * 2 / 3):
            for x, y in offsets:
                # Offset directions follow the selected symmetric grasp frame.
                local = np.array([x, y, 0])
                turn = grasp_rotation @ hand.ROTATION.T
                target = body.xpos + [0, 0, height] + turn @ local
                hand.active_rotation = grasp_rotation.copy()
                prediction = hand.predict(
                    target, allowed_object=product.name, alternatives=False, check_path=False
                )
                clearance = open_hand_clearance(hand, prediction, product.name)
                script._stage("grasp_candidate")
                script.events[-1].update(
                    ik_prediction=prediction.report(),
                    object_aligned_yaw_deg=yaw,
                    object_frame_offset_m=[x, y, height],
                    open_hand_clearance=clearance,
                    minimum_open_clearance_m=0.002,
                    decision="try_next_candidate",
                )
                if not prediction.feasible:
                    continue
                if clearance["minimum_distance_m"] < 0.002:
                    script.events[-1]["decision"] = "reject_open_hand_object_overlap"
                    continue
                script.events[-1]["decision"] = "select_candidate"
                return target.copy(), prediction
    raise InteractionFailure("No checked object-aligned grasp candidate; no grasp motion commanded")


def plan_pregrasp_route(script, target, rotation):
    """Check empty-hand transit around the shelf roof on independent data.

    A reachable pregrasp endpoint does not certify the route from the raised
    hand after a base relocation. Recheck each segment again before execution.
    """
    from abstract_stack.interactions import CartesianHand

    env, live = script.env, script.controller
    target = np.asarray(target)
    start = live.position().copy()
    script._stage("pregrasp_route")
    event = script.events[-1]
    event.update(candidates=[], wall_budget_seconds=12.0, empty_hand=True)
    routes = [[]]
    for clearance in (0.20, 0.26):
        front = env.config.front_x + clearance
        routes.append(
            [
                np.array([front, start[1], start[2]]),
                np.array([front, target[1], target[2]]),
            ]
        )
        sign = -1 if target[1] < env.config.center_y else 1
        outside = target[1] + sign * 0.08
        routes.append(
            [
                np.array([front, start[1], start[2]]),
                np.array([front, outside, target[2]]),
                np.array([target[0], outside, target[2]]),
            ]
        )
    started = time.monotonic()
    for waypoints in routes:
        if time.monotonic() - started > 12:
            break
        clone = CartesianHand(env.clone_for_planning(), script.config)
        clone.command = live.command.copy()
        clone.abduction_targets = dict(live.abduction_targets)
        clone.open_posture_overrides = dict(live.open_posture_overrides)
        clone.active_rotation = np.asarray(rotation).copy()
        row = dict(waypoints_m=[p.tolist() for p in waypoints], predictions=[])
        event["candidates"].append(row)
        for point in [*waypoints, target]:
            prediction = clone.predict(
                point, alternatives=False, grip=0, carrying=False, path_mode="cartesian"
            )
            row["predictions"].append(prediction.report())
            if not prediction.feasible:
                break
            script._planning_pose(clone, prediction)
        else:
            event.update(
                selected_waypoints_m=row["waypoints_m"],
                wall_seconds=time.monotonic() - started,
                decision="execute_rechecked_empty_cartesian_route",
            )
            return waypoints
    event.update(wall_seconds=time.monotonic() - started, decision="hold_for_base_or_route_replan")
    raise InteractionFailure("pregrasp_route: no checked empty-hand route around shelf fixtures")


class GraspSkillProbe(StockingScript):
    """Pick one declared carton in the original inventory, without stocking credit."""

    TASK_SCOPE = "grasp_lift_hold"

    def __init__(self, *args, product, entry_checkpoint=None, **kwargs):
        self.entry_checkpoint = entry_checkpoint
        if entry_checkpoint is not None and (
            entry_checkpoint.controller.get("kind") != "stocking_skill_entry"
            or entry_checkpoint.context["skill_entry"]["skill"] != "grasp_lift"
            or entry_checkpoint.context["skill_entry"]["active_product"] != product
            or entry_checkpoint.controller["active_grip"] is not None
            or entry_checkpoint.controller["carry_reference"] is not None
        ):
            raise ValueError("Grasp probe requires a matching empty-hand grasp entry checkpoint")
        super().__init__(*args, **kwargs)
        self.target_product = next(p for p in self.env.products if p.name == product)
        self.grasp_skill_verified = False

    def _prepare_episode(self):
        if self.entry_checkpoint is None:
            return super()._prepare_episode()
        checkpoint = self.entry_checkpoint
        directory = self.directory / "initial-scene"
        directory.mkdir()
        scene = copy.deepcopy(checkpoint.scene)
        scene["continuation"] = dict(
            checkpoint=str(checkpoint.directory),
            manifest_sha256=hashlib.sha256(
                (checkpoint.directory / "manifest.json").read_bytes()
            ).hexdigest(),
            fresh_stocking_credit=0,
        )
        (directory / "manifest.json").write_text(json.dumps(scene, indent=2) + "\n")

    def _new_cartesian_hand(self):
        hand = super()._new_cartesian_hand()
        if self.entry_checkpoint is not None:
            state = self.entry_checkpoint.controller
            hand.command = np.array(
                [state["hand_nominal_subset"][n] for n in [*hand.names, "Lift"]]
            )
            hand.active_rotation = np.array(state["active_rotation_world"])
        return hand

    def _new_tracker(self):
        if self.entry_checkpoint is None:
            return super()._new_tracker()
        from abstract_stack.mabel_controller import MabelController, TrackingConfig

        state = self.entry_checkpoint.controller
        return MabelController(
            self.env.model,
            self.env.data,
            TrackingConfig(**state["tracking_config"]),
            nominal_references=state["nominal_subset"],
        )

    def _new_observer(self):
        observer = super()._new_observer()
        if self.entry_checkpoint is not None:
            history = self.entry_checkpoint.context["observer_history"]
            for name, value in copy.deepcopy(history).items():
                if name == "seen":
                    value = {tuple(item) for item in value}
                if name == "completed":
                    value = set(value)
                setattr(observer, name, value)
        return observer

    def _execute_task(self, result):
        selected = self._select_grasp([self.target_product])
        grip = self._grasp_and_lift(selected)
        self._stage("grasp_hold")
        target = self.controller.position().copy()
        for _ in range(int(np.ceil(0.5 / self.env.dt))):
            self._tick(target, grip)
            if (
                not self.last_forces["opposing_contacts"]
                or self.last_forces["hand"] < 0.2
                or self.last_forces["table"] > 0.15
            ):
                raise InteractionFailure(
                    "grasp_hold: loaded opposing contact or table clearance lost"
                )
        if self.observer.incidents:
            raise InteractionFailure("grasp_hold: an inventory product tipped or dropped")
        self.grasp_skill_verified = True
        self.events[-1].update(
            verified_grasp_skill=True,
            required_hold_seconds=0.5,
            measured_hand_normal_N=self.last_forces["hand"],
            measured_table_normal_N=self.last_forces["table"],
        )

    def episode_evidence(self):
        return dict(
            kind="isolated_grasp_lift_hold",
            scheduled_grasp_products=[self.target_product.name],
            grasp_skill_verified=self.grasp_skill_verified,
            full_stocking_task_success=False,
            entry_scope="fresh_reset" if self.entry_checkpoint is None else "retained_skill_entry",
            scope="One physical pickup, not transfer/placement/retreat or a whole shelf",
        )
