"""Measured-grasp corridor search with scratch-only rigid-load forecasts."""

from dataclasses import dataclass
from itertools import product as combinations

import mujoco
import numpy as np

from abstract_stack.interactions import CartesianHand


def planning_hand(snapshot, controller, config):
    """Clone nominal planning constraints without aliasing their live mapping."""
    hand = CartesianHand(snapshot, config)
    hand.command = controller.command.copy()
    for index, (lower, upper) in controller.ik.reference_ranges.items():
        hand.ik.constrain_reference_range(controller.ik.names[index], lower, upper)
    for index, value in controller.ik.posture_constraints.items():
        hand.ik.constrain_posture(controller.ik.names[index], value)
    return hand


def object_tool_goal(hand, product_name, object_goal, rotation_goal):
    body = hand.data.body(product_name)
    rotation = body.xmat.reshape(3, 3)
    relative = rotation.T @ (hand.position() - body.xpos)
    palm = hand.data.xmat[hand.palm].reshape(3, 3)
    return (
        np.asarray(object_goal) + rotation_goal @ relative,
        rotation_goal @ rotation.T @ palm,
    )


def advance_forecast(hand, prediction):
    """Advance only a planning clone, including its forecast carried-object pose."""
    hand.ik.forward(prediction.q, prediction.reference_qpos)
    hand.data.qpos[:] = hand.ik.scratch.qpos
    mujoco.mj_forward(hand.model, hand.data)
    hand.command = prediction.q.copy()
    hand.active_rotation = prediction.requested_rotation.copy()


def premature_grasp_contacts(hand, product, prediction):
    """Reject target penetration before the tool enters the bounded grasp region.

    Active-object contact is allowed by ordinary grasp IK. This additional
    admission screen distinguishes a side grasp from catching a top edge with
    the thumb during descent. It predicts geometry, not a contact force.
    """
    body = hand.data.body(product.name)
    center, rotation = body.xpos.copy(), body.xmat.reshape(3, 3).copy()
    bounds = np.asarray(product.half_size) + [0.002, 0.002, 0]
    start = hand.ik.measured_q()
    samples = max(31, int(np.ceil(prediction.minimum_seconds / 0.04)))
    for fraction in np.linspace(0, 1, samples):
        q = start + fraction * (prediction.q - start)
        point, _ = hand.ik.forward(q, prediction.reference_qpos)
        if np.all(np.abs(rotation.T @ (point - center)) <= bounds):
            continue
        contacts = [
            info
            for info in hand.ik._contacts(q, prediction.reference_qpos, None).values()
            if f"{product.name}_geom" in info["geoms"] and info["distance_m"] < 0.001
        ]
        if contacts:
            return {
                "fraction": float(fraction),
                "tool_position_m": point.tolist(),
                "contacts": contacts,
            }
    return None


@dataclass(frozen=True)
class LoadedRoute:
    accepted: bool
    object_waypoints: tuple
    candidates: tuple
    object_rotations: tuple = ()


def carried_product_clearance(hand, prediction, active, products, margin_m):
    """Check the retained carried-box sweep against every other product in scratch."""
    if not np.isfinite(margin_m) or not 0 < margin_m <= 0.03:
        raise ValueError("Carried-product clearance must be positive and at most 30 mm")
    if prediction.joint_path is None or prediction.carried_object != active.name:
        raise ValueError("Product clearance requires a retained rigid-carry forecast")
    model, ik = hand.model, hand.ik
    active_geom = model.geom(f"{active.name}_geom").id
    others = [(p.name, model.geom(f"{p.name}_geom").id) for p in products if p.name != active.name]
    minimum, samples = margin_m, 0
    for index, q in enumerate(prediction.joint_path):
        ik.forward(q, prediction.reference_qpos)
        mujoco.mj_fwdPosition(model, ik.scratch)
        for name, geom in others:
            distance = float(
                mujoco.mj_geomDistance(model, ik.scratch, active_geom, geom, margin_m, None)
            )
            samples += 1
            minimum = min(minimum, distance)
            if distance < margin_m - 1e-9:
                return dict(
                    accepted=False,
                    required_clearance_m=margin_m,
                    minimum_sampled_distance_m=minimum,
                    pair=[active.name, name],
                    retained_waypoint=index,
                    checked_pairs=samples,
                )
    return dict(
        accepted=True,
        required_clearance_m=margin_m,
        minimum_sampled_distance_m=minimum,
        checked_pairs=samples,
        distance_reporting="capped at required clearance; sampled kinematic forecast",
    )


def _inventory_transit_height(hand, products, active, end_xy, margin_m, minimum_z):
    """Clear products inside the measured XY transfer envelope before lateral motion."""
    body = hand.data.body(active.name)
    extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(active.half_size)
    lower = np.minimum(body.xpos[:2], end_xy) - extent[:2] - margin_m
    upper = np.maximum(body.xpos[:2], end_xy) + extent[:2] + margin_m
    height = max(float(body.xpos[2]), minimum_z)
    for other in products:
        if other.name == active.name:
            continue
        neighbor = hand.data.body(other.name)
        other_extent = np.abs(neighbor.xmat.reshape(3, 3)) @ np.asarray(other.half_size)
        if np.all(neighbor.xpos[:2] + other_extent[:2] >= lower) and np.all(
            neighbor.xpos[:2] - other_extent[:2] <= upper
        ):
            height = max(height, float(neighbor.xpos[2] + other_extent[2] + extent[2] + margin_m))
    return height


def plan_loaded_route(
    env,
    controller,
    config,
    product,
    rotation_goal,
    *,
    grip,
    start_state=None,
    front_offset=None,
    product_clearance_m=0.0,
    transit_yaws_degrees=(0, -20, 20),
):
    """Search a low front corridor; refine every stage from fresh measured state.

    This predicts a rigid measured grasp, not contact forces, retention or tracking.
    A hypothetical corridor updates independent MjData only. Execution must check
    each segment again; returned poses are not cached execution certificates.
    """
    reports = []
    if (
        not 1 <= len(transit_yaws_degrees) <= 7
        or not np.isfinite(transit_yaws_degrees).all()
        or any(abs(yaw) > 60 for yaw in transit_yaws_degrees)
    ):
        raise ValueError("Use at most seven bounded carried-object yaw alternatives")
    if not np.isfinite(product_clearance_m) or not 0 <= product_clearance_m <= 0.03:
        raise ValueError("Loaded route product margin must be between zero and 30 mm")
    front_offset = -config.front_staging_depth if front_offset is None else front_offset
    middle_offset = (
        (0.10 + front_offset) / 2 if front_offset >= 0 else -min(0.04, -front_offset * 0.6)
    )
    for yaw_degrees, clearance in combinations(transit_yaws_degrees, (0.03, 0.01, 0.06)):
        snapshot = env.clone_for_planning()
        if start_state is not None:
            mujoco.mj_copyData(snapshot.data, snapshot.model, start_state.data)
        hand = planning_hand(snapshot, controller, config)
        z = env.config.levels[product.level] + product.half_size[2] + clearance
        goals = tuple(
            (env.config.front_x + offset, product.target[1], z)
            for offset in (0.10, middle_offset, front_offset)
        )
        angle = np.radians(yaw_degrees)
        c, s = np.cos(angle), np.sin(angle)
        transit_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation_goal
        rotations = (transit_rotation, transit_rotation, rotation_goal)
        if product_clearance_m > 0:
            end_xy = (env.config.front_x + front_offset, product.target[1])
            transit_z = _inventory_transit_height(
                hand, env.products, product, end_xy, product_clearance_m, z
            )
            current = hand.data.body(product.name).xpos.copy()
            goals = (
                (float(current[0]), float(current[1]), transit_z),
                (env.config.front_x + 0.10, product.target[1], transit_z),
                (*end_xy, transit_z),
                (*end_xy, z),
            )
            rotations = (rotation_goal, transit_rotation, rotation_goal, rotation_goal)
        stages = []
        for goal, object_rotation in zip(goals, rotations, strict=True):
            target, rotation = object_tool_goal(hand, product.name, goal, object_rotation)
            hand.active_rotation = rotation
            prediction = hand.predict(
                target,
                alternatives=False,
                allowed_object=product.name,
                carrying=True,
                grip=grip,
                path_mode="cartesian",
            )
            accepted = (
                prediction.feasible
                and prediction.minimum_seconds * 1.05 <= config.max_motion_seconds
            )
            stages.append(
                {
                    "object_goal_m": goal,
                    "object_rotation": object_rotation.tolist(),
                    "prediction": prediction.report(),
                    "accepted": accepted,
                }
            )
            if accepted and product_clearance_m > 0:
                clearance_check = carried_product_clearance(
                    hand, prediction, product, env.products, product_clearance_m
                )
                stages[-1]["product_clearance_check"] = clearance_check
                accepted = clearance_check["accepted"]
                stages[-1]["accepted"] = accepted
            if not accepted:
                break
            advance_forecast(hand, prediction)
        report = {
            "clearance_m": clearance,
            "transit_object_yaw_degrees": yaw_degrees,
            "stages": stages,
            "accepted": len(stages) == len(goals) and all(s["accepted"] for s in stages),
            "required_product_clearance_m": product_clearance_m,
            "vertical_extraction_before_transit": product_clearance_m > 0,
        }
        reports.append(report)
        if report["accepted"]:
            return LoadedRoute(
                True,
                goals,
                tuple(reports),
                tuple(tuple(tuple(map(float, row)) for row in rotation) for rotation in rotations),
            )
    return LoadedRoute(False, (), tuple(reports))


def admit_loaded_route(
    env,
    controller,
    config,
    product,
    grasp_target,
    grasp_prediction,
    *,
    front_offset=None,
    allow_lower_clearance=False,
    home_displacement_xy=None,
    product_clearance_m=0.0,
    transit_yaws_degrees=(0, -20, 20),
):
    """Necessary geometric forecast before pickup; no grasp/contact guarantee.

    The assumed closure is explicitly 0.4 of native finger range. Actual contact
    may close differently, so live execution still replans with its measured
    object/tool transform and retains all force/slip guards.
    """
    hand = planning_hand(env.clone_for_planning(), controller, config)
    stages = []
    home_displacement = (
        np.zeros(2)
        if home_displacement_xy is None
        else np.asarray(home_displacement_xy, dtype=float)
    )
    if home_displacement.shape != (2,) or not np.isfinite(home_displacement).all():
        raise ValueError("Home waypoint displacement must be a finite XY vector")
    empty_goals = [
        ("clear_high", np.array([-0.30, -0.20, 1.35])),
        ("retract", np.array([-0.44, -0.20, 1.0])),
        (
            "raise",
            np.array([-0.44, -0.20, env.data.body(product.name).xpos[2] + config.pregrasp_height]),
        ),
        ("pregrasp", np.asarray(grasp_target) + [0, 0, config.pregrasp_height]),
        ("approach", np.asarray(grasp_target)),
    ]
    for name, target in empty_goals:
        if name in ("clear_high", "retract", "raise"):
            target = target.copy()
            target[:2] += home_displacement
        if name == "approach":
            hand.active_rotation = grasp_prediction.requested_rotation.copy()
        alternatives = name != "approach"
        candidates = (
            [np.array([-0.30, -0.20, z]) for z in (1.35, 1.25, 1.20)]
            if name == "clear_high" and allow_lower_clearance
            else [target]
        )
        if name == "clear_high" and allow_lower_clearance:
            for candidate in candidates:
                candidate[:2] += home_displacement
        accepted = False
        for candidate in candidates:
            prediction = hand.predict(
                candidate,
                alternatives=alternatives,
                carrying=False,
                allowed_object=product.name if name == "approach" else None,
                path_mode="cartesian"
                if name == "clear_high" and allow_lower_clearance
                else "joint",
            )
            accepted = (
                prediction.feasible
                and prediction.minimum_seconds * 1.05 <= config.max_motion_seconds
            )
            stage = {"phase": name, "prediction": prediction.report(), "accepted": accepted}
            if accepted and name == "approach":
                premature = premature_grasp_contacts(hand, product, prediction)
                if premature is not None:
                    accepted = False
                    stage.update(accepted=False, premature_target_contact=premature)
            stages.append(stage)
            if accepted:
                break
        if not accepted:
            return {"accepted": False, "stages": stages, "failed_phase": name, "assumed_grip": 0.4}
        advance_forecast(hand, prediction)
    for name, value in hand.finger_targets(0.4).items():
        hand.data.qpos[int(hand.model.joint(name).qposadr[0])] = value
    mujoco.mj_forward(hand.model, hand.data)
    rotation_goal = hand.data.body(product.name).xmat.reshape(3, 3).copy()
    for name, rise in (("lift", config.lift_height), ("carry_clear", config.carry_clearance)):
        body = hand.data.body(product.name)
        hand.active_rotation = (
            rotation_goal @ body.xmat.reshape(3, 3).T @ hand.data.xmat[hand.palm].reshape(3, 3)
        )
        target = (
            np.asarray(grasp_target) + [0, 0, rise]
            if name == "lift"
            else hand.position() + [0, 0, rise]
        )
        prediction = hand.predict(
            target,
            alternatives=False,
            allowed_object=product.name,
            carrying=True,
            grip=0.4,
        )
        accepted = (
            prediction.feasible and prediction.minimum_seconds * 1.05 <= config.max_motion_seconds
        )
        stages.append({"phase": name, "prediction": prediction.report(), "accepted": accepted})
        if not accepted:
            return {"accepted": False, "stages": stages, "assumed_grip": 0.4}
        advance_forecast(hand, prediction)
    route = plan_loaded_route(
        env,
        hand,
        config,
        product,
        rotation_goal,
        grip=0.4,
        start_state=hand.env,
        front_offset=front_offset,
        product_clearance_m=product_clearance_m,
        transit_yaws_degrees=transit_yaws_degrees,
    )
    return {
        "accepted": route.accepted,
        "stages": stages,
        "corridor_candidates": route.candidates,
        "assumed_grip": 0.4,
        "scope": "kinematic necessary condition, not a physical grasp guarantee",
    }
