"""Perfect-state carton grasp geometry, independent of live actuation/evaluation."""

import mujoco
import numpy as np


def object_aligned_rotation(object_rotation, canonical):
    """Choose the nearest upright carton-symmetric yaw, bounded to +/-90 deg.

    This specifies a grasp orientation from object geometry; it does not relax
    IK orientation tolerance or rotate any live body.
    """
    rotation = np.asarray(object_rotation, float)
    canonical = np.asarray(canonical, float)
    for value in (rotation, canonical):
        if (
            value.shape != (3, 3)
            or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(value), 1, atol=1e-7)
        ):
            raise ValueError("Grasp orientation requires proper rotations")
    if rotation[2, 2] < np.cos(np.radians(5)):
        raise ValueError("Upright carton grasp skill requires tilt <=5 degrees")
    yaw = (np.arctan2(rotation[1, 0], rotation[0, 0]) + np.pi / 2) % np.pi - np.pi / 2
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return turn @ canonical, float(np.degrees(yaw))


def open_hand_clearance(hand, prediction, product_name):
    """Measure native colliding hand geometry against the active carton in scratch."""
    model, scratch = hand.model, hand.ik.scratch
    hand.ik.forward(prediction.q, prediction.reference_qpos)
    mujoco.mj_fwdPosition(model, scratch)
    product = model.geom(f"{product_name}_geom").id
    best, nearest = 0.2, None
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in hand.ik.hand or not (
            model.geom_contype[geom] & model.geom_conaffinity[product]
            or model.geom_contype[product] & model.geom_conaffinity[geom]
        ):
            continue
        distance = float(mujoco.mj_geomDistance(model, scratch, geom, product, 0.2, None))
        if distance < best:
            best = distance
            nearest = model.geom(geom).name or f"geom_{geom}"
    return dict(minimum_distance_m=best, nearest_hand_geom=nearest)


def timed_joint_path(path, rates):
    """Time a retained path by segment rate cost, including a quintic peak factor."""
    path, rates = np.asarray(path, float), np.asarray(rates, float)
    if (
        path.ndim != 2
        or len(path) < 2
        or rates.shape != (path.shape[1],)
        or not np.isfinite(path).all()
        or not np.isfinite(rates).all()
        or np.any(rates <= 0)
    ):
        raise ValueError("Invalid retained path or native coordinate rates")
    cost = np.max(np.abs(np.diff(path, axis=0)) / rates, axis=1)
    keep = np.r_[True, cost > 1e-12]
    path = path[keep]
    if len(path) == 1:
        return np.repeat(path, 2, axis=0), np.array([0.0, 1.0]), 0.0
    times = np.r_[0.0, np.cumsum(cost[cost > 1e-12])]
    return path, times / times[-1], float(1.875 * times[-1])
