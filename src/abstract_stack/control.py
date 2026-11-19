"""Actuator targets for MABEL's position servos, tendon lift and velocity wheels."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np


def stationary_targets(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Hold current actuator lengths for position servos; stop velocity servos.

    Position bias is -kp * length, velocity bias is -kv * velocity. Inspecting
    these coefficients avoids treating MABEL's wheel speeds as joint angles or
    losing its tendon-driven lift (which spans two slide joints).
    """
    targets = np.zeros(model.nu)
    position = model.actuator_biasprm[:, 1] < 0
    targets[position] = data.actuator_length[position]
    return targets


def validate_targets(
    model: mujoco.MjModel,
    current: np.ndarray,
    action: Mapping[str, float] | np.ndarray,
) -> np.ndarray:
    """Validate a complete vector or named partial update, without changing state."""
    if isinstance(action, Mapping):
        targets = current.copy()
        for name, value in action.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Unknown actuator: {name}")
            if np.iscomplexobj(value):
                raise ValueError("Actuator targets must be real")
            targets[aid] = value
    else:
        values = np.asarray(action)
        if np.iscomplexobj(values) or (
            values.dtype.kind == "O" and any(np.iscomplexobj(value) for value in values.flat)
        ):
            raise ValueError("Actuator targets must be real")
        targets = values.astype(float)
    if targets.shape != (model.nu,) or not np.isfinite(targets).all():
        raise ValueError(f"Action must contain {model.nu} finite actuator targets")
    for aid in np.flatnonzero(model.actuator_ctrllimited):
        low, high = model.actuator_ctrlrange[aid]
        if not low <= targets[aid] <= high:
            raise ValueError(
                f"{model.actuator(aid).name}: {targets[aid]} is outside [{low}, {high}]"
            )
    return targets
