"""Bounded corrective actions for guarded, stationary MABEL task phases."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import mujoco
import numpy as np

from abstract_stack.ik import rotation_error
from abstract_stack.sim_sensors import ContactSensorSuite
from abstract_stack.whole_body_control import PalmGoal, StationaryWholeBodyController


def _vector(value, size):
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite vector of length {size}")
    vector.setflags(write=False)
    return vector


def _clip_norm(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


@dataclass(frozen=True)
class PalmDelta:
    """World translation in metres and world rotation vector in radians."""

    translation_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3))
        object.__setattr__(self, "rotation_rad", _vector(self.rotation_rad, 3))


@dataclass(frozen=True)
class CorrectionLimits:
    translation_m: float = 0.005
    rotation_rad: float = 0.04

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Correction limits must be finite and positive")


@dataclass(frozen=True)
class WristCompliance:
    """Offset gains per world axis at the wrist; zero gain holds an axis fixed."""

    translation_m_per_N: np.ndarray
    rotation_rad_per_Nm: np.ndarray
    desired_wrench_world: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self):
        for name, size in (
            ("translation_m_per_N", 3),
            ("rotation_rad_per_Nm", 3),
            ("desired_wrench_world", 6),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size))
        if np.any(self.translation_m_per_N < 0) or np.any(self.rotation_rad_per_Nm < 0):
            raise ValueError("Wrist compliance gains must be nonnegative")

    def correction(self, wrench_world):
        error = _vector(wrench_world, 6) - self.desired_wrench_world
        return PalmDelta(
            self.translation_m_per_N * error[:3],
            self.rotation_rad_per_Nm * error[3:],
        )


class WholeBodyTaskAdapter:
    """Execute one guarded native interval from a task goal and optional deltas.

    The caller retains phase completion, collision/contact semantics and guard
    ownership. A guard exception propagates, including if physics already
    advanced; the caller must treat that action as interrupted.
    """

    def __init__(self, env, *, guard, controller=None, limits=None):
        if not callable(guard):
            raise ValueError("A per-physics-step task guard is required")
        self.env = env
        self.guard = guard
        self.controller = controller or StationaryWholeBodyController(env.model, env.data)
        if self.controller.model is not env.model or self.controller.data is not env.data:
            raise ValueError("The coordinator must be bound to this environment")
        self.limits = limits or CorrectionLimits()
        self.sensors = ContactSensorSuite.mabel(env.model)

    def contact_wrenches_world(self):
        """Native contact-only force/torque ON each distal subtree at its wrist."""
        self._require_current_env()
        data = self.env.data
        sample = self.sensors.sample(data)
        result = {}
        for wrist, body, wrench in zip(
            self.sensors.wrists, self.sensors.bodies, sample["contact_wrench"], strict=True
        ):
            rotation = data.xmat[body].reshape(3, 3)
            result[wrist.name] = np.r_[rotation @ wrench[:3], rotation @ wrench[3:]]
        return result

    def _require_current_env(self):
        if self.controller.model is not self.env.model or self.controller.data is not self.env.data:
            raise ValueError("Recreate the task adapter after reset or scene replacement")

    def advance(
        self, goals, *, deltas=None, compliance=None, finger_targets=None, load_feedforward=None
    ):
        """Apply one interval, or report a rejected preview without stepping.

        Corrections are re-anchored to uncorrected world goals every call.
        Supply a goal for every corrected palm, including compliance-only palms.
        Fingers remain named absolute native servo references.
        """
        self._require_current_env()
        deltas = {} if deltas is None else deltas
        compliance = {} if compliance is None else compliance
        if (
            not all(isinstance(mapping, Mapping) for mapping in (goals, deltas, compliance))
            or set(goals) - set(self.controller.palms)
            or any(not isinstance(goal, PalmGoal) for goal in goals.values())
            or (set(deltas) | set(compliance)) - set(goals)
            or any(not isinstance(delta, PalmDelta) for delta in deltas.values())
            or any(not isinstance(config, WristCompliance) for config in compliance.values())
        ):
            raise ValueError("Expected world palm goals and corrections for targeted palms")
        wrench = self.contact_wrenches_world() if compliance else {}
        targets, offsets = dict(goals), {}
        for side, goal in goals.items():
            if side not in deltas and side not in compliance:
                continue
            delta = deltas.get(side, PalmDelta())
            feedback = (
                compliance[side].correction(wrench[side]) if side in compliance else PalmDelta()
            )
            translation = _clip_norm(
                delta.translation_m + feedback.translation_m, self.limits.translation_m
            )
            vector = _clip_norm(
                delta.rotation_rad + feedback.rotation_rad, self.limits.rotation_rad
            )
            angle = np.linalg.norm(vector)
            rotation = goal.rotation_world
            if angle:
                quaternion, matrix = np.empty(4), np.empty(9)
                mujoco.mju_axisAngle2Quat(quaternion, vector / angle, angle)
                mujoco.mju_quat2Mat(matrix, quaternion)
                rotation = matrix.reshape(3, 3) @ rotation
            targets[side] = PalmGoal(goal.position_m + translation, rotation)
            offsets[side] = dict(translation_m=translation.tolist(), rotation_rad=vector.tolist())
        preview = self.controller.predict_step(
            targets,
            dt=self.env.dt,
            finger_targets=finger_targets,
            load_feedforward=load_feedforward,
        )
        result = dict(
            advanced=False,
            prediction=preview.report,
            corrections=offsets,
            contact_wrenches_world={side: value.tolist() for side, value in wrench.items()},
        )
        if not preview.report["accepted"]:
            return result
        baseline = float(self.env.data.time)
        _, _, terminated, truncated, _ = self.env.step(
            self.controller.accept(preview), on_physics_step=self.guard
        )
        tracking = {}
        for side, goal in targets.items():
            body = self.controller.palms[side]
            position = self.env.data.xpos[body]
            rotation = self.env.data.xmat[body].reshape(3, 3)
            tracking[side] = dict(
                corrected_goal_position_error_m=float(np.linalg.norm(goal.position_m - position)),
                corrected_goal_orientation_error_deg=float(
                    np.degrees(np.linalg.norm(rotation_error(rotation, goal.rotation_world)))
                ),
                task_goal_position_error_m=float(np.linalg.norm(goals[side].position_m - position)),
                task_goal_orientation_error_deg=float(
                    np.degrees(np.linalg.norm(rotation_error(rotation, goals[side].rotation_world)))
                ),
            )
        return dict(
            result,
            advanced=True,
            elapsed_s=float(self.env.data.time) - baseline,
            terminated=terminated,
            truncated=truncated,
            measured_tracking=tracking,
        )
