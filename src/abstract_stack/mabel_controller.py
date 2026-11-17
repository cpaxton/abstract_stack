"""Named bimanual joint/lift tracking through unmodified MABEL position servos."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.control import validate_targets


@dataclass(frozen=True)
class TrackingConfig:
    gravity_compensation: bool = True
    joint_speed: float = 1.0
    lift_speed: float = 0.08
    joint_offset_limit: float = 0.10
    lift_offset_limit: float = 0.01
    wrist_damping: float = 0.2

    def __post_init__(self):
        if not isinstance(self.gravity_compensation, bool):
            raise ValueError("gravity_compensation must be boolean")
        for value in (
            self.joint_speed,
            self.lift_speed,
            self.joint_offset_limit,
            self.lift_offset_limit,
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("tracking limits must be finite and positive")
        if not np.isfinite(self.wrist_damping) or self.wrist_damping < 0:
            raise ValueError("wrist damping must be finite and nonnegative")


class MabelController:
    """Track named references for either arm, both wrists and the shared lift.

    Adds gravity feedforward as a bounded position-servo reference offset,
    never as external forces. Fingers and wheels pass through their native
    controls; this class does not implement a grasp or a navigation policy.
    A learned policy can supply the same nominal references after its own
    contract and rate validation. Model admission and hardware validation are
    separate from this simulator controller.
    """

    NAMES = tuple(f"{side} Arm {i}" for side in ("Left", "Right") for i in range(1, 8)) + (
        "Left Wrist",
        "Right Wrist",
        "Lift",
    )

    def __init__(self, model, data, config: TrackingConfig | None = None):
        self.model, self.data = model, data
        self.config = config or TrackingConfig()
        self.ids = np.array([model.actuator(n).id for n in self.NAMES])
        self.kp = model.actuator_gainprm[self.ids, 0].copy()
        if not np.all(self.kp > 0) or not np.allclose(
            model.actuator_biasprm[self.ids, 1], -self.kp
        ):
            raise ValueError("MABEL tracking requires native position servos")
        if np.any(model.actuator_gaintype[self.ids] != mujoco.mjtGain.mjGAIN_FIXED) or np.any(
            model.actuator_biastype[self.ids] != mujoco.mjtBias.mjBIAS_AFFINE
        ):
            raise ValueError("MABEL tracking requires fixed-gain affine position servos")
        self.lower, self.upper = model.actuator_ctrlrange[self.ids].T.copy()
        # Restrict direct joint references by both actuator and joint bounds.
        for i, name in enumerate(self.NAMES[:-1]):
            joint = model.joint(name)
            self.lower[i] = max(self.lower[i], joint.range[0])
            self.upper[i] = min(self.upper[i], joint.range[1])
        self.rates = np.r_[
            np.full(len(self.ids) - 1, self.config.joint_speed), self.config.lift_speed
        ]
        self.offset_limits = np.r_[
            np.full(len(self.ids) - 1, self.config.joint_offset_limit),
            self.config.lift_offset_limit,
        ]
        self.nominal = np.clip(data.ctrl[self.ids], self.lower, self.upper)
        self.applied = self.nominal.copy()
        self.scratch = mujoco.MjData(model)
        self.last_report = {}

    def _gravity_forces(self):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = 0
        mujoco.mj_forward(self.model, self.scratch)
        moment = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            moment,
            self.scratch.actuator_moment,
            self.scratch.moment_rownnz,
            self.scratch.moment_rowadr,
            self.scratch.moment_colind,
        )
        transmission = moment[self.ids].T
        driven = np.any(np.abs(transmission) > 1e-10, axis=1)
        # One tendon drives two lift stages: fit its common force rather than
        # counting the upper-body weight twice. Unactuated base DOFs are omitted.
        forces = np.linalg.lstsq(transmission[driven], self.scratch.qfrc_bias[driven], rcond=None)[
            0
        ]
        return forces

    def targets(self, requested, *, dt):
        """Return native actuator targets; computation never steps or mutates live data."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("controller dt must be finite and positive")
        desired = validate_targets(self.model, self.data.ctrl, requested)
        reference = desired[self.ids]
        # Omitted tracking dimensions retain their nominal reference, even when
        # live ctrl already contains a feedforward offset.
        if hasattr(requested, "keys"):
            reference = np.array(
                [
                    requested[n] if n in requested else self.nominal[i]
                    for i, n in enumerate(self.NAMES)
                ]
            )
        if np.any(reference < self.lower) or np.any(reference > self.upper):
            raise ValueError("MABEL joint reference exceeds joint bounds")
        nominal = self.nominal + np.clip(
            reference - self.nominal, -self.rates * dt, self.rates * dt
        )
        nominal = np.clip(nominal, self.lower, self.upper)
        gravity = (
            self._gravity_forces() if self.config.gravity_compensation else np.zeros(len(self.ids))
        )
        force = gravity.copy()
        # Canonical wrists have kp=1, kv=0; add bounded measured velocity
        # feedback through references without editing native actuator gains.
        for name in ("Left Wrist", "Right Wrist"):
            i = self.NAMES.index(name)
            velocity = (nominal[i] - self.nominal[i]) / dt
            force[i] += self.config.wrist_damping * (
                velocity - self.data.actuator_velocity[self.ids[i]]
            )
        force_limited = []
        for i, aid in enumerate(self.ids):
            if self.model.actuator_forcelimited[aid]:
                limited = np.clip(force[i], *self.model.actuator_forcerange[aid])
                if abs(limited - force[i]) > 1e-9:
                    force_limited.append(self.NAMES[i])
                force[i] = limited
        raw_offset = force / self.kp
        offset = np.clip(raw_offset, -self.offset_limits, self.offset_limits)
        bounded = np.clip(nominal + offset, self.lower, self.upper)
        applied = self.applied + np.clip(bounded - self.applied, -self.rates * dt, self.rates * dt)
        applied = np.clip(applied, self.lower, self.upper)
        desired[self.ids] = applied
        desired = validate_targets(self.model, self.data.ctrl, desired)
        self.nominal, self.applied = nominal, applied
        self.last_report = {
            "gravity_compensation": self.config.gravity_compensation,
            "nominal": dict(zip(self.NAMES, nominal.tolist(), strict=True)),
            "measured": dict(
                zip(self.NAMES, self.data.actuator_length[self.ids].tolist(), strict=True)
            ),
            "servo_target": dict(zip(self.NAMES, applied.tolist(), strict=True)),
            "gravity_force": dict(zip(self.NAMES, gravity.tolist(), strict=True)),
            "force_limited": force_limited,
            "feedforward_limited": [
                n for i, n in enumerate(self.NAMES) if abs(raw_offset[i] - offset[i]) > 1e-9
            ],
            "target_limited": [
                n
                for i, n in enumerate(self.NAMES)
                if abs(nominal[i] + offset[i] - bounded[i]) > 1e-9
            ],
            "rate_limited": [
                n for i, n in enumerate(self.NAMES) if abs(bounded[i] - applied[i]) > 1e-9
            ],
            "scope": "robot gravity only; payload/contact forces are not compensated",
        }
        return desired
