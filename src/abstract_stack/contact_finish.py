"""Measured supported-backstop dwell, separate from release and task success."""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class BackstopDwell:
    physics_dt_s: float
    force_threshold_N: float
    required_seconds: float
    upright_limit_degrees: float
    yaw_limit_degrees: float
    held_seconds: float = field(default=0.0, init=False)
    maximum_held_seconds: float = field(default=0.0, init=False)
    last_time_s: float | None = field(default=None, init=False)
    verified: bool = field(default=False, init=False)

    def __post_init__(self):
        values = (
            self.physics_dt_s,
            self.force_threshold_N,
            self.required_seconds,
            self.upright_limit_degrees,
            self.yaw_limit_degrees,
        )
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Contact dwell requires explicit finite positive thresholds")

    def update(self, time_s, forces, rotation):
        """Caller must first admit every native step with all-product guards.

        The first sample establishes the clock without awarding elapsed time.
        Missing, repeated or reordered samples cannot accumulate contact dwell.
        No grasp, release, final-position or fresh-episode success is implied.
        """
        rotation = np.asarray(rotation, float)
        required = ("back_N", "hand_N", "support_N")
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Dwell requires every physics step at a finite nonnegative time")
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)
            or any(k not in forces or not np.isfinite(forces[k]) or forces[k] < 0 for k in required)
        ):
            raise ValueError("Dwell requires fresh measured forces and a proper rotation")
        elapsed = 0.0 if self.last_time_s is None else time_s - self.last_time_s
        if self.last_time_s is not None and not np.isclose(
            elapsed, self.physics_dt_s, atol=1e-9, rtol=0
        ):
            raise ValueError("Dwell requires every physics step exactly once")
        self.last_time_s = float(time_s)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        accepted = (
            forces["back_N"] >= self.force_threshold_N
            and forces["hand_N"] > 0.2
            and forces["support_N"] > 0.2
            and rotation[2, 2] >= np.cos(np.radians(self.upright_limit_degrees))
            and abs(yaw) <= np.radians(self.yaw_limit_degrees)
        )
        self.held_seconds = self.held_seconds + elapsed if accepted else 0.0
        self.maximum_held_seconds = max(self.maximum_held_seconds, self.held_seconds)
        self.verified = self.held_seconds + 1e-12 >= self.required_seconds
        return self.verified

    def report(self):
        return dict(
            **asdict(self),
            kind="measured_supported_squared_backstop_dwell",
            task_success=False,
            release_verified=False,
            fresh_task_credit=0,
        )

    def restore_state(self, state, time_s):
        for name in (
            "physics_dt_s",
            "force_threshold_N",
            "required_seconds",
            "upright_limit_degrees",
            "yaw_limit_degrees",
        ):
            if state[name] != getattr(self, name):
                raise ValueError("Dwell checkpoint changes an acceptance threshold")
        held, maximum = state["held_seconds"], state["maximum_held_seconds"]
        last = state["last_time_s"]
        if (
            not np.isfinite([held, maximum]).all()
            or not 0 <= held <= maximum
            or last is not None
            and (not np.isfinite(last) or abs(last - time_s) > 1e-8)
            or last is None
            and held != 0
            or type(state["verified"]) is not bool
            or state["verified"] != (held + 1e-12 >= self.required_seconds)
        ):
            raise ValueError("Invalid dwell checkpoint clock/history")
        self.held_seconds, self.maximum_held_seconds = float(held), float(maximum)
        self.last_time_s, self.verified = last, state["verified"]
