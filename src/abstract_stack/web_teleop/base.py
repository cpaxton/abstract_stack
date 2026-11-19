"""Operator base driving through MABEL's native three-module swerve actuators.

A body twist in the ``mabel_chassis`` frame (forward/left speed, turn rate) is
acceleration-limited, then converted per module into a steer angle (position servo)
and a wheel speed (velocity servo). Module positions, zero-steer rolling directions
and the wheel radius come from the compiled model rather than constants. Wheels only
drive once their steer angle is close to the target, so modules re-aim instead of
scrubbing sideways. Nothing here moves the base directly: it only proposes native
actuator targets that the simulation owner applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

MODULES = ("FL", "FR", "B")


@dataclass(frozen=True)
class BaseConfig:
    max_linear: float = 0.45  # m/s
    max_angular: float = 0.9  # rad/s
    accel_linear: float = 0.5  # m/s^2 when speeding up
    decel_linear: float = 0.8  # m/s^2 when slowing or stopping
    accel_angular: float = 1.2  # rad/s^2
    decel_angular: float = 2.0
    steer_rate: float = 8.0  # rad/s
    align_tolerance: float = math.radians(25)  # no wheel drive beyond this steer error
    idle_speed: float = 0.005  # m/s module speed treated as stopped


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class SwerveBase:
    def __init__(self, model, data, config: BaseConfig | None = None):
        self.model, self.data = model, data
        self.config = config or BaseConfig()
        self.chassis = model.body("mabel_chassis").id
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        self.steer_adr, self.steer_aid, self.drive_aid = {}, {}, {}
        for name in MODULES:
            joint = model.joint(f"{name} Steer")
            self.steer_adr[name] = int(joint.qposadr[0])
            self.steer_aid[name] = model.actuator(f"{name} Steer").id
            self.drive_aid[name] = model.actuator(f"{name} Drive").id
            scratch.qpos[self.steer_adr[name]] = 0.0
        mujoco.mj_kinematics(model, scratch)
        root = scratch.xmat[self.chassis].reshape(3, 3)
        origin = scratch.xpos[self.chassis]
        self.position, self.zero_heading = {}, {}
        for name in MODULES:
            drive = model.joint(f"{name} Drive")
            body = int(drive.bodyid[0])
            axis = root.T @ (scratch.xmat[body].reshape(3, 3) @ drive.axis)
            # No-slip rolling: v_centre = omega * r * (axis x up) for positive spin.
            rolling = np.cross(axis, [0.0, 0.0, 1.0])
            self.zero_heading[name] = math.atan2(rolling[1], rolling[0])
            self.position[name] = (root.T @ (scratch.xpos[body] - origin))[:2]
        self.radius = self._wheel_radius(model, model.joint("FL Drive"))
        self.drive_limit = float(model.actuator_ctrlrange[self.drive_aid["FL"], 1])
        self.reset()

    @staticmethod
    def _wheel_radius(model, drive) -> float:
        body = int(drive.bodyid[0])
        radius = 0.0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            mid = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            vertices = model.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
            vertices = vertices + model.geom_pos[g]
            perpendicular = vertices - np.outer(vertices @ drive.axis, drive.axis)
            radius = max(radius, float(np.linalg.norm(perpendicular, axis=1).max()))
        if radius <= 0:
            raise ValueError("could not determine MABEL wheel radius from collision geometry")
        return radius

    def reset(self) -> None:
        self.goal = np.zeros(3)  # requested (vx, vy, wz) in the chassis frame
        self.twist = np.zeros(3)  # applied, acceleration-limited twist
        self.steer = {n: float(self.data.qpos[self.steer_adr[n]]) for n in MODULES}
        self.last = {"wheel_speed": dict.fromkeys(MODULES, 0.0), "aligned": True}

    @property
    def moving(self) -> bool:
        return bool(np.any(np.abs(self.twist) > 1e-4))

    def command(self, forward: float, left: float, turn: float, dt: float) -> None:
        """Ramp toward a requested twist (operator frame: forward, left, turn-left)."""
        cfg = self.config
        # MABEL faces chassis -X with its left at -Y; a left turn is +Z.
        self.goal = np.array(
            [
                -np.clip(forward, -cfg.max_linear, cfg.max_linear),
                -np.clip(left, -cfg.max_linear, cfg.max_linear),
                np.clip(turn, -cfg.max_angular, cfg.max_angular),
            ]
        )
        # Steer first: while any module is still re-aiming, slow down instead of
        # speeding up, so wheels never engage abruptly and jolt the tall body.
        goal = self.goal if self.last["aligned"] else np.zeros(3)
        linear = goal[:2] - self.twist[:2]
        speeding_up = float(np.linalg.norm(goal[:2])) > float(np.linalg.norm(self.twist[:2]))
        limit = (cfg.accel_linear if speeding_up else cfg.decel_linear) * dt
        norm = float(np.linalg.norm(linear))
        if norm > limit:
            linear *= limit / norm
        self.twist[:2] += linear
        angular_limit = (
            cfg.accel_angular if abs(goal[2]) > abs(self.twist[2]) else cfg.decel_angular
        ) * dt
        self.twist[2] += float(np.clip(goal[2] - self.twist[2], -angular_limit, angular_limit))

    def stop(self, dt: float) -> None:
        self.command(0.0, 0.0, 0.0, dt)

    def _module_velocity(self, twist: np.ndarray, name: str) -> tuple[float, float]:
        px, py = self.position[name]
        return twist[0] - twist[2] * py, twist[1] + twist[2] * px

    def targets(self, dt: float) -> dict[str, float]:
        """Native steer angles and wheel speeds realising the current twist."""
        cfg = self.config
        result, aligned = {}, True
        for name in MODULES:
            gvx, gvy = self._module_velocity(self.goal, name)
            mvx, mvy = self._module_velocity(self.twist, name)
            # Aim at the requested motion (pre-aim from standstill); while stopping,
            # keep following the decaying motion; when idle, hold the steer angle.
            if math.hypot(gvx, gvy) > cfg.idle_speed:
                aim = (gvx, gvy)
            elif math.hypot(mvx, mvy) > cfg.idle_speed:
                aim = (mvx, mvy)
            else:
                aim = None
            if aim is not None:
                desired = math.atan2(aim[1], aim[0]) - self.zero_heading[name]
                # Use the equivalent angle (or its reverse with negated spin) nearest
                # the current command, so a module turns at most 90 degrees.
                delta = _wrap(desired - self.steer[name])
                if abs(delta) > math.pi / 2:
                    delta = _wrap(delta + math.pi)
                step = float(np.clip(delta, -cfg.steer_rate * dt, cfg.steer_rate * dt))
                self.steer[name] += step
                measured = float(self.data.qpos[self.steer_adr[name]])
                if (
                    abs(delta - step) + abs(_wrap(measured - self.steer[name]))
                    > cfg.align_tolerance
                ):
                    aligned = False
            # Drive only the component of the module velocity along the wheel's actual
            # rolling direction (smooth cos falloff; sign handles reversed modules).
            heading = self.zero_heading[name] + float(self.data.qpos[self.steer_adr[name]])
            along = mvx * math.cos(heading) + mvy * math.sin(heading)
            wheel = along / self.radius if aligned or not self.moving else 0.0
            wheel = float(np.clip(wheel, -self.drive_limit, self.drive_limit))
            result[f"{name} Steer"] = self.steer[name]
            result[f"{name} Drive"] = wheel
            self.last["wheel_speed"][name] = wheel
        self.last["aligned"] = aligned
        return result

    def report(self) -> dict:
        forward, left = -self.twist[0], -self.twist[1]
        return {
            "forward_mps": round(float(forward), 4),
            "left_mps": round(float(left), 4),
            "turn_radps": round(float(self.twist[2]), 4),
            "aligned": self.last["aligned"],
            "limits": {
                "linear_mps": self.config.max_linear,
                "angular_radps": self.config.max_angular,
            },
        }
