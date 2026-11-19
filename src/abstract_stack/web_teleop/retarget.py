"""Map absolute web operator intents to complete, bounded MABEL actuator targets.

Arms: palm control-point pose in the ``mabel_chassis`` frame → per-side ``PoseIK`` on
scratch data → nominal joint references. When the full pose is out of reach the palm
point takes priority and orientation is recovered as far as the arm allows; a
kinematic collision guard stops the arm short of fixtures and the robot's own body.
Grip: one fraction per hand → native finger flexion references. Head, lift and
torso: clamped, rate-limited absolute references.
Every tick the complete reference set passes through ``MabelController`` (rate limits,
robot-gravity feedforward) and ``validate_targets``. Nothing here writes live
``qpos``/``ctrl`` or steps physics; the simulation owner applies the returned vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.control import validate_targets
from abstract_stack.ik import PoseIK, rotation_error
from abstract_stack.mabel_controller import MabelController, TrackingConfig
from abstract_stack.recording import HeadGaze
from abstract_stack.web_teleop.base import BaseConfig, SwerveBase

SIDES = ("left", "right")
# Palm-frame control point used by the scripted stocking controller (interactions.py).
# MABEL's palm frames are mirrored, so the same local offset points inward for both.
PALM_OFFSET = np.array([0.0, 0.045, 0.04])
FLEXION = (
    "Thumb PIP",
    "Thumb DIP",
    "Index MCP",
    "Index PIP",
    "Middle MCP",
    "Middle PIP",
    "Ring MCP",
    "Ring PIP",
    "Pinky MCP",
    "Pinky PIP",
)
# Same thumb-base abduction the scripted grasp keeps to clear carton edges.
THUMB_PRESHAPE = 0.4
# Elbow-bent posture with the palms ~0.4 m ahead of the chassis at waist height. The
# reset posture has both arms straight ahead (an elbow singularity), where Cartesian
# teleop converges poorly; "ready" moves there through rate-limited joint references.
READY_POSTURE = {
    "Left Arm 1": 1.2,
    "Left Arm 4": -0.9,
    "Left Wrist": 0.15,
    "Right Arm 1": -1.2,
    "Right Arm 4": 0.75,
    "Right Wrist": -0.15,
}


@dataclass(frozen=True)
class RetargetConfig:
    ik_iterations: int = 40
    reachable_position_m: float = 0.02
    reachable_orientation_rad: float = 0.25
    # Kinematic guard against fixtures (shelf, table, floor, walls) and self-contact;
    # products are excluded so hands can touch, push and grasp them.
    collision_guard: bool = True
    collision_margin_m: float = 0.005
    grip_rate: float = 2.0  # fraction per second
    head_rate: float = 1.5  # rad/s
    torso_rate: float = 0.5  # rad/s
    tracking: TrackingConfig = TrackingConfig()
    base: BaseConfig = BaseConfig()
    tilt_guard_rad: float = 0.14  # ~8 deg chassis tilt brakes the base


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix, dtype=float).ravel())
    return quat


def _rate(current: float, goal: float, limit: float) -> float:
    return current + float(np.clip(goal - current, -limit, limit))


class WebRetargeter:
    def __init__(self, model, data, config: RetargetConfig | None = None):
        self.model, self.data = model, data
        self.config = config or RetargetConfig()
        self.chassis = model.body("mabel_chassis").id
        self.palms = {side: model.body(f"{side}_palm").id for side in SIDES}
        self._ranges = {}
        for name in ("Neck Yaw", "Neck Pitch", "Torso"):
            joint, actuator = model.joint(name), model.actuator(name)
            self._ranges[name] = (
                max(joint.range[0], actuator.ctrlrange[0]),
                min(joint.range[1], actuator.ctrlrange[1]),
            )
        self._ranges["Lift"] = tuple(model.actuator("Lift").ctrlrange)
        robot_root = int(model.body_rootid[self.chassis])
        free_roots = {
            int(model.body_rootid[model.jnt_bodyid[j]])
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        } - {robot_root}
        # Free bodies other than the robot are manipulable products, never obstacles.
        self._movable = {b for b in range(model.nbody) if model.body_rootid[b] in free_roots}
        self._robot_root = robot_root
        self.base = SwerveBase(model, data, self.config.base)
        self.fingers = {
            side: {
                f"{side.title()} {suffix}": tuple(
                    model.actuator(f"{side.title()} {suffix}").ctrlrange
                )
                for suffix in FLEXION
            }
            for side in SIDES
        }
        self.reset()

    # -- lifecycle -----------------------------------------------------------------
    def reset(self) -> None:
        """Rebuild controller state from the freshly reset live targets."""
        self.controller = MabelController(self.model, self.data, self.config.tracking)
        names = MabelController.NAMES
        self.ik, self.arm_names = {}, {}
        for side in SIDES:
            arm = [f"{side.title()} Arm {i}" for i in range(1, 8)] + [f"{side.title()} Wrist"]
            index = [names.index(n) for n in arm] + [names.index("Lift")]
            lower = self.controller.lower[index] + 1e-5
            upper = self.controller.upper[index] - 1e-5
            home = np.r_[[self.model.qpos0[self.model.joint(n).qposadr[0]] for n in arm], 0.3175]
            self.ik[side] = PoseIK(
                self.model,
                self.data,
                arm,
                lower,
                upper,
                home,
                PALM_OFFSET,
                palm=f"{side}_palm",
            )
            self.arm_names[side] = arm
        # The controller clips its initial nominal into joint ∩ actuator bounds; start
        # from the same values so held references never fail its bounds check.
        self.reference = dict(zip(names, self.controller.nominal.tolist(), strict=True))
        for name in ("Neck Yaw", "Neck Pitch", "Torso"):
            ctrl = float(self.data.ctrl[self.model.actuator(name).id])
            self.reference[name] = float(np.clip(ctrl, *self._ranges[name]))
        self.grip = {side: 0.0 for side in SIDES}
        self._ik_cache = {side: None for side in SIDES}
        self._ik_last = {side: None for side in SIDES}
        self.arm_report = {side: None for side in SIDES}
        self.target_epoch = getattr(self, "target_epoch", 0) + 1
        if hasattr(self, "base"):
            self.base.reset()
        self.tilt_guard = False
        # Gaze state follows the episode clock, so rebuild it on every reset.
        self.gaze = HeadGaze(self.model, self.data)
        self._gazing = False

    def resync(self) -> None:
        """Invalidate client arm targets: commands must be rebased on ``arm_targets``."""
        self._ik_cache = {side: None for side in SIDES}
        self._ik_last = {side: None for side in SIDES}
        self.target_epoch += 1

    def ready(self) -> None:
        """Move both arms to ``READY_POSTURE`` (other arm joints to zero) via references."""
        for side in SIDES:
            for name in self.arm_names[side]:
                self.reference[name] = READY_POSTURE.get(name, 0.0)
        self.resync()

    def arm_targets(self) -> dict:
        """Root-frame palm pose implied by the current nominal arm/lift references."""
        origin, root = self.root_pose()
        result = {}
        for side in SIDES:
            q = np.r_[[self.reference[n] for n in self.arm_names[side]], self.reference["Lift"]]
            point, rotation = self.ik[side].forward(q)
            result[side] = {
                "position_m": (root.T @ (point - origin)).tolist(),
                "quaternion_wxyz": _mat_to_quat(root.T @ rotation).tolist(),
            }
        return result

    # -- frames --------------------------------------------------------------------
    def root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.xpos[self.chassis].copy(),
            self.data.xmat[self.chassis].reshape(3, 3).copy(),
        )

    def palm_pose_root(self, side: str) -> dict:
        """Measured palm control point in the chassis frame (for client ghost initialisation)."""
        origin, root = self.root_pose()
        rotation = self.data.xmat[self.palms[side]].reshape(3, 3)
        point = self.data.xpos[self.palms[side]] + rotation @ PALM_OFFSET
        return {
            "position_m": (root.T @ (point - origin)).tolist(),
            "quaternion_wxyz": _mat_to_quat(root.T @ rotation).tolist(),
        }

    def channel_state(self) -> dict:
        return {
            "head": {
                "yaw_rad": self.reference["Neck Yaw"],
                "pitch_rad": self.reference["Neck Pitch"],
            },
            "lift_m": self.reference["Lift"],
            "torso_rad": self.reference["Torso"],
            "grip": dict(self.grip),
            "ranges": {
                "head": {
                    "yaw_rad": list(self._ranges["Neck Yaw"]),
                    "pitch_rad": list(self._ranges["Neck Pitch"]),
                },
                "lift_m": list(self._ranges["Lift"]),
                "torso_rad": list(self._ranges["Torso"]),
            },
        }

    # -- intents -------------------------------------------------------------------
    def _solve_arm(self, side: str, intent: dict) -> None:
        position, quaternion = intent["position"], intent["quaternion"]
        cached = self._ik_cache[side]
        lift = self.reference["Lift"]
        if (
            cached is not None
            and np.array_equal(cached[0], position)
            and np.array_equal(cached[1], quaternion)
            and cached[2] == lift
        ):
            return
        origin, root = self.root_pose()
        target = origin + root @ position
        rotation = root @ _quat_to_mat(quaternion)
        ik = self.ik[side]
        # The operator owns Lift through its own channel; pin it for both arms' IK.
        ik.lower[-1] = ik.upper[-1] = lift
        seed = np.r_[[self.reference[n] for n in self.arm_names[side]], lift]
        iterations = self.config.ik_iterations
        # Palm point first, orientation recovered in its null space: identical to a
        # full-pose solve whenever the pose is reachable, and otherwise keeps the hand
        # where the operator put it, twisting only as much as the arm requires.
        q, point, actual = ik.solve_position_first(target, rotation, seed, iterations=iterations)
        position_error = float(np.linalg.norm(target - point))
        if position_error > self.config.reachable_position_m:
            # Stuck against joint limits: restart from a position-only solve, which
            # escapes them, and keep it only if it gets the palm closer.
            restart = iterations // 2
            q_position, _, _ = ik.solve(
                target, rotation, seed, iterations=restart, rotation_weight=0.0
            )
            q2, point2, actual2 = ik.solve_position_first(
                target, rotation, q_position, iterations=restart
            )
            error2 = float(np.linalg.norm(target - point2))
            if error2 < position_error - 1e-3:
                q, point, actual, position_error = q2, point2, actual2, error2
        orientation_error = float(np.linalg.norm(rotation_error(actual, rotation)))
        blocked = self._guard(side, seed, q) if self.config.collision_guard else []
        if blocked:
            q = self._guard_fraction
            point, actual = ik.forward(q)
            position_error = float(np.linalg.norm(target - point))
            orientation_error = float(np.linalg.norm(rotation_error(actual, rotation)))
        position_ok = position_error <= self.config.reachable_position_m
        reachable = position_ok and orientation_error <= self.config.reachable_orientation_rad
        # Teleop applies the closest bounded solution and reports the residual; the
        # operator sees the miss. This is never evidence of a grasp or placement.
        for name, value in zip(self.arm_names[side], q[:-1], strict=True):
            self.reference[name] = float(value)
        # Unconverged solves (e.g. near the straight-arm singularity) keep refining
        # from this warm start on later ticks. Cache a converged answer, or one that
        # has stopped improving for an unchanged target, so a held unreachable target
        # does not re-run the full search every tick.
        key = (position.copy(), quaternion.copy(), lift)
        converged = position_error < 1e-3 and orientation_error < 1e-2
        last = self._ik_last[side]
        stalled = (
            last is not None
            and np.array_equal(last[0][0], position)
            and np.array_equal(last[0][1], quaternion)
            and last[0][2] == lift
            and abs(last[1] - position_error) < 1e-5
        )
        self._ik_last[side] = (key, position_error)
        self._ik_cache[side] = key if converged or stalled else None
        self.arm_report[side] = {
            "reachable": reachable,
            "ik_position_error_m": position_error,
            "ik_orientation_error_rad": orientation_error,
            "orientation_relaxed": position_ok and not reachable,
            "blocked_by": blocked,
        }

    def _collisions(self, ik: PoseIK, q: np.ndarray, reference, baseline) -> list[str]:
        """Names of fixtures (or "self") the arm at ``q`` would penetrate beyond the margin."""
        model, names = self.model, []
        for pair, info in ik._contacts(q, reference, None).items():
            bodies = [int(model.geom_bodyid[g]) for g in pair]
            if any(b in self._movable for b in bodies):
                continue
            allowed = min(
                -self.config.collision_margin_m,
                baseline.get(pair, {"distance_m": 0.0})["distance_m"] - 0.001,
            )
            if info["distance_m"] >= allowed:
                continue
            roots = [int(model.body_rootid[b]) for b in bodies]
            if all(r == self._robot_root for r in roots):
                label = "self"
            else:
                other = pair[0] if roots[1] == self._robot_root else pair[1]
                body = int(model.geom_bodyid[other])
                label = (
                    model.geom(other).name
                    if body == 0
                    else model.body(model.body_rootid[body]).name
                )
            if label not in names:
                names.append(label)
        return names

    def _guard(self, side: str, start: np.ndarray, goal: np.ndarray) -> list[str]:
        """Keep the farthest collision-free point on the joint segment start → goal.

        Contacts already present in the measured state (plus 1 mm) are allowed, so a
        hand resting on a surface is never frozen. Sets ``_guard_fraction``.
        """
        ik = self.ik[side]
        reference = self.data.qpos.copy()
        baseline = ik._contacts(ik.measured_q(), reference, None)
        blocked = self._collisions(ik, goal, reference, baseline)
        if not blocked:
            return []
        low, high = 0.0, 1.0
        for _ in range(6):  # ~1.6 % of the segment
            middle = (low + high) / 2
            if self._collisions(ik, start + middle * (goal - start), reference, baseline):
                high = middle
            else:
                low = middle
        self._guard_fraction = start + low * (goal - start)
        return blocked

    def _look_at(self, point_root: np.ndarray, dt: float) -> None:
        """Aim the head camera at a chassis-frame point with the repository's gaze solver."""
        if not self._gazing:
            # Start from the current references so switching modes never jumps.
            self.gaze.command = np.array([self.reference["Neck Yaw"], self.reference["Neck Pitch"]])
            self.gaze.next_update = 0.0
            self._gazing = True
        origin, root = self.root_pose()
        for name, value in self.gaze.targets(origin + root @ point_root, dt).items():
            self.reference[name] = float(np.clip(value, *self._ranges[name]))

    def tilt(self) -> float:
        """Chassis tilt from vertical in radians."""
        return float(math.acos(min(1.0, max(-1.0, self.data.xmat[self.chassis][8]))))

    def drive(self, base: dict | None, dt: float) -> None:
        """Ramp the base toward a velocity intent, or brake when ``None`` / tilting."""
        self.tilt_guard = self.tilt() > self.config.tilt_guard_rad
        if base is None or self.tilt_guard:
            self.base.stop(dt)
        else:
            self.base.command(base["forward_mps"], base["left_mps"], base["turn_radps"], dt)

    def apply_intent(self, intent: dict, dt: float) -> None:
        """Update nominal references from one validated command (see protocol.parse_command)."""
        cfg = self.config
        if intent["lift"] is not None:
            self.reference["Lift"] = float(np.clip(intent["lift"], *self._ranges["Lift"]))
        if intent["torso"] is not None:
            goal = float(np.clip(intent["torso"], *self._ranges["Torso"]))
            self.reference["Torso"] = _rate(self.reference["Torso"], goal, cfg.torso_rate * dt)
        head = intent["head"]
        if head is not None and "look_at" in head:
            self._look_at(head["look_at"], dt)
        elif head is not None:
            self._gazing = False
            for name, value in head.items():
                goal = float(np.clip(value, *self._ranges[name]))
                self.reference[name] = _rate(self.reference[name], goal, cfg.head_rate * dt)
        if intent["target_epoch"] != self.target_epoch:
            return  # arm targets predate a host-side move; wait for the client to rebase
        for side in SIDES:
            arm = intent["arms"][side]
            if arm is None:
                continue
            self._solve_arm(side, arm)
            self.grip[side] = _rate(self.grip[side], arm["grip"], cfg.grip_rate * dt)

    def targets(self, dt: float) -> np.ndarray:
        """Complete validated actuator vector for the current (possibly held) references."""
        requested = dict(self.reference)
        for side in SIDES:
            for name, (low, high) in self.fingers[side].items():
                requested[name] = min(high, max(low, low + self.grip[side] * (high - low)))
            requested[f"{side.title()} Thumb MCP"] = THUMB_PRESHAPE
        requested.update(self.base.targets(dt))
        desired = self.controller.targets(requested, dt=dt)
        return validate_targets(self.model, self.data.ctrl, desired)

    def report(self) -> dict:
        last = self.controller.last_report
        return {
            "target_epoch": self.target_epoch,
            "arm_targets": self.arm_targets(),
            "arms": {
                side: {
                    "palm": self.palm_pose_root(side),
                    "ik": self.arm_report[side],
                }
                for side in SIDES
            },
            "channels": {**self.channel_state(), "head_mode": "gaze" if self._gazing else "angles"},
            "base": {
                **self.base.report(),
                "tilt_deg": round(math.degrees(self.tilt()), 1),
                "tilt_guard": self.tilt_guard,
            },
            "limits": {
                key: list(last.get(key, []))
                for key in ("rate_limited", "target_limited", "force_limited")
            },
        }
