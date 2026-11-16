"""First-pass actuator-driven stocking, with measured contact forces and failure traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.control import stationary_targets


class InteractionFailure(RuntimeError):
    """A measured acceptance or force limit failed; no success is implied."""


@dataclass(frozen=True)
class ScriptConfig:
    approach_seconds: float = 2.0
    descend_seconds: float = 8.0
    pregrasp_height: float = 0.35
    close_seconds: float = 1.5
    lift_height: float = 0.07
    grip: float = 0.8
    max_force: float = 25.0
    max_joint_speed: float = 1.0
    tracking_tolerance: float = 0.035
    push_speed: float = 0.025
    push_distance: float = 0.16
    contact_threshold: float = 0.4
    contact_dwell: float = 0.12

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Script settings must be finite and positive")
        if self.grip > 1 or self.max_force <= self.contact_threshold:
            raise ValueError("grip must be <= 1 and max_force must exceed contact_threshold")


def _descendants(model, root_name):
    root = model.body(root_name).id
    result = set()
    for bid in range(model.nbody):
        parent = bid
        while parent:
            if parent == root:
                result.add(bid)
                break
            parent = model.body_parentid[parent]
    return result


def contact_forces(env, product_name: str, hand_bodies: set[int]) -> dict:
    """Sum measured nonnegative contact-normal forces, in newtons, by counterpart."""
    model, data = env.model, env.data
    product = model.body(product_name).id
    supply = model.body("supply").id
    shelf = model.body("shelf").id
    back = model.geom("shelf_back").id
    result = {
        "hand": 0.0,
        "support": 0.0,
        "back": 0.0,
        "arm_fixture": 0.0,
        "shelf": 0.0,
        "table": 0.0,
    }
    hand_contacts = set()
    normals = []
    for i, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal = max(0.0, float(force[0]))
        if product in (b1, b2):
            other, other_geom = (b2, g2) if b1 == product else (b1, g1)
            if other in hand_bodies:
                result["hand"] += normal
                if normal > 0.05:
                    hand_contacts.add(other)
                    normals.append(contact.frame[:3].copy() * (1 if b1 == product else -1))
            if other == supply:
                result["table"] += normal
                result["support"] += normal
            if (model.geom(other_geom).name or "").startswith("shelf_board_"):
                result["shelf"] += normal
                result["support"] += normal
            if other_geom == back:
                result["back"] += normal
        if (b1 in hand_bodies and b2 in (supply, shelf)) or (
            b2 in hand_bodies and b1 in (supply, shelf)
        ):
            result["arm_fixture"] += normal
    result["hand_contact_bodies"] = len(hand_contacts)
    result["opposing_contacts"] = any(
        np.dot(a, b) < -0.25 for i, a in enumerate(normals) for b in normals[i + 1 :]
    )
    return result


class CartesianHand:
    """Damped pose IK on scratch data, followed by bounded native actuator targets.

    The lift contributes one coordinate: both stages move by half its extension.
    Live robot/object qpos is never changed by the solver.
    """

    OFFSET = np.array([0.0, 0.045, 0.04])
    ROTATION = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
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

    def __init__(self, env, config: ScriptConfig):
        self.env, self.config = env, config
        self.model, self.data = env.model, env.data
        self.scratch = mujoco.MjData(self.model)
        self.palm = self.model.body("left_palm").id
        self.names = [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.names])
        self.dofs = np.array([self.model.joint(n).dofadr[0] for n in self.names])
        self.aids = np.array([self.model.actuator(n).id for n in self.names])
        self.lift_joints = [self.model.joint(n) for n in ("Lift Lower", "Lift Upper")]
        self.lower = np.array(
            [
                max(self.model.joint(n).range[0], self.model.actuator_ctrlrange[a, 0]) + 1e-5
                for n, a in zip(self.names, self.aids, strict=True)
            ]
            + [0.18]
        )
        self.upper = np.array(
            [
                min(self.model.joint(n).range[1], self.model.actuator_ctrlrange[a, 1]) - 1e-5
                for n, a in zip(self.names, self.aids, strict=True)
            ]
            + [0.55]
        )
        self.command = np.r_[self.data.ctrl[self.aids], self.data.actuator("Lift").ctrl[0]]
        self.home = np.r_[self.model.qpos0[self.qadr], 0.3175]
        self.last_target = self.position()
        self.last_ik_error = 0.0
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)

    def position(self):
        return self.data.xpos[self.palm] + self.data.xmat[self.palm].reshape(3, 3) @ self.OFFSET

    def targets(self, target, grip):
        target = np.asarray(target, dtype=float)
        self.last_target = target.copy()
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = 0
        q = self.command.copy()
        best_error, best_q = float("inf"), q.copy()
        # Retry from the model's home pose if a warm start becomes trapped at a
        # joint limit. Both searches remain in scratch data; actuator rate limits
        # still bound any change of IK branch in the physical simulation.
        for iteration in range(100):
            if iteration == 40:
                if best_error < 0.005:
                    break
                q = np.clip(self.home, self.lower, self.upper)
            self.scratch.qpos[self.qadr] = q[:-1]
            for joint in self.lift_joints:
                self.scratch.qpos[joint.qposadr[0]] = q[-1] / 2
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
            rotation = self.scratch.xmat[self.palm].reshape(3, 3)
            point = self.scratch.xpos[self.palm] + rotation @ self.OFFSET
            quat, angular_error = np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(quat, (self.ROTATION @ rotation.T).ravel())
            if quat[0] < 0:
                quat *= -1
            mujoco.mju_quat2Vel(angular_error, quat, 1)
            error = np.r_[target - point, 0.15 * angular_error]
            residual = float(np.linalg.norm(error))
            if residual < best_error:
                best_error, best_q = residual, q.copy()
            if residual < 1e-4:
                break
            mujoco.mj_jac(self.model, self.scratch, self.jacp, self.jacr, point, self.palm)
            lift_dofs = [j.dofadr[0] for j in self.lift_joints]
            jacobian = np.column_stack(
                (
                    np.vstack((self.jacp[:, self.dofs], 0.15 * self.jacr[:, self.dofs])),
                    np.r_[self.jacp[:, lift_dofs].mean(axis=1), np.zeros(3)],
                )
            )
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(6),
                error,
            )
            q = np.clip(q + np.clip(delta, -0.06, 0.06), self.lower, self.upper)
        q = best_q
        self.last_ik_error = best_error
        # Metres for Lift, radians for joints: use a lower physical lift rate.
        rates = np.r_[np.full(len(self.names), self.config.max_joint_speed), 0.08]
        self.command += np.clip(q - self.command, -rates * self.env.dt, rates * self.env.dt)
        action = dict(zip(self.names, self.command[:-1].tolist(), strict=True))
        action["Lift"] = float(self.command[-1])
        for suffix in self.FLEXION:
            name = f"Left {suffix}"
            lo, hi = self.model.actuator_ctrlrange[self.model.actuator(name).id]
            action[name] = float(lo + grip * (hi - lo))
        return action


class StockingScript:
    """Stage machine; failure stops the run and retains its measured evidence."""

    def __init__(
        self, env, directory: str | Path, config: ScriptConfig | None = None, *, render=False
    ):
        if not env.robot or not env.products or any(p.kind != "carton" for p in env.products):
            raise ValueError("The script requires MABEL and a carton-only scene (pick_place)")
        if env.config.environment != "studio" or tuple(env.config.robot_xy) != (0.0, 0.0):
            raise ValueError("This first script supports the stationary pick_place bench only")
        self.env, self.config = env, config or ScriptConfig()
        self.directory, self.render = Path(directory), render
        self.directory.mkdir(parents=True, exist_ok=True)
        self.controller = CartesianHand(env, self.config)
        self.hand_bodies = _descendants(env.model, "left_palm")
        self.arm_bodies = _descendants(env.model, "l_shoulder_1")
        self.rows, self.events, self.controls = [], [], []
        self.terminal = False
        self.product = env.products[0]
        self.phase = "settle"
        self.last_forces = {}

    def _sample(self, env):
        if self.rows and env.data.time <= self.rows[-1][0]:
            raise InteractionFailure("MuJoCo time did not advance; possible physics reset")
        forces = contact_forces(env, self.product.name, self.hand_bodies)
        arm_forces = contact_forces(env, self.product.name, self.arm_bodies)
        forces["arm_fixture"] = arm_forces["arm_fixture"]
        self.last_forces = forces
        position = env.data.body(self.product.name).xpos.copy()
        error = float(np.linalg.norm(self.controller.last_target - self.controller.position()))
        self.rows.append(
            [
                float(env.data.time),
                *position,
                *self.controller.position(),
                forces["hand"],
                forces["support"],
                forces["back"],
                forces["arm_fixture"],
                error,
            ]
        )
        self.controls.append(env.data.ctrl.copy())
        if max(forces["hand"], forces["back"], forces["arm_fixture"]) > self.config.max_force:
            raise InteractionFailure(
                f"{self.phase}: contact force exceeded {self.config.max_force} N"
            )
        if env.data.body("mabel_chassis").xmat.reshape(3, 3)[2, 2] < 0.95:
            raise InteractionFailure(f"{self.phase}: robot tilt limit exceeded")

    def _tick(self, target, grip):
        _, _, terminated, truncated, _ = self.env.step(
            self.controller.targets(target, grip),
            on_physics_step=self._sample,
        )
        if truncated:
            raise InteractionFailure(f"{self.phase}: episode time limit")
        self.terminal = terminated
        return terminated

    def _stage(self, name):
        self.phase = name
        self.events.append(
            {"phase": name, "product": self.product.name, "sim_time": float(self.env.data.time)}
        )

    def _move(self, name, target, grip=0.0, seconds=None, *, check_tracking=True):
        self._stage(name)
        start = self.controller.position().copy()
        target = np.asarray(target, dtype=float)
        ticks = max(1, int(np.ceil((seconds or self.config.approach_seconds) / self.env.dt)))
        for k in range(ticks):
            t = (k + 1) / ticks
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            if self._tick(start + blend * (target - start), grip):
                if name != "clear":
                    raise InteractionFailure(f"{name}: task ended before push/contact validation")
                break
        error = float(np.linalg.norm(self.controller.position() - target))
        self.events[-1].update(end_time=float(self.env.data.time), tracking_error=error)
        if self.render:
            from PIL import Image

            Image.fromarray(self.env.render("overview")).save(
                self.directory / f"{self.product.name}-{name}.png",
            )
        if check_tracking and not self.terminal and error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: measured tracking error {error:.3f} m")

    def _one_box(self):
        env, config, product = self.env, self.config, self.product
        box = env.data.body(product.name).xpos.copy()
        self._move("retract", [-0.44, -0.20, 1.00])
        self._move("raise", [-0.44, -0.20, box[2] + config.pregrasp_height])
        self._move("pregrasp", box + [0, 0, config.pregrasp_height])
        # The shared lift is capped at 8 cm/s; allow the full descent rather
        # than treating lag from a short pose command as a grasp failure.
        self._move("approach", box, seconds=config.descend_seconds)
        self._stage("close")
        for g in np.linspace(0, config.grip, max(1, int(config.close_seconds / env.dt))):
            self._tick(box, float(g))
        if not self.last_forces["opposing_contacts"]:
            raise InteractionFailure("close: no opposing multi-body hand contact")
        self._move("lift", box + [0, 0, config.lift_height], config.grip)
        actual = env.data.body(product.name).xpos
        if (
            actual[2] < box[2] + config.lift_height * 0.65
            or self.last_forces["hand"] < 0.2
            or self.last_forces["table"] > 0.15
        ):
            raise InteractionFailure("lift: object did not rise with the hand under contact")
        # Retain the measured relative grasp offset; never attach or move the object directly.
        offset = self.controller.position() - actual
        placement = np.array(product.target)
        placement[0] = env.config.front_x - product.half_size[0] - 0.006
        self._move("transfer", placement + offset + [0, 0, config.lift_height], config.grip)
        self._move("place", placement + offset, config.grip)
        self._stage("release")
        for g in np.linspace(config.grip, 0, max(1, int(config.close_seconds / env.dt))):
            self._tick(placement + offset, float(g))
        if self.last_forces["shelf"] < 0.2 or self.last_forces["hand"] > 0.2:
            raise InteractionFailure("release: no measured shelf support")
        placed = env.data.body(product.name).xpos.copy()
        self._move("withdraw", placed + [0.13, -0.04, 0.03])
        self._stage("push")
        start = self.controller.position().copy()
        held = 0.0
        for k in range(int(config.push_distance / config.push_speed / env.dt)):
            target = start - [(k + 1) * config.push_speed * env.dt, 0, 0]
            self._tick(target, 0.0)
            contacted = (
                self.last_forces["back"] >= config.contact_threshold
                and self.last_forces["hand"] >= 0.2
            )
            held = held + env.dt if contacted else 0.0
            if held >= config.contact_dwell:
                self.events[-1]["back_contact_verified"] = True
                break
            if self.terminal:
                raise InteractionFailure("push: task ended before contact dwell validation")
        else:
            raise InteractionFailure("push: shelf-back contact was not established")
        self._move("clear", self.controller.position() + [0.13, 0, 0.03])

    def run(self) -> dict:
        from abstract_stack.bundle import build_scene_bundle

        build_scene_bundle(self.env, self.directory / "initial-scene")
        # The bundle settles the world; initialize IK targets from that measured state.
        self.controller = CartesianHand(self.env, self.config)
        self.base_start = self.env.data.body("mabel_chassis").xpos.copy()
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.env.model, state_spec))
        mujoco.mj_getState(self.env.model, self.env.data, state, state_spec)
        np.savez_compressed(
            self.directory / "rollout_start.npz", state=state, state_spec=int(state_spec)
        )
        result = {
            "physical_success": False,
            "failure": None,
            "completed_products": [],
            "controller": asdict(self.config),
            "assistance": "none",
        }
        try:
            for product in self.env.products:
                self.product = product
                self._one_box()
                result["completed_products"].append(product.name)
            self._stage("settle")
            for _ in range(
                0 if self.terminal else int(self.env.config.stable_seconds / self.env.dt) + 10
            ):
                if self._tick(self.controller.position(), 0):
                    break
            result["physical_success"] = self.env.info()["success"]
            if not result["physical_success"]:
                result["failure"] = "Final stocking stability criteria were not met"
        except (RuntimeError, ValueError, FloatingPointError) as error:
            result["failure"] = str(error)
        finally:
            self.env.data.ctrl[:] = stationary_targets(self.env.model, self.env.data)
            result["events"] = self.events
            result["last_ik_residual"] = self.controller.last_ik_error
            result["base_displacement_m"] = float(
                np.linalg.norm(
                    self.env.data.body("mabel_chassis").xpos - self.base_start,
                )
            )
            result["base_orientation_wxyz"] = self.env.data.body("mabel_chassis").xquat.tolist()
            result["last_actuator_targets"] = dict(
                zip(
                    [*self.controller.names, "Lift"],
                    self.controller.command.tolist(),
                    strict=True,
                )
            )
            result["last_measured_actuator_lengths"] = {
                name: float(self.env.data.actuator(name).length[0])
                for name in [*self.controller.names, "Lift"]
            }
            result["metrics"] = self.env.info()
            result["trace_columns"] = [
                "sim_time",
                "box_x",
                "box_y",
                "box_z",
                "hand_x",
                "hand_y",
                "hand_z",
                "hand_normal_N",
                "support_normal_N",
                "back_normal_N",
                "arm_fixture_normal_N",
                "tracking_error_m",
            ]
            trace = np.asarray(self.rows, dtype=float).reshape(-1, 12)
            np.savez_compressed(
                self.directory / "trace.npz",
                samples=trace,
                controls=np.asarray(self.controls).reshape(-1, self.env.model.nu),
            )
            if len(trace):
                result["peak_forces_N"] = dict(
                    zip(
                        ("hand", "support", "back", "arm_fixture"),
                        trace[:, 7:11].max(axis=0).tolist(),
                        strict=True,
                    )
                )
            if self.render:
                from PIL import Image

                Image.fromarray(self.env.render()).save(self.directory / "final.png")
            (self.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
