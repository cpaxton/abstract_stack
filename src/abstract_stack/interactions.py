"""First-pass actuator-driven stocking, with measured contact forces and failure traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.control import stationary_targets
from abstract_stack.ik import IKPrediction, PoseIK, rotation_error


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
    ik_position_tolerance: float = 0.015
    ik_orientation_tolerance_deg: float = 12.0
    ik_orientation_span_deg: float = 40.0
    staging_distance_limit: float = 0.10
    best_effort_staging: bool = True
    max_motion_seconds: float = 12.0

    def __post_init__(self):
        values = asdict(self)
        values.pop("best_effort_staging")
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Script settings must be finite and positive")
        if self.grip > 1 or self.max_force <= self.contact_threshold:
            raise ValueError("grip must be <= 1 and max_force must exceed contact_threshold")
        if not isinstance(self.best_effort_staging, bool):
            raise ValueError("best_effort_staging must be true or false")
        if self.ik_orientation_span_deg > 90 or self.ik_orientation_tolerance_deg > 45:
            raise ValueError("IK orientation relaxation/tolerance exceeds the grasp search bounds")


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
        self.active_rotation = self.ROTATION.copy()
        self.ik = PoseIK(
            self.model, self.data, self.names, self.lower, self.upper, self.home, self.OFFSET
        )
        self.last_prediction = None

    def position(self):
        return self.data.xpos[self.palm] + self.data.xmat[self.palm].reshape(3, 3) @ self.OFFSET

    def targets(self, target, grip):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Hand target must contain three finite coordinates")
        self.last_target = target.copy()
        q, point, rotation = self.ik.solve(
            target, self.active_rotation, self.command, iterations=30
        )
        self.last_ik_error = float(
            np.linalg.norm(
                np.r_[target - point, 0.15 * rotation_error(rotation, self.active_rotation)]
            )
        )
        position_error = float(np.linalg.norm(target - point))
        angle_error = float(np.linalg.norm(rotation_error(rotation, self.active_rotation)))
        if position_error > self.config.ik_position_tolerance or angle_error > np.radians(
            self.config.ik_orientation_tolerance_deg
        ):
            raise InteractionFailure(
                f"Online IK rejected before actuation: predicted miss {position_error:.3f} m, "
                f"orientation error {np.degrees(angle_error):.1f} deg"
            )
        return self.joint_targets(q, grip)

    def joint_targets(self, q, grip):
        q = np.asarray(q, dtype=float)
        if q.shape != self.command.shape or not np.isfinite(q).all():
            raise ValueError("Joint target must contain finite arm, wrist and lift coordinates")
        if not np.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError("Grip must be between zero and one")
        # Metres for Lift, radians for joints: use a lower physical lift rate.
        rates = np.r_[np.full(len(self.names), self.config.max_joint_speed), 0.08]
        bounded_q = np.clip(q, self.lower, self.upper)
        self.command = np.clip(
            self.command
            + np.clip(
                bounded_q - self.command,
                -rates * self.env.dt,
                rates * self.env.dt,
            ),
            self.lower,
            self.upper,
        )
        action = dict(zip(self.names, self.command[:-1].tolist(), strict=True))
        action["Lift"] = float(self.command[-1])
        for suffix in self.FLEXION:
            name = f"Left {suffix}"
            lo, hi = self.model.actuator_ctrlrange[self.model.actuator(name).id]
            action[name] = float(lo + grip * (hi - lo))
        return action

    def predict(self, target, *, allowed_object=None, alternatives=True, check_path=True, grip=0):
        fingers = {}
        if grip == 0:
            for suffix in self.FLEXION:
                name = f"Left {suffix}"
                fingers[int(self.model.joint(name).qposadr[0])] = self.model.actuator_ctrlrange[
                    self.model.actuator(name).id, 0
                ]
        prediction = self.ik.predict(
            target,
            self.ROTATION if alternatives else self.active_rotation,
            self.command,
            position_tolerance=self.config.ik_position_tolerance,
            angle_tolerance_degrees=self.config.ik_orientation_tolerance_deg,
            orientation_span_degrees=self.config.ik_orientation_span_deg,
            max_joint_speed=self.config.max_joint_speed,
            allowed_object=allowed_object,
            check_path=check_path,
            alternatives=alternatives,
            finger_positions=fingers,
        )
        self.last_prediction = prediction
        return prediction

    def execution_diagnostics(self, prediction):
        """Attribute FK displacement in a fixed order, without changing live state."""
        measured = self.ik.measured_q()
        at_base, _ = self.ik.forward(prediction.q)
        arm_q = prediction.q.copy()
        arm_q[:-1] = measured[:-1]
        at_arm, _ = self.ik.forward(arm_q)
        at_lift, _ = self.ik.forward(measured)
        return {
            "base_and_uncontrolled_xyz_m": (at_base - prediction.position).tolist(),
            "arm_and_wrist_xyz_m": (at_arm - at_base).tolist(),
            "lift_xyz_m": (at_lift - at_arm).tolist(),
            "remaining_model_difference_xyz_m": (self.position() - at_lift).tolist(),
            "joint_error_deg": dict(
                zip(self.names, np.degrees(measured[:-1] - prediction.q[:-1]).tolist(), strict=True)
            ),
            "lift_error_m": float(measured[-1] - prediction.q[-1]),
            "scope": "ordered FK decomposition, not an independent force-causality estimate",
        }

    def staging_pose(self, prediction: IKPrediction):
        if (
            not prediction.path_checked
            or not prediction.safe
            or prediction.baseline_qpos is None
            or not np.array_equal(prediction.baseline_qpos, self.data.qpos)
            or prediction.preferred_orientation_error_rad
            > np.radians(
                self.config.ik_orientation_span_deg + self.config.ik_orientation_tolerance_deg
            )
        ):
            return None
        start = self.ik.measured_q()
        point, _ = self.ik.forward(start, prediction.reference_qpos)
        initial_error = float(np.linalg.norm(point - prediction.target))
        best = None
        for t in np.linspace(0, 1, 51)[1:]:
            q = start + t * (prediction.q - start)
            position, rotation = self.ik.forward(q, prediction.reference_qpos)
            if np.linalg.norm(position - point) > self.config.staging_distance_limit:
                break
            error = float(np.linalg.norm(position - prediction.target))
            if error < initial_error - 0.005 and (best is None or error < best.position_error_m):
                rates = np.r_[np.full(len(self.names), self.config.max_joint_speed), 0.08]
                best = replace(
                    prediction,
                    q=q.copy(),
                    position=position,
                    rotation=rotation,
                    position_error_m=error,
                    feasible=False,
                    orientation_error_rad=float(
                        np.linalg.norm(rotation_error(rotation, prediction.requested_rotation))
                    ),
                    preferred_orientation_error_rad=float(
                        np.linalg.norm(rotation_error(rotation, self.ROTATION))
                    ),
                    candidate="staging:" + prediction.candidate,
                    minimum_seconds=float(1.875 * np.max(np.abs(q - start) / rates)),
                    active_limits=[],
                )
        if best is not None and self.ik.check_path(
            best.q,
            best.reference_qpos,
            self.config.max_joint_speed,
            allowed_object=best.allowed_contact_object,
            baseline_reference=best.baseline_qpos,
        ):
            return None
        return best


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
        return self._apply_tick(self.controller.targets(target, grip))

    def _apply_tick(self, action):
        _, _, terminated, truncated, _ = self.env.step(
            action,
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
        target = np.asarray(target, dtype=float)
        allow_object = self.product.name if (grip > 0 or name in ("approach", "place")) else None
        prediction = self.controller.predict(
            target,
            allowed_object=allow_object,
            alternatives=name != "approach",
            grip=grip,
        )
        event = self.events[-1]
        event["ik_prediction"] = prediction.report()
        staging = False
        if not prediction.feasible:
            event["decision"] = "hold_for_replan"
            if self.config.best_effort_staging and grip == 0:
                safe_prediction = self.controller.predict(target) if allow_object else prediction
                candidate = self.controller.staging_pose(safe_prediction)
                if candidate is not None:
                    prediction, staging = candidate, True
                    event["decision"] = "bounded_staging_then_replan"
                    event["staging_prediction"] = prediction.report()
            if not staging:
                raise InteractionFailure(
                    f"{name}: IK prefilter found no acceptable pose/path; "
                    f"predicted miss {prediction.position_error_m:.3f} m; replan required"
                )
        else:
            event["decision"] = "execute_accepted_pose"
        self.controller.active_rotation = prediction.requested_rotation.copy()
        self.controller.last_prediction = prediction
        start = self.controller.ik.measured_q()
        duration = max(seconds or self.config.approach_seconds, prediction.minimum_seconds * 1.05)
        event["planned_duration_s"] = duration
        if duration > self.config.max_motion_seconds:
            event["decision"] = "hold_for_duration_replan"
            raise InteractionFailure(
                f"{name}: predicted rate-limited duration exceeds motion limit"
            )
        ticks = max(1, int(np.ceil(duration / self.env.dt)))
        for k in range(ticks):
            t = (k + 1) / ticks
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            action = self.controller.joint_targets(start + blend * (prediction.q - start), grip)
            self.controller.last_target, _ = self.controller.ik.forward(
                self.controller.command,
                prediction.reference_qpos,
            )
            if self._apply_tick(action):
                if name != "clear":
                    raise InteractionFailure(f"{name}: task ended before push/contact validation")
                break
        error = float(np.linalg.norm(self.controller.position() - target))
        event.update(
            end_time=float(self.env.data.time),
            tracking_error=error,
            requested_position_error_m=error,
            measured_position=self.controller.position().tolist(),
            execution_diagnostics=self.controller.execution_diagnostics(prediction),
            prediction_execution_gap_m=float(
                np.linalg.norm(
                    self.controller.position() - prediction.position,
                )
            ),
            measured_orientation_error_deg=float(
                np.degrees(
                    np.linalg.norm(
                        rotation_error(
                            self.env.data.xmat[self.controller.palm].reshape(3, 3),
                            prediction.requested_rotation,
                        )
                    )
                )
            ),
        )
        if self.render:
            from PIL import Image

            Image.fromarray(self.env.render("overview")).save(
                self.directory / f"{self.product.name}-{name}.png",
            )
        if staging:
            raise InteractionFailure(
                f"{name}: bounded staging complete; predicted remaining miss "
                f"{prediction.position_error_m:.3f} m; replan required before grasping"
            )
        if check_tracking and not self.terminal and error > self.config.tracking_tolerance:
            raise InteractionFailure(f"{name}: measured tracking error {error:.3f} m")

    def _one_box(self):
        env, config, product = self.env, self.config, self.product
        box = env.data.body(product.name).xpos.copy()
        self._stage("grasp_precheck")
        grasp = self.controller.predict(box, allowed_object=product.name, check_path=False)
        self.events[-1]["ik_prediction"] = grasp.report()
        if not grasp.feasible:
            self.events[-1]["decision"] = "reject_grasp_candidate"
            self._move("grasp_staging", box)
            raise InteractionFailure("Grasp candidate rejected; replan required")
        self.events[-1]["decision"] = "accept_grasp_candidate"
        grasp_rotation = grasp.requested_rotation.copy()
        self._move("retract", [-0.44, -0.20, 1.00])
        self._move("raise", [-0.44, -0.20, box[2] + config.pregrasp_height])
        self._move("pregrasp", box + [0, 0, config.pregrasp_height])
        self.controller.active_rotation = grasp_rotation
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
            if self.controller.last_prediction is not None:
                result["last_ik_prediction"] = self.controller.last_prediction.report()
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
