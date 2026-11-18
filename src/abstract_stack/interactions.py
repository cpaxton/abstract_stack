"""First-pass actuator-driven stocking, with measured contact forces and failure traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.control import stationary_targets
from abstract_stack.ik import IKPrediction, PoseIK, rotation_error
from abstract_stack.mabel_controller import MabelController, TrackingConfig


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
    gravity_compensation: bool = True
    adaptive_grasps: bool = True
    grasp_force_target: float = 4.5
    push_force_target: float = 1.875
    grip_force_gain: float = 0.04
    grip_max_rate: float = 0.25
    carry_clearance: float = 0.10
    release_forward: float = 0.12
    release_up: float = 0.025
    release_seconds: float = 3.0
    release_drag_tolerance: float = 0.025
    payload_orientation_speed: float = 0.30
    payload_upright_degrees: float = 5.0
    push_timeout_seconds: float = 12.0
    insertion_clearance: float = 0.025
    insertion_vertical_speed: float = 0.01
    head_tracking: bool = True
    contact_aware: bool = True
    guarded_approach: bool = False
    grasp_skill: bool = False
    record_skill_starts: bool = False
    fixture_avoidance: bool = True
    approach_force_target: float = 15.0
    release_lateral_limit: float = 0.04
    contact_grasp_seconds: float = 8.0
    contact_descent_speed: float = 0.015
    fixture_force_target: float = 3.0
    fixture_unload_speed: float = 0.015
    fixture_unload_distance: float = 0.025
    fixture_clearance: float = 0.004
    thumb_preshape: bool = True
    alignment_timeout_seconds: float = 6.0
    insertion_yaw_lever_gain: float = 0.12
    insertion_back_force_target: float = 0.6
    front_staging_depth: float = 0.07
    insertion_lateral_position_gain: float = 0.2
    insertion_tangent_admittance: float = 0.004

    def __post_init__(self):
        values = asdict(self)
        values.pop("best_effort_staging")
        values.pop("gravity_compensation")
        values.pop("adaptive_grasps")
        values.pop("head_tracking")
        values.pop("contact_aware")
        values.pop("guarded_approach")
        values.pop("grasp_skill")
        values.pop("record_skill_starts")
        values.pop("thumb_preshape")
        values.pop("fixture_avoidance")
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Script settings must be finite and positive")
        if self.grip > 1 or self.max_force <= self.contact_threshold:
            raise ValueError("grip must be <= 1 and max_force must exceed contact_threshold")
        if self.front_staging_depth > 0.12:
            raise ValueError("Front staging depth must be within the bounded 0.12 m corridor")
        if (
            max(
                self.grasp_force_target,
                self.push_force_target,
                self.approach_force_target,
                self.fixture_force_target,
                self.insertion_back_force_target,
            )
            >= self.max_force
        ):
            raise ValueError("force targets must be below the contact abort limit")
        if not isinstance(self.best_effort_staging, bool):
            raise ValueError("best_effort_staging must be true or false")
        if not isinstance(self.gravity_compensation, bool):
            raise ValueError("gravity_compensation must be true or false")
        if not isinstance(self.adaptive_grasps, bool):
            raise ValueError("adaptive_grasps must be true or false")
        if not isinstance(self.grasp_skill, bool) or not isinstance(self.record_skill_starts, bool):
            raise ValueError("grasp_skill and record_skill_starts must be boolean")
        if not isinstance(self.thumb_preshape, bool):
            raise ValueError("thumb_preshape must be boolean")
        if not isinstance(self.head_tracking, bool):
            raise ValueError("head_tracking must be true or false")
        if not isinstance(self.contact_aware, bool):
            raise ValueError("contact_aware must be true or false")
        if not isinstance(self.guarded_approach, bool) or not isinstance(
            self.fixture_avoidance, bool
        ):
            raise ValueError("guarded_approach and fixture_avoidance must be boolean")
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


def contact_forces(
    env, product_name: str, hand_bodies: set[int], *, all_products=None, fixture_vector_bodies=None
) -> dict:
    """Sum measured nonnegative contact-normal forces, in newtons, by counterpart."""
    model, data = env.model, env.data
    if fixture_vector_bodies is None:
        fixture_vector_bodies = hand_bodies
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
        "other_product": 0.0,
        "thumb_normal": 0.0,
        "nonthumb_normal": 0.0,
    }
    hand_contacts = set()
    normals = []
    normal_vector, hand_force, fixture_normal = np.zeros(3), np.zeros(3), np.zeros(3)
    hand_moment, contact_position = np.zeros(3), np.zeros(3)
    for i, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal = max(0.0, float(force[0]))
        if all_products is not None and (
            (b1 in hand_bodies and b2 in all_products and b2 != product)
            or (b2 in hand_bodies and b1 in all_products and b1 != product)
        ):
            result["other_product"] += normal
        if product in (b1, b2):
            other, other_geom = (b2, g2) if b1 == product else (b1, g1)
            if other in hand_bodies:
                result["hand"] += normal
                result[
                    "thumb_normal" if "thumb" in model.body(other).name else "nonthumb_normal"
                ] += normal
                sign = -1 if b1 == product else 1
                normal_vector += sign * normal * contact.frame[:3]
                hand_force += sign * contact.frame.reshape(3, 3).T @ force[:3]
                world_force = sign * contact.frame.reshape(3, 3).T @ force[:3]
                hand_moment += np.cross(contact.pos - data.xpos[product], world_force)
                hand_moment += sign * contact.frame.reshape(3, 3).T @ force[3:]
                contact_position += normal * contact.pos
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
            if b1 in fixture_vector_bodies or b2 in fixture_vector_bodies:
                fixture_normal += (-1 if b1 in hand_bodies else 1) * normal * contact.frame[:3]
    result["hand_contact_bodies"] = len(hand_contacts)
    result["grip_normal"] = max(0.0, result["hand"] - float(np.linalg.norm(normal_vector)))
    result["push_force"] = max(0.0, -float(hand_force[0]))
    result["hand_force_world_N"] = hand_force.tolist()
    result["hand_moment_about_body_origin_Nm"] = hand_moment.tolist()
    result["hand_contact_position_world_m"] = (
        (contact_position / result["hand"]).tolist() if result["hand"] > 1e-9 else None
    )
    result["fixture_normal_world_N"] = fixture_normal.tolist()
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
            + [
                max(
                    sum(j.range[0] for j in self.lift_joints),
                    self.model.actuator("Lift").ctrlrange[0],
                )
                + 1e-5
            ]
        )
        self.upper = np.array(
            [
                min(self.model.joint(n).range[1], self.model.actuator_ctrlrange[a, 1]) - 1e-5
                for n, a in zip(self.names, self.aids, strict=True)
            ]
            + [
                min(
                    sum(j.range[1] for j in self.lift_joints),
                    self.model.actuator("Lift").ctrlrange[1],
                )
                - 1e-5
            ]
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
        tower = self.model.body("left_tower").id
        self.tower_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == tower
            and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
        ]

    def tower_clearance(self, prediction, level):
        """Signed native collision-geometry distance, evaluated only on scratch data."""
        self.ik.forward(prediction.q, prediction.reference_qpos)
        board = self.model.geom(f"shelf_board_{level}").id
        best, direction = float("inf"), np.array([0.0, 0.0, 1.0])
        for geom in self.tower_geoms:
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(
                self.model, self.ik.scratch, geom, board, 0.2, segment
            )
            if distance < best:
                best = float(distance)
                if abs(distance) > 1e-9:
                    direction = (segment[:3] - segment[3:]) / distance
                    norm = np.linalg.norm(direction)
                    direction = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        return best, direction

    def check_hand_step(self, prediction, grip, allowed_object, *, targets=None):
        """Check three coupled arm/finger poses on scratch data, never live qpos."""
        start = self.ik.measured_q()
        reference = self.data.qpos.copy()
        object_id = None if allowed_object is None else self.model.body(allowed_object).id
        baseline = self.ik._contacts(start, reference, object_id)
        for alpha in (0.0, 0.5, 1.0):
            fingers = reference.copy()
            for name, goal in (self.finger_targets(grip) if targets is None else targets).items():
                adr = int(self.model.joint(name).qposadr[0])
                fingers[adr] = reference[adr] + alpha * (goal - reference[adr])
            errors = self.ik.collision_errors(
                start + alpha * (prediction.q - start), fingers, baseline, object_id
            )
            if errors:
                return errors
        return []

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

    def finger_targets(self, grip):
        """Native finger references, including clearance at the thumb base."""
        action = {}
        for suffix in self.FLEXION:
            name = f"Left {suffix}"
            lo, hi = self.model.actuator(name).ctrlrange
            action[name] = float(lo + grip * (hi - lo))
        if self.config.thumb_preshape:
            # The neutral thumb metacarpal catches the carton's top edge during
            # descent. Abduct it before approach and retain that grasp posture
            # during carry/opening; this changes references, never the model.
            name = "Left Thumb MCP"
            if (
                not self.model.actuator(name).ctrlrange[0]
                <= 0.4
                <= self.model.actuator(name).ctrlrange[1]
            ):
                raise ValueError("MABEL thumb preshape exceeds native actuator limits")
            action[name] = 0.4
        return action

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
        action.update(self.finger_targets(grip))
        return action

    def predict(
        self,
        target,
        *,
        allowed_object=None,
        alternatives=True,
        check_path=True,
        grip=0,
        carrying=None,
        sliding=False,
        path_mode="joint",
    ):
        fingers = {}
        if grip == 0:
            fingers = {
                int(self.model.joint(name).qposadr[0]): value
                for name, value in self.finger_targets(0).items()
            }
        elif self.config.thumb_preshape:
            fingers[int(self.model.joint("Left Thumb MCP").qposadr[0])] = 0.4
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
            carrying=grip > 0 if carrying is None else carrying,
            sliding=sliding,
            path_mode=path_mode,
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
            or prediction.carried_object is not None
            or prediction.sliding_object is not None
            or self.ik.carry is not None
            or self.ik.slide is not None
            or prediction.reference_ranges != self.ik.reference_range_report()
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
                    path_mode="joint",
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

    FRONT_SUPPORT = "shelf"
    CONTACT_PROTOCOL = "grasped_supported_slide"
    ALLOW_OFFSET_START = False
    TASK_SCOPE = "stocking"

    def episode_evidence(self):
        """Read-only controller-specific diagnostics for the terminal artifact."""
        return {}

    def __init__(
        self,
        env,
        directory: str | Path,
        config: ScriptConfig | None = None,
        *,
        render=False,
        video=False,
        video_fps=15,
        video_width=640,
        video_height=480,
        first_person="zed_left",
        third_person="stocking_third_person",
        force_profile=None,
    ):
        if not env.robot or not env.products or any(p.kind != "carton" for p in env.products):
            raise ValueError("The script requires MABEL and a carton-only scene (pick_place)")
        if env.config.environment != "studio" or (
            not self.ALLOW_OFFSET_START and tuple(env.config.robot_xy) != (0.0, 0.0)
        ):
            raise ValueError("This first script supports the stationary pick_place bench only")
        self.env, self.config = env, config or ScriptConfig()
        self.force_profile = force_profile
        self.directory, self.render = Path(directory), render
        self.directory.mkdir(parents=True, exist_ok=True)
        self.controller = self._new_cartesian_hand()
        self.hand_bodies = _descendants(env.model, "left_palm")
        self.left_arm_bodies = _descendants(env.model, "l_shoulder_1")
        self.arm_bodies = self.left_arm_bodies | _descendants(env.model, "r_shoulder_1")
        self.rows, self.events, self.controls = [], [], []
        self.terminal = False
        self.product = env.products[0]
        self.product_bodies = {env.model.body(p.name).id for p in env.products}
        self.phase = "settle"
        self.last_forces = {}
        self.carry_reference = None
        self.active_grip = None
        self.release_origin = None
        self.payload_rotation_goal = None
        self.supported_slide = False
        self.support_contact_loss = 0.0
        self.video_options = (
            {
                "fps": video_fps,
                "width": video_width,
                "height": video_height,
                "first_person": first_person,
                "third_person": third_person,
            }
            if video
            else None
        )
        self.recorder, self.gaze = None, None
        self.observer = None
        self.fixture_unloading = False
        self.fixture_feedback = np.zeros(3)
        self.skill_entries = []

    def _new_cartesian_hand(self):
        """Construct the empty hand again after settling, preserving subclass policy."""
        return CartesianHand(self.env, self.config)

    def _tracking_config(self):
        return TrackingConfig(
            gravity_compensation=self.config.gravity_compensation,
            joint_speed=self.config.max_joint_speed,
        )

    def _check_tracked_targets(self, targets):
        """Optional final coupled geometry guard, before native actuation."""

    def _sample(self, env):
        if self.observer is not None:
            self.observer.sample(self.phase, self.product.name)
        if self.rows and env.data.time <= self.rows[-1][0]:
            raise InteractionFailure("MuJoCo time did not advance; possible physics reset")
        forces = contact_forces(env, self.product.name, self.hand_bodies)
        arm_forces = contact_forces(
            env,
            self.product.name,
            self.arm_bodies,
            all_products=self.product_bodies,
            fixture_vector_bodies=self.left_arm_bodies,
        )
        forces["arm_fixture"] = arm_forces["arm_fixture"]
        forces["other_product"] = arm_forces["other_product"]
        forces["arm_product"] = max(0.0, arm_forces["hand"] - forces["hand"])
        forces["robot_product"] = arm_forces["hand"]
        forces["fixture_normal_world_N"] = arm_forces["fixture_normal_world_N"]
        vector = np.array(forces["fixture_normal_world_N"])
        if np.linalg.norm(vector) > np.linalg.norm(self.fixture_feedback):
            self.fixture_feedback = vector
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
                *env.data.body(self.product.name).xquat,
                forces["grip_normal"],
                forces["push_force"],
                *forces["fixture_normal_world_N"],
                forces["arm_product"],
                forces["other_product"],
            ]
        )
        self.controls.append(env.data.ctrl.copy())
        if self.recorder is not None:
            self.recorder.capture(phase=self.phase, product=self.product.name, forces=forces)
        if self.supported_slide:
            body = env.data.body(self.product.name)
            extent = np.abs(body.xmat.reshape(3, 3)) @ np.array(self.product.half_size)
            gap = body.xpos[2] - extent[2] - env.config.levels[self.product.level]
            # Uprighting changes the box's vertical extent and can briefly
            # lift it a few millimetres. Permit this bounded, hand-supported
            # mode; the vertical feedback restores board support.
            loaded = forces["hand"] > 0.2 and (
                forces["shelf"] > 0.2 or -0.003 <= gap <= self.config.insertion_clearance
            )
            self.support_contact_loss = (
                0.0 if loaded else self.support_contact_loss + env.config.timestep
            )
            if self.support_contact_loss > 0.10:
                raise InteractionFailure("push: lost shelf support or hand contact")
        if self.carry_reference is not None and self.phase not in ("release", "clear"):
            relative = env.data.xmat[self.controller.palm].reshape(3, 3).T @ (
                position - env.data.xpos[self.controller.palm]
            )
            if np.linalg.norm(relative - self.carry_reference) > 0.03:
                raise InteractionFailure(
                    f"{self.phase}: carried object slipped from measured grasp"
                )
        if (
            self.release_origin is not None
            and forces["hand"] > 0.2
            and np.linalg.norm(position - self.release_origin) > self.config.release_drag_tolerance
        ):
            raise InteractionFailure("release: product moved beyond bounded ungrasp allowance")
        if (
            max(
                # Combined hand + non-grasping arm normal load. Components
                # remain separate in diagnostics; their sum guards contact.
                forces["robot_product"],
                forces["back"],
                forces["arm_fixture"],
                forces["other_product"],
            )
            > self.config.max_force
        ):
            raise InteractionFailure(
                f"{self.phase}: contact force exceeded {self.config.max_force} N"
            )
        if env.data.body("mabel_chassis").xmat.reshape(3, 3)[2, 2] < 0.95:
            raise InteractionFailure(f"{self.phase}: robot tilt limit exceeded")

    def _tick(self, target, grip):
        if self.phase == "push" and self.payload_rotation_goal is not None:
            actual = self.env.data.body(self.product.name).xmat.reshape(3, 3)
            palm_rotation = self.env.data.xmat[self.controller.palm].reshape(3, 3)
            desired = self.payload_rotation_goal @ actual.T @ palm_rotation
            vector = rotation_error(self.controller.active_rotation, desired)
            angle = float(np.linalg.norm(vector))
            if self.fixture_unloading:
                self.controller.active_rotation = palm_rotation.copy()
            elif angle > 1e-9:
                quat, matrix = np.empty(4), np.empty(9)
                mujoco.mju_axisAngle2Quat(
                    quat,
                    vector / angle,
                    min(angle, self.config.payload_orientation_speed * self.env.dt),
                )
                mujoco.mju_quat2Mat(matrix, quat)
                self.controller.active_rotation = (
                    matrix.reshape(3, 3) @ self.controller.active_rotation
                )
            prediction = self.controller.predict(
                target,
                allowed_object=self.product.name,
                alternatives=False,
                grip=grip,
            )
            event = self.events[-1]
            if self.config.contact_aware and self.config.fixture_avoidance and prediction.feasible:
                clearance, normal = self.controller.tower_clearance(prediction, self.product.level)
                event["minimum_predicted_tower_clearance_m"] = min(
                    event.get("minimum_predicted_tower_clearance_m", float("inf")), clearance
                )
                if clearance < self.config.fixture_clearance:
                    correction = self.config.fixture_clearance - clearance
                    # Prefer lifting above a board edge over pushing back out of
                    # the shelf. The measured support/clearance guard still applies.
                    if normal[2] > 0.2:
                        target[2] += correction / normal[2]
                    else:
                        target += correction * normal
                    origin = np.array(event["push_origin_m"])
                    if (
                        not origin[0] - self.config.push_distance <= target[0] <= origin[0] + 0.03
                        or target[2] - origin[2] > self.config.fixture_unload_distance
                    ):
                        raise InteractionFailure(
                            "push: fixture clearance correction outside bounded motion"
                        )
                    prediction = self.controller.predict(
                        target, allowed_object=self.product.name, alternatives=False, grip=grip
                    )
                    event["clearance_corrections"] = event.get("clearance_corrections", 0) + 1
            origin = np.array(event["push_origin_m"])
            if (
                not origin[0] - self.config.push_distance <= target[0] <= origin[0] + 0.03
                or target[2] - origin[2] > self.config.fixture_unload_distance
            ):
                raise InteractionFailure("push: final feedback target outside bounded motion")
            event["feedback_checks"] = event.get("feedback_checks", 0) + 1
            if not prediction.feasible:
                event["feedback_rejection"] = prediction.report()
                raise InteractionFailure("push: feedback pose/path rejected before actuation")
            self.controller.last_target = np.asarray(target).copy()
            return self._apply_tick(self.controller.joint_targets(prediction.q, grip))
        previous = self.controller.command.copy()
        action = self.controller.targets(target, grip)
        if self.config.grasp_skill and self.phase == "close":
            collisions = self.controller.check_hand_step(
                SimpleNamespace(q=self.controller.command.copy()), grip, self.product.name
            )
            if collisions:
                self.controller.command = previous
                self.events[-1]["finger_sweep_rejection"] = collisions
                raise InteractionFailure(
                    "close: coupled arm/finger geometry rejected before actuation"
                )
        return self._apply_tick(action)

    def _load_feedforward(self):
        return None

    def _apply_tick(self, action):
        if self.gaze is not None:
            action = dict(action)
            action.update(
                self.gaze.targets(self.env.data.body(self.product.name).xpos.copy(), self.env.dt)
            )
        if self.active_grip is not None and self.phase != "release":
            force_error = self.config.grasp_force_target - self.last_forces.get("grip_normal", 0)
            self.active_grip = float(
                np.clip(
                    self.active_grip
                    + np.clip(
                        self.config.grip_force_gain * force_error,
                        -self.config.grip_max_rate,
                        self.config.grip_max_rate,
                    )
                    * self.env.dt,
                    0,
                    self.config.grip,
                )
            )
            action = dict(action)
            for suffix in self.controller.FLEXION:
                name = f"Left {suffix}"
                lo, hi = self.env.model.actuator(name).ctrlrange
                action[name] = float(lo + self.active_grip * (hi - lo))
        self.fixture_feedback = np.zeros(3)
        targets = self.tracker.targets(
            action, dt=self.env.dt, load_feedforward=self._load_feedforward()
        )
        self._check_tracked_targets(targets)
        _, _, terminated, truncated, _ = self.env.step(
            targets,
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
        (self.directory / "physics-progress.json").write_text(
            json.dumps(self.events[-1], indent=2) + "\n"
        )
        if self.config.record_skill_starts:
            from abstract_stack.skill_entries import record_skill_entry

            record_skill_entry(self, name)

    def _move(
        self,
        name,
        target,
        grip=0.0,
        seconds=None,
        *,
        check_tracking=True,
        carrying=None,
        orientation_alternatives=None,
    ):
        self._stage(name)
        if grip > 0 and self.payload_rotation_goal is not None:
            # Align the actual load, rather than assuming a constant hand pose
            # prevents angular slip. This is an explicit payload goal, not an
            # orientation relaxation to satisfy IK.
            object_rotation = self.env.data.body(self.product.name).xmat.reshape(3, 3)
            palm_rotation = self.env.data.xmat[self.controller.palm].reshape(3, 3)
            self.controller.active_rotation = (
                self.payload_rotation_goal @ object_rotation.T @ palm_rotation
            )
            self.events[-1]["payload_orientation_goal"] = self.payload_rotation_goal.tolist()
        target = np.asarray(target, dtype=float)
        cartesian_entry = self.config.grasp_skill and name == "approach" and grip == 0
        allow_object = (
            self.product.name
            if not cartesian_entry and (grip > 0 or name in ("approach", "place"))
            else None
        )
        prediction = self.controller.predict(
            target,
            allowed_object=allow_object,
            alternatives=(name != "approach" and grip == 0)
            if orientation_alternatives is None
            else orientation_alternatives,
            grip=grip,
            carrying=carrying,
            path_mode="cartesian" if cartesian_entry else "joint",
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
        path, knots = None, None
        if cartesian_entry and not staging:
            from abstract_stack.grasping import timed_joint_path

            if prediction.joint_path is None:
                raise InteractionFailure("approach: Cartesian entry lacks checked waypoints")
            rates = np.r_[np.full(len(self.controller.names), self.config.max_joint_speed), 0.08]
            path, knots, minimum = timed_joint_path(prediction.joint_path, rates)
            duration = max(duration, minimum * 1.05)
            event.update(
                decision="execute_checked_cartesian_grasp_entry",
                active_object_contact_allowed=False,
                retained_waypoints=len(path),
                reference_timing="segment joint-rate cost with quintic peak bound",
            )
        event["planned_duration_s"] = duration
        if duration > self.config.max_motion_seconds:
            event["decision"] = "hold_for_duration_replan"
            raise InteractionFailure(
                f"{name}: predicted rate-limited duration exceeds motion limit"
            )
        ticks = max(1, int(np.ceil(duration / self.env.dt)))
        contact_stopped = False
        for k in range(ticks):
            t = (k + 1) / ticks
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            q = (
                start + blend * (prediction.q - start)
                if path is None
                else np.array([np.interp(blend, knots, path[:, i]) for i in range(path.shape[1])])
            )
            action = self.controller.joint_targets(q, grip)
            self.controller.last_target, _ = self.controller.ik.forward(
                self.controller.command,
                prediction.reference_qpos,
            )
            if self._apply_tick(action):
                if name != "clear":
                    raise InteractionFailure(f"{name}: task ended before push/contact validation")
                break
            if (
                name == "approach"
                and self.config.contact_aware
                and self.config.guarded_approach
                and self.last_forces["robot_product"] >= self.config.approach_force_target
            ):
                contact_stopped = True
                event.update(
                    decision="cancel_descent_and_check_measured_grasp",
                    measured_contact_stop_N=self.last_forces["robot_product"],
                )
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
        if (
            check_tracking
            and not contact_stopped
            and not self.terminal
            and error > self.config.tracking_tolerance
        ):
            raise InteractionFailure(f"{name}: measured tracking error {error:.3f} m")
        return contact_stopped

    def _select_grasp(self, pending):
        """Regenerate object/height choices before committing to an approach."""
        if self.config.grasp_skill:
            from abstract_stack.grasp_skill import select_grasp

            return select_grasp(self, pending)
        base = self.env.data.body("mabel_chassis").xpos
        for product in sorted(
            pending, key=lambda p: np.linalg.norm(self.env.data.body(p.name).xpos[:2] - base[:2])
        ):
            self.product = product
            box = self.env.data.body(product.name).xpos.copy()
            # Stay within the product's height; fingers must still establish
            # opposing contact and a physical lift after this geometric screen.
            for height in (product.half_size[2] / 3, 0.0, product.half_size[2] * 2 / 3):
                target = box + [0, 0, height]
                prediction = self.controller.predict(
                    target, allowed_object=product.name, check_path=False
                )
                self._stage("grasp_candidate")
                self.events[-1].update(
                    ik_prediction=prediction.report(),
                    decision="select_candidate" if prediction.feasible else "try_next_candidate",
                )
                if prediction.feasible:
                    return target, prediction
        raise InteractionFailure("No acceptable grasp candidate found; no grasp motion commanded")

    def _one_box(self, selected=None):
        grasp_grip = self._pick_and_place_front(selected)
        self._push_grasped(grasp_grip)

    def _front_center_x(self, extent):
        return self.env.config.front_x - extent - 0.006

    def _pick_and_place_front(self, selected=None):
        """Compose independently callable grasp/lift and loaded transfer skills."""
        grasp_grip = self._grasp_and_lift(selected)
        return self._transfer_and_place_front(grasp_grip)

    def _grasp_and_lift(self, selected=None):
        env, config, product = self.env, self.config, self.product
        box = env.data.body(product.name).xpos.copy()
        self.payload_rotation_goal = env.data.body(product.name).xmat.reshape(3, 3).copy()
        self._stage("grasp_precheck")
        grasp_target, grasp = (
            selected
            if selected is not None
            else (box, self.controller.predict(box, allowed_object=product.name, check_path=False))
        )
        self.events[-1]["ik_prediction"] = grasp.report()
        if not grasp.feasible:
            self.events[-1]["decision"] = "reject_grasp_candidate"
            self._move("grasp_staging", box)
            raise InteractionFailure("Grasp candidate rejected; replan required")
        self.events[-1]["decision"] = "accept_grasp_candidate"
        grasp_rotation = grasp.requested_rotation.copy()
        self._move("retract", [-0.44, -0.20, 1.00])
        self._move("raise", [-0.44, -0.20, box[2] + config.pregrasp_height])
        if config.grasp_skill:
            self.controller.active_rotation = grasp_rotation.copy()
        self._move(
            "pregrasp",
            grasp_target + [0, 0, config.pregrasp_height],
            orientation_alternatives=False if config.grasp_skill else None,
        )
        self.controller.active_rotation = grasp_rotation
        # The shared lift is capped at 8 cm/s; allow the full descent rather
        # than treating lag from a short pose command as a grasp failure.
        contact_stopped = self._move("approach", grasp_target, seconds=config.descend_seconds)
        if contact_stopped:
            requested_grasp_target = grasp_target.copy()
            grasp_target = self.controller.position().copy()
            body = env.data.body(product.name)
            if (
                np.any(
                    np.abs(grasp_target[:2] - body.xpos[:2])
                    > np.array(product.half_size[:2]) + 0.002
                )
                or abs(grasp_target[2] - body.xpos[2]) > product.half_size[2]
            ):
                raise InteractionFailure("approach: contact outside bounded grasp region")
            measured_grasp = self.controller.predict(
                grasp_target,
                allowed_object=product.name,
                alternatives=False,
                check_path=False,
                grip=0,
                carrying=False,
            )
            self.events[-1].update(
                measured_grasp_target=grasp_target.tolist(),
                contact_grasp_prediction=measured_grasp.report(),
            )
            if not measured_grasp.feasible:
                raise InteractionFailure("approach: measured contact grasp rejected before closure")
        self._stage("close")
        grasp_grip = config.grip
        if contact_stopped:
            target, fraction = grasp_target.copy(), 0.0
            self.events[-1].update(
                contact_descent=True, requested_grasp_target=requested_grasp_target.tolist()
            )
            for _ in range(int(config.contact_grasp_seconds / env.dt)):
                at_depth = self.controller.position()[2] <= requested_grasp_target[2] + 0.005
                limit = config.grip if at_depth else min(0.2, config.grip)
                fraction = min(limit, fraction + config.grip_max_rate * env.dt)
                if self.last_forces["robot_product"] < config.approach_force_target:
                    target[2] = max(
                        requested_grasp_target[2], target[2] - config.contact_descent_speed * env.dt
                    )
                else:
                    target[2] = self.controller.position()[2]
                prediction = self.controller.predict(
                    target,
                    allowed_object=product.name,
                    alternatives=False,
                    grip=fraction,
                    carrying=False,
                )
                if not prediction.feasible:
                    self.events[-1]["feedback_rejection"] = prediction.report()
                    raise InteractionFailure(
                        "close: contact-descent path rejected before actuation"
                    )
                collisions = self.controller.check_hand_step(prediction, fraction, product.name)
                if collisions:
                    self.events[-1]["finger_sweep_rejection"] = collisions
                    raise InteractionFailure("close: coupled hand step rejected before actuation")
                self.controller.last_target = target.copy()
                self._apply_tick(self.controller.joint_targets(prediction.q, fraction))
                if (
                    at_depth
                    and self.last_forces["opposing_contacts"]
                    and (self.last_forces["grip_normal"] >= config.grasp_force_target)
                ):
                    break
            else:
                raise InteractionFailure("close: guarded descent/closure timeout")
            grasp_target, grasp_grip = self.controller.position().copy(), fraction
            self.events[-1].update(
                contact_descent=True, measured_grasp_target=grasp_target.tolist()
            )
        else:
            for g in np.linspace(0, config.grip, max(1, int(config.close_seconds / env.dt))):
                self._tick(grasp_target, float(g))
                if self.last_forces["opposing_contacts"] and (
                    self.last_forces["grip_normal"] >= config.grasp_force_target
                ):
                    grasp_grip = float(g)
                    break
        if not self.last_forces["opposing_contacts"]:
            raise InteractionFailure("close: no opposing multi-body hand contact")
        self.events[-1].update(
            closure_fraction=grasp_grip,
            measured_hand_normal_N=self.last_forces["hand"],
        )
        self.active_grip = grasp_grip
        self._move("lift", grasp_target + [0, 0, config.lift_height], grasp_grip)
        actual = env.data.body(product.name).xpos
        if (
            actual[2] < box[2] + config.lift_height * 0.65
            or self.last_forces["hand"] < 0.2
            or self.last_forces["table"] > 0.15
        ):
            raise InteractionFailure("lift: object did not rise with the hand under contact")
        self.events[-1].update(
            verified_lift=True,
            object_rise_m=float(actual[2] - box[2]),
            measured_hand_normal_N=self.last_forces["hand"],
            measured_table_normal_N=self.last_forces["table"],
        )
        self.carry_reference = env.data.xmat[self.controller.palm].reshape(3, 3).T @ (
            actual - env.data.xpos[self.controller.palm]
        )
        return grasp_grip

    def _transfer_and_place_front(self, grasp_grip):
        env, config, product = self.env, self.config, self.product
        actual = env.data.body(product.name).xpos.copy()
        if config.grasp_skill:
            # The table yaw specifies the grasp, not the shelf placement goal.
            # Reorient only after the measured lift has cleared the table.
            # Every loaded move still checks the measured grasp transform and
            # native carried-object path; no object pose is edited here.
            self.payload_rotation_goal = np.eye(3)
        # Retain the measured relative grasp offset; never attach or move the object directly.
        offset = self.controller.position() - actual
        placement = np.array(product.target)
        placement[0] = self._front_center_x(product.half_size[0])
        self._move(
            "carry_clear", self.controller.position() + [0, 0, config.carry_clearance], grasp_grip
        )
        # Route in front of the shelf before insertion, rather than moving
        # diagonally through a side panel with the carried box.
        staging = placement + offset + [0, 0, config.lift_height]
        staging[0] = env.config.front_x + 0.20 + offset[0]
        staging[2] = max(staging[2], self.controller.position()[2])
        if config.grasp_skill:
            # A near-body lateral waypoint can require shoulder adduction that
            # the native arm cannot supply. Search outward shelf approaches
            # from the same measured grasp instead of accepting a poor IK fit.
            self._stage("loaded_stage_plan")
            attempts = []
            body = env.data.body(product.name)
            self.controller.active_rotation = (
                self.payload_rotation_goal
                @ body.xmat.reshape(3, 3).T
                @ env.data.xmat[self.controller.palm].reshape(3, 3)
            )
            for front_clearance in (0.20, 0.16, 0.12, 0.08):
                proposed = staging.copy()
                proposed[0] = env.config.front_x + front_clearance + offset[0]
                prediction = self.controller.predict(
                    proposed,
                    allowed_object=product.name,
                    alternatives=False,
                    grip=grasp_grip,
                )
                attempts.append(
                    dict(front_clearance_m=front_clearance, prediction=prediction.report())
                )
                if (
                    prediction.feasible
                    and prediction.minimum_seconds * 1.05 <= config.max_motion_seconds
                ):
                    staging = proposed
                    break
            else:
                self.events[-1].update(candidates=attempts, decision="hold_for_replan")
                raise InteractionFailure("loaded_stage: no checked shelf approach candidate")
            self.events[-1].update(
                candidates=attempts,
                decision="execute_checked_shelf_approach",
                selected_position_world_m=staging.tolist(),
            )
        self._move("align_shelf", staging, grasp_grip)
        transfer = placement + offset + [0, 0, config.lift_height]
        transfer[2] = max(transfer[2], staging[2])
        self._move("transfer", transfer, grasp_grip)
        return self._seat_front(grasp_grip)

    def _seat_front(self, grasp_grip):
        """Seat an already carried carton from its current measured transform."""
        env, product = self.env, self.product
        placement = np.array(product.target)
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.array(product.half_size)
        # Use the measured rotated box bounds and grasp offset; nominal carton
        # half-sizes can request penetration when the grasp has a small tilt.
        placement[0] = self._front_center_x(extent[0])
        placement[2] = env.config.levels[product.level] + extent[2] + 0.001
        offset = self.controller.position() - body.xpos
        self._move("place", placement + offset, grasp_grip)
        if self.last_forces[self.FRONT_SUPPORT] < 0.2:
            body = env.data.body(product.name)
            extent = np.abs(body.xmat.reshape(3, 3)) @ np.array(product.half_size)
            gap = float(body.xpos[2] - extent[2] - env.config.levels[product.level])
            if not -0.002 <= gap <= 0.025:
                raise InteractionFailure("place: shelf support outside bounded seating range")
            seat = self.controller.position() - [0, 0, max(0, gap) + 0.0005]
            self._move("seat", seat, grasp_grip)
            if self.last_forces[self.FRONT_SUPPORT] < 0.2:
                raise InteractionFailure(f"place: no measured {self.FRONT_SUPPORT} support")
        self.events[-1][f"verified_{self.FRONT_SUPPORT}_support"] = True
        return grasp_grip

    def _push_grasped(self, grasp_grip):
        env, config, product = self.env, self.config, self.product
        # Contact mode changes here: a shelf-supported load may slide within
        # the fingers. Monitor support/contact instead of enforcing a rigid
        # free-space grasp transform throughout the supported slide.
        self.carry_reference = None
        self.supported_slide = True
        self.support_contact_loss = 0.0
        # Maintain the physical grasp during the supported slide to stabilize
        # tall cartons. Release only after contact, rather than pushing a free
        # carton above its tipping threshold.
        self._stage("push")
        object_rotation = env.data.body(product.name).xmat.reshape(3, 3)
        self.controller.active_rotation = (
            self.payload_rotation_goal
            @ object_rotation.T
            @ env.data.xmat[self.controller.palm].reshape(3, 3)
        )
        start = self.controller.position().copy()
        box_position = env.data.body(product.name).xpos.copy()
        box_extent = np.abs(env.data.body(product.name).xmat.reshape(3, 3)) @ np.array(
            product.half_size
        )
        contact_target = start.copy()
        contact_target[0] += (
            env.config.front_x - env.config.depth + box_extent[0] - 0.0005
        ) - box_position[0]
        push_prediction = self.controller.predict(
            contact_target,
            allowed_object=product.name,
            alternatives=False,
            grip=grasp_grip,
            path_mode="cartesian",
        )
        self.events[-1]["ik_prediction"] = push_prediction.report()
        if not push_prediction.feasible:
            self.events[-1]["decision"] = "hold_for_replan"
            raise InteractionFailure("push: no acceptable supported slide path; replan required")
        self.events[-1]["decision"] = "execute_supported_slide"
        target = start.copy()
        self.events[-1]["push_origin_m"] = start.tolist()
        held = 0.0
        for _ in range(int(config.push_timeout_seconds / env.dt)):
            previous_target = target.copy()
            upright = env.data.body(product.name).xmat.reshape(3, 3)[2, 2] >= np.cos(
                np.radians(config.payload_upright_degrees)
            )
            if self.last_forces["back"] >= config.contact_threshold:
                # Remove accumulated servo penetration once the back supports
                # the product; hold measured position for the contact dwell.
                target = self.controller.position().copy()
                if not upright:
                    target += [config.push_speed * env.dt, 0, 0]
            else:
                scale = np.clip(
                    (config.push_force_target - self.last_forces["push_force"])
                    / config.push_force_target,
                    -1.0,
                    1.0,
                )
                target = target - [config.push_speed * env.dt * scale, 0, 0]
            if not start[0] - config.push_distance <= target[0] <= start[0] + 0.03:
                raise InteractionFailure("push: bounded advance/retreat distance exceeded")
            body = env.data.body(product.name)
            extent = np.abs(body.xmat.reshape(3, 3)) @ np.array(product.half_size)
            gap = body.xpos[2] - extent[2] - env.config.levels[product.level]
            target[2] -= (
                np.clip(
                    2 * (gap + 0.0002),
                    -config.insertion_vertical_speed,
                    config.insertion_vertical_speed,
                )
                * env.dt
            )
            if config.contact_aware and config.fixture_avoidance:
                normal = self.fixture_feedback.copy()
                force = float(np.linalg.norm(normal))
                threshold = config.fixture_force_target * (0.5 if self.fixture_unloading else 1)
                self.fixture_unloading = force > threshold
                if self.fixture_unloading:
                    normal /= force
                    distance = (
                        config.fixture_unload_speed
                        * env.dt
                        * min(1.0, force / config.fixture_force_target)
                    )
                    target = previous_target.copy()
                    target += normal * (
                        np.dot(self.controller.position() - target, normal) + distance
                    )
                    event = self.events[-1]
                    event["fixture_unload_steps"] = event.get("fixture_unload_steps", 0) + 1
                    event["fixture_unload_distance_m"] = (
                        event.get("fixture_unload_distance_m", 0.0) + distance
                    )
                    if event["fixture_unload_distance_m"] > config.fixture_unload_distance:
                        raise InteractionFailure(
                            "push: fixture unload distance exhausted; replan required"
                        )
            self._tick(target, grasp_grip)
            contacted = (
                self.last_forces["back"] >= config.contact_threshold
                and self.last_forces["hand"] >= 0.2
                and env.data.body(product.name).xmat.reshape(3, 3)[2, 2]
                >= np.cos(np.radians(config.payload_upright_degrees))
            )
            held = held + env.dt if contacted else 0.0
            if held >= config.contact_dwell:
                self.events[-1]["back_contact_verified"] = True
                break
            if self.terminal:
                raise InteractionFailure("push: task ended before contact dwell validation")
        else:
            raise InteractionFailure("push: shelf-back contact was not established")
        self.supported_slide = False
        self.fixture_unloading = False
        self.observer.begin_retreat(self.product.name)
        self._release_grasp()
        self.payload_rotation_goal = None
        self._move("clear", self.controller.position() + [0.04, 0, 0.02])
        self.observer.end_retreat(self.product.name)

    def _release_grasp(self, name="release", *, support="shelf"):
        """Check opening fingers and retreat together, then execute via servos."""
        env, config, controller = self.env, self.config, self.controller
        self._stage(name)
        self.active_grip, self.carry_reference = None, None
        self.release_origin = env.data.body(self.product.name).xpos.copy()
        reference = env.data.qpos.copy()
        start = controller.position().copy()
        start_q = controller.ik.measured_q()
        controller.ik.carry = None
        controller.ik.slide = None
        object_id = env.model.body(self.product.name).id
        baseline = controller.ik._contacts(start_q, reference, object_id)
        fingers = {}
        for name, opened in controller.finger_targets(0).items():
            adr = int(env.model.joint(name).qposadr[0])
            lo, hi = env.model.actuator(name).ctrlrange
            fingers[name] = (adr, float(np.clip(reference[adr], lo, hi)), opened)
        displacements = [np.array([config.release_forward, 0, config.release_up])]
        if config.contact_aware:
            others = [p for p in env.products if p.name != self.product.name]
            nearest = min(
                others,
                key=lambda p: np.linalg.norm(
                    env.data.body(p.name).xpos - env.data.body(self.product.name).xpos
                ),
                default=None,
            )
            away = (
                -1.0
                if nearest is not None
                and env.data.body(nearest.name).xpos[1] > env.data.body(self.product.name).xpos[1]
                else 1.0
            )
            for amount in (config.release_lateral_limit / 2, config.release_lateral_limit):
                for sign in (away, -away):
                    displacements.append(
                        np.array([config.release_forward, sign * amount, config.release_up])
                    )
        attempts = []
        for displacement in displacements:
            plan, q = [], start_q.copy()
            for alpha in np.linspace(0, 1, 41):
                finger_reference = reference.copy()
                for adr, value, opened in fingers.values():
                    finger_reference[adr] = value + alpha * (opened - value)
                target = start + alpha * displacement
                q, point, rotation = controller.ik.solve(
                    target, controller.active_rotation, q, reference=finger_reference
                )
                errors = controller.ik.collision_errors(q, finger_reference, baseline, object_id)
                pos_error = float(np.linalg.norm(point - target))
                angle_error = float(
                    np.linalg.norm(rotation_error(rotation, controller.active_rotation))
                )
                if (
                    pos_error > config.ik_position_tolerance
                    or angle_error > np.radians(config.ik_orientation_tolerance_deg)
                    or errors
                ):
                    attempts.append(
                        dict(
                            displacement_m=displacement.tolist(),
                            accepted=False,
                            collisions=errors,
                            position_error_m=pos_error,
                            orientation_error_deg=float(np.degrees(angle_error)),
                        )
                    )
                    break
                plan.append(q.copy())
            else:
                attempts.append(dict(displacement_m=displacement.tolist(), accepted=True))
                break
        else:
            self.events[-1].update(
                decision="hold_for_replan",
                release_candidates=attempts,
                collisions=attempts[-1]["collisions"],
            )
            raise InteractionFailure("release: opening/retreat path rejected before actuation")
        self.events[-1].update(
            decision="execute_checked_opening_retreat",
            sampled_poses=len(plan),
            release_candidates=attempts,
            assumption="released object remains on shelf; measured displacement is guarded",
        )
        plan = np.asarray(plan)
        rates = np.r_[np.full(len(controller.names), config.max_joint_speed), 0.08]
        minimum = float(1.875 * np.max(np.abs(np.diff(plan, axis=0)).sum(axis=0) / rates))
        ticks = int(np.ceil(max(config.release_seconds, minimum) / env.dt))
        for k in range(ticks):
            t = (k + 1) / ticks
            alpha = 10 * t**3 - 15 * t**4 + 6 * t**5
            index = alpha * (len(plan) - 1)
            low = min(int(index), len(plan) - 2)
            q = plan[low] + (index - low) * (plan[low + 1] - plan[low])
            action = controller.joint_targets(q, 0)
            for name, (_, value, opened) in fingers.items():
                action[name] = float(
                    np.clip(value + alpha * (opened - value), *env.model.actuator(name).ctrlrange)
                )
            controller.last_target, _ = controller.ik.forward(controller.command, reference)
            self._apply_tick(action)
        for _ in range(int(0.3 / env.dt)):
            self._tick(controller.position(), 0)
        if self.last_forces[support] < 0.2 or self.last_forces["hand"] > 0.2:
            raise InteractionFailure(f"release: no measured {support} support")
        if env.data.body(self.product.name).xmat.reshape(3, 3)[2, 2] < np.cos(np.radians(15)):
            raise InteractionFailure("release: product is not upright")
        self.events[-1].update(
            verified_release=True,
            support_surface=support,
            placement_conditions=bool(
                env._placement_conditions()[env.products.index(self.product)]
            ),
            measured_shelf_normal_N=self.last_forces["shelf"],
            measured_hand_normal_N=self.last_forces["hand"],
        )
        self.release_origin = None

    def _execute_task(self, result):
        """Execute the stocking composition; isolated skills override this loop."""
        pending = [p for p in self.env.products if not p.initially_stocked]
        while pending:
            selected = self._select_grasp(pending) if self.config.adaptive_grasps else None
            if selected is None:
                self.product = pending[0]
            self._one_box(selected)
            result["completed_products"].append(self.product.name)
            pending.remove(self.product)
        self._stage("settle")
        for _ in range(
            0 if self.terminal else int(self.env.config.stable_seconds / self.env.dt) + 10
        ):
            if self._tick(self.controller.position(), 0):
                break
        result["physical_success"] = self.env.info()["success"]
        if not result["physical_success"]:
            result["failure"] = "Final stocking stability criteria were not met"

    def _prepare_episode(self):
        from abstract_stack.bundle import build_scene_bundle

        build_scene_bundle(self.env, self.directory / "initial-scene")

    def _new_observer(self):
        from abstract_stack.evaluation import StageObserver

        return StageObserver(self.env)

    def _new_tracker(self):
        return MabelController(self.env.model, self.env.data, self._tracking_config())

    def run(self) -> dict:
        self._prepare_episode()
        self.observer = self._new_observer()
        self.observer.completed.update(
            p.name
            for i, p in enumerate(self.env.products)
            if p.initially_stocked and self.env._stocked[i]
        )
        # The bundle settles the world; initialize IK targets from that measured state.
        self.controller = self._new_cartesian_hand()
        self.tracker = self._new_tracker()
        self.base_start = self.env.data.body("mabel_chassis").xpos.copy()
        if self.config.head_tracking:
            from abstract_stack.recording import HeadGaze

            self.gaze = HeadGaze(self.env.model, self.env.data)
        if self.video_options is not None:
            from abstract_stack.recording import DualVideoRecorder

            self.recorder = DualVideoRecorder(
                self.env, self.directory / "videos", **self.video_options
            )
            self.recorder.capture(phase="initial", product=self.product.name, forces={})
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.env.model, state_spec))
        mujoco.mj_getState(self.env.model, self.env.data, state, state_spec)
        np.savez_compressed(
            self.directory / "rollout_start.npz", state=state, state_spec=int(state_spec)
        )
        result = {
            "schema_version": 3,
            "task_scope": self.TASK_SCOPE,
            "physical_success": False,
            "failure": None,
            "completed_products": [],
            "initially_stocked_products": [
                p.name for p in self.env.products if p.initially_stocked
            ],
            "scheduled_pick_products": [
                p.name for p in self.env.products if not p.initially_stocked
            ],
            "controller": asdict(self.config),
            "force_profile": self.force_profile,
            "assistance": "none",
            "state_access": "privileged_mujoco_world_state",
            "policy_kind": "handwritten_feedback_program",
            "contact_protocol": self.CONTACT_PROTOCOL,
        }
        try:
            self._execute_task(result)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            result["failure"] = str(error)
        finally:
            terminal_state = np.empty(mujoco.mj_stateSize(self.env.model, state_spec))
            mujoco.mj_getState(self.env.model, self.env.data, terminal_state, state_spec)
            np.savez_compressed(
                self.directory / "rollout_terminal.npz",
                state=terminal_state,
                state_spec=int(state_spec),
            )
            self.env.data.ctrl[:] = stationary_targets(self.env.model, self.env.data)
            result["events"] = self.events
            result["stage_evaluation"] = self.observer.report(self.events)
            result["last_ik_residual"] = self.controller.last_ik_error
            result["tracking_controller"] = self.tracker.last_report
            result["tracking_config"] = asdict(self.tracker.config)
            if self.controller.last_prediction is not None:
                result["last_ik_prediction"] = self.controller.last_prediction.report()
            result["base_displacement_m"] = float(
                np.linalg.norm(
                    self.env.data.body("mabel_chassis").xpos - self.base_start,
                )
            )
            result["base_orientation_wxyz"] = self.env.data.body("mabel_chassis").xquat.tolist()
            result["last_nominal_targets"] = dict(
                zip(
                    [*self.controller.names, "Lift"],
                    self.controller.command.tolist(),
                    strict=True,
                )
            )
            result["last_actuator_targets"] = {
                name: self.tracker.last_report.get("servo_target", {}).get(name, value)
                for name, value in result["last_nominal_targets"].items()
            }
            result["final_product_poses"] = {
                p.name: np.r_[
                    self.env.data.body(p.name).xpos, self.env.data.body(p.name).xquat
                ].tolist()
                for p in self.env.products
            }
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
                "object_qw",
                "object_qx",
                "object_qy",
                "object_qz",
                "grip_normal_N",
                "push_force_N",
                "fixture_normal_world_x_N",
                "fixture_normal_world_y_N",
                "fixture_normal_world_z_N",
                "arm_product_normal_N",
                "other_product_normal_N",
            ]
            trace = np.asarray(self.rows, dtype=float).reshape(-1, len(result["trace_columns"]))
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
                result["peak_forces_N"]["other_product"] = float(trace[:, -1].max())
                result["peak_forces_N"]["arm_product"] = float(trace[:, -2].max())
                result["peak_forces_N"]["robot_product"] = float((trace[:, 7] + trace[:, -2]).max())
            if self.render:
                from PIL import Image

                Image.fromarray(self.env.render()).save(self.directory / "final.png")
            from abstract_stack.evaluation import manipulation_evaluation

            result["manipulation_evaluation"] = manipulation_evaluation(result)
            result["task_success"] = result["manipulation_evaluation"]["clean_success"]
            result["controller_evidence"] = self.episode_evidence()
            if (
                self.TASK_SCOPE == "stocking"
                and not result["task_success"]
                and result["failure"] is None
            ):
                result["failure"] = (
                    "Physical goal met but scheduled manipulation evidence incomplete"
                )
            if self.recorder is not None:
                try:
                    result["video"] = self.recorder.finish(result)
                except (OSError, RuntimeError) as error:
                    result["video_error"] = str(error)
            (self.directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
