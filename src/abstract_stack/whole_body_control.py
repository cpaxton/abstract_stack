"""Stationary bimanual Cartesian coordination through native MABEL servos.

This is an opt-in kinematic/reference layer, not a contact, grasp or mobile
whole-body policy. Both palms share ONE lift coordinate in ONE bounded solve.
All forecasts use scratch data. ``accept`` returns controls; the caller owns
native stepping, per-step force/stability guards and task evaluation.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.cartesian_feedback import box_damped_least_squares
from abstract_stack.control import validate_targets
from abstract_stack.ik import rotation_error
from abstract_stack.mabel_controller import MabelController, TrackingConfig
from abstract_stack.native_views import retain_data_views, retain_model_views


@dataclass(frozen=True)
class WholeBodyConfig:
    linear_speed_m_s: float = 0.10
    angular_speed_rad_s: float = 0.5
    rotation_weight_m_rad: float = 0.12
    damping_m: float = 0.0001
    finger_speed_rad_s: float = 1.0
    max_control_dt_s: float = 0.05
    position_tolerance_m: float = 0.015
    orientation_tolerance_deg: float = 12.0
    reserve_compensation_headroom: bool = True
    base_linear_speed_tolerance_m_s: float = 0.002
    base_angular_speed_tolerance_rad_s: float = 0.01

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != "reserve_compensation_headroom"]
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Whole-body settings must be finite and positive")
        if type(self.reserve_compensation_headroom) is not bool:
            raise ValueError("Headroom reservation must be boolean")


@dataclass(frozen=True)
class PalmGoal:
    position_m: np.ndarray
    rotation_world: np.ndarray

    def __post_init__(self):
        position = np.array(self.position_m, float, copy=True)
        rotation = np.array(self.rotation_world, float, copy=True)
        if (
            position.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)
        ):
            raise ValueError("Palm goal requires a finite world pose with a proper rotation")
        position.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "rotation_world", rotation)


@dataclass(frozen=True)
class WholeBodyStep:
    nominal: np.ndarray
    applied_targets: np.ndarray
    finger_nominal: dict
    tracking_report: dict
    report: dict
    baseline_state: np.ndarray


def _state(model, data):
    result = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, result, mujoco.mjtState.mjSTATE_INTEGRATION)
    return result


def _descendants(model, root):
    selected = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in selected:
            selected.add(body)
    return selected


class StationaryWholeBodyController:
    """Coordinate two palms, both wrists and a shared lift in a 17D rate box.

    Native steering references pass through unchanged and drive commands must
    already be zero. Loaded/mobile-base control is deliberately not admitted.
    A returned accepted step is only a bounded forecast; it never passes a
    grasp, measured tracking, contact or stocking gate.
    """

    NAMES = MabelController.NAMES

    def __init__(self, model, data, config=None, tracking=None, *, nominal=None):
        self.model, self.data = model, data
        self.config = config or WholeBodyConfig()
        self.tracker = MabelController(model, data, tracking or TrackingConfig())
        self._views = retain_model_views(model), retain_data_views(model, data)
        self.scratch = mujoco.MjData(model)
        self._scratch_views = retain_data_views(model, self.scratch)
        self.ids = self.tracker.ids.copy()
        self.joints = [model.joint(name) for name in self.NAMES[:-1]]
        self.qadr = np.array([int(j.qposadr[0]) for j in self.joints])
        self.dofs = np.array([int(j.dofadr[0]) for j in self.joints])
        self.lifts = [model.joint(name) for name in ("Lift Lower", "Lift Upper")]
        self.lift_qadr = np.array([int(j.qposadr[0]) for j in self.lifts])
        self.lift_dofs = np.array([int(j.dofadr[0]) for j in self.lifts])
        if any(int(j.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE) for j in self.joints) or any(
            int(j.type[0]) != int(mujoco.mjtJoint.mjJNT_SLIDE) for j in self.lifts
        ):
            raise ValueError("Expected native MABEL hinge arms/wrists and two sliding lift stages")
        moment = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            moment,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
        expected = np.zeros(model.nv)
        expected[self.lift_dofs] = 1.0
        if not np.allclose(moment[self.ids[-1]], expected, atol=1e-12, rtol=0):
            raise ValueError("Whole-body lift mapping requires the native common 1:1 tendon")
        self.palms = {side: model.body(f"{side}_palm").id for side in ("left", "right")}
        self.base = model.body("mabel_chassis").id
        self.robot = _descendants(model, self.base)
        self.wheels = [model.actuator(f"{prefix} Drive").id for prefix in ("FL", "FR", "B")]
        self._require_stationary_commands()
        self.lower, self.upper = self.tracker.lower.copy(), self.tracker.upper.copy()
        if self.config.reserve_compensation_headroom:
            self.lower += self.tracker.offset_limits
            self.upper -= self.tracker.offset_limits
        initial = np.clip(data.actuator_length[self.ids], self.tracker.lower, self.tracker.upper)
        if nominal is not None:
            if not hasattr(nominal, "keys") or set(nominal) - set(self.NAMES):
                raise ValueError("Initial nominal must contain named arm/wrist/lift references")
            initial = np.array([nominal.get(name, initial[i]) for i, name in enumerate(self.NAMES)])
        if (
            not np.isfinite(initial).all()
            or np.any(initial < self.lower)
            or np.any(initial > self.upper)
        ):
            raise ValueError(
                "Initial nominal is outside planning headroom; stage before coordination"
            )
        self.nominal = initial.copy()
        self.tracker.nominal = self.nominal.copy()
        self.fingers = {}
        self.finger_joints = {}
        hand_bodies = set().union(*(_descendants(model, body) for body in self.palms.values()))
        for aid in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(model.actuator_trnid[aid, 0])
            if jid < 0 or model.jnt_bodyid[jid] not in hand_bodies or aid in self.ids:
                continue
            if (
                model.actuator_biasprm[aid, 1] >= 0
                or not np.isclose(model.actuator_gear[aid, 0], 1.0)
                or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise ValueError("Whole-body hand mapping requires native direct position servos")
            self.fingers[name] = float(data.ctrl[aid])
            self.finger_joints[name] = model.joint(jid)
        self.pending = None
        self._pending_admission = False
        self._pending_fingers = None

    def _require_stationary_commands(self):
        if np.any(np.abs(self.data.ctrl[self.wheels]) > 1e-9):
            raise ValueError("Whole-body prototype requires a separately stopped native base")
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0
        )
        if (
            np.linalg.norm(velocity[3:]) > self.config.base_linear_speed_tolerance_m_s
            or np.linalg.norm(velocity[:3]) > self.config.base_angular_speed_tolerance_rad_s
            or self.data.xmat[self.base].reshape(3, 3)[2, 2] < np.cos(np.radians(2))
        ):
            raise ValueError("Whole-body prototype requires a measured stationary upright base")

    def _forward(self, reference, q, fingers):
        self.scratch.qpos[:] = reference
        self.scratch.qpos[self.qadr] = q[:-1]
        total = float(reference[self.lift_qadr].sum())
        # The single tendon cannot independently command its two stages.
        # Retain their measured imbalance and forecast a common equal increment.
        self.scratch.qpos[self.lift_qadr] = reference[self.lift_qadr] + (q[-1] - total) / 2
        for name, value in fingers.items():
            self.scratch.qpos[int(self.finger_joints[name].qposadr[0])] = value
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        return {
            side: (self.scratch.xpos[bid].copy(), self.scratch.xmat[bid].reshape(3, 3).copy())
            for side, bid in self.palms.items()
        }

    def nominal_poses(self):
        return self._forward(self.data.qpos.copy(), self.nominal, self.fingers)

    def _contacts(self, reference, q, fingers):
        self._forward(reference, q, fingers)
        mujoco.mj_fwdPosition(self.model, self.scratch)
        result = {}
        for contact in self.scratch.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if b1 not in self.robot and b2 not in self.robot:
                continue
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in (b1, b2)
            ]
            geoms = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                for g in (g1, g2)
            ]
            if "floor" in geoms and any(name.endswith("wheel") for name in names):
                continue
            pair = tuple(sorted((g1, g2)))
            if pair not in result or contact.dist < result[pair]["distance_m"]:
                result[pair] = dict(geoms=geoms, distance_m=float(contact.dist))
        return result

    def _geometry_errors(self, reference, nominal, applied, fingers):
        start = np.r_[reference[self.qadr], reference[self.lift_qadr].sum()]
        start_fingers = {
            name: float(reference[int(self.finger_joints[name].qposadr[0])]) for name in fingers
        }
        baseline = self._contacts(reference, start, start_fingers)
        errors = []
        for mode, goal in (("nominal", nominal), ("applied", applied[self.ids])):
            for fraction in np.linspace(0, 1, 5):
                posture = {
                    name: start_fingers[name] + fraction * (value - start_fingers[name])
                    for name, value in fingers.items()
                }
                contacts = self._contacts(reference, start + fraction * (goal - start), posture)
                for pair, item in contacts.items():
                    if item["distance_m"] < min(
                        -0.002, baseline.get(pair, {"distance_m": 0.0})["distance_m"] - 0.001
                    ):
                        errors.append(dict(mode=mode, fraction=float(fraction), **item))
        return errors

    def predict_step(self, goals, *, dt, finger_targets=None, load_feedforward=None):
        """Preview one local step; absent palm goals hold the previous nominal pose."""
        self.pending = None
        self._pending_admission = False
        if not np.isfinite(dt) or not 0 < dt <= self.config.max_control_dt_s:
            raise ValueError("Whole-body dt is outside the declared control interval")
        if (
            not hasattr(goals, "keys")
            or set(goals) - set(self.palms)
            or any(not isinstance(v, PalmGoal) for v in goals.values())
        ):
            raise ValueError("Goals must map left/right to PalmGoal")
        self._require_stationary_commands()
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError("Whole-body prototype does not admit external-force inputs")
        baseline_state = _state(self.model, self.data)
        reference = self.data.qpos.copy()
        fingers = self.fingers.copy()
        for name, value in (finger_targets or {}).items():
            if name not in fingers or not np.isfinite(value):
                raise ValueError("Finger targets must be finite native hand-position references")
            aid, joint = self.model.actuator(name).id, self.finger_joints[name]
            if (
                not max(joint.range[0], self.model.actuator_ctrlrange[aid, 0])
                <= value
                <= min(joint.range[1], self.model.actuator_ctrlrange[aid, 1])
            ):
                raise ValueError("Finger target exceeds native bounds")
            fingers[name] += float(
                np.clip(
                    value - fingers[name],
                    -self.config.finger_speed_rad_s * dt,
                    self.config.finger_speed_rad_s * dt,
                )
            )
        lower, upper = self.lower.copy(), self.upper.copy()
        measured_sum = float(reference[self.lift_qadr].sum())
        for joint in self.lifts:
            measured = reference[int(joint.qposadr[0])]
            lower[-1] = max(lower[-1], measured_sum + 2 * (joint.range[0] - measured))
            upper[-1] = min(upper[-1], measured_sum + 2 * (joint.range[1] - measured))
        scales = self.tracker.rates * dt
        lo = np.maximum(-1.0, (lower - self.nominal) / scales)
        hi = np.minimum(1.0, (upper - self.nominal) / scales)
        if np.any(lo > hi):
            raise ValueError("Measured lift distribution leaves no rate-bounded nominal interval")
        previous_poses = self._forward(reference, self.nominal, self.fingers)
        bounded_goals = {}
        for side, (position, rotation) in previous_poses.items():
            goal = goals.get(side, PalmGoal(position, rotation))
            delta = goal.position_m - position
            delta *= min(1.0, self.config.linear_speed_m_s * dt / max(np.linalg.norm(delta), 1e-12))
            vector = rotation_error(rotation, goal.rotation_world)
            angle = float(np.linalg.norm(vector))
            quat, matrix = np.empty(4), np.empty(9)
            if angle > 1e-12:
                mujoco.mju_axisAngle2Quat(
                    quat, vector / angle, min(angle, self.config.angular_speed_rad_s * dt)
                )
                mujoco.mju_quat2Mat(matrix, quat)
                rotation = matrix.reshape(3, 3) @ rotation
            bounded_goals[side] = PalmGoal(position + delta, rotation)
        current = self.nominal.copy()
        solver = {}
        for _ in range(2):
            poses = self._forward(reference, current, fingers)
            matrices, errors = [], []
            for side, goal in bounded_goals.items():
                position, rotation = poses[side]
                jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.scratch, jacp, jacr, self.palms[side])
                weight = self.config.rotation_weight_m_rad
                full = np.vstack((jacp, weight * jacr))
                jac = np.column_stack((full[:, self.dofs], full[:, self.lift_dofs].mean(axis=1)))
                matrices.append(jac)
                errors.append(
                    np.r_[
                        goal.position_m - position,
                        weight * rotation_error(rotation, goal.rotation_world),
                    ]
                    + jac @ (current - self.nominal)
                )
            fraction, solver = box_damped_least_squares(
                np.vstack(matrices) * scales,
                np.concatenate(errors),
                lo,
                hi,
                damping=self.config.damping_m,
            )
            current = np.clip(self.nominal + fraction * scales, lower, upper)
        poses = self._forward(reference, current, fingers)
        palm_reports = {}
        motion_bounded = True
        for side, (position, rotation) in poses.items():
            requested = goals.get(side, PalmGoal(*previous_poses[side]))
            displacement = float(np.linalg.norm(position - previous_poses[side][0]))
            angular = float(np.linalg.norm(rotation_error(previous_poses[side][1], rotation)))
            motion_bounded &= (
                displacement <= self.config.linear_speed_m_s * dt + 1e-6
                and angular <= self.config.angular_speed_rad_s * dt + 1e-6
            )
            palm_reports[side] = dict(
                position_error_m=float(np.linalg.norm(position - requested.position_m)),
                orientation_error_deg=float(
                    np.degrees(np.linalg.norm(rotation_error(rotation, requested.rotation_world)))
                ),
                nominal_displacement_m=displacement,
                nominal_rotation_step_rad=angular,
            )
            palm_reports[side]["goal_within_kinematic_tolerances"] = (
                palm_reports[side]["position_error_m"] <= self.config.position_tolerance_m
                and palm_reports[side]["orientation_error_deg"]
                <= self.config.orientation_tolerance_deg
            )
        old_nominal, old_applied, old_report = (
            self.tracker.nominal,
            self.tracker.applied,
            self.tracker.last_report,
        )
        try:
            requested = {**dict(zip(self.NAMES, current.tolist(), strict=True)), **fingers}
            applied = self.tracker.targets(requested, dt=dt, load_feedforward=load_feedforward)
            tracking_report = self.tracker.last_report
        finally:
            self.tracker.nominal, self.tracker.applied, self.tracker.last_report = (
                old_nominal,
                old_applied,
                old_report,
            )
        errors = self._geometry_errors(reference, current, applied, fingers)
        accepted = motion_bounded and not errors and solver["projected_gradient_inf"] <= 1e-6
        report = dict(
            accepted=bool(accepted),
            scope="stationary bounded reference step; no measured tracking, grasp or task credit",
            palms=palm_reports,
            motion_bounded=bool(motion_bounded),
            collisions=errors,
            solver=solver,
            active_joint_limits=[
                name
                for i, name in enumerate(self.NAMES)
                if min(current[i] - lower[i], upper[i] - current[i]) <= 1e-8
            ],
            nominal=dict(zip(self.NAMES, current.tolist(), strict=True)),
            planning_lower=dict(zip(self.NAMES, lower.tolist(), strict=True)),
            planning_upper=dict(zip(self.NAMES, upper.tolist(), strict=True)),
            base_mode="held; no wheel velocity or base pose command",
            lift_forecast=(
                "one shared tendon coordinate; preserve measured stage imbalance "
                "with equal increments"
            ),
        )
        for array in (current, applied, baseline_state):
            array.setflags(write=False)
        step = WholeBodyStep(current, applied, fingers, tracking_report, report, baseline_state)
        self.pending = step
        self._pending_admission = bool(accepted)
        self._pending_fingers = fingers.copy()
        return step

    def accept(self, step):
        """Accept only the last unchanged-state preview; return native controls."""
        if step is not self.pending or not np.array_equal(
            step.baseline_state, _state(self.model, self.data)
        ):
            raise ValueError("Whole-body prediction is stale or belongs to another coordinator")
        if not self._pending_admission:
            raise ValueError("Whole-body prediction failed motion/collision/solver admission")
        result = validate_targets(self.model, self.data.ctrl, step.applied_targets)
        self.nominal = step.nominal.copy()
        self.tracker.nominal = self.nominal.copy()
        self.tracker.applied = result[self.ids].copy()
        self.tracker.last_report = step.tracking_report
        self.fingers = self._pending_fingers.copy()
        self.pending = None
        self._pending_admission = False
        return result
