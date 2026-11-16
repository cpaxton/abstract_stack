"""Geometric IK predictions, sampled collision checks and bounded alternatives."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np


def rotation_error(actual, desired) -> np.ndarray:
    quat, vector = np.empty(4), np.empty(3)
    mujoco.mju_mat2Quat(quat, (desired @ actual.T).ravel())
    if quat[0] < 0:
        quat *= -1
    mujoco.mju_quat2Vel(vector, quat, 1.0)
    return vector


def _descendants(model, root):
    result = set()
    for bid in range(model.nbody):
        parent = bid
        while parent:
            if parent == root:
                result.add(bid)
                break
            parent = int(model.body_parentid[parent])
    return result


@dataclass
class IKPrediction:
    q: np.ndarray
    target: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    requested_rotation: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    preferred_orientation_error_rad: float
    active_limits: list[str]
    collisions: list[dict]
    path_collisions: list[dict]
    feasible: bool
    candidate: str
    minimum_seconds: float
    reference_qpos: np.ndarray = field(repr=False)
    path_checked: bool = False
    blocked_by: list[dict] = field(default_factory=list)
    position_tolerance_m: float = 0.015
    angle_tolerance_degrees: float = 12.0
    allowed_contact_object: str | None = None

    @property
    def safe(self):
        return not self.collisions and not self.path_collisions

    def report(self):
        reasons = []
        if self.position_error_m > self.position_tolerance_m:
            reasons.append("position_tolerance")
        if self.orientation_error_rad > np.radians(self.angle_tolerance_degrees):
            reasons.append("orientation_tolerance")
        if self.collisions:
            reasons.append("endpoint_collision")
        if self.path_collisions:
            reasons.append("sampled_path_collision")
        if self.blocked_by:
            reasons.append("blocked_continuation")
        return {
            "status": "accepted" if self.feasible else "no_acceptable_solution_found",
            "requested_position": self.target.tolist(),
            "predicted_position": self.position.tolist(),
            "predicted_position_error_m": self.position_error_m,
            "predicted_position_error_xyz_m": (self.position - self.target).tolist(),
            "predicted_orientation_error_deg": float(np.degrees(self.orientation_error_rad)),
            "deviation_from_preferred_orientation_deg": float(
                np.degrees(self.preferred_orientation_error_rad)
            ),
            "selected_orientation": self.requested_rotation.tolist(),
            "active_limits": self.active_limits,
            "collisions": self.collisions,
            "sampled_path_collisions": self.path_collisions,
            "endpoint_within_collision_tolerance": not self.collisions,
            "sampled_path_checked": self.path_checked,
            "safe_sampled_path": self.path_checked and self.safe,
            "blocked_continuation": self.blocked_by,
            "rejection_reasons": reasons if not self.feasible else [],
            "allowed_target_hand_contact": self.allowed_contact_object,
            "collision_tolerance_m": 0.002,
            "existing_contact_allowance_m": 0.001,
            "candidate": self.candidate,
            "predicted_minimum_seconds": self.minimum_seconds,
            "prediction_scope": (
                "kinematic; measured base held at current pose; not a dynamics guarantee"
            ),
        }


class PoseIK:
    """MABEL arm/wrist/lift search using scratch data only.

    Failure means the deterministic candidate search found no acceptable pose.
    It does not establish that the target is globally unreachable.
    """

    def __init__(self, model, data, names, lower, upper, home, offset):
        self.model, self.data = model, data
        self.names = [*names, "Lift"]
        self.joints = [model.joint(n) for n in names]
        self.qadr = np.array([j.qposadr[0] for j in self.joints])
        self.dofs = np.array([j.dofadr[0] for j in self.joints])
        self.lifts = [model.joint(n) for n in ("Lift Lower", "Lift Upper")]
        self.lower, self.upper, self.home = lower.copy(), upper.copy(), home.copy()
        self.offset = offset.copy()
        self.palm = model.body("left_palm").id
        self.robot = _descendants(model, model.body("mabel_chassis").id)
        self.hand = _descendants(model, self.palm)
        self.scratch = mujoco.MjData(model)
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def measured_q(self):
        return np.r_[
            self.data.qpos[self.qadr], sum(self.data.qpos[j.qposadr[0]] for j in self.lifts)
        ]

    def forward(self, q, reference=None):
        self.scratch.qpos[:] = self.data.qpos if reference is None else reference
        self.scratch.qpos[self.qadr] = q[:-1]
        for joint in self.lifts:
            self.scratch.qpos[joint.qposadr[0]] = q[-1] / 2
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        rotation = self.scratch.xmat[self.palm].reshape(3, 3).copy()
        point = self.scratch.xpos[self.palm] + rotation @ self.offset
        return point.copy(), rotation

    def solve(
        self, target, rotation, seed, *, iterations=100, rotation_weight=0.12, reference=None
    ):
        q = np.clip(seed, self.lower, self.upper).copy()
        best, best_q = float("inf"), q.copy()
        for _ in range(iterations):
            point, actual_rotation = self.forward(q, reference)
            error = np.r_[
                target - point, rotation_weight * rotation_error(actual_rotation, rotation)
            ]
            cost = float(error @ error)
            if cost < best:
                best, best_q = cost, q.copy()
            if cost < 1e-8:
                break
            mujoco.mj_jac(self.model, self.scratch, self.jacp, self.jacr, point, self.palm)
            lift_dofs = [j.dofadr[0] for j in self.lifts]
            jacobian = np.column_stack(
                (
                    np.vstack((self.jacp[:, self.dofs], rotation_weight * self.jacr[:, self.dofs])),
                    np.r_[self.jacp[:, lift_dofs].mean(axis=1), np.zeros(3)],
                )
            )
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(6),
                error,
            )
            q = np.clip(q + np.clip(delta, -0.08, 0.08), self.lower, self.upper)
        point, actual_rotation = self.forward(best_q, reference)
        return best_q, point, actual_rotation

    def _contacts(self, q, reference, allowed_object):
        self.forward(q, reference)
        mujoco.mj_fwdPosition(self.model, self.scratch)
        contacts = {}
        for contact in self.scratch.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if b1 not in self.robot and b2 not in self.robot:
                continue
            names = [self.model.body(b).name for b in (b1, b2)]
            geoms = [
                self.model.geom(g).name or f"{n}:geom{g}"
                for g, n in zip((g1, g2), names, strict=True)
            ]
            if "floor" in geoms and any(n.endswith("wheel") for n in names):
                continue
            if allowed_object is not None and (
                (b1 == allowed_object and b2 in self.hand)
                or (b2 == allowed_object and b1 in self.hand)
            ):
                continue
            pair = tuple(sorted((g1, g2)))
            if pair not in contacts or contact.dist < contacts[pair]["distance_m"]:
                contacts[pair] = {"geoms": geoms, "distance_m": float(contact.dist)}
        return contacts

    def collision_errors(self, q, reference, baseline, allowed_object=None):
        contacts = self._contacts(q, reference, allowed_object)
        return [
            info
            for pair, info in contacts.items()
            if info["distance_m"]
            < min(-0.002, baseline.get(pair, {"distance_m": 0.0})["distance_m"] - 0.001)
        ]

    def check_path(self, q, reference, max_joint_speed, allowed_object=None):
        """Recheck a selected joint segment, including its exact endpoint."""
        start = self.measured_q()
        object_id = self.model.body(allowed_object).id if allowed_object else None
        baseline = self._contacts(start, self.data.qpos.copy(), object_id)
        rates = np.r_[np.full(len(self.joints), max_joint_speed), 0.08]
        samples = max(8, int(np.ceil(np.max(np.abs(q - start) / rates) / 0.08)))
        for t in np.linspace(0, 1, samples):
            errors = self.collision_errors(start + t * (q - start), reference, baseline, object_id)
            if errors:
                return errors
        return []

    @staticmethod
    def orientations(preferred, span_degrees):
        yield "preferred", preferred
        for degrees in (span_degrees / 2, span_degrees):
            for axis in range(3):
                for sign in (-1, 1):
                    angle = np.radians(degrees * sign)
                    vector = np.eye(3)[axis]
                    skew = np.array(
                        [
                            [0, -vector[2], vector[1]],
                            [vector[2], 0, -vector[0]],
                            [-vector[1], vector[0], 0],
                        ]
                    )
                    rotation = (
                        np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
                    )
                    yield f"axis{axis}:{degrees * sign:g}deg", preferred @ rotation

    def predict(
        self,
        target,
        preferred,
        command,
        *,
        position_tolerance=0.015,
        angle_tolerance_degrees=12.0,
        orientation_span_degrees=40.0,
        max_joint_speed=1.0,
        allowed_object=None,
        check_path=True,
        alternatives=True,
        finger_positions=None,
    ):
        target = np.asarray(target, dtype=float)
        preferred = np.asarray(preferred, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("IK target must contain three finite coordinates")
        if (
            preferred.shape != (3, 3)
            or not np.isfinite(preferred).all()
            or not np.allclose(preferred.T @ preferred, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(preferred), 1.0)
        ):
            raise ValueError("IK orientation must be a proper rotation matrix")
        baseline_reference = self.data.qpos.copy()
        reference = baseline_reference.copy()
        for index, value in (finger_positions or {}).items():
            reference[index] = value
        start = self.measured_q()
        object_id = self.model.body(allowed_object).id if allowed_object else None
        baseline = self._contacts(start, baseline_reference, object_id)
        seeds = [command.copy(), start, self.home.copy(), (self.lower + self.upper) / 2]
        for sign in (-1, 1):
            seed = self.home.copy()
            seed[[0, 1, 2, 3]] += [sign * 0.6, 0.5, -sign * 0.4, 0.7]
            seeds.append(np.clip(seed, self.lower, self.upper))
        rotations = (
            self.orientations(preferred, orientation_span_degrees)
            if alternatives
            else [("preferred", preferred)]
        )
        best_safe, best_any = None, None
        rates = np.r_[np.full(len(self.joints), max_joint_speed), 0.08]
        for label, rotation in rotations:
            for seed in seeds:
                q, point, actual_rotation = self.solve(target, rotation, seed, reference=reference)
                position_error = float(np.linalg.norm(point - target))
                angular_error = float(np.linalg.norm(rotation_error(actual_rotation, rotation)))
                collisions = self.collision_errors(q, reference, baseline, object_id)
                path_collisions, prefix = [], None
                if check_path:
                    samples = max(8, int(np.ceil(np.max(np.abs(q - start) / rates) / 0.08)))
                    for t in np.linspace(0, 1, samples):
                        path_q = start + t * (q - start)
                        path_collisions = self.collision_errors(
                            path_q,
                            reference,
                            baseline,
                            object_id,
                        )
                        if path_collisions:
                            break
                        path_point, path_rotation = self.forward(path_q, reference)
                        path_error = float(np.linalg.norm(path_point - target))
                        if (
                            t > 0
                            and (prefix is None or path_error < prefix[3])
                            and np.linalg.norm(rotation_error(path_rotation, preferred))
                            <= np.radians(orientation_span_degrees + angle_tolerance_degrees)
                        ):
                            prefix = (path_q.copy(), path_point, path_rotation, path_error)
                limits = [
                    name
                    for name, value, lo, hi in zip(
                        self.names,
                        q,
                        self.lower,
                        self.upper,
                        strict=True,
                    )
                    if min(value - lo, hi - value) < 0.002
                ]
                feasible = (
                    position_error <= position_tolerance
                    and angular_error <= np.radians(angle_tolerance_degrees)
                    and not collisions
                    and not path_collisions
                )
                prediction = IKPrediction(
                    q,
                    target.copy(),
                    point,
                    actual_rotation,
                    rotation.copy(),
                    position_error,
                    angular_error,
                    float(np.linalg.norm(rotation_error(actual_rotation, preferred))),
                    limits,
                    collisions,
                    path_collisions,
                    feasible,
                    label,
                    float(1.875 * np.max(np.abs(q - start) / rates)),
                    reference.copy(),
                    path_checked=check_path,
                    position_tolerance_m=position_tolerance,
                    angle_tolerance_degrees=angle_tolerance_degrees,
                    allowed_contact_object=allowed_object,
                )
                key = (position_error, angular_error)
                if best_any is None or key < (
                    best_any.position_error_m,
                    best_any.orientation_error_rad,
                ):
                    best_any = prediction
                if prediction.safe and (
                    best_safe is None
                    or key
                    < (
                        best_safe.position_error_m,
                        best_safe.orientation_error_rad,
                    )
                ):
                    best_safe = prediction
                if feasible:
                    return prediction
                if prefix is not None and (collisions or path_collisions):
                    pq, pp, pr, pe = prefix
                    candidate = replace(
                        prediction,
                        q=pq,
                        position=pp,
                        rotation=pr,
                        position_error_m=pe,
                        orientation_error_rad=float(np.linalg.norm(rotation_error(pr, rotation))),
                        preferred_orientation_error_rad=float(
                            np.linalg.norm(rotation_error(pr, preferred))
                        ),
                        collisions=[],
                        path_collisions=[],
                        feasible=False,
                        candidate=label + ":closest_safe_prefix",
                        minimum_seconds=float(1.875 * np.max(np.abs(pq - start) / rates)),
                        active_limits=[
                            name
                            for name, value, lo, hi in zip(
                                self.names,
                                pq,
                                self.lower,
                                self.upper,
                                strict=True,
                            )
                            if min(value - lo, hi - value) < 0.002
                        ],
                        blocked_by=path_collisions or collisions,
                    )
                    if best_safe is None or (pe, candidate.orientation_error_rad) < (
                        best_safe.position_error_m,
                        best_safe.orientation_error_rad,
                    ):
                        best_safe = candidate
        return best_safe if best_safe is not None else best_any
