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


def interpolate_rotation(start, goal, fraction):
    vector = rotation_error(start, goal)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-9:
        return start.copy()
    quat, matrix = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quat, vector / angle, angle * fraction)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3) @ start


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
    carried_object: str | None = None
    sliding_object: str | None = None
    path_mode: str = "joint"
    path_ik_error: dict | None = None
    posture_constraints: dict = field(default_factory=dict)
    baseline_qpos: np.ndarray | None = field(default=None, repr=False)
    joint_path: np.ndarray | None = field(default=None, repr=False)
    bounded_feedback: dict | None = field(default=None, repr=False)
    initial_cartesian_transition: dict | None = field(default=None, repr=False)

    @property
    def safe(self):
        return not self.collisions and not self.path_collisions and self.path_ik_error is None

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
        if self.path_ik_error:
            reasons.append("sampled_path_ik_error")
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
            "planning_posture_constraints": self.posture_constraints,
            "collisions": self.collisions,
            "sampled_path_collisions": self.path_collisions,
            "sampled_path_model": self.path_mode,
            "sampled_path_ik_error": self.path_ik_error,
            "retained_cartesian_joint_waypoints": (
                0 if self.joint_path is None else len(self.joint_path)
            ),
            "bounded_feedback": self.bounded_feedback,
            "initial_cartesian_transition": self.initial_cartesian_transition,
            "endpoint_within_collision_tolerance": not self.collisions,
            "sampled_path_checked": self.path_checked,
            "safe_sampled_path": self.path_checked and self.safe,
            "blocked_continuation": self.blocked_by,
            "rejection_reasons": reasons if not self.feasible else [],
            "allowed_target_hand_contact": self.allowed_contact_object,
            "carried_object_swept_volume": self.carried_object,
            "supported_slide_prediction": None
            if self.sliding_object is None
            else {
                "object": self.sliding_object,
                "motion": "translate along -X after closing the measured tool/front-face gap",
                "backstop": (
                    "stop at geometric shelf-back contact when Y/Z overlap; tool preload allowed"
                ),
                "fixed": "current object Y, Z and orientation",
                "scope": "scratch kinematics only; no attachment or dynamics guarantee",
            },
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
        self.carry = None
        self.slide = None
        self.posture_constraints = {}

    def constrain_posture(self, name, value):
        """Constrain a planning coordinate without editing native joint bounds."""
        index = self.names.index(name)
        if not np.isfinite(value) or not self.lower[index] <= value <= self.upper[index]:
            raise ValueError("Planning posture must remain inside native coordinate limits")
        self.posture_constraints[index] = float(value)

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
        if self.carry is not None:
            _, qadr, relative_position, relative_rotation = self.carry
            self.scratch.qpos[qadr : qadr + 3] = (
                self.scratch.xpos[self.palm] + rotation @ relative_position
            )
            mujoco.mju_mat2Quat(
                self.scratch.qpos[qadr + 3 : qadr + 7], (rotation @ relative_rotation).ravel()
            )
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
        if self.slide is not None:
            _, qadr, position, start_x, approach_gap, minimum_x = self.slide
            self.scratch.qpos[qadr : qadr + 3] = position
            advance = max(0.0, start_x - point[0] - approach_gap)
            self.scratch.qpos[qadr] -= min(advance, max(0.0, position[0] - minimum_x))
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
        return point.copy(), rotation

    def _cartesian_start_transition(
        self,
        start,
        first_q,
        start_point,
        start_rotation,
        reference,
        baseline,
        allowed_object,
        rates,
        position_tolerance,
        angle_tolerance_degrees,
    ):
        """Check a real joint-space prefix into the first constrained IK pose.

        A Cartesian t=0 solve may change redundant joints or enforce a different
        wrist posture. Its FK matching the initial tool pose does not make that
        joint change instantaneous. Return sampled prefix poses for execution.
        """
        if np.array_equal(first_q, start):
            return [], [], None, None
        normalized_gap = float(np.max(np.abs(first_q - start) / rates))
        steps = max(2, int(np.ceil(normalized_gap / 0.04)))
        metadata = dict(
            joint_gap=(first_q - start).tolist(),
            samples=steps + 1,
            checked_samples=0,
            minimum_seconds=1.875 * normalized_gap,
            maximum_hold_position_error_m=0.0,
            maximum_hold_orientation_error_deg=0.0,
            accepted=False,
        )
        path = []
        for fraction in np.linspace(0, 1, steps + 1):
            q = start + fraction * (first_q - start)
            point, rotation = self.forward(q, reference)
            pe = float(np.linalg.norm(point - start_point))
            ae = float(np.linalg.norm(rotation_error(rotation, start_rotation)))
            metadata["checked_samples"] += 1
            metadata["maximum_hold_position_error_m"] = max(
                metadata["maximum_hold_position_error_m"], pe
            )
            metadata["maximum_hold_orientation_error_deg"] = max(
                metadata["maximum_hold_orientation_error_deg"], float(np.degrees(ae))
            )
            errors = self.collision_errors(q, reference, baseline, allowed_object)
            if errors:
                return path, errors, None, metadata
            if pe > position_tolerance or ae > np.radians(angle_tolerance_degrees):
                return (
                    path,
                    [],
                    dict(
                        stage="initial_cartesian_transition",
                        fraction=float(fraction),
                        position_error_m=pe,
                        orientation_error_deg=float(np.degrees(ae)),
                    ),
                    metadata,
                )
            if fraction < 1:
                path.append(q.copy())
        metadata["accepted"] = True
        return path, [], None, metadata

    def solve(
        self, target, rotation, seed, *, iterations=100, rotation_weight=0.12, reference=None
    ):
        q = np.clip(seed, self.lower, self.upper).copy()
        for index, value in self.posture_constraints.items():
            q[index] = value
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
            for index in self.posture_constraints:
                jacobian[:, index] = 0
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
            carried = (
                self.carry[0]
                if self.carry is not None
                else (self.slide[0] if self.slide is not None else None)
            )
            if b1 not in self.robot and b2 not in self.robot and carried not in (b1, b2):
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

    def check_path(
        self, q, reference, max_joint_speed, allowed_object=None, *, baseline_reference=None
    ):
        """Recheck a joint segment from its measured-state baseline snapshot.

        ``reference`` includes the proposed finger posture; existing-contact
        allowances come from ``baseline_reference`` (live state by default).
        Callers must reject stale predictions before executing the segment.
        """
        baseline_reference = (
            self.data.qpos.copy() if baseline_reference is None else baseline_reference
        )
        start = np.r_[
            baseline_reference[self.qadr],
            sum(baseline_reference[j.qposadr[0]] for j in self.lifts),
        ]
        object_id = self.model.body(allowed_object).id if allowed_object else None
        baseline = self._contacts(start, baseline_reference, object_id)
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
        carrying=False,
        sliding=False,
        path_mode="joint",
    ):
        if path_mode not in ("joint", "cartesian"):
            raise ValueError("IK path_mode must be joint or cartesian")
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
        self.carry = None
        self.slide = None
        if carrying and sliding:
            raise ValueError("Rigid carrying and supported translation are different contact modes")
        if carrying:
            if allowed_object is None:
                raise ValueError("A carried product must be named")
            bid = self.model.body(allowed_object).id
            qadr = int(self.model.joint(f"{allowed_object}_joint").qposadr[0])
            palm_rotation = self.data.xmat[self.palm].reshape(3, 3)
            self.carry = (
                bid,
                qadr,
                palm_rotation.T @ (self.data.xpos[bid] - self.data.xpos[self.palm]),
                palm_rotation.T @ self.data.xmat[bid].reshape(3, 3),
            )
        if sliding:
            if allowed_object is None:
                raise ValueError("A sliding carton must be named")
            bid = self.model.body(allowed_object).id
            geom = self.model.geom(f"{allowed_object}_geom").id
            if int(self.model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("Supported translation prediction currently requires a box")
            point, _ = self.forward(self.measured_q(), baseline_reference)
            extent = np.abs(self.data.xmat[bid].reshape(3, 3)) @ self.model.geom_size[geom]
            position = self.data.xpos[bid].copy()
            qadr = int(self.model.joint(f"{allowed_object}_joint").qposadr[0])
            back = self.model.geom("shelf_back").id
            back_extent = (
                np.abs(self.data.geom_xmat[back].reshape(3, 3)) @ self.model.geom_size[back]
            )
            back_position = self.data.geom_xpos[back]
            overlaps_back = np.all(
                np.abs(position[1:] - back_position[1:]) < extent[1:] + back_extent[1:]
            )
            self.slide = (
                bid,
                qadr,
                position,
                point[0],
                max(0.0, point[0] - position[0] - extent[0]),
                float(back_position[0] + back_extent[0] + extent[0]) if overlaps_back else -np.inf,
            )
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
        if path_mode == "cartesian":
            seeds = seeds[:1]
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
                path_collisions, prefix, path_ik_error = [], None, None
                retained_path = None
                initial_transition = None
                duration = float(1.875 * np.max(np.abs(q - start) / rates))
                if check_path:
                    start_point, start_rotation = self.forward(start, reference)
                    if path_mode == "cartesian":
                        samples = max(
                            8, int(np.ceil(np.linalg.norm(target - start_point) / 0.008)) + 1
                        )
                    else:
                        samples = max(8, int(np.ceil(np.max(np.abs(q - start) / rates) / 0.08)))
                    path_q, path = start.copy(), []
                    for t in np.linspace(0, 1, samples):
                        if path_mode == "cartesian":
                            desired_point = start_point + t * (target - start_point)
                            desired_rotation = interpolate_rotation(start_rotation, rotation, t)
                            path_q, path_point, path_rotation = self.solve(
                                desired_point, desired_rotation, path_q, reference=reference
                            )
                            pe = float(np.linalg.norm(path_point - desired_point))
                            ae = float(
                                np.linalg.norm(rotation_error(path_rotation, desired_rotation))
                            )
                            if pe > position_tolerance or ae > np.radians(angle_tolerance_degrees):
                                path_ik_error = {
                                    "position_error_m": pe,
                                    "orientation_error_deg": float(np.degrees(ae)),
                                }
                                break
                            if t == 0:
                                initial_path, path_collisions, path_ik_error, initial_transition = (
                                    self._cartesian_start_transition(
                                        start,
                                        path_q,
                                        start_point,
                                        start_rotation,
                                        reference,
                                        baseline,
                                        object_id,
                                        rates,
                                        position_tolerance,
                                        angle_tolerance_degrees,
                                    )
                                )
                                if path_collisions or path_ik_error:
                                    break
                                path.extend(initial_path)
                        else:
                            path_q = start + t * (q - start)
                        path_collisions = self.collision_errors(
                            path_q,
                            reference,
                            baseline,
                            object_id,
                        )
                        if path_collisions:
                            break
                        path.append(path_q.copy())
                        path_point, path_rotation = self.forward(path_q, reference)
                        path_error = float(np.linalg.norm(path_point - target))
                        if (
                            t > 0
                            and (prefix is None or path_error < prefix[3])
                            and np.linalg.norm(rotation_error(path_rotation, preferred))
                            <= np.radians(orientation_span_degrees + angle_tolerance_degrees)
                        ):
                            prefix = (path_q.copy(), path_point, path_rotation, path_error)
                    if path_mode == "cartesian" and not path_collisions and not path_ik_error:
                        q = path_q.copy()
                        point, actual_rotation = self.forward(q, reference)
                        position_error = float(np.linalg.norm(point - target))
                        angular_error = float(
                            np.linalg.norm(rotation_error(actual_rotation, rotation))
                        )
                        collisions = self.collision_errors(q, reference, baseline, object_id)
                        duration = float(
                            1.875 * np.max(np.abs(np.diff(path, axis=0)).sum(axis=0) / rates)
                        )
                        if initial_transition is not None:
                            # Execution interpolates by waypoint index. Include
                            # the prefix AND the largest segment rate so a
                            # denser Cartesian tail cannot rush the prefix.
                            segment_times = np.abs(np.diff(path, axis=0)) / rates
                            duration = max(
                                duration, float(1.875 * (len(path) - 1) * np.max(segment_times))
                            )
                            initial_transition["included_in_duration"] = True
                        retained_path = np.asarray(path).copy()
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
                    and not path_ik_error
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
                    duration,
                    reference.copy(),
                    path_checked=check_path,
                    position_tolerance_m=position_tolerance,
                    angle_tolerance_degrees=angle_tolerance_degrees,
                    allowed_contact_object=allowed_object,
                    carried_object=allowed_object if carrying else None,
                    sliding_object=allowed_object if sliding else None,
                    path_mode=path_mode,
                    path_ik_error=path_ik_error,
                    posture_constraints={
                        self.names[i]: value for i, value in self.posture_constraints.items()
                    },
                    baseline_qpos=baseline_reference.copy(),
                    joint_path=retained_path,
                    initial_cartesian_transition=initial_transition,
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
                if prefix is not None and (collisions or path_collisions or path_ik_error):
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
                        path_ik_error=None,
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
                        blocked_by=path_collisions or collisions or [{"ik_error": path_ik_error}],
                    )
                    if best_safe is None or (pe, candidate.orientation_error_rad) < (
                        best_safe.position_error_m,
                        best_safe.orientation_error_rad,
                    ):
                        best_safe = candidate
        return best_safe if best_safe is not None else best_any
