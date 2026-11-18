"""Bilateral arm/head pose proposals against a private native MuJoCo model.

No live state, control vector, model settings or external forces are changed.
The caller owns session freshness/calibration and the final composed-command gate.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from abstract_stack.env import ShelfStockingEnv
from abstract_stack.ik import rotation_error
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.calibration import Pose
from abstract_stack.teleop.sim import SimState

ARM_NAMES = {
    side: tuple(f"{side.title()} Arm {i}" for i in range(1, 8)) + (f"{side.title()} Wrist",)
    for side in ("left", "right")
}
HEAD_NAMES = ("Neck Yaw", "Neck Pitch", "Neck Roll")
ALL_NAMES = ARM_NAMES["left"] + ARM_NAMES["right"] + HEAD_NAMES


@dataclass(frozen=True)
class ArmHeadSettings:
    iterations: int = 80
    position_tolerance_m: float = 0.015
    orientation_tolerance_rad: float = 0.15
    orientation_weight_m: float = 0.15
    damping: float = 0.02
    max_ik_step_rad: float = 0.10
    max_reach_m: float = 0.90
    minimum_palm_height_m: float = 0.20
    max_joint_speed_rad_s: float = 0.8
    max_joint_acceleration_rad_s2: float = 3.0
    filter_time_s: float = 0.06
    max_control_dt_s: float = 0.10
    elbow_gain: float = 0.05
    collision_tolerance_m: float = 0.002
    existing_contact_allowance_m: float = 0.001
    path_samples: int = 8
    orientation_alternative_rad: float = 0.08

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ("iterations", "path_samples"):
                if type(value) is not int or not 1 <= value <= 200:
                    raise ValueError(f"{name} must be an integer in[1,200]")
            elif type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.path_samples < 2:
            raise ValueError("path_samples must be at least2")


@dataclass(frozen=True)
class ArmHeadProposal:
    valid: bool
    targets: Mapping[str, float]
    reason: str
    diagnostics: dict


class ArmHeadRetargeter:
    """Scratch-only DLS with a complete snapshot and fixed torso/lift/base.

    ``palm_targets`` are calibrated native root-relative palm origins/orientations,
    not raw wrists. ``head_target`` is the calibrated head body orientation in the
    same frame; its translation is not commanded. #16 owns wrist-to-palm offsets.
    Optional elbow points guide null-space posture only. Outputs are partial
    named native actuator maps (arms16, head3), never complete live commands.
    """

    def __init__(
        self,
        config: ShelfConfig | None = None,
        *,
        mabel_root=None,
        seed=0,
        settings: ArmHeadSettings | None = None,
        expected_model_sha256=None,
    ):
        self._owner = threading.current_thread()
        self.settings = settings or ArmHeadSettings()
        self._env = ShelfStockingEnv(config, mabel_root=mabel_root, seed=seed)
        self.model, self.data = self._env.model, self._env.data
        self._closed = False
        self._model_hash = hashlib.sha256(self._env.xml.encode()).hexdigest()
        if expected_model_sha256 is not None and self._model_hash != expected_model_sha256:
            self._env.close()
            raise ValueError("retargeting scratch model differs from simulator model")
        self._seed = seed
        self._root = self.model.body("mabel_chassis").id
        self._bodies = {side: self.model.body(f"{side}_palm").id for side in ARM_NAMES}
        self._bodies["head"] = self.model.body("head").id
        self._elbows = {
            "left": self.model.body("l_forearm").id,
            "right": self.model.body("r_forearm").id,
        }
        self._shoulders = {
            "left": self.model.body("l_arm_mount").id,
            "right": self.model.body("r_arm_mount").id,
        }
        self._ids = {name: self.model.actuator(name).id for name in ALL_NAMES}
        self._qadr, self._dofs, self._bounds = {}, {}, {}
        for name in ALL_NAMES:
            joint, actuator = self.model.joint(name), self.model.actuator(name)
            aid = actuator.id
            if (
                int(actuator.trntype[0]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                or int(actuator.trnid[0]) != joint.id
                or not np.array_equal(actuator.gear, [1, 0, 0, 0, 0, 0])
                or self.model.actuator_biasprm[aid, 1] >= 0
                or self.model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise ValueError(f"unsupported native position channel: {name}")
            self._qadr[name] = int(joint.qposadr[0])
            self._dofs[name] = int(joint.dofadr[0])
            lo = max(float(actuator.ctrlrange[0]), float(joint.range[0]))
            hi = min(float(actuator.ctrlrange[1]), float(joint.range[1]))
            if not lo < hi:
                raise ValueError(f"empty native bounds: {name}")
            self._bounds[name] = (lo, hi)
        self._robot = set()
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor:
                if ancestor == self._root:
                    self._robot.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self._last = None
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _check(self):
        if self._closed or threading.current_thread() is not self._owner:
            raise RuntimeError("retargeter must remain open on its creating thread")

    def describe(self):
        self._check()
        return dict(
            config=asdict(self._env.config),
            seed=self._seed,
            model_xml_sha256=self._model_hash,
            settings=asdict(self.settings),
            mujoco_version=mujoco.__version__,
            actuator_names=list(ALL_NAMES),
            scope="scratch pose proposals; caller gates and applies through native physics",
        )

    def reset(self, state: SimState | None = None):
        """Clear filtering/rate history on session/generation/loss or gate rejection."""
        self._check()
        self._last = None
        if state is not None:
            self._load(state)

    def _load(self, state):
        if not isinstance(state, SimState):
            raise ValueError("a detached SimState is required")
        if any(
            type(value) is not int or value < 0
            for value in (state.reset_generation, state.step_index)
        ):
            raise ValueError("snapshot generation/step must be nonnegative integers")
        for field, count in (
            ("qpos", self.model.nq),
            ("qvel", self.model.nv),
            ("applied_targets", self.model.nu),
        ):
            value = getattr(state, field)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (count,)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid snapshot {field}")
        if tuple(state.actuator_names) != self._env.actuator_names:
            raise ValueError("snapshot actuator ownership differs from model")
        if (
            type(state.sim_time_s) not in (int, float)
            or not np.isfinite(state.sim_time_s)
            or state.sim_time_s < 0
        ):
            raise ValueError("invalid simulation time")
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel
        self.data.ctrl[:] = state.applied_targets
        self.data.time = state.sim_time_s
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.xpos).all() or not np.isfinite(self.data.xmat).all():
            raise ValueError("snapshot produced nonfinite geometry")

    def _root_pose(self, body):
        root_rotation = self.data.xmat[self._root].reshape(3, 3)
        position = root_rotation.T @ (self.data.xpos[body] - self.data.xpos[self._root])
        rotation = root_rotation.T @ self.data.xmat[body].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        return Pose(tuple(position), tuple(quaternion))

    def measured_poses(self, state):
        self._check()
        self._load(state)
        return {key: self._root_pose(body) for key, body in self._bodies.items()}

    def _contacts(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        result = {}
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if b1 not in self._robot and b2 not in self._robot:
                continue
            bodies = (self.model.body(b1).name, self.model.body(b2).name)
            geoms = (self.model.geom(g1).name, self.model.geom(g2).name)
            if "floor" in geoms and any(name.endswith("wheel") for name in bodies):
                continue
            key = tuple(sorted((g1, g2)))
            result[key] = min(result.get(key, 0), float(contact.dist))
        return result

    def _collision_errors(self, baseline):
        cfg = self.settings
        return [
            dict(geom_ids=list(pair), distance_m=distance, baseline_distance_m=baseline.get(pair))
            for pair, distance in self._contacts().items()
            if distance
            < min(
                -cfg.collision_tolerance_m, baseline.get(pair, 0) - cfg.existing_contact_allowance_m
            )
        ]

    def _forward(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _solve(self, names, body, desired, *, position=True, elbow=None, side=None):
        cfg = self.settings
        qadr = np.array([self._qadr[n] for n in names])
        dofs = np.array([self._dofs[n] for n in names])
        bounds = np.array([self._bounds[n] for n in names])
        seed = self.data.qpos[qadr].copy()
        q = np.clip(seed, bounds[:, 0], bounds[:, 1])
        best, best_q = np.inf, q.copy()
        for _ in range(cfg.iterations):
            self.data.qpos[qadr] = q
            self._forward()
            actual = self.data.xmat[body].reshape(3, 3)
            angular = rotation_error(actual, desired[1])
            error = (
                np.r_[desired[0] - self.data.xpos[body], cfg.orientation_weight_m * angular]
                if position
                else angular
            )
            cost = float(error @ error)
            if cost < best:
                best, best_q = cost, q.copy()
            if np.linalg.norm(angular) < min(cfg.orientation_tolerance_rad / 5, 1e-4) and (
                not position
                or np.linalg.norm(error[:3]) < min(cfg.position_tolerance_m / 5, 0.0002)
            ):
                break
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, body)
            jac = (
                np.vstack((self._jacp[:, dofs], cfg.orientation_weight_m * self._jacr[:, dofs]))
                if position
                else self._jacr[:, dofs].copy()
            )
            inverse = jac.T @ np.linalg.solve(
                jac @ jac.T + cfg.damping**2 * np.eye(len(error)), np.eye(len(error))
            )
            delta = inverse @ error
            if elbow is not None:
                mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self._elbows[side])
                guidance = self._jacp[:, dofs].T @ (elbow - self.data.xpos[self._elbows[side]])
                delta += cfg.elbow_gain * (np.eye(len(names)) - inverse @ jac) @ guidance
            delta = np.clip(delta, -cfg.max_ik_step_rad, cfg.max_ik_step_rad)
            q = np.clip(q + delta, bounds[:, 0], bounds[:, 1])
        self.data.qpos[qadr] = best_q
        self._forward()
        position_error = (
            float(np.linalg.norm(desired[0] - self.data.xpos[body])) if position else None
        )
        angle_error = float(
            np.linalg.norm(rotation_error(self.data.xmat[body].reshape(3, 3), desired[1]))
        )
        feasible = (
            not position or position_error <= cfg.position_tolerance_m
        ) and angle_error <= cfg.orientation_tolerance_rad
        return best_q, dict(
            position_error_m=position_error,
            orientation_error_rad=angle_error,
            feasible=bool(feasible),
            elbow_guidance=elbow is not None,
            active_limits=[
                n
                for i, n in enumerate(names)
                if min(best_q[i] - bounds[i, 0], bounds[i, 1] - best_q[i]) < 0.002
            ],
        )

    def propose(
        self,
        state: SimState,
        palm_targets: Mapping[str, Pose] | None = None,
        head_target: Pose | None = None,
        *,
        elbows=None,
        channels=("arms", "head"),
    ):
        self._check()
        diagnostics = {}
        try:
            if (
                not channels
                or len(set(channels)) != len(channels)
                or set(channels) - {"arms", "head"}
            ):
                raise ValueError("channels must select arms and/or head exactly once")
            self._load(state)
            if state.terminated or state.truncated:
                raise ValueError("episode finished")
            dt = self._env.dt
            if self._last is not None and state.reset_generation == self._last[0]:
                if state.step_index <= self._last[1] or state.sim_time_s <= self._last[2]:
                    raise ValueError("snapshot did not advance; reset history after realignment")
                dt = state.sim_time_s - self._last[2]
            if not 0 < dt <= self.settings.max_control_dt_s:
                raise ValueError("control time gap exceeds configured bound")
            baseline = self._contacts()
            diagnostics["baseline_penetrating_contacts"] = len(
                [v for v in baseline.values() if v < -0.002]
            )
            root_rotation = self.data.xmat[self._root].reshape(3, 3).copy()
            root_position = self.data.xpos[self._root].copy()
            nominal, solve_reports = {}, {}
            if "arms" in channels:
                if not isinstance(palm_targets, Mapping) or set(palm_targets) != {"left", "right"}:
                    raise ValueError("both named calibrated palm targets are required")
                elbows = {} if elbows is None else elbows
                if not isinstance(elbows, Mapping) or set(elbows) - {"left", "right"}:
                    raise ValueError("unknown elbow guidance source")
                for side in ("left", "right"):
                    pose = palm_targets[side]
                    if not isinstance(pose, Pose):
                        raise ValueError("palm targets must be validated calibrated Pose objects")
                    target = root_position + root_rotation @ pose.position_m
                    shoulder = self.data.xpos[self._shoulders[side]]
                    if (
                        np.linalg.norm(target - shoulder) > self.settings.max_reach_m
                        or pose.position_m[2] < self.settings.minimum_palm_height_m
                    ):
                        raise ValueError(f"{side} target outside configured workspace")
                    guidance = elbows.get(side)
                    good_guidance = False
                    if guidance is not None:
                        array = np.asarray(guidance)
                        good_guidance = (
                            array.shape == (3,)
                            and array.dtype.kind in "iuf"
                            and np.isfinite(array).all()
                            and np.linalg.norm(root_position + root_rotation @ array - shoulder)
                            <= self.settings.max_reach_m
                        )
                        guidance = root_position + root_rotation @ array if good_guidance else None
                    # Bound alternative orientation search without relaxing the
                    # residual limit against the ORIGINAL requested orientation.
                    original_q = self.data.qpos.copy()
                    preferred = root_rotation @ pose.rotation
                    alternatives = [(None, 0.0)] + [
                        (axis, sign * self.settings.orientation_alternative_rad)
                        for axis in range(3)
                        for sign in (-1, 1)
                    ]
                    best = None
                    for axis, angle in alternatives:
                        self.data.qpos[:] = original_q
                        selected = preferred.copy()
                        if axis is not None:
                            quaternion, matrix = np.empty(4), np.empty(9)
                            mujoco.mju_axisAngle2Quat(quaternion, np.eye(3)[axis], angle)
                            mujoco.mju_quat2Mat(matrix, quaternion)
                            selected = preferred @ matrix.reshape(3, 3)
                        q, report = self._solve(
                            ARM_NAMES[side],
                            self._bodies[side],
                            (target, selected),
                            elbow=guidance,
                            side=side,
                        )
                        report["selected_orientation_error_rad"] = report["orientation_error_rad"]
                        requested_error = float(
                            np.linalg.norm(
                                rotation_error(
                                    self.data.xmat[self._bodies[side]].reshape(3, 3), preferred
                                )
                            )
                        )
                        report["orientation_error_rad"] = requested_error
                        report["orientation_alternative_axis"] = axis
                        report["orientation_alternative_rad"] = angle
                        report["feasible"] = (
                            report["feasible"]
                            and requested_error <= self.settings.orientation_tolerance_rad
                        )
                        report["endpoint_collisions"] = self._collision_errors(baseline)
                        report["feasible"] = (
                            report["feasible"] and not report["endpoint_collisions"]
                        )
                        if best is None or (report["position_error_m"], requested_error) < (
                            best[1]["position_error_m"],
                            best[1]["orientation_error_rad"],
                        ):
                            best = (q, report)
                        if report["feasible"]:
                            best = (q, report)
                            break
                    q, report = best
                    self.data.qpos[:] = original_q
                    self.data.qpos[[self._qadr[n] for n in ARM_NAMES[side]]] = q
                    self._forward()
                    report["elbow_fallback"] = (
                        "measured-posture seed" if not good_guidance else None
                    )
                    solve_reports[side] = report
                    nominal.update(zip(ARM_NAMES[side], q.tolist(), strict=True))
                diagnostics["ik"] = solve_reports
                if not all(r["feasible"] for r in solve_reports.values()):
                    raise ValueError("no acceptable arm IK solution found")
            if "head" in channels:
                if not isinstance(head_target, Pose):
                    raise ValueError("a validated calibrated head orientation is required")
                q, report = self._solve(
                    HEAD_NAMES,
                    self._bodies["head"],
                    (None, root_rotation @ head_target.rotation),
                    position=False,
                )
                solve_reports["head"] = report
                diagnostics["ik"] = solve_reports
                if not report["feasible"]:
                    raise ValueError("no acceptable neck solution found")
                nominal.update(zip(HEAD_NAMES, q.tolist(), strict=True))
            endpoint_qpos = self.data.qpos.copy()
            errors = self._collision_errors(baseline)
            if errors:
                diagnostics["nominal_collisions"] = errors
                raise ValueError("nominal coupled endpoint collision")
            names = tuple(nominal)
            ids = [self._ids[n] for n in names]
            current = state.applied_targets[ids]
            previous_velocity = np.zeros(len(names))
            if self._last is not None and state.reset_generation == self._last[0]:
                previous_velocity = (current - self._last[3][ids]) / dt
            bounds = np.array([self._bounds[n] for n in names])
            cfg = self.settings
            if np.any(current < bounds[:, 0]) or np.any(current > bounds[:, 1]):
                raise ValueError("applied references outside native joint/actuator bounds")
            desired = current + dt / (cfg.filter_time_s + dt) * (
                np.array(list(nominal.values())) - current
            )
            low = np.maximum.reduce(
                (
                    np.full(len(names), -cfg.max_joint_speed_rad_s),
                    previous_velocity - cfg.max_joint_acceleration_rad_s2 * dt,
                    (bounds[:, 0] - current) / dt,
                )
            )
            high = np.minimum.reduce(
                (
                    np.full(len(names), cfg.max_joint_speed_rad_s),
                    previous_velocity + cfg.max_joint_acceleration_rad_s2 * dt,
                    (bounds[:, 1] - current) / dt,
                )
            )
            if np.any(low > high):
                raise ValueError("no feasible rate/acceleration/native-limit intersection")
            velocity = np.clip((desired - current) / dt, low, high)
            bounded = current + velocity * dt
            # Check nominal path and final bounded path, including every robot
            # link and both hands in the complete measured finger posture.
            for label, goal in (("nominal", endpoint_qpos), ("bounded", None)):
                if goal is None:
                    goal = state.qpos.copy()
                    goal[[self._qadr[n] for n in names]] = bounded
                for fraction in np.linspace(0, 1, cfg.path_samples):
                    self.data.qpos[:] = state.qpos + fraction * (goal - state.qpos)
                    # Free-joint quaternion is unchanged between snapshots/goals.
                    errors = self._collision_errors(baseline)
                    if errors:
                        diagnostics[label + "_path_collisions"] = errors
                        raise ValueError(f"sampled {label} coupled path collision")
            self._last = (
                state.reset_generation,
                state.step_index,
                state.sim_time_s,
                state.applied_targets.copy(),
            )
            diagnostics.update(
                nominal_targets=nominal,
                bounded_targets=dict(zip(names, bounded.tolist(), strict=True)),
                proposal_velocity_rad_s=dict(zip(names, velocity.tolist(), strict=True)),
                dt_s=dt,
                path_samples=cfg.path_samples,
                scope="kinematic prefilter; native physics execution measured separately",
            )
            return ArmHeadProposal(
                True, dict(zip(names, bounded.tolist(), strict=True)), "accepted", diagnostics
            )
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            self._last = None
            return ArmHeadProposal(False, {}, str(error), diagnostics)

    def check_composed(
        self,
        state: SimState,
        complete_named_targets: Mapping[str, float],
        *,
        body_hold: Mapping[str, float] | None = None,
    ):
        """Post-composition scratch check, including the proposed finger posture.

        Inject this into #20's final gate after all target/rate corrections.
        Body retargeting is not enabled: only unchanged body references or a
        measured torso/lift hold are admitted. The final owner may additionally
        supply its latched measured body_hold reference for a bounded transition
        from the current command to that hold; this is not a human body target.
        Existing penetrations are not
        certified safe; only new/deeper penetrations beyond allowances reject.
        """
        self._check()
        try:
            self._load(state)
            if not isinstance(complete_named_targets, Mapping) or set(
                complete_named_targets
            ) != set(state.actuator_names):
                raise ValueError("composed check requires every named actuator exactly once")
            baseline = self._contacts()
            if body_hold is not None and (
                not isinstance(body_hold, Mapping) or set(body_hold) != {"Lift", "Torso"}
            ):
                raise ValueError("body hold must contain exactly Lift and Torso")
            goal = state.qpos.copy()
            for aid, name in enumerate(state.actuator_names):
                value = complete_named_targets[name]
                if (
                    isinstance(value, (bool, np.bool_))
                    or not np.isscalar(value)
                    or not np.isfinite(value)
                ):
                    raise ValueError("composed targets must be finite scalars")
                value = float(value)
                if (
                    self.model.actuator_ctrllimited[aid]
                    and not self.model.actuator_ctrlrange[aid, 0]
                    <= value
                    <= self.model.actuator_ctrlrange[aid, 1]
                ):
                    raise ValueError(f"composed native actuator bound: {name}")
                if name.endswith(" Steer") or name.endswith(" Drive"):
                    if value != state.applied_targets[aid]:
                        raise ValueError("composed base references must remain unchanged")
                    continue
                if name in ("Torso", "Lift"):
                    admitted_hold = False
                    if body_hold is not None:
                        hold = body_hold[name]
                        if (
                            isinstance(hold, (bool, np.bool_))
                            or not np.isscalar(hold)
                            or not np.isfinite(hold)
                            or not self.model.actuator_ctrlrange[aid, 0]
                            <= hold
                            <= self.model.actuator_ctrlrange[aid, 1]
                        ):
                            raise ValueError("body hold reference outside native bounds")
                        low, high = sorted((state.applied_targets[aid], hold))
                        admitted_hold = low - 1e-12 <= value <= high + 1e-12
                    if not (
                        admitted_hold
                        or np.isclose(value, state.applied_targets[aid], rtol=0, atol=1e-12)
                        or np.isclose(value, self.data.actuator_length[aid], rtol=0, atol=1e-12)
                    ):
                        raise ValueError("new human body commands require the #19 mapping")
                if name == "Lift":
                    tendon = self.model.tendon("Total Lift")
                    joints = [self.model.joint(n) for n in ("Lift Lower", "Lift Upper")]
                    start, count = (
                        int(self.model.tendon_adr[tendon.id]),
                        int(self.model.tendon_num[tendon.id]),
                    )
                    if (
                        self.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON
                        or self.model.actuator_trnid[aid, 0] != tendon.id
                        or count != 2
                        or set(self.model.wrap_objid[start : start + count])
                        != {j.id for j in joints}
                        or not np.all(
                            self.model.wrap_type[start : start + count]
                            == mujoco.mjtWrap.mjWRAP_JOINT
                        )
                        or not np.all(self.model.wrap_prm[start : start + count] == 1)
                    ):
                        raise ValueError("unsupported native lift transmission")
                    delta = (value - self.data.actuator_length[aid]) / 2
                    for joint in joints:
                        goal[int(joint.qposadr[0])] += delta
                        if not joint.range[0] <= goal[int(joint.qposadr[0])] <= joint.range[1]:
                            raise ValueError("composed lift stage outside native joint bound")
                    continue
                if self.model.actuator_trntype[
                    aid
                ] != mujoco.mjtTrn.mjTRN_JOINT or not np.array_equal(
                    self.model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]
                ):
                    raise ValueError("unsupported composed joint transmission")
                joint = self.model.joint(int(self.model.actuator_trnid[aid, 0]))
                if (
                    self.model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE
                    or not joint.range[0] <= value <= joint.range[1]
                ):
                    raise ValueError(f"composed native joint bound: {name}")
                goal[int(joint.qposadr[0])] = value
            for fraction in np.linspace(0, 1, self.settings.path_samples):
                self.data.qpos[:] = state.qpos + fraction * (goal - state.qpos)
                collisions = self._collision_errors(baseline)
                if collisions:
                    return dict(
                        valid=False,
                        reason="composed sampled path collision",
                        collisions=collisions,
                        fraction=float(fraction),
                    )
            return dict(
                valid=True,
                reason="accepted",
                collisions=[],
                path_samples=self.settings.path_samples,
                scope="sampled scratch geometry; not dynamics or continuous collision proof",
            )
        except (ValueError, KeyError, TypeError) as error:
            return dict(valid=False, reason=str(error), collisions=[])

    def close(self):
        self._check()
        self._env.close()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
