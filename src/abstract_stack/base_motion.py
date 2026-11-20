"""Native MABEL swerve commands and read-only whole-robot base path screening.

The controller returns position targets for steering and rad/s targets for drive
servos. It never assigns a base pose or velocity. Collision forecasts mutate
independent scratch data only and do not certify tracking/contact dynamics.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.interactions import InteractionFailure


def closest_steering(target, current):
    """Choose continuous steering within +/-pi/2, reversing drive if shorter."""
    if not np.isfinite([target, current]).all():
        raise ValueError("Steering angles must be finite")
    difference = float(np.arctan2(np.sin(target - current), np.cos(target - current)))
    sign = 1.0
    if difference > np.pi / 2:
        difference -= np.pi
        sign = -1.0
    elif difference < -np.pi / 2:
        difference += np.pi
        sign = -1.0
    return current + difference, sign


@dataclass(frozen=True)
class BaseMotionConfig:
    max_speed_m_s: float = 0.025
    max_yaw_rate_rad_s: float = 0.08
    steering_rate_rad_s: float = 1.0
    drive_accel_rad_s2: float = 0.8
    steering_alignment_rad: float = 0.08
    position_tolerance_m: float = 0.002
    yaw_tolerance_rad: float = 0.01
    motion_timeout_s: float = 20.0
    fixture_force_limit_N: float = 25.0
    path_margin_m: float = 0.008
    path_yaw_margin_rad: float = 0.03

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Base motion settings must be finite and positive")
        if self.max_speed_m_s > 0.1 or self.fixture_force_limit_N > 25:
            raise ValueError("Base probe exceeds its declared speed/force bounds")


class NativeSwerve:
    """Derive wheel mounts, rolling direction and radius from the native model."""

    PREFIXES = ("FL", "FR", "B")

    def __init__(self, model, data, config=None):
        self.model, self.data = model, data
        self.config = config or BaseMotionConfig()
        self.base = model.body("mabel_chassis").id
        rotation = data.xmat[self.base].reshape(3, 3)
        self.modules = []
        for prefix in self.PREFIXES:
            steer, drive = (model.joint(prefix + suffix) for suffix in (" Steer", " Drive"))
            sa, da = (model.actuator(prefix + suffix) for suffix in (" Steer", " Drive"))
            if (
                sa.biasprm[1] >= 0
                or da.biasprm[2] >= 0
                or da.biasprm[1] != 0
                or bool(steer.limited)
                or not bool(da.ctrllimited)
                or not np.isclose(sa.gear[0], 1)
                or not np.isclose(da.gear[0], 1)
            ):
                raise ValueError("Expected native continuous steer and velocity-drive servos")
            axis = data.xaxis[drive.id]
            rolling = rotation.T @ np.cross(axis, [0, 0, 1])
            angle = float(data.qpos[int(steer.qposadr[0])])
            zero_heading = float(np.arctan2(rolling[1], rolling[0]) - angle)
            wheel = int(model.jnt_bodyid[drive.id])
            radii = []
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] != wheel or not (
                    model.geom_contype[geom] or model.geom_conaffinity[geom]
                ):
                    continue
                if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise ValueError("Native wheel-radius derivation expects collision meshes")
                mesh = int(model.geom_dataid[geom])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                points = (
                    model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                    + data.geom_xpos[geom]
                )
                radial = np.linalg.norm(np.cross(points - data.xanchor[drive.id], axis), axis=1)
                radii.append(float(radial.max()))
            if not radii or not 0.02 < max(radii) < 0.06:
                raise ValueError("Native wheel radius outside expected geometry bounds")
            self.modules.append(
                dict(
                    prefix=prefix,
                    steer=steer.id,
                    drive=drive.id,
                    steer_aid=sa.id,
                    drive_aid=da.id,
                    zero_heading_rad=zero_heading,
                    radius_m=max(radii),
                    position_body_m=(rotation.T @ (data.xanchor[steer.id] - data.xpos[self.base]))[
                        :2
                    ],
                )
            )
        self.last_steer = np.array([data.ctrl[m["steer_aid"]] for m in self.modules])
        self.last_drive = np.array([data.ctrl[m["drive_aid"]] for m in self.modules])
        self.last_report = {}

    def module_requests(self, world_twist):
        twist = np.asarray(world_twist, dtype=float).copy()
        if twist.shape != (3,) or not np.isfinite(twist).all():
            raise ValueError("World twist must contain finite vx, vy and yaw rate")
        rotation = self.data.xmat[self.base].reshape(3, 3)
        velocity = rotation[:2, :2].T @ twist[:2]
        steering, drive, errors = [], [], []
        for i, module in enumerate(self.modules):
            x, y = module["position_body_m"]
            point_velocity = velocity + twist[2] * np.array([-y, x])
            speed = float(np.linalg.norm(point_velocity))
            current = float(self.data.qpos[int(self.model.jnt_qposadr[module["steer"]])])
            if speed < 1e-8:
                goal, sign = self.last_steer[i], 1.0
            else:
                goal, sign = closest_steering(
                    np.arctan2(point_velocity[1], point_velocity[0]) - module["zero_heading_rad"],
                    current,
                )
            error = abs(goal - current)
            wheel_speed = sign * speed / module["radius_m"]
            if error > self.config.steering_alignment_rad:
                wheel_speed = 0.0
            else:
                wheel_speed *= np.cos(error)
            steering.append(goal)
            drive.append(wheel_speed)
            errors.append(error)
        return np.asarray(steering), np.asarray(drive), np.asarray(errors)

    def targets(self, world_twist, *, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Swerve dt must be finite and positive")
        twist = np.asarray(world_twist, dtype=float).copy()
        if twist.shape != (3,) or not np.isfinite(twist).all():
            raise ValueError("World twist must contain finite vx, vy and yaw rate")
        length = float(np.linalg.norm(twist[:2]))
        if length > self.config.max_speed_m_s:
            twist = twist.copy()
            twist[:2] *= self.config.max_speed_m_s / length
        twist[2] = np.clip(
            twist[2], -self.config.max_yaw_rate_rad_s, self.config.max_yaw_rate_rad_s
        )
        desired_steer, desired_drive, errors = self.module_requests(twist)
        steer = self.last_steer + np.clip(
            desired_steer - self.last_steer,
            -self.config.steering_rate_rad_s * dt,
            self.config.steering_rate_rad_s * dt,
        )
        drive = self.last_drive + np.clip(
            desired_drive - self.last_drive,
            -self.config.drive_accel_rad_s2 * dt,
            self.config.drive_accel_rad_s2 * dt,
        )
        result = {}
        native_limited = []
        for i, module in enumerate(self.modules):
            for suffix, value in ((" Steer", steer[i]), (" Drive", drive[i])):
                actuator = self.model.actuator(module["prefix"] + suffix)
                if actuator.ctrllimited:
                    bounded = np.clip(value, *actuator.ctrlrange)
                    if abs(bounded - value) > 1e-12:
                        native_limited.append(module["prefix"] + suffix)
                    value = bounded
                result[module["prefix"] + suffix] = float(value)
            steer[i], drive[i] = (
                result[module["prefix"] + " Steer"],
                result[module["prefix"] + " Drive"],
            )
        self.last_steer, self.last_drive = steer, drive
        self.last_report = dict(
            world_twist_m_s_rad_s=twist.tolist(),
            steering_error_rad=errors.tolist(),
            steering_rate_limited=[
                m["prefix"]
                for i, m in enumerate(self.modules)
                if abs(desired_steer[i] - steer[i]) > 1e-9
            ],
            drive_rate_limited=[
                m["prefix"]
                for i, m in enumerate(self.modules)
                if abs(desired_drive[i] - drive[i]) > 1e-9
            ],
            drive_waiting_for_steering=[
                m["prefix"]
                for i, m in enumerate(self.modules)
                if errors[i] > self.config.steering_alignment_rad
            ],
            native_target_limited=native_limited,
            targets=result,
        )
        return result


def screen_base_translation(
    env, displacement_xy, *, config=None, carried_product=None, turn_rad=0.0
):
    """Screen translation or an unloaded in-place turn using scratch data.

    A named ``carried_product`` moves rigidly with the robot: its geoms are
    removed from the fixture set and its free-joint position tracks the base, so
    a retained load is not mistaken for a stationary obstacle.
    """
    config = config or BaseMotionConfig()
    displacement = np.asarray(displacement_xy, dtype=float)
    if (
        displacement.shape != (2,)
        or not np.isfinite(displacement).all()
        or np.linalg.norm(displacement) > 0.15
    ):
        raise ValueError("Base probe displacement must be finite and within 0.15 m")
    if carried_product is not None and not isinstance(carried_product, str):
        raise ValueError("carried_product must be a product name or None")
    if (
        not np.isfinite(turn_rad)
        or abs(turn_rad) > np.pi / 2
        or (turn_rad and (np.any(displacement) or carried_product is not None))
    ):
        raise ValueError("A screened turn must be in place, unloaded and within 90 degrees")
    snapshot = env.clone_for_planning()
    model, data = snapshot.model, snapshot.data
    swerve = NativeSwerve(model, data, config)
    desired, _, _ = swerve.module_requests([*displacement, turn_rad])
    initial = data.qpos.copy()
    base_adr = int(model.joint("base_root").qposadr[0])
    carried_body = model.body(carried_product).id if carried_product is not None else None
    carried_joint_adr = (
        int(model.joint(f"{carried_product}_joint").qposadr[0])
        if carried_product is not None
        else None
    )
    root, robot = model.body("mabel_chassis").id, set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == root:
                robot.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    robot_geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in robot and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    fixtures = np.array(
        [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] not in robot
            and model.geom_bodyid[g] != carried_body
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
    )
    floor = model.geom("floor").id
    samples = 0
    segments = int(np.ceil(max(np.linalg.norm(displacement) / 0.005, abs(turn_rad) / 0.025)))
    for alpha in np.linspace(0, 1, max(2, segments + 1)):
        for margin in (-config.path_yaw_margin_rad, 0, config.path_yaw_margin_rad):
            yaw = alpha * turn_rad + margin
            for steer_alpha in (0, 0.5, 1):
                data.qpos[:] = initial
                data.qpos[base_adr : base_adr + 2] += alpha * displacement
                if carried_joint_adr is not None:
                    data.qpos[carried_joint_adr : carried_joint_adr + 2] += alpha * displacement
                quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                mujoco.mju_mulQuat(
                    data.qpos[base_adr + 3 : base_adr + 7],
                    quat,
                    initial[base_adr + 3 : base_adr + 7],
                )
                for i, module in enumerate(swerve.modules):
                    adr = int(model.jnt_qposadr[module["steer"]])
                    data.qpos[adr] += steer_alpha * (desired[i] - initial[adr])
                mujoco.mj_forward(model, data)
                samples += 1
                for geom in robot_geoms:
                    delta = data.geom_xpos[fixtures] - data.geom_xpos[geom]
                    possible = (
                        np.linalg.norm(delta, axis=1)
                        - model.geom_rbound[fixtures]
                        - model.geom_rbound[geom]
                        < config.path_margin_m
                    )
                    possible |= fixtures == floor
                    for fixture in fixtures[possible]:
                        wheel_floor = fixture == floor and model.body(
                            model.geom_bodyid[geom]
                        ).name.endswith("wheel")
                        if wheel_floor:
                            continue
                        margin = 0.0 if fixture == floor else config.path_margin_m
                        distance = float(
                            mujoco.mj_geomDistance(
                                model, data, geom, fixture, max(0.001, margin), None
                            )
                        )
                        if distance < margin - 1e-9:
                            return dict(
                                accepted=False,
                                checked_samples=samples,
                                rejection=dict(
                                    translation_fraction=float(alpha),
                                    yaw_rad=yaw,
                                    steering_fraction=steer_alpha,
                                    distance_m=distance,
                                    robot_body=model.body(model.geom_bodyid[geom]).name,
                                    fixture=model.geom(fixture).name,
                                ),
                            )
    return dict(
        accepted=True,
        checked_samples=samples,
        clearance_lower_bound_m=config.path_margin_m,
        yaw_envelope_rad=config.path_yaw_margin_rad,
        scope="kinematic geometry; no tracking or tire-dynamics certificate",
    )


class BaseMotionFailure(InteractionFailure):
    """A base action failed screening, tracking or a measured physical guard."""


class BaseContactGuard:
    """Per-physics-step base-action checks without changing episode/observer state.

    ``step(env)`` must run after every native physics step, including steering
    alignment and braking. It never resets observers, sets targets or changes
    physics. An optional displacement bound applies relative to construction.
    Force aborts are terminal action failures, not permission to drive again.
    """

    def __init__(
        self,
        env,
        config=None,
        *,
        reference_yaw=None,
        max_product_displacement_m=None,
        exclude_products=(),
        turn_rad=0.0,
        max_base_displacement_m=None,
    ):
        self.model = env.model
        self.config = config or BaseMotionConfig()
        if max_product_displacement_m is not None and (
            not np.isfinite(max_product_displacement_m) or max_product_displacement_m <= 0
        ):
            raise ValueError("Product displacement bound must be finite and positive")
        self.displacement_limit = max_product_displacement_m
        excluded = set(exclude_products)
        if not excluded <= {p.name for p in env.products}:
            raise ValueError("Excluded products must be named scene products")
        self.base = self.model.body("mabel_chassis").id
        rotation = env.data.xmat[self.base].reshape(3, 3)
        self.reference_yaw = (
            float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            if reference_yaw is None
            else float(reference_yaw)
        )
        if not np.isfinite(self.reference_yaw):
            raise ValueError("Base guard yaw reference must be finite")
        if not np.isfinite(turn_rad) or abs(turn_rad) > np.pi / 2:
            raise ValueError("Base guard turn must be finite and within 90 degrees")
        if max_base_displacement_m is not None and (
            not np.isfinite(max_base_displacement_m) or max_base_displacement_m <= 0
        ):
            raise ValueError("Base drift bound must be finite and positive")
        self.turn_rad = turn_rad
        self.base_displacement_limit = max_base_displacement_m
        self.initial_base_xy = env.data.xpos[self.base, :2].copy()
        self.robot = set()
        for body in range(self.model.nbody):
            ancestor = body
            while ancestor:
                if ancestor == self.base:
                    self.robot.add(body)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        self.products = {self.model.body(p.name).id: p.name for p in env.products}
        self.initial_products = {p.name: env.data.body(p.name).xpos.copy() for p in env.products}
        # Absolute world-frame displacement guards the products that must stay
        # put; a retained carried load is excluded and monitored by the caller's
        # relative grasp-slip check instead.
        self.displacement_products = {
            body_id: name for body_id, name in self.products.items() if name not in excluded
        }
        self.maximum_product_displacement_m = dict.fromkeys(self.initial_products, 0.0)
        self.floor = self.model.geom("floor").id
        self.actuator_ids = [
            self.model.actuator(prefix + suffix).id
            for prefix in NativeSwerve.PREFIXES
            for suffix in (" Steer", " Drive")
        ]
        self.peaks = dict(robot_fixture_N=0.0, robot_product_N=0.0, carton_pair_N=0.0)
        self.peak_actuator_effort_Nm = {self.model.actuator(a).name: 0.0 for a in self.actuator_ids}
        self.incidents = []
        self.velocity = np.zeros(6)
        self.last_sample = {}

    def step(self, env):
        if env.model is not self.model:
            raise ValueError("Base guard belongs to a different MuJoCo model")
        model, data = env.model, env.data
        fixture, product, pairs = 0.0, 0.0, {}
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = max(0.0, float(wrench[0]))
            if (b1 in self.robot) != (b2 in self.robot):
                rbody, other, other_geom = (b1, b2, g2) if b1 in self.robot else (b2, b1, g1)
                if other_geom == self.floor and model.body(rbody).name.endswith("wheel"):
                    continue
                if other in self.products:
                    product += normal
                else:
                    fixture += normal
            if b1 in self.products and b2 in self.products:
                key = tuple(sorted((b1, b2)))
                pairs[key] = pairs.get(key, 0.0) + normal
        pair = max(pairs.values(), default=0.0)
        loads = dict(robot_fixture_N=fixture, robot_product_N=product, carton_pair_N=pair)
        for name, value in loads.items():
            self.peaks[name] = max(self.peaks[name], value)
        for aid in self.actuator_ids:
            name = model.actuator(aid).name
            self.peak_actuator_effort_Nm[name] = max(
                self.peak_actuator_effort_Nm[name], abs(float(data.actuator_force[aid]))
            )
        rotation = data.xmat[self.base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        yaw_error = float(
            np.arctan2(np.sin(yaw - self.reference_yaw), np.cos(yaw - self.reference_yaw))
        )
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.base, self.velocity, 0)
        for body, name in self.displacement_products.items():
            displacement = float(np.linalg.norm(data.xpos[body] - self.initial_products[name]))
            self.maximum_product_displacement_m[name] = max(
                self.maximum_product_displacement_m[name], displacement
            )
        for body, name in self.products.items():
            tipped = data.xmat[body].reshape(3, 3)[2, 2] < np.cos(np.radians(45))
            dropped = data.xpos[body, 2] < 0.25
            if tipped or dropped:
                self.incidents.append(
                    dict(
                        product=name,
                        kind="tipped" if tipped else "dropped",
                        sim_time=float(data.time),
                    )
                )
        self.last_sample = dict(
            sim_time=float(data.time),
            base_xyz_m=data.xpos[self.base].tolist(),
            base_wxyz=data.xquat[self.base].tolist(),
            velocity_world=self.velocity.tolist(),
            yaw_error_rad=yaw_error,
            **loads,
        )
        if max(loads.values()) > self.config.fixture_force_limit_N:
            raise BaseMotionFailure("base_motion: native contact-force guard exceeded")
        if self.incidents:
            raise BaseMotionFailure("base_motion: carton tip/drop incident")
        if rotation[2, 2] < np.cos(np.radians(2)):
            raise BaseMotionFailure("base_motion: base tilt guard")
        margin = self.config.path_yaw_margin_rad
        if not min(0, self.turn_rad) - margin <= yaw_error <= max(0, self.turn_rad) + margin:
            raise BaseMotionFailure("base_motion: yaw left screened envelope")
        if (
            self.base_displacement_limit is not None
            and np.linalg.norm(data.xpos[self.base, :2] - self.initial_base_xy)
            > self.base_displacement_limit
        ):
            raise BaseMotionFailure("base_motion: turn drift left screened envelope")
        if (
            self.displacement_limit is not None
            and max(self.maximum_product_displacement_m.values(), default=0)
            > self.displacement_limit
        ):
            raise BaseMotionFailure("base_motion: product disturbance guard")
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise BaseMotionFailure("base_motion: unexpected external force channels")
        return self.last_sample

    def report(self):
        return dict(
            peak_contact_normal_N=self.peaks.copy(),
            peak_actuator_effort_Nm=self.peak_actuator_effort_Nm.copy(),
            maximum_product_displacement_m=self.maximum_product_displacement_m.copy(),
            incidents=self.incidents.copy(),
            reference_yaw_rad=self.reference_yaw,
            screened_turn_rad=self.turn_rad,
            base_displacement_limit_m=self.base_displacement_limit,
            product_displacement_limit_m=self.displacement_limit,
        )
