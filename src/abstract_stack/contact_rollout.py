"""Independent native-physics forecasts; never a fresh manipulation episode.

Each candidate owns its data. Only its actuator controls change after reset.
Model arrays are shared read-only. Forces and every product are checked at each
physics step, rather than inferred from sparsely saved trajectory endpoints.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class RolloutScene:
    products: tuple[str, ...]
    active: str
    robot_root: str
    hand_root: str
    support_geoms: tuple[str, ...]
    active_half_size: tuple[float, float, float]
    support_height: float
    back_geom: str | None = None
    floor_geom: str = "floor"
    base_root: str | None = None
    force_limit_N: float = 25.0
    active_tilt_degrees: float = 15.0
    other_tilt_degrees: float = 45.0
    support_loss_seconds: float = 0.10
    contact_loss_seconds: float = 0.04
    support_clearance_m: float = 0.025
    drop_height_m: float = 0.25

    def __post_init__(self):
        for name in ("products", "support_geoms", "active_half_size"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            not self.products
            or len(set(self.products)) != len(self.products)
            or self.active not in self.products
            or not np.isfinite(self.support_height)
            or len(self.active_half_size) != 3
            or any(not np.isfinite(v) or v <= 0 for v in self.active_half_size)
            or not 0 < self.force_limit_N <= 25
        ):
            raise ValueError("Invalid rollout scene or force guard")
        checks = (
            self.active_tilt_degrees,
            self.other_tilt_degrees,
            self.support_loss_seconds,
            self.contact_loss_seconds,
            self.support_clearance_m,
            self.drop_height_m,
        )
        if (
            any(not np.isfinite(v) or v <= 0 for v in checks)
            or self.active_tilt_degrees > 15
            or self.other_tilt_degrees > 45
            or self.support_loss_seconds > 0.10
            or self.contact_loss_seconds > 0.04
            or self.support_clearance_m > 0.025
            or self.drop_height_m < 0.25
        ):
            raise ValueError("Rollout scene may not weaken the native push guards")

    @classmethod
    def from_env(cls, env, active):
        product = next(p for p in env.products if p.name == active)
        return cls(
            products=tuple(p.name for p in env.products),
            active=active,
            robot_root="mabel_chassis",
            hand_root="left_palm",
            support_geoms=("supply_table", f"shelf_board_{product.level}"),
            active_half_size=tuple(product.half_size),
            support_height=env.config.levels[product.level],
            back_geom="shelf_back",
            base_root="mabel_chassis",
        )


def _id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"Missing rollout model entity: {name}")
    return value


def _descendants(model, root):
    selected = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in selected:
            selected.add(body)
    return selected


class ContactMonitor:
    """Conservative released/supported push guard, not a grasp/carry controller."""

    def __init__(self, model, data, scene):
        self.model, self.scene = model, scene
        body = mujoco.mjtObj.mjOBJ_BODY
        geom = mujoco.mjtObj.mjOBJ_GEOM
        self.products = {_id(model, body, n): n for n in scene.products}
        self.active = _id(model, body, scene.active)
        if self.active not in self.products:
            raise ValueError("Active product is absent from scheduled products")
        self.robot = _descendants(model, _id(model, body, scene.robot_root))
        self.hand = _descendants(model, _id(model, body, scene.hand_root))
        if not self.hand <= self.robot or set(self.products) & self.robot:
            raise ValueError("Hand/product membership does not match the native robot")
        self.support = {_id(model, geom, n) for n in scene.support_geoms}
        self.back = None if scene.back_geom is None else _id(model, geom, scene.back_geom)
        floor = _id(model, geom, scene.floor_geom)
        self.floor_wheels = {
            b for b in self.robot if (mujoco.mj_id2name(model, body, b) or "").endswith("wheel")
        }
        self.floor = floor
        self.base = None if scene.base_root is None else _id(model, body, scene.base_root)
        self.initial = {b: data.xpos[b].copy() for b in self.products}
        self.initial_yaw = 0.0
        if self.base is not None:
            rotation = data.xmat[self.base].reshape(3, 3)
            self.initial_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        self.contact_seen = False
        self.contact_loss = self.support_loss = 0.0
        self.max_displacement = dict.fromkeys(scene.products, 0.0)
        self.peaks = dict(
            robot_product_N=0.0,
            nonhand_product_N=0.0,
            robot_fixture_N=0.0,
            carton_pair_N=0.0,
            back_N=0.0,
        )
        self.last = {}
        self.incidents = []

    def sample(self, data, *, elapsed):
        model, scene = self.model, self.scene
        pairs, robot_product, nonhand, fixture, hand, support = {}, 0.0, 0.0, 0.0, 0.0, 0.0
        active_force = np.zeros(3)
        back = 0.0
        wrench = np.empty(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            mujoco.mj_contactForce(model, data, index, wrench)
            if not np.isfinite(wrench).all():
                return "nonfinite_contact_force"
            normal = max(0.0, float(wrench[0]))
            if b1 in self.products and b2 in self.products:
                pair = tuple(sorted((b1, b2)))
                pairs[pair] = pairs.get(pair, 0.0) + normal
            if (b1 in self.robot) != (b2 in self.robot):
                robot, other, other_geom = (b1, b2, g2) if b1 in self.robot else (b2, b1, g1)
                if other in self.products:
                    robot_product += normal
                    if robot not in self.hand:
                        nonhand += normal
                elif not (other_geom == self.floor and robot in self.floor_wheels):
                    fixture += normal
            if self.active in (b1, b2):
                other, other_geom = (b2, g2) if b1 == self.active else (b1, g1)
                if other in self.hand:
                    hand += normal
                    active_force += (-1 if b1 == self.active else 1) * (
                        contact.frame.reshape(3, 3).T @ wrench[:3]
                    )
                if other_geom in self.support:
                    support += normal
                if other_geom == self.back:
                    back += normal
        loads = dict(
            robot_product_N=robot_product,
            nonhand_product_N=nonhand,
            robot_fixture_N=fixture,
            carton_pair_N=max(pairs.values(), default=0.0),
            back_N=back,
        )
        for name, value in loads.items():
            self.peaks[name] = max(self.peaks[name], value)
        self.last = dict(
            sim_time=float(data.time),
            hand_N=hand,
            support_N=support,
            active_force_world_N=active_force.tolist(),
            **loads,
        )
        for body, name in self.products.items():
            displacement = float(np.linalg.norm(data.xpos[body] - self.initial[body]))
            self.max_displacement[name] = max(self.max_displacement[name], displacement)
            angle = scene.active_tilt_degrees if body == self.active else scene.other_tilt_degrees
            if data.xmat[body].reshape(3, 3)[2, 2] < np.cos(np.radians(angle)):
                self.incidents.append(dict(product=name, kind="tipped", sim_time=float(data.time)))
            if data.xpos[body, 2] < scene.drop_height_m:
                self.incidents.append(dict(product=name, kind="dropped", sim_time=float(data.time)))
        if max(loads.values()) > scene.force_limit_N:
            return "contact_force_guard"
        if self.incidents:
            return "product_tip_or_drop"
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            return "external_force_input"
        if self.base is not None:
            rotation = data.xmat[self.base].reshape(3, 3)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0]) - self.initial_yaw
            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
            if rotation[2, 2] < np.cos(np.radians(2)) or abs(yaw) > 0.03:
                return "base_left_screened_envelope"
        extent = np.abs(data.xmat[self.active].reshape(3, 3)) @ scene.active_half_size
        gap = data.xpos[self.active, 2] - extent[2] - scene.support_height
        supported = support > 0.2 and -0.003 <= gap <= scene.support_clearance_m
        self.support_loss = 0.0 if supported else self.support_loss + elapsed
        self.contact_seen = self.contact_seen or hand > 0.2
        self.contact_loss = 0.0 if hand > 0.2 else self.contact_loss + elapsed
        if self.support_loss > scene.support_loss_seconds:
            return "lost_surface_support"
        if self.contact_loss > scene.contact_loss_seconds:
            return "lost_push_contact"
        return None


def integration_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def evaluate_batch(model, initial_state, candidates, scene, *, rates, workers=1):
    """Forecast explicit applied-target arrays, checking every native step.

    Candidates are dictionaries with unique ``name`` and ``controls[T, nu]``.
    Geometry/IK admission belongs to candidate generation; this function checks
    native bounds/rates and the physical result. No accepted forecast earns task
    credit or authorizes execution without a separate fresh-state admission.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use one through eight rollout workers")
    if not 1 <= len(candidates) <= 32 or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Use one through 32 uniquely named candidates")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
        raise ValueError("Rollouts require native contacts")
    rates = np.asarray(rates, float)
    initial_state = np.asarray(initial_state, float).copy()
    if (
        rates.shape != (model.nu,)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or initial_state.shape != (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),)
        or not np.isfinite(initial_state).all()
    ):
        raise ValueError("Invalid rollout state or native target rates")
    prepared = [{**c, "controls": np.array(c["controls"], float, copy=True)} for c in candidates]

    def evaluate(candidate):
        started = time.monotonic()
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        monitor = ContactMonitor(model, data, scene)
        origin = data.xpos[monitor.active].copy()
        controls = candidate["controls"]
        failure, steps = None, 0
        samples = []
        if (
            controls.ndim != 2
            or controls.shape[1] != model.nu
            or not 1 <= len(controls) <= int(2 / model.opt.timestep)
        ):
            failure = "invalid_control_shape_or_horizon"
        elif not np.isfinite(controls).all():
            failure = "nonfinite_controls"
        elif np.any(
            (
                controls[:, model.actuator_ctrllimited != 0]
                < model.actuator_ctrlrange[model.actuator_ctrllimited != 0, 0]
            )
            | (
                controls[:, model.actuator_ctrllimited != 0]
                > model.actuator_ctrlrange[model.actuator_ctrllimited != 0, 1]
            )
        ):
            failure = "native_control_limit"
        elif np.any(
            np.abs(np.diff(np.vstack((data.ctrl, controls)), axis=0))
            > rates * model.opt.timestep + 1e-10
        ):
            failure = "native_target_rate"
        else:
            failure = monitor.sample(data, elapsed=0.0)
        if failure is None:
            initial_time = float(data.time)
            for index, control in enumerate(controls):
                data.ctrl[:] = control
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                steps = index + 1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    failure = "nonfinite_state"
                elif abs(data.time - initial_time - steps * model.opt.timestep) > 1e-7:
                    failure = "physics_time_reset"
                elif np.any(data.warning.number):
                    failure = "native_warning"
                else:
                    failure = monitor.sample(data, elapsed=model.opt.timestep)
                samples.append(
                    [
                        float(data.time),
                        *data.xpos[monitor.active],
                        monitor.last.get("hand_N", 0),
                        monitor.last.get("support_N", 0),
                        *monitor.last.get("active_force_world_N", [0, 0, 0]),
                    ]
                )
                if failure:
                    break
        if failure is None and not monitor.contact_seen:
            failure = "no_push_contact_observed"
        result = dict(
            name=candidate["name"],
            eligible=failure is None,
            failure=failure,
            completed_steps=steps,
            requested_steps=len(controls),
            progress_minus_x_m=float(origin[0] - data.xpos[monitor.active, 0]),
            displacement_xyz_m=(data.xpos[monitor.active] - origin).tolist(),
            peak_forces_N=monitor.peaks,
            incidents=monitor.incidents,
            maximum_product_displacement_m=monitor.max_displacement,
            last_sample=monitor.last,
            wall_seconds=time.monotonic() - started,
            fresh_task_credit=0,
            live_execution_admitted=False,
            contact_seen=monitor.contact_seen,
        )
        return result, np.asarray(samples, float), integration_state(model, data)

    started = time.monotonic()
    if workers == 1:
        results = [evaluate(c) for c in prepared]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate, prepared))
    return results, time.monotonic() - started


def evaluate_screened_batch(
    model, initial_state, candidates, scene, *, rates, workers=4, shortlist=4
):
    """Rank with C++ trajectories, then guard a bounded shortlist from reset.

    Unchecked candidates never become eligible. Final selection uses guarded
    progress, not the optimistic screening score. This is an open-loop forecast
    primitive, not a stocked-task success or a live receding-horizon controller.
    """
    import mujoco.rollout

    if type(shortlist) is not int or not 1 <= shortlist <= len(candidates) <= 32:
        raise ValueError("Shortlist must be between one and the bounded candidate count")
    # Validate every complete applied-target sequence before launching the
    # screen. Neither ranking nor finite endpoint state establishes force safety.
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use one through eight rollout workers")
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
        raise ValueError("Rollouts require native contacts")
    initial = np.asarray(initial_state, float).copy()
    if (
        initial.shape != (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Invalid rollout state")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("External-force inputs are not admitted")
    rates = np.asarray(rates, float)
    if rates.shape != (model.nu,) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Invalid target-rate bounds")
    if len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    prepared = [{**c, "controls": np.array(c["controls"], float, copy=True)} for c in candidates]
    length = None
    for candidate in prepared:
        controls = candidate["controls"]
        if (
            controls.ndim != 2
            or controls.shape[1] != model.nu
            or not np.isfinite(controls).all()
            or not 1 <= len(controls) <= int(2 / model.opt.timestep)
        ):
            raise ValueError("Invalid candidate control shape or horizon")
        if length is not None and len(controls) != length:
            raise ValueError("C++ screen requires equal candidate horizons")
        length = len(controls)
        limited = model.actuator_ctrllimited != 0
        if (
            np.any(controls[:, limited] < model.actuator_ctrlrange[limited, 0])
            or np.any(controls[:, limited] > model.actuator_ctrlrange[limited, 1])
            or np.any(
                np.abs(np.diff(np.vstack((data.ctrl, controls)), axis=0))
                > rates * model.opt.timestep + 1e-10
            )
        ):
            raise ValueError("Candidate exceeds native control or declared target-rate bounds")
    user_spec = int(mujoco.mjtState.mjSTATE_USER)
    if user_spec & (int(mujoco.mjtState.mjSTATE_CTRL) - 1):
        raise ValueError("Unsupported native user-state layout")
    user = np.empty(mujoco.mj_stateSize(model, user_spec))
    mujoco.mj_getState(model, data, user, user_spec)
    controls = np.tile(user, (len(prepared), length, 1))
    for index, candidate in enumerate(prepared):
        controls[index, :, : model.nu] = candidate["controls"]
    full = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, full, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    body = _id(model, mujoco.mjtObj.mjOBJ_BODY, scene.active)
    joint = int(model.body_jntadr[body])
    if (
        joint < 0
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
        or model.body_parentid[body] != 0
    ):
        raise ValueError("Screen ranking requires a world-root free-joint product")
    x_index = 1 + int(model.jnt_qposadr[joint])
    started = time.monotonic()
    with mujoco.rollout.Rollout(nthread=workers) as pool:
        states, _ = pool.rollout(
            model,
            [mujoco.MjData(model) for _ in range(workers)],
            full,
            controls,
            control_spec=user_spec,
            initial_warmstart=data.qacc_warmstart.copy(),
        )
    screen_seconds = time.monotonic() - started
    ranking = []
    rows = []
    for index, candidate in enumerate(prepared):
        valid = bool(
            np.isfinite(states[index]).all()
            and np.allclose(
                states[index, :, 0],
                data.time + (np.arange(length) + 1) * model.opt.timestep,
                atol=1e-7,
                rtol=0,
            )
        )
        progress = float(data.qpos[x_index - 1] - states[index, -1, x_index]) if valid else None
        rows.append(
            dict(
                name=candidate["name"],
                screen_progress_minus_x_m=progress,
                eligible=False,
                verification_status="not_force_checked",
                fresh_task_credit=0,
            )
        )
        if valid:
            ranking.append((index, progress))
        else:
            rows[-1]["verification_status"] = "invalid_screen_trajectory"
    ranking.sort(key=lambda item: (-item[1], item[0]))
    selected = ranking[:shortlist]
    if selected:
        guarded, guard_seconds = evaluate_batch(
            model, initial, [prepared[i] for i, _ in selected], scene, rates=rates, workers=workers
        )
    else:
        guarded, guard_seconds = [], 0.0
    for (index, _), result in zip(selected, guarded, strict=True):
        rows[index].update(result[0], verification_status="guarded_forecast")
    admitted = [r for r in rows if r["eligible"]]
    best = max(admitted, key=lambda r: r["progress_minus_x_m"]) if admitted else None
    return dict(
        kind="screen_then_guard_forecast_zero_task_credit",
        candidates=rows,
        selected_name=None if best is None else best["name"],
        screen_seconds=screen_seconds,
        guard_seconds=guard_seconds,
        total_seconds=screen_seconds + guard_seconds,
        shortlist_limit=shortlist,
        screened_count=len(prepared),
        guarded_count=len(guarded),
        fresh_task_credit=0,
        live_execution_admitted=False,
    )
