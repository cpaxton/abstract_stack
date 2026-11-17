"""Explicit two-carton corner contact contracts for experimental row compaction.

This module grants no general row collision permission. Callers retain existing
IK, force, episode and final-fit checks. Forecasts use scratch data only.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.finger_load_control import FingerLoadCompensator
from abstract_stack.interactions import InteractionFailure


@dataclass(frozen=True)
class AdjacentContactPlan:
    active: str
    neighbor: str
    direction: int
    finger: str = "Middle"

    def __post_init__(self):
        if self.active == self.neighbor or self.direction not in (-1, 1):
            raise ValueError("A contact plan needs one distinct directional neighbor")
        if self.finger not in ("Index", "Middle", "Ring"):
            raise ValueError("Contact plan requires a supported native finger")

    @property
    def cartons(self):
        return (self.active, self.neighbor)

    @property
    def distal_body(self):
        return f"left_{self.finger.lower()}_ip"


def plan_adjacent_contact(env, active_name, direction, *, finger="Middle"):
    """Name the next physically shelf-supported, X/Z-compatible measured neighbor.

    The active identity stays explicit. Neither initial-stock labels nor target
    Y positions substitute for measured row geometry and actual board support.
    """
    if direction not in (-1, 1):
        raise ValueError("Contact direction must be -1 or +1")
    active = next((p for p in env.products if p.name == active_name), None)
    if active is None:
        raise ValueError("Active carton is not in this scene")
    # Task-level labels do not identify occupancy: an unstocked table carton
    # can have the same target level and a Y between two real shelf cartons.
    # Admit neighbors only from measured overlapping X/Z envelopes with physical
    # contact on the expected shelf board, close to that board's actual height.
    active_body = env.data.body(active.name)
    active_extent = np.abs(active_body.xmat.reshape(3, 3)) @ np.asarray(active.half_size)
    row = [active]
    for product in env.products:
        if product.name == active.name or product.level != active.level:
            continue
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        overlap = np.minimum(active_body.xpos + active_extent, body.xpos + extent) - np.maximum(
            active_body.xpos - active_extent, body.xpos - extent
        )
        gap = body.xpos[2] - extent[2] - env.config.levels[product.level]
        if np.any(overlap[[0, 2]] <= 0) or not -0.003 <= gap <= 0.003:
            continue
        board = env.model.geom(f"shelf_board_{product.level}").id
        support = 0.0
        for index, contact in enumerate(env.data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = map(int, env.model.geom_bodyid[[g1, g2]])
            if (b1 == body.id and g2 == board) or (b2 == body.id and g1 == board):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, index, wrench)
                support += max(0.0, float(wrench[0]))
        if support > 0.2:
            row.append(product)
    row.sort(key=lambda p: float(env.data.body(p.name).xpos[1]))
    index = row.index(active) + direction
    if not 0 <= index < len(row):
        raise ValueError("No adjacent carton in the requested direction")
    return AdjacentContactPlan(active_name, row[index].name, direction, finger)


def _validate_plan(env, plan):
    try:
        current = plan_adjacent_contact(env, plan.active, plan.direction, finger=plan.finger)
    except ValueError as error:
        raise InteractionFailure(
            "contact_set: measured next-neighbor identity changed or row support lost"
        ) from error
    if current != plan:
        raise InteractionFailure("contact_set: measured next-neighbor identity changed")


def assess_neighbor_contact(plan, *, body, distance, local_position, half_size, force_direction):
    """Conservative front/side lower-corner window, independently of force caps.

    The 0.5 mm forecast ceiling is stricter than existing 2 mm IK rejection.
    Force direction is the normal ON the neighbor, in its carton frame.
    """
    if body != plan.distal_body:
        return "neighbor contact is not on the selected distal finger"
    if not np.isfinite(distance) or distance < -0.0005:
        return "neighbor penetration exceeds 0.5 mm forecast ceiling"
    position, half, normal = map(np.asarray, (local_position, half_size, force_direction))
    if not all(a.shape == (3,) and np.isfinite(a).all() for a in (position, half, normal)):
        return "nonfinite neighbor contact geometry"
    if (
        abs(position[0] - half[0]) > 0.003
        or abs(position[1] + plan.direction * half[1]) > 0.003
        or not 0.010 <= position[2] + half[2] <= 0.045
    ):
        return "neighbor contact is outside the declared lower front-side corner"
    if normal[0] > -0.25 or plan.direction * normal[1] < 0.15 or abs(normal[2]) > 0.3:
        return "neighbor contact normal does not push inward and toward row side"
    return None


def check_coupled_contact_set(hand, plan, prediction_q, nominal_targets, applied_targets):
    """Check arm/finger approach, loaded posture and unloading on a fixed grid.

    Native contacts stay enabled. Existing collision_errors is always evaluated
    with its original single active-object permission. No error is filtered out.
    A named neighbor must ALSO pass the stricter corner/penetration contract.
    """
    _validate_plan(hand.env, plan)
    ik, model = hand.ik, hand.model
    q = np.asarray(prediction_q, float)
    if q.shape != hand.command.shape or not np.isfinite(q).all():
        raise ValueError("Contact forecast requires finite native arm references")
    if np.any(q < ik.lower) or np.any(q > ik.upper):
        raise ValueError("Contact forecast arm reference exceeds native bounds")
    if set(nominal_targets) != set(applied_targets):
        raise ValueError("Nominal and applied finger references must name identical joints")
    addresses = {}
    for name in nominal_targets:
        joint, actuator = model.joint(name), model.actuator(name)
        lo = max(float(joint.range[0]), float(actuator.ctrlrange[0]))
        hi = min(float(joint.range[1]), float(actuator.ctrlrange[1]))
        values = [nominal_targets[name], applied_targets[name]]
        if not np.isfinite(values).all() or any(not lo <= value <= hi for value in values):
            raise ValueError("Contact forecast finger reference exceeds native bounds")
        addresses[name] = int(joint.qposadr[0])
    start, reference = ik.measured_q(), hand.data.qpos.copy()
    active_id, neighbor_id = (model.body(name).id for name in plan.cartons)
    product = next(p for p in hand.env.products if p.name == plan.neighbor)
    baseline = ik._contacts(start, reference, active_id)
    existing, violations, worst = {}, {}, None
    samples = []
    fractions = np.linspace(0, 1, 9)
    for arm_fraction in fractions:
        arm = start + arm_fraction * (q - start)
        for mode in ("nominal", "applied", "unloading"):
            # The complete nominal<->applied sweep includes zero-load release;
            # sample it at every arm fraction, rather than only the endpoint.
            finger_fractions = fractions if mode == "unloading" else (arm_fraction,)
            for finger_fraction in finger_fractions:
                fingers = reference.copy()
                for name, adr in addresses.items():
                    if mode == "unloading":
                        low, high = nominal_targets[name], applied_targets[name]
                    else:
                        low = reference[adr]
                        high = (nominal_targets if mode == "nominal" else applied_targets)[name]
                    fingers[adr] = low + finger_fraction * (high - low)
                errors = ik.collision_errors(arm, fingers, baseline, active_id)
                for error in errors:
                    key = tuple(sorted(error["geoms"]))
                    if key not in existing or error["distance_m"] < existing[key]["distance_m"]:
                        existing[key] = error
                for contact in ik.scratch.contact:
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    b1, b2 = map(int, model.geom_bodyid[[g1, g2]])
                    if neighbor_id not in (b1, b2):
                        continue
                    other = b2 if b1 == neighbor_id else b1
                    if other not in ik.robot or contact.dist > 0:
                        continue
                    rotation = ik.scratch.xmat[neighbor_id].reshape(3, 3)
                    local = rotation.T @ (contact.pos - ik.scratch.xpos[neighbor_id])
                    normal = rotation.T @ contact.frame[:3] * (-1 if b1 == neighbor_id else 1)
                    body = model.body(other).name
                    reason = assess_neighbor_contact(
                        plan,
                        body=body,
                        distance=contact.dist,
                        local_position=local,
                        half_size=product.half_size,
                        force_direction=normal,
                    )
                    sample = dict(
                        arm_fraction=float(arm_fraction),
                        mode=mode,
                        finger_fraction=float(finger_fraction),
                        body=body,
                        distance_m=float(contact.dist),
                        local_position_m=local.tolist(),
                        normal_on_carton_local=normal.tolist(),
                    )
                    if worst is None or contact.dist < worst["distance_m"]:
                        worst = sample
                    if reason:
                        violations.setdefault(reason, sample)
                samples.append((float(arm_fraction), mode, float(finger_fraction)))
    return dict(
        accepted=not existing and not violations,
        plan=dict(
            active=plan.active,
            neighbor=plan.neighbor,
            direction=plan.direction,
            distal_body=plan.distal_body,
        ),
        samples=len(samples),
        maximum_neighbor_penetration_m=0.0005,
        worst_neighbor_contact=worst,
        existing_collision_errors=list(existing.values()),
        contact_set_violations=[dict(reason=k, **v) for k, v in violations.items()],
        scope="sampled scratch geometry, not a force or dynamics certificate",
    )


class NamedContactFingerLoadCompensator(FingerLoadCompensator):
    """Aggregate native holding effort over exactly the two declared cartons."""

    def __init__(self, model, data, joint_names, *, plan, **kwargs):
        super().__init__(model, data, joint_names, product_name=plan.active, **kwargs)
        self.plan = plan
        self.products = {model.body(name).id: name for name in plan.cartons}

    def targets(self, nominal, *, dt):
        # Reuse original native/rate validation without changing its state.
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Finger compensation dt must be finite and positive")
        if any(name not in nominal or not np.isfinite(nominal[name]) for name in self.names):
            raise ValueError("Finger compensation requires finite nominal targets")
        if any(not self.limits[n][0] <= nominal[n] <= self.limits[n][1] for n in self.names):
            raise ValueError("Nominal finger reference exceeds native position bounds")
        effort = {name: dict.fromkeys(self.plan.cartons, 0.0) for name in self.names}
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            b1, b2 = map(int, self.model.geom_bodyid[[contact.geom1, contact.geom2]])
            for name in self.names:
                if b1 in self.descendants[name] and b2 in self.products:
                    carton, sign = b2, -1
                elif b2 in self.descendants[name] and b1 in self.products:
                    carton, sign = b1, 1
                else:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                jid = self.joints[name]
                rotation = contact.frame.reshape(3, 3).T
                force = sign * rotation @ wrench[:3]
                moment = sign * rotation @ wrench[3:]
                moment += np.cross(contact.pos - self.data.xanchor[jid], force)
                effort[name][self.products[carton]] += float(moment @ self.data.xaxis[jid])
        result, report = dict(nominal), {}
        for name in self.names:
            jid = self.joints[name]
            contact_effort = sum(effort[name].values())
            requested = float(self.data.qfrc_bias[self.model.jnt_dofadr[jid]]) - contact_effort
            desired = requested / self.gains[name]
            bounded = np.clip(desired, -self.offset_limit, self.offset_limit)
            offset = self.offsets[name] + np.clip(
                bounded - self.offsets[name], -self.rate_limit * dt, self.rate_limit * dt
            )
            applied = float(np.clip(nominal[name] + offset, *self.limits[name]))
            self.offsets[name] = applied - nominal[name]
            clipped = abs(desired - self.offsets[name]) > 1e-9
            self.clipped_updates += clipped
            self.peak_offset = max(self.peak_offset, abs(self.offsets[name]))
            result[name] = applied
            report[name] = dict(
                nominal_reference_rad=float(nominal[name]),
                applied_reference_rad=applied,
                measured_angle_rad=float(self.data.qpos[self.model.jnt_qposadr[jid]]),
                measured_contact_effort_Nm=contact_effort,
                contact_effort_by_carton_Nm=effort[name],
                requested_holding_effort_Nm=requested,
                applied_offset_rad=self.offsets[name],
                clipped=bool(clipped),
            )
        self.last_report = report
        return result


class ContactSetMonitor:
    """Additional per-physics-step guards; existing episode guards still run.

    This monitor does not reset observers or modify controls. It must be called
    once at every physics step, including unloading and retreat.
    """

    def __init__(self, env, plan):
        _validate_plan(env, plan)
        self.plan = plan
        model = env.model
        self.products = {model.body(p.name).id: p for p in env.products}
        root = model.body("mabel_chassis").id
        self.robot = set()
        for body in range(model.nbody):
            parent = body
            while parent:
                if parent == root:
                    self.robot.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.support_loss = dict.fromkeys(plan.cartons, 0.0)
        self.initial_positions = {p.name: env.data.body(p.name).xpos.copy() for p in env.products}
        self.last_time = float(env.data.time)
        self.peak_robot_normals = dict.fromkeys((p.name for p in env.products), 0.0)
        self.peak_pairs, self.maximum_tilt_deg, self.last_sample = {}, {}, {}

    def step(self, env):
        elapsed = float(env.data.time) - self.last_time
        if not 0 < elapsed <= env.config.timestep * 1.01:
            raise InteractionFailure("contact_set: monitor missed a physics step")
        self.last_time = float(env.data.time)
        _validate_plan(env, self.plan)
        normals = dict.fromkeys((p.name for p in env.products), 0.0)
        support = normals.copy()
        pairs, contact_violations = {}, []
        model, data = env.model, env.data
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = map(int, model.geom_bodyid[[g1, g2]])
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal = max(0.0, float(wrench[0]))
            if b1 in self.products and b2 in self.products:
                pair = tuple(sorted((self.products[b1].name, self.products[b2].name)))
                pairs[pair] = pairs.get(pair, 0.0) + normal
            for carton, other, other_geom in ((b1, b2, g2), (b2, b1, g1)):
                if carton not in self.products:
                    continue
                product = self.products[carton]
                if other in self.robot:
                    normals[product.name] += normal
                if model.body(other).name == "supply" or (
                    model.geom(other_geom).name or ""
                ).startswith("shelf_board_"):
                    support[product.name] += normal
                if product.name != self.plan.neighbor or other not in self.robot or normal <= 0.01:
                    continue
                rotation = data.xmat[carton].reshape(3, 3)
                reason = assess_neighbor_contact(
                    self.plan,
                    body=model.body(other).name,
                    distance=contact.dist,
                    local_position=rotation.T @ (contact.pos - data.xpos[carton]),
                    half_size=product.half_size,
                    force_direction=rotation.T @ contact.frame[:3] * (-1 if carton == b1 else 1),
                )
                if reason:
                    contact_violations.append(reason)
        for name, normal in normals.items():
            self.peak_robot_normals[name] = max(self.peak_robot_normals[name], normal)
        for pair, normal in pairs.items():
            key = "/".join(pair)
            self.peak_pairs[key] = max(self.peak_pairs.get(key, 0), normal)
        self.last_sample = dict(
            sim_time=float(data.time),
            robot_normal_N=normals,
            support_N=support,
            carton_pair_normal_N={"/".join(k): v for k, v in pairs.items()},
        )
        if max([*normals.values(), *pairs.values()], default=0) > 25.0:
            raise InteractionFailure("contact_set: per-carton or summed-pair force exceeded 25 N")
        if contact_violations:
            raise InteractionFailure("contact_set: " + contact_violations[0])
        for bid, product in self.products.items():
            rotation = data.xmat[bid].reshape(3, 3)
            tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))
            self.maximum_tilt_deg[product.name] = max(
                self.maximum_tilt_deg.get(product.name, 0), tilt
            )
            if (
                tilt > (15.0 if product.name in self.plan.cartons else 45.0)
                or data.xpos[bid, 2] < 0.25
            ):
                raise InteractionFailure("contact_set: carton tilt/drop guard")
            if product.name in self.plan.cartons:
                extent = np.abs(rotation) @ np.asarray(product.half_size)
                gap = data.xpos[bid, 2] - extent[2] - env.config.levels[product.level]
                loaded = support[product.name] > 0.2 and -0.003 <= gap <= 0.025
                self.support_loss[product.name] = (
                    0.0 if loaded else self.support_loss[product.name] + elapsed
                )
                if self.support_loss[product.name] > 0.1:
                    raise InteractionFailure("contact_set: named carton lost physical support")
                if np.linalg.norm(data.xpos[bid] - self.initial_positions[product.name]) > 0.025:
                    raise InteractionFailure("contact_set: named carton exceeded 25 mm travel")
        return self.last_sample

    def report(self):
        return dict(
            plan=dict(
                active=self.plan.active,
                neighbor=self.plan.neighbor,
                direction=self.plan.direction,
                distal_body=self.plan.distal_body,
            ),
            peak_robot_normal_N=self.peak_robot_normals.copy(),
            peak_carton_pair_normal_N=self.peak_pairs.copy(),
            maximum_tilt_deg=self.maximum_tilt_deg.copy(),
            last_sample=self.last_sample,
            scope=(
                "additional guards only; caller retains original guards "
                "and final 0.2 mm row-fit test"
            ),
        )
