"""One coordinated, rate-bounded Cartesian correction from a nominal reference.

Only PoseIK's scratch kinematics are evaluated. The caller owns collision checks,
force guards, actuator compensation and execution. A returned residual is not a
reachability or tracking certificate.
"""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from abstract_stack.ik import rotation_error


@dataclass(frozen=True)
class CartesianFeedbackConfig:
    rotation_weight_m_rad: float = 0.12
    damping_m: float = 0.0001
    posture_rate_rad_s: float = 0.5
    posture_gain_s: float = 4.0
    solver_iterations: int = 96
    solver_tolerance: float = 1e-8
    linearizations: int = 2

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Cartesian correction settings must be finite and positive")
        if (
            not isinstance(self.solver_iterations, int)
            or isinstance(self.solver_iterations, bool)
            or not isinstance(self.linearizations, int)
            or isinstance(self.linearizations, bool)
        ):
            raise ValueError("Cartesian solver iteration counts must be integers")


def box_damped_least_squares(
    matrix, error, lower, upper, *, damping=0.0001, iterations=96, tolerance=1e-8
):
    """Solve min ||A u-b||² + damping²||u||² over a box, using NumPy active sets.

    Small free-coordinate systems are solved exactly; a bound is activated when
    a trial crosses it and released when its KKT sign is wrong. The final
    projected-gradient residual reports convergence rather than assuming it.
    """
    matrix, error = np.asarray(matrix, float), np.asarray(error, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        matrix.ndim != 2
        or matrix.shape[1] < 1
        or error.shape != (matrix.shape[0],)
        or lower.shape != (matrix.shape[1],)
        or upper.shape != lower.shape
        or not all(np.isfinite(a).all() for a in (matrix, error, lower, upper))
        or np.any(lower > upper)
        or not np.isfinite(damping)
        or damping <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Invalid bounded least-squares problem")
    hessian = matrix.T @ matrix + damping**2 * np.eye(matrix.shape[1])
    rhs = matrix.T @ error
    lipschitz = float(np.linalg.eigvalsh(hessian)[-1])
    current = np.clip(np.zeros(matrix.shape[1]), lower, upper)
    fixed = upper - lower <= 1e-14
    active = fixed | (current <= lower + 1e-12) | (current >= upper - 1e-12)
    used = 0
    for _ in range(iterations):
        used += 1
        free = ~active
        destination = current.copy()
        if free.any():
            destination[free] = np.linalg.solve(
                hessian[np.ix_(free, free)],
                rhs[free] - hessian[np.ix_(free, active)] @ current[active],
            )
        direction = destination - current
        fraction = 1.0
        for index in np.flatnonzero(free):
            if direction[index] > 1e-14:
                fraction = min(fraction, (upper[index] - current[index]) / direction[index])
            elif direction[index] < -1e-14:
                fraction = min(fraction, (lower[index] - current[index]) / direction[index])
        current = np.clip(current + max(0.0, fraction) * direction, lower, upper)
        if fraction < 1 - 1e-12:
            active |= (current <= lower + 1e-12) & (direction < 0) | (current >= upper - 1e-12) & (
                direction > 0
            )
            continue
        gradient = hessian @ current - rhs
        violations = np.zeros_like(current)
        at_lower = active & ~fixed & (current <= lower + 1e-12)
        at_upper = active & ~fixed & (current >= upper - 1e-12)
        violations[at_lower] = np.maximum(0, -gradient[at_lower])
        violations[at_upper] = np.maximum(0, gradient[at_upper])
        if np.max(violations, initial=0) <= tolerance * lipschitz:
            break
        active[int(np.argmax(violations))] = False
    projected = current - np.clip(current - (hessian @ current - rhs) / lipschitz, lower, upper)
    return current, dict(
        method="numpy_active_set_box_dls",
        iterations=used,
        projected_gradient_inf=float(np.max(np.abs(projected), initial=0)),
        linear_cost=float(
            np.linalg.norm(matrix @ current - error) ** 2 + damping**2 * np.dot(current, current)
        ),
    )


@dataclass
class CartesianStep:
    q: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    previous_position: np.ndarray
    requested_position: np.ndarray
    active_rate_limits: list[str]
    active_joint_limits: list[str]
    ramped_postures: dict
    solver: dict

    def report(self):
        return dict(
            predicted_position_m=self.position.tolist(),
            requested_position_m=self.requested_position.tolist(),
            previous_nominal_position_m=self.previous_position.tolist(),
            position_error_m=self.position_error_m,
            position_error_xyz_m=(self.position - self.requested_position).tolist(),
            orientation_error_deg=float(np.degrees(self.orientation_error_rad)),
            active_rate_limits=self.active_rate_limits,
            active_joint_limits=self.active_joint_limits,
            ramped_postures=self.ramped_postures,
            solver=self.solver,
            scope="bounded nominal scratch FK; collisions and dynamics are caller responsibilities",
        )


def bounded_cartesian_step(
    ik, previous_q, target, rotation, *, dt, rates=None, reference=None, config=None
):
    """Coordinate all arm/wrist/lift increments inside one fixed native/rate box.

    ``previous_q`` is the caller's previous NOMINAL command, never compensated
    actuator controls or measured sag. Existing PoseIK posture goals are ramped
    within this box, without editing those goals. Multiple linearizations share
    the same original rate box; they cannot accumulate extra motion budget.
    """
    config = config or CartesianFeedbackConfig()
    previous = np.asarray(previous_q, float).copy()
    target, rotation = np.asarray(target, float), np.asarray(rotation, float)
    size = len(ik.names)
    rates = np.r_[np.ones(size - 1), 0.08] if rates is None else np.asarray(rates, float)
    if (
        previous.shape != (size,)
        or target.shape != (3,)
        or rotation.shape != (3, 3)
        or rates.shape != (size,)
        or not np.isfinite(dt)
        or dt <= 0
        or not all(np.isfinite(a).all() for a in (previous, target, rotation, rates))
        or np.any(rates <= 0)
        or np.any(previous < ik.lower - 1e-10)
        or np.any(previous > ik.upper + 1e-10)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
    ):
        raise ValueError("Invalid nominal Cartesian step or native/rate bounds")
    reference = ik.data.qpos.copy() if reference is None else np.asarray(reference, float).copy()
    if reference.shape != ik.data.qpos.shape or not np.isfinite(reference).all():
        raise ValueError("Invalid reference qpos snapshot")
    scales = rates * dt
    lower = np.maximum(-np.ones(size), (ik.lower - previous) / scales)
    upper = np.minimum(np.ones(size), (ik.upper - previous) / scales)
    postures = {}
    for index, goal in ik.posture_constraints.items():
        increment = float(
            np.clip(
                config.posture_gain_s * (goal - previous[index]) * dt,
                -config.posture_rate_rad_s * dt,
                config.posture_rate_rad_s * dt,
            )
        )
        fraction = float(np.clip(increment / scales[index], lower[index], upper[index]))
        lower[index] = upper[index] = fraction
        postures[ik.names[index]] = dict(
            requested=float(goal),
            previous=float(previous[index]),
            ramped=float(previous[index] + fraction * scales[index]),
        )
    previous_position, _ = ik.forward(previous, reference)
    current = previous.copy()
    best, best_cost, solver = None, float("inf"), {}
    for _ in range(config.linearizations):
        point, actual_rotation = ik.forward(current, reference)
        weight = config.rotation_weight_m_rad
        error = np.r_[target - point, weight * rotation_error(actual_rotation, rotation)]
        mujoco.mj_jac(ik.model, ik.scratch, ik.jacp, ik.jacr, point, ik.palm)
        lift_dofs = [int(j.dofadr[0]) for j in ik.lifts]
        jacobian = np.column_stack(
            (
                np.vstack((ik.jacp[:, ik.dofs], weight * ik.jacr[:, ik.dofs])),
                np.r_[ik.jacp[:, lift_dofs].mean(axis=1), np.zeros(3)],
            )
        )
        # Optimize TOTAL displacement from the original nominal reference,
        # even while relinearizing about a candidate within the fixed box.
        rhs = error + jacobian @ (current - previous)
        fraction, info = box_damped_least_squares(
            jacobian * scales,
            rhs,
            lower,
            upper,
            damping=config.damping_m,
            iterations=config.solver_iterations,
            tolerance=config.solver_tolerance,
        )
        current = np.clip(previous + fraction * scales, ik.lower, ik.upper)
        new_point, new_rotation = ik.forward(current, reference)
        cost = float(
            np.linalg.norm(new_point - target) ** 2
            + weight**2 * np.linalg.norm(rotation_error(new_rotation, rotation)) ** 2
        )
        if cost < best_cost:
            best, best_cost, solver = current.copy(), cost, info
    point, actual_rotation = ik.forward(best, reference)
    step = best - previous
    return CartesianStep(
        q=best,
        position=point,
        rotation=actual_rotation,
        position_error_m=float(np.linalg.norm(point - target)),
        orientation_error_rad=float(np.linalg.norm(rotation_error(actual_rotation, rotation))),
        previous_position=previous_position,
        requested_position=target.copy(),
        active_rate_limits=[
            name for i, name in enumerate(ik.names) if abs(abs(step[i]) - scales[i]) <= 1e-8
        ],
        active_joint_limits=[
            name
            for i, name in enumerate(ik.names)
            if min(best[i] - ik.lower[i], ik.upper[i] - best[i]) <= 1e-8
        ],
        ramped_postures=postures,
        solver=solver,
    )


def bounded_contact_prediction(
    hand,
    proposal,
    target,
    dt,
    rates,
    *,
    config=None,
    reference_envelope=None,
    max_translation_step_m=None,
    max_rotation_step_rad=None,
):
    """Replace a fresh contact proposal with a checked bounded nominal step.

    Existing ``hand.predict`` must initialize matching carry/slide context first.
    The proposal's contact permission, tolerances and reference snapshot remain
    authoritative. This function changes scratch data only; logging, command
    application and projection anti-windup belong to the caller.
    """
    ik = hand.ik
    if proposal.baseline_qpos is None or not np.array_equal(proposal.baseline_qpos, hand.data.qpos):
        raise ValueError("Bounded contact prediction requires a fresh baseline")
    if proposal.reference_ranges != ik.reference_range_report():
        raise ValueError("Bounded contact prediction has stale reference-range constraints")
    for context, expected in (
        (ik.carry, proposal.carried_object),
        (ik.slide, proposal.sliding_object),
    ):
        actual = None if context is None else hand.model.body(context[0]).name
        if actual != expected:
            raise ValueError("Bounded contact prediction has stale carry/slide context")
    previous = hand.command.copy()
    rates = np.asarray(rates, dtype=float)
    step = bounded_cartesian_step(
        ik,
        previous,
        target,
        proposal.requested_rotation,
        dt=dt,
        rates=rates,
        reference=proposal.reference_qpos,
        config=config,
    )
    motion_projection = None
    if max_translation_step_m is not None or max_rotation_step_rad is not None:
        if (
            max_translation_step_m is None
            or max_rotation_step_rad is None
            or not np.isfinite([max_translation_step_m, max_rotation_step_rad]).all()
            or min(max_translation_step_m, max_rotation_step_rad) <= 0
        ):
            raise ValueError(
                "Cartesian motion projection needs positive translation/rotation bounds"
            )
        start_point, start_rotation = ik.forward(previous, proposal.reference_qpos)
        for power in range(17):
            fraction = 2.0**-power if power < 16 else 0.0
            q = previous + fraction * (step.q - previous)
            position, rotation = ik.forward(q, proposal.reference_qpos)
            linear = float(np.linalg.norm(position - start_point))
            angular = float(np.linalg.norm(rotation_error(start_rotation, rotation)))
            if linear <= max_translation_step_m and angular <= max_rotation_step_rad:
                break
        step = replace(
            step,
            q=q,
            position=position,
            rotation=rotation,
            position_error_m=float(np.linalg.norm(position - target)),
            orientation_error_rad=float(
                np.linalg.norm(rotation_error(rotation, proposal.requested_rotation))
            ),
            ramped_postures={
                name: dict(value, ramped=float(q[ik.names.index(name)]))
                for name, value in step.ramped_postures.items()
            },
            active_rate_limits=[
                name
                for i, name in enumerate(ik.names)
                if abs(abs(q[i] - previous[i]) - rates[i] * dt) <= 1e-8
            ],
            active_joint_limits=[
                name
                for i, name in enumerate(ik.names)
                if min(q[i] - ik.lower[i], ik.upper[i] - q[i]) <= 1e-8
            ],
        )
        motion_projection = dict(
            fraction=fraction,
            translation_step_m=linear,
            rotation_step_rad=angular,
            max_translation_step_m=max_translation_step_m,
            max_rotation_step_rad=max_rotation_step_rad,
        )
    projection = None
    if reference_envelope is not None:
        q, projection = reference_envelope.project(
            ik,
            previous,
            step.q,
            rates=rates,
            dt=dt,
            reference=proposal.reference_qpos,
            postures=step.ramped_postures,
        )
        position, rotation = ik.forward(q, proposal.reference_qpos)
        step = replace(
            step,
            q=q,
            position=position,
            rotation=rotation,
            position_error_m=float(np.linalg.norm(position - target)),
            orientation_error_rad=float(
                np.linalg.norm(rotation_error(rotation, proposal.requested_rotation))
            ),
            active_rate_limits=[
                name
                for i, name in enumerate(ik.names)
                if abs(abs(q[i] - previous[i]) - rates[i] * dt) <= 1e-8
            ],
            active_joint_limits=[
                name
                for i, name in enumerate(ik.names)
                if min(q[i] - ik.lower[i], ik.upper[i] - q[i]) <= 1e-8
            ],
        )
    object_name = proposal.allowed_contact_object
    object_id = None if object_name is None else hand.model.body(object_name).id
    start = ik.measured_q()
    baseline = ik._contacts(start, proposal.baseline_qpos, object_id)
    collisions = ik.collision_errors(step.q, proposal.reference_qpos, baseline, object_id)
    path_collisions = []
    for fraction in (0.0, 0.5, 1.0):
        path_collisions = ik.collision_errors(
            start + fraction * (step.q - start), proposal.reference_qpos, baseline, object_id
        )
        if path_collisions:
            break
    diagnostics = {
        **step.report(),
        "reference_mode": "rate_bounded_nominal_step_with_projection_antiwindup",
        "control_dt_seconds": float(dt),
        "global_proposal_position_error_m": proposal.position_error_m,
        "allowed_contact_object": object_name,
        "paired_reference_projection": projection,
        "cartesian_motion_projection": motion_projection,
    }
    motion_admitted = True
    if motion_projection is not None:
        linear = float(np.linalg.norm(step.position - start_point))
        angular = float(np.linalg.norm(rotation_error(start_rotation, step.rotation)))
        motion_admitted = linear <= max_translation_step_m and angular <= max_rotation_step_rad
        motion_projection.update(
            final_translation_step_m=linear,
            final_rotation_step_rad=angular,
            admitted=bool(motion_admitted),
        )
    feasible = (
        step.position_error_m <= proposal.position_tolerance_m
        and step.orientation_error_rad <= np.radians(proposal.angle_tolerance_degrees)
        and not collisions
        and not path_collisions
        and diagnostics["solver"]["projected_gradient_inf"] <= 1e-6
        and (projection is None or projection["feasible"])
        and motion_admitted
    )
    return replace(
        proposal,
        q=step.q,
        position=step.position,
        rotation=step.rotation,
        position_error_m=step.position_error_m,
        orientation_error_rad=step.orientation_error_rad,
        preferred_orientation_error_rad=float(
            np.linalg.norm(rotation_error(step.rotation, hand.active_rotation))
        ),
        collisions=collisions,
        path_collisions=path_collisions,
        path_ik_error=None,
        blocked_by=[],
        feasible=feasible,
        candidate="bounded_nominal_cartesian_step",
        joint_path=None,
        initial_cartesian_transition=None,
        path_checked=True,
        path_mode="joint",
        minimum_seconds=float(np.max(np.abs(step.q - previous) / rates)),
        active_limits=[
            name
            for name, q, lower, upper in zip(ik.names, step.q, ik.lower, ik.upper, strict=True)
            if min(q - lower, upper - q) < 0.002
        ],
        posture_constraints={name: value["ramped"] for name, value in step.ramped_postures.items()},
        bounded_feedback=diagnostics,
    )
