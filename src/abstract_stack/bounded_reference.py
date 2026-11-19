"""Scratch-only projection of nominal and compensated joint reference increments."""

from dataclasses import dataclass

import mujoco
import numpy as np


def project_halfspaces(desired, lower, upper, matrix, bounds, *, iterations=128):
    """Project onto a box and linear inequalities with bounded Dykstra search.

    Callers must check convergence and their exact nonlinear geometry afterward.
    """
    desired, lower, upper, matrix, bounds = (
        np.asarray(a, float) for a in (desired, lower, upper, matrix, bounds)
    )
    if (
        desired.ndim != 1
        or lower.shape != desired.shape
        or upper.shape != desired.shape
        or bounds.ndim != 1
        or matrix.shape != (len(bounds), len(desired))
        or not all(np.isfinite(a).all() for a in (desired, lower, upper, matrix, bounds))
        or np.any(lower > upper)
        or type(iterations) is not int
        or iterations < 1
    ):
        raise ValueError("Invalid box/halfspace projection")
    current = desired.copy()
    corrections = np.zeros((len(bounds) + 1, len(desired)))
    used = 0
    for _ in range(iterations):
        used += 1
        previous = current.copy()
        point = current + corrections[0]
        current = np.clip(point, lower, upper)
        corrections[0] = point - current
        for index, (normal, limit) in enumerate(zip(matrix, bounds, strict=True), 1):
            point = current + corrections[index]
            norm = float(normal @ normal)
            distance = max(0.0, float(normal @ point - limit))
            current = point - (distance / norm * normal if norm > 0 else 0)
            corrections[index] = point - current
        violation = max(
            float(np.max(lower - current, initial=0)),
            float(np.max(current - upper, initial=0)),
            float(np.max(matrix @ current - bounds, initial=0)),
        )
        if violation <= 1e-11 and np.max(np.abs(current - previous), initial=0) < 1e-10:
            break
    return current, dict(iterations=used, maximum_violation=violation)


@dataclass(frozen=True)
class ReferenceEnvelope:
    """Existing Cartesian corridors and prior APPLIED actuator reference.

    Both references receive the same joint increment; compensation stays
    separate and constant. Applied references are virtual servo attractors,
    not observations of the robot's physical configuration.
    """

    nominal_lower: np.ndarray
    nominal_upper: np.ndarray
    applied_lower: np.ndarray
    applied_upper: np.ndarray
    previous_applied: np.ndarray

    def __post_init__(self):
        for name in vars(self):
            value = np.array(getattr(self, name), float, copy=True)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError("Reference envelope requires finite vectors")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for prefix in ("nominal", "applied"):
            low, high = getattr(self, prefix + "_lower"), getattr(self, prefix + "_upper")
            if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
                raise ValueError("Reference envelope requires ordered XYZ bounds")

    def project(self, ik, previous, proposed, *, rates, dt, reference, postures):
        """Relinearize both FK maps within one original native/rate box.

        A 1 micrometre inward margin protects against FK curvature at active
        boundaries. It narrows the corridor; acceptance is never relaxed.
        """
        previous, proposed, rates = (np.asarray(a, float) for a in (previous, proposed, rates))
        applied = self.previous_applied
        if (
            previous.shape != proposed.shape
            or previous.shape != applied.shape
            or previous.shape != rates.shape
            or previous.shape != ik.lower.shape
            or not all(np.isfinite(a).all() for a in (previous, proposed, rates))
            or np.any(rates <= 0)
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError("Invalid paired reference projection")
        scales = rates * dt
        lower = np.maximum.reduce(
            [
                -np.ones_like(previous),
                (ik.lower - previous) / scales,
                (ik.original_lower - applied) / scales,
            ]
        )
        upper = np.minimum.reduce(
            [
                np.ones_like(previous),
                (ik.upper - previous) / scales,
                (ik.original_upper - applied) / scales,
            ]
        )
        for index in ik.posture_constraints:
            value = (postures[ik.names[index]]["ramped"] - previous[index]) / scales[index]
            lower[index], upper[index] = max(lower[index], value), min(upper[index], value)
        if np.any(lower > upper):
            return proposed.copy(), dict(feasible=False, failure="native_rate_or_posture_box_empty")
        desired = (proposed - previous) / scales
        fraction = np.clip(desired, lower, upper)
        margin = 1e-6
        lift_dofs = [int(j.dofadr[0]) for j in ik.lifts]
        linearization = 0
        for _ in range(4):
            linearization += 1
            matrices, limits = [], []
            for origin, low, high in (
                (previous, self.nominal_lower, self.nominal_upper),
                (applied, self.applied_lower, self.applied_upper),
            ):
                point, _ = ik.forward(origin + fraction * scales, reference)
                mujoco.mj_jac(ik.model, ik.scratch, ik.jacp, ik.jacr, point, ik.palm)
                jacobian = (
                    np.column_stack((ik.jacp[:, ik.dofs], ik.jacp[:, lift_dofs].mean(axis=1)))
                    * scales
                )
                matrices.extend((jacobian, -jacobian))
                limits.extend(
                    (
                        high - margin - point + jacobian @ fraction,
                        point - low - margin - jacobian @ fraction,
                    )
                )
            fraction, info = project_halfspaces(
                desired, lower, upper, np.vstack(matrices), np.concatenate(limits)
            )
            q = previous + fraction * scales
            nominal_point, _ = ik.forward(q, reference)
            applied_point, _ = ik.forward(applied + fraction * scales, reference)
            feasible = bool(
                info["maximum_violation"] <= 1e-9
                and np.all(fraction >= lower - 1e-10)
                and np.all(fraction <= upper + 1e-10)
                and np.all(nominal_point >= self.nominal_lower)
                and np.all(nominal_point <= self.nominal_upper)
                and np.all(applied_point >= self.applied_lower)
                and np.all(applied_point <= self.applied_upper)
            )
            if feasible:
                break
        return q, dict(
            feasible=feasible,
            failure=None if feasible else "paired_reference_projection_infeasible_or_budget",
            method="relinearized_dykstra_nominal_and_applied_fk",
            linearizations=linearization,
            inward_margin_m=margin,
            projection=info,
            correction_joint_norm=float(np.linalg.norm(q - proposed)),
            nominal_position_m=nominal_point.tolist(),
            applied_virtual_position_m=applied_point.tolist(),
            compensation_joint_offset=(applied - previous).tolist(),
        )
