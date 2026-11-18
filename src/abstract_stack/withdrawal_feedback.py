"""Measured contact-load feedback through the existing native MABEL tracker."""

from dataclasses import dataclass

import mujoco
import numpy as np

from abstract_stack.mabel_controller import MabelController, TrackingConfig


@dataclass(frozen=True)
class WithdrawalFeedbackConfig:
    # Virtual compensated targets need room to remove stale load offsets.
    # Nominal Cartesian speed, corridor, native rates and task gates stay separate.
    servo_speed_m_s: float = 0.02
    contact_effort_limit_Nm: float = 0.3
    wrist_offset_limit_rad: float = 0.5
    yaw_feedback: bool = False
    yaw_lever_gain_m_rad: float = 0.12
    yaw_lever_damping_m_s_rad: float = 0.01
    yaw_moment_admittance_m_Nms: float = 1.0
    unload_force_goal_N: float = 0.03
    unload_dwell_s: float = 0.2
    unload_speed_m_s: float = 0.004
    unload_seconds: float = 5.0
    yaw_moment_stiffness_Nm_rad: float = 1.0
    yaw_moment_damping_Nms_rad: float = 0.1
    straight_retreat_first: bool = True
    yaw_reference_deg: float = 0.0
    paired_plane_feedback: bool = False

    def __post_init__(self):
        if (
            not np.isfinite(list(vars(self).values())).all()
            or not 0 < self.servo_speed_m_s <= 0.025
            or not 0 < self.contact_effort_limit_Nm <= 0.3
            or not 0 < self.wrist_offset_limit_rad <= 0.5
            or type(self.yaw_feedback) is not bool
            or type(self.straight_retreat_first) is not bool
            or type(self.paired_plane_feedback) is not bool
            or abs(self.yaw_reference_deg) > 0.13
            or not 0 < self.yaw_lever_gain_m_rad <= 0.2
            or not 0 < self.yaw_lever_damping_m_s_rad <= 0.05
            or not 0 < self.yaw_moment_admittance_m_Nms <= 2.0
            or not 0 < self.unload_force_goal_N <= 0.2
            or not 0.12 <= self.unload_dwell_s <= 0.5
            or not 0.001 <= self.unload_speed_m_s <= 0.004
            or not 5 <= self.unload_seconds <= 10
            or not 0.25 <= self.yaw_moment_stiffness_Nm_rad <= 2
            or not 0 < self.yaw_moment_damping_Nms_rad <= 0.2
        ):
            raise ValueError("Withdrawal compensation settings exceed bounded development limits")


def contact_effort_on_joint(model, data, joint, descendants, product):
    """Native contact effort ON the distal robot subtree, in joint-axis Nm.

    Include friction and contact moments as well as normals. The result is
    reaction load, so its negative is the actuator's holding feedforward.
    """
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError("Contact torque feedback requires a hinge joint")
    effort = 0.0
    wrench = np.empty(6)
    for index, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if b1 in descendants and b2 == product:
            sign = -1
        elif b2 in descendants and b1 == product:
            sign = 1
        else:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        rotation = contact.frame.reshape(3, 3).T
        force, torque = sign * rotation @ wrench[:3], sign * rotation @ wrench[3:]
        torque += np.cross(contact.pos - data.xanchor[joint], force)
        effort += float(data.xaxis[joint] @ torque)
    return effort


def compensation_translation(ik, entry_nominal, nominal, applied, reference):
    """Virtual-frame translation from admitted bias, anchored at entry posture.

    Changing a force offset changes the virtual attractor's coordinate frame.
    It does not change the physical nominal/measured corridor. Use the same
    entry nominal posture for both FK evaluations, so the translation cannot
    absorb the skill's commanded motion.
    """
    anchor, _ = ik.forward(entry_nominal, reference)
    virtual_anchor, _ = ik.forward(entry_nominal + applied - nominal, reference)
    return virtual_anchor - anchor


class WithdrawalTracker:
    """Propose compensated references; the withdrawal skill owns admission.

    Unknown nominal dimensions, wheels and all fingers retain their native
    controls. The caller must acknowledge the final admitted controls so tracker
    state cannot integrate an unexecuted compensation proposal.
    """

    def __init__(self, hand, nominal, active, config=None):
        self.config = config or WithdrawalFeedbackConfig()
        self.hand = hand
        self.names = (*hand.names, "Lift")
        self.tracker = MabelController(
            hand.model,
            hand.data,
            TrackingConfig(
                joint_speed=hand.config.max_joint_speed,
                wrist_offset_limit=self.config.wrist_offset_limit_rad,
            ),
            nominal_references=dict(zip(self.names, nominal, strict=True)),
        )
        self.joint = hand.model.joint("Left Wrist").id
        self.product = hand.model.body(active).id
        root = int(hand.model.jnt_bodyid[self.joint])
        self.descendants = {root}
        for body in range(root + 1, hand.model.nbody):
            if int(hand.model.body_parentid[body]) in self.descendants:
                self.descendants.add(body)
        self.last_report = {}

    def propose(self, nominal):
        effort = contact_effort_on_joint(
            self.hand.model, self.hand.data, self.joint, self.descendants, self.product
        )
        load = float(
            np.clip(
                -effort, -self.config.contact_effort_limit_Nm, self.config.contact_effort_limit_Nm
            )
        )
        targets = self.tracker.targets(
            dict(zip(self.names, nominal, strict=True)),
            dt=self.hand.env.dt,
            load_feedforward={"Left Wrist": load},
        )
        self.last_report = dict(
            contact_reaction_effort_Nm=effort,
            requested_contact_holding_effort_Nm=-effort,
            bounded_contact_holding_effort_Nm=load,
            contact_effort_limited=bool(abs(load + effort) > 1e-9),
            requested_tracker=self.tracker.last_report,
        )
        fingers = getattr(self.hand, "finger_load_compensation", None)
        if fingers is not None:
            targets, report = fingers.propose(targets, self.hand.model.body(self.product).name)
            self.last_report["selected_finger_tracking"] = report
        return targets

    def contact_center(self):
        points, weights = [], []
        wrench = np.empty(6)
        for index, contact in enumerate(self.hand.data.contact):
            if contact.efc_address < 0:
                continue
            b1, b2 = (int(self.hand.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if not (
                b1 in self.descendants
                and b2 == self.product
                or b2 in self.descendants
                and b1 == self.product
            ):
                continue
            mujoco.mj_contactForce(self.hand.model, self.hand.data, index, wrench)
            if wrench[0] > 0:
                points.append(contact.pos.copy())
                weights.append(float(wrench[0]))
        return None if not weights else np.average(points, axis=0, weights=weights)

    def contact_wrench_on_product(self):
        """World wrench ON carton about its measured origin; force then moment."""
        result = np.zeros(6)
        wrench = np.empty(6)
        model, data = self.hand.model, self.hand.data
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            b1, b2 = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            if b1 == self.product and b2 in self.descendants:
                sign = -1
            elif b2 == self.product and b1 in self.descendants:
                sign = 1
            else:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            rotation = contact.frame.reshape(3, 3).T
            force = sign * rotation @ wrench[:3]
            result[:3] += force
            result[3:] += sign * rotation @ wrench[3:] + np.cross(
                contact.pos - data.xpos[self.product], force
            )
        return result

    def acknowledge(self, nominal, applied):
        # MabelController orders the known subset by its own canonical names.
        lookup = dict(zip(self.names, nominal, strict=True))
        self.tracker.nominal = np.array([lookup[n] for n in self.tracker.NAMES])
        self.tracker.applied = np.array(applied[self.tracker.ids], copy=True)
        fingers = getattr(self.hand, "finger_load_compensation", None)
        if fingers is not None:
            fingers.acknowledge(applied)
        self.last_report["accepted_nominal"] = lookup
        self.last_report["accepted_servo_targets"] = dict(
            zip(self.tracker.NAMES, self.tracker.applied.tolist(), strict=True)
        )
