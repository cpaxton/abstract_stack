"""Bounded native selected-finger load compensation; references stay separate."""

import numpy as np

from abstract_stack.withdrawal_feedback import contact_effort_on_joint


class FingerLoadCompensation:
    """Refresh gravity/contact holding torque through existing position servos.

    No native gains, inertias, ranges or forces are changed. Nominal preshapes
    remain physical planning references; the caller must admit the final applied
    arm/finger targets together before executing any native step.
    """

    def __init__(self, hand):
        self.hand = hand
        self.joints = {}
        model = hand.model
        suffixes = ("Abd", "MCP", "PIP") if getattr(hand, "fingers", ()) else ("MCP", "PIP")
        names = [
            f"Left {finger} {suffix}"
            for finger in getattr(hand, "fingers", (hand.finger,))
            for suffix in suffixes
        ]
        for name in names:
            joint, actuator = model.joint(name), model.actuator(name)
            kp = float(actuator.gainprm[0])
            if (
                kp <= 0
                or not np.isclose(actuator.biasprm[1], -kp)
                or not np.isclose(actuator.gear[0], 1)
            ):
                raise ValueError("Finger compensation requires native position servos")
            root = int(model.jnt_bodyid[joint.id])
            descendants = {root}
            for body in range(root + 1, model.nbody):
                if int(model.body_parentid[body]) in descendants:
                    descendants.add(body)
            self.joints[name] = dict(
                joint=joint.id,
                actuator=actuator.id,
                kp=kp,
                descendants=descendants,
                previous=float(hand.data.ctrl[actuator.id]),
                lower=max(float(joint.range[0]), float(actuator.ctrlrange[0])),
                upper=min(float(joint.range[1]), float(actuator.ctrlrange[1])),
            )

    def propose(self, targets, product):
        hand = self.hand
        model, data = hand.model, hand.data
        applied = np.array(targets, copy=True)
        nominal = hand.finger_targets(0)
        product_id = model.body(product).id
        report = dict(
            maximum_additional_effort_Nm=0.15,
            maximum_feedforward_offset_rad=0.15,
            maximum_servo_rate_rad_s=0.5,
            joints={},
        )
        for name, state in self.joints.items():
            joint, aid = state["joint"], state["actuator"]
            contact = contact_effort_on_joint(model, data, joint, state["descendants"], product_id)
            gravity = float(data.qfrc_bias[int(model.jnt_dofadr[joint])])
            effort = gravity - contact
            bounded_effort = float(np.clip(effort, -0.15, 0.15))
            offset = float(np.clip(bounded_effort / state["kp"], -0.15, 0.15))
            wanted = float(np.clip(nominal[name] + offset, state["lower"], state["upper"]))
            value = float(
                state["previous"]
                + np.clip(wanted - state["previous"], -0.5 * hand.env.dt, 0.5 * hand.env.dt)
            )
            applied[aid] = value
            report["joints"][name] = dict(
                nominal_rad=nominal[name],
                applied_servo_rad=value,
                measured_rad=float(data.qpos[int(model.jnt_qposadr[joint])]),
                contact_reaction_Nm=contact,
                bias_Nm=gravity,
                requested_holding_Nm=effort,
                bounded_holding_Nm=bounded_effort,
                requested_feedforward_offset_rad=offset,
                applied_minus_preshape_rad=value - nominal[name],
                effort_limited=abs(effort - bounded_effort) > 1e-9,
                reference_limited=abs(nominal[name] + offset - wanted) > 1e-9,
                rate_limited=abs(wanted - value) > 1e-9,
            )
        return applied, report

    def acknowledge(self, targets):
        for state in self.joints.values():
            state["previous"] = float(targets[state["actuator"]])
