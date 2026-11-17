"""Bounded native position-reference compensation for measured finger loads.

Callers retain nominal targets for planning and MUST collision-check coupled
arm/finger motion to the returned applied targets before commanding them.
"""

import mujoco
import numpy as np


class FingerLoadCompensator:
    def __init__(self, model, data, joint_names, *, product_name, offset_limit=0.5, rate_limit=1.0):
        if (
            not np.isfinite(offset_limit)
            or not 0 < offset_limit <= 0.5
            or not np.isfinite(rate_limit)
            or not 0 < rate_limit <= 2
        ):
            raise ValueError("Finger reference compensation exceeds finite bounds")
        self.model, self.data = model, data
        self.names = tuple(joint_names)
        if not self.names:
            raise ValueError("Finger load compensation requires named joints")
        self.product = model.body(product_name).id
        self.offset_limit, self.rate_limit = offset_limit, rate_limit
        self.offsets = dict.fromkeys(self.names, 0.0)
        self.joints, self.descendants, self.gains, self.limits = {}, {}, {}, {}
        self.last_report, self.peak_offset, self.clipped_updates = {}, 0.0, 0
        for name in self.names:
            joint, actuator = model.joint(name), model.actuator(name)
            gain = float(actuator.gainprm[0])
            if (
                gain <= 0
                or not np.isclose(actuator.biasprm[1], -gain)
                or not np.isclose(actuator.gear[0], 1)
                or int(actuator.trnid[0]) != joint.id
            ):
                raise ValueError("Finger compensation requires native scalar position servos")
            if not joint.limited or not actuator.ctrllimited:
                raise ValueError("Finger compensation requires explicit native position bounds")
            self.limits[name] = (
                max(float(joint.range[0]), float(actuator.ctrlrange[0])),
                min(float(joint.range[1]), float(actuator.ctrlrange[1])),
            )
            self.joints[name], self.gains[name] = joint.id, gain
            root = int(model.jnt_bodyid[joint.id])
            descendants = set()
            for body in range(model.nbody):
                ancestor = body
                while ancestor:
                    if ancestor == root:
                        descendants.add(body)
                        break
                    ancestor = int(model.body_parentid[ancestor])
            self.descendants[name] = descendants

    def targets(self, nominal, *, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Finger compensation dt must be finite and positive")
        if any(name not in nominal or not np.isfinite(nominal[name]) for name in self.names):
            raise ValueError("Finger compensation requires finite nominal targets")
        if any(
            not self.limits[name][0] <= nominal[name] <= self.limits[name][1] for name in self.names
        ):
            raise ValueError("Nominal finger reference exceeds native position bounds")
        result = dict(nominal)
        report = {}
        for name in self.names:
            jid = self.joints[name]
            dof = int(self.model.jnt_dofadr[jid])
            external = 0.0
            for index, contact in enumerate(self.data.contact):
                if contact.efc_address < 0:
                    continue
                b1 = int(self.model.geom_bodyid[contact.geom1])
                b2 = int(self.model.geom_bodyid[contact.geom2])
                if not (
                    (b1 in self.descendants[name] and b2 == self.product)
                    or (b2 in self.descendants[name] and b1 == self.product)
                ):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                sign = -1 if b1 in self.descendants[name] else 1
                rotation = contact.frame.reshape(3, 3).T
                force = sign * rotation @ wrench[:3]
                moment = sign * rotation @ wrench[3:]
                moment += np.cross(contact.pos - self.data.xanchor[jid], force)
                external += float(np.dot(self.data.xaxis[jid], moment))
            requested_effort = float(self.data.qfrc_bias[dof]) - external
            desired_offset = requested_effort / self.gains[name]
            bounded = float(np.clip(desired_offset, -self.offset_limit, self.offset_limit))
            limited = self.offsets[name] + float(
                np.clip(
                    bounded - self.offsets[name],
                    -self.rate_limit * dt,
                    self.rate_limit * dt,
                )
            )
            joint = self.model.joint(name)
            lower, upper = self.limits[name]
            applied = float(np.clip(nominal[name] + limited, lower, upper))
            self.offsets[name] = applied - nominal[name]
            clipped = abs(desired_offset - self.offsets[name]) > 1e-9
            self.clipped_updates += clipped
            self.peak_offset = max(self.peak_offset, abs(self.offsets[name]))
            result[name] = applied
            report[name] = dict(
                nominal_reference_rad=float(nominal[name]),
                applied_reference_rad=applied,
                measured_angle_rad=float(self.data.qpos[int(joint.qposadr[0])]),
                measured_contact_effort_Nm=external,
                requested_holding_effort_Nm=requested_effort,
                applied_offset_rad=self.offsets[name],
                clipped=bool(clipped),
            )
        self.last_report = report
        return result
