"""Virtual wrist F/T and coarse rigid-geometry tactile force proxies.

No model or live-data writes. Native parent-interaction wrenches are computed
on a data copy and match MuJoCo force/torque sensor conventions. Contact-only
wrenches use the different, explicit force-ON-hand convention. Neither channel
models transducer electronics, soft skin or optical tactile images.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class WristFrame:
    name: str
    joint: str


def _entity(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"Missing sensor model entity: {name}")
    return index


def _subtree(model, root):
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


class ContactSensorSuite:
    """Stateless measurements of a fresh native state; caller owns sample rate.

    Frame origins are the named joint's native world anchor. Axes follow that
    joint's child body. A tactile cell is one configured collision geom, with
    force expressed in its geom frame. This is an explicit coarse proxy, not a
    calibrated spatial taxel grid. Call after native forward/stepping; this class
    does not repair stale contacts, advance physics or change control targets.
    """

    def __init__(self, model, wrists, *, taxel_geoms=None):
        self.model = model
        self.wrists = tuple(wrists)
        if (
            not self.wrists
            or any(not w.name or not w.joint for w in self.wrists)
            or len({w.name for w in self.wrists}) != len(self.wrists)
        ):
            raise ValueError("Virtual wrist names must be nonempty and unique")
        self.joints = [_entity(model, mujoco.mjtObj.mjOBJ_JOINT, w.joint) for w in self.wrists]
        self.bodies = [int(model.jnt_bodyid[j]) for j in self.joints]
        self.subtrees = [_subtree(model, b) for b in self.bodies]
        if any(a & b for i, a in enumerate(self.subtrees) for b in self.subtrees[i + 1 :]):
            raise ValueError("Virtual wrist subtrees must not overlap")
        all_bodies = set().union(*self.subtrees)
        collision = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in all_bodies
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        if taxel_geoms is None:
            self.geoms = tuple(collision)
        else:
            self.geoms = tuple(_entity(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in taxel_geoms)
            if len(set(self.geoms)) != len(self.geoms) or not set(self.geoms) <= set(collision):
                raise ValueError("Taxels require unique native collision geoms below the wrists")
        self.taxel_index = {geom: index for index, geom in enumerate(self.geoms)}
        self.scratch = mujoco.MjData(model)

    @classmethod
    def mabel(cls, model):
        return cls(model, (WristFrame("left", "Left Wrist"), WristFrame("right", "Right Wrist")))

    def metadata(self):
        return dict(
            kind="native_simulator_sensor_approximation",
            wrist_order=[w.name for w in self.wrists],
            wrist_components=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            wrist_units=["N", "N", "N", "Nm", "Nm", "Nm"],
            wrist_frames=[
                dict(
                    name=w.name,
                    joint=w.joint,
                    origin="native joint anchor",
                    axes="joint child-body rotation",
                )
                for w in self.wrists
            ],
            wrist_wrench_provenance=(
                "native_parent_interaction; matches MuJoCo force/torque sensors"
            ),
            contact_wrench_provenance=(
                "net external contact wrench ON distal robot subtree; excludes gravity and inertia"
            ),
            tactile_provenance=(
                "coarse collision-geom force proxy; "
                "no soft skin, optical image or spatial taxel calibration"
            ),
            tactile_units="N",
            tactile_frame="each configured native geom frame",
            tactile_order=[
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                for g in self.geoms
            ],
            noise="none",
            sample_rate="caller supplied, native simulation timestamp",
        )

    def sample(self, data):
        model = self.model
        if (
            data.qpos.shape != (model.nq,)
            or data.qvel.shape != (model.nv,)
            or data.ctrl.shape != (model.nu,)
            or data.xpos.shape != (model.nbody, 3)
            or data.geom_xpos.shape != (model.ngeom, 3)
        ):
            raise ValueError("Sensor data dimensions do not match the admitted model")
        if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT):
            raise ValueError("Contact sensors require native contacts enabled")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Cannot sample nonfinite native state")
        mujoco.mj_copyData(self.scratch, model, data)
        mujoco.mj_rnePostConstraint(model, self.scratch)
        reaction = np.zeros((len(self.wrists), 6))
        contact_world = np.zeros_like(reaction)
        taxel_world = np.zeros((len(self.geoms), 3))
        for i, (joint, body) in enumerate(zip(self.joints, self.bodies, strict=True)):
            rotation = data.xmat[body].reshape(3, 3)
            origin = data.xanchor[joint]
            raw = self.scratch.cfrc_int[body]
            force = raw[3:]
            moment = raw[:3] - np.cross(
                origin - self.scratch.subtree_com[int(model.body_rootid[body])], force
            )
            reaction[i] = np.r_[rotation.T @ force, rotation.T @ moment]
        wrench = np.empty(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            if not np.isfinite(wrench).all():
                raise ValueError("Nonfinite native contact wrench")
            frame = contact.frame.reshape(3, 3)
            force = frame.T @ wrench[:3]
            moment = frame.T @ wrench[3:]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            for i, (joint, subtree) in enumerate(zip(self.joints, self.subtrees, strict=True)):
                if (b1 in subtree) == (b2 in subtree):
                    continue
                signed_force = (-1 if b1 in subtree else 1) * force
                signed_moment = (-1 if b1 in subtree else 1) * moment
                contact_world[i, :3] += signed_force
                contact_world[i, 3:] += signed_moment + np.cross(
                    contact.pos - data.xanchor[joint], signed_force
                )
            for geom, sign in ((g1, -1), (g2, 1)):
                if geom in self.taxel_index:
                    taxel_world[self.taxel_index[geom]] += sign * force
        contact_local = np.zeros_like(contact_world)
        for i, body in enumerate(self.bodies):
            rotation = data.xmat[body].reshape(3, 3)
            contact_local[i] = np.r_[
                rotation.T @ contact_world[i, :3], rotation.T @ contact_world[i, 3:]
            ]
        tactile = np.asarray(
            [data.geom_xmat[g].reshape(3, 3).T @ taxel_world[i] for i, g in enumerate(self.geoms)],
            dtype=float,
        ).reshape(-1, 3)
        if not np.isfinite(reaction).all():
            raise ValueError("Nonfinite parent-interaction wrench")
        return dict(
            timestamp_s=float(data.time),
            wrist_wrench=reaction,
            contact_wrench=contact_local,
            tactile=tactile,
            wrist_available=np.ones(len(self.wrists), dtype=bool),
            tactile_available=np.ones(len(self.geoms), dtype=bool),
        )
