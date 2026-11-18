"""Native selected-finger pusher shared by stocking protocols."""

import mujoco
import numpy as np

from abstract_stack.interactions import CartesianHand


class FingerPusher(CartesianHand):
    """Geometry-derived extended fingertip, with the other fingers folded.

    Offset comes from native collision geometry in the straight-finger posture;
    it is a kinematic reference, not an attachment or a measured contact point.
    """

    OFFSET = np.array([0.002, 0.006, 0.139])
    BASE_ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def finger_targets(self, grip):
        targets = super().finger_targets(0)
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            for joint in ("MCP", "PIP"):
                targets[f"Left {finger} {joint}"] = (
                    (-0.2 if joint == "MCP" else 0.0) if finger == self.finger else 1.0
                )
        return targets

    def __init__(self, env, config, *, finger="Middle", tilt_degrees=20, wrist_posture=0.4):
        if finger not in ("Index", "Middle", "Ring", "Pinky"):
            raise ValueError("Unsupported native pushing finger")
        if not np.isfinite(tilt_degrees) or not 0 <= tilt_degrees <= 60:
            raise ValueError("Pusher tilt must be within 0–60 degrees")
        self.finger = finger
        self.tilt_degrees = float(tilt_degrees)
        self.wrist_posture = float(wrist_posture)
        tilt = np.radians(tilt_degrees)
        self.ROTATION = (
            np.array(
                [
                    [np.cos(tilt), 0, -np.sin(tilt)],
                    [0, 1, 0],
                    [np.sin(tilt), 0, np.cos(tilt)],
                ]
            )
            @ self.BASE_ROTATION
        )
        super().__init__(env, config)
        # Keep the native wrist mount above the board; unrestricted redundant
        # IK can realize the palm angle with a drooping mount. This is an
        # explicit planning constraint, not a change to joint/actuator limits.
        self.ik.constrain_posture("Left Wrist", wrist_posture)
        # Derive the nominal tip from unmodified collision meshes in the
        # proposed finger posture. Actual compliant contact remains measured.
        self.OFFSET = self.native_tip_offset(finger)
        self.ik.offset = self.OFFSET.copy()
        self.last_target = self.position().copy()
        self.finger_load_compensation = None
        if config.ungrasped_insertion:
            from abstract_stack.finger_control import FingerLoadCompensation

            self.finger_load_compensation = FingerLoadCompensation(self)

    def native_tip_offset(self, finger):
        return self.native_tip_reference(finger)[0]

    def native_tip_reference(self, finger):
        """Read a native mesh tip in the proposed posture, using independent data."""
        scratch = mujoco.MjData(self.model)
        mujoco.mj_copyData(scratch, self.model, self.data)
        for name, value in self.finger_targets(0).items():
            scratch.qpos[int(self.model.joint(name).qposadr[0])] = value
        mujoco.mj_forward(self.model, scratch)
        body = self.model.body(f"left_{finger.lower()}_ip").id
        points = []
        palm_rotation = scratch.xmat[self.palm].reshape(3, 3)
        for geom in range(self.model.ngeom):
            if self.model.geom_bodyid[geom] != body or not (
                self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]
            ):
                continue
            if int(self.model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError("Native tip calibration requires collision meshes")
            mesh = self.model.geom_dataid[geom]
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start : start + count]
            world = vertices @ scratch.geom_xmat[geom].reshape(3, 3).T + scratch.geom_xpos[geom]
            points.append((world - scratch.xpos[self.palm]) @ palm_rotation)
        if not points:
            raise ValueError("Selected native finger has no collision meshes")
        points = np.vstack(points)
        front = points[points[:, 2] >= points[:, 2].max() - 0.002]
        offset = front.mean(axis=0)
        offset[2] = points[:, 2].max()
        world = scratch.xpos[self.palm] + palm_rotation @ offset
        body_offset = scratch.xmat[body].reshape(3, 3).T @ (world - scratch.xpos[body])
        return offset, body_offset


class PairedFingerPusher(FingerPusher):
    """Two native fingertip contacts form a wider, compliant squaring surface."""

    fingers = ("Index", "Ring")
    WRISTS = (0.2, 0.0)

    def finger_targets(self, grip):
        targets = super().finger_targets(0)
        # Bring both contact pads inside a 50 mm carton face. Neutral/native
        # abduction leaves these two tips about 59 mm apart; home is wider still.
        targets["Left Index Abd"] = 0.15
        targets["Left Ring Abd"] = -0.15
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            for joint in ("MCP", "PIP"):
                targets[f"Left {finger} {joint}"] = (
                    (-0.2 if joint == "MCP" else 0.0) if finger in self.fingers else 1.5
                )
        return targets

    def __init__(self, env, config, **kwargs):
        kwargs.setdefault("wrist_posture", 0.2)
        super().__init__(env, config, **kwargs)
        index = self.ik.names.index("Left Wrist")
        self.ik.constrain_reference_range(
            "Left Wrist", self.ik.original_lower[index], self.ik.original_upper[index] - 0.35
        )
        references = [self.native_tip_reference(finger) for finger in self.fingers]
        tips = np.array([reference[0] for reference in references])
        self.tip_body_references = {
            f"left_{finger.lower()}_ip": reference[1].copy()
            for finger, reference in zip(self.fingers, references, strict=True)
        }
        delta = self.ROTATION @ (tips[1] - tips[0])
        angle = (np.arctan2(delta[0], delta[1]) + np.pi / 2) % np.pi - np.pi / 2
        angle = float(np.clip(angle, -np.radians(15), np.radians(15)))
        c, s = np.cos(angle), np.sin(angle)
        self.ROTATION = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.ROTATION
        self.active_rotation = self.ROTATION.copy()
        self.OFFSET = tips.mean(axis=0)
        self.OFFSET[2] = tips[:, 2].max()
        self.ik.offset = self.OFFSET.copy()
        self.last_target = self.position().copy()
        self.plane_yaw_correction_deg = float(np.degrees(angle))

    def contact_plane_yaw(self):
        """Measured native pad-plane yaw, including finger-servo deflection."""
        points = []
        for name, offset in self.tip_body_references.items():
            body = self.model.body(name).id
            points.append(self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ offset)
        delta = points[1] - points[0]
        return float((-np.arctan2(delta[0], delta[1]) + np.pi / 2) % np.pi - np.pi / 2)


class WallFingerPusher(FingerPusher):
    """Use an outer native finger while keeping the palm inside a sidewall."""

    WRISTS = (0.2, 0.0)

    def finger_targets(self, grip):
        targets = super().finger_targets(grip)
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            if finger != self.finger:
                for joint in ("MCP", "PIP"):
                    targets[f"Left {finger} {joint}"] = 1.5
        return targets

    def __init__(self, env, config, **kwargs):
        super().__init__(env, config, **kwargs)
        index = self.ik.names.index("Left Wrist")
        self.ik.constrain_reference_range(
            "Left Wrist", self.ik.original_lower[index], self.ik.original_upper[index] - 0.35
        )
