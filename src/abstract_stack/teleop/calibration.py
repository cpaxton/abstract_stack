"""Generation-scoped relative pose calibration; never writes robot state.

Inputs are validated global operator poses. Transport freshness, lifecycle,
collision checks and the final engagement/command gate remain their owners'
responsibility. A locally eligible alignment is not permission to actuate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import mujoco
import numpy as np

UNITY_TO_OPERATOR = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
OPERATOR_TO_ROBOT = np.diag([-1.0, -1.0, 1.0])
REQUIRED_POSES = frozenset(("head", "left_hand", "right_hand"))
OPTIONAL_POSES = frozenset(("Body_Chest", "Body_Hips"))


def _vector(value, size, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return array


def _rotation(quaternion):
    q = _vector(quaternion, 4, "wxyz quaternion")
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("quaternion must have unit norm within 1e-3")
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, q / norm)
    return result.reshape(3, 3)


def _quaternion(rotation):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(rotation).reshape(9))
    # q and -q are equivalent; deterministic storage avoids artificial sign jumps.
    if result[np.argmax(np.abs(result))] < 0:
        result *= -1
    return tuple(float(x) for x in result)


@dataclass(frozen=True)
class Pose:
    """Active local-to-parent rigid pose, metres and wxyz; immutable storage."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        p = _vector(self.position_m, 3, "position_m")
        r = _rotation(self.quaternion_wxyz)
        object.__setattr__(self, "position_m", tuple(float(x) for x in p))
        object.__setattr__(self, "quaternion_wxyz", _quaternion(r))

    @property
    def rotation(self):
        return _rotation(self.quaternion_wxyz)

    def compose(self, local: Pose) -> Pose:
        return Pose(
            tuple(np.asarray(self.position_m) + self.rotation @ local.position_m),
            _quaternion(self.rotation @ local.rotation),
        )

    def to_dict(self):
        return {"position_m": list(self.position_m), "quaternion_wxyz": list(self.quaternion_wxyz)}


IDENTITY = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def change_basis(pose: Pose, basis) -> Pose:
    """Conjugate rotations, including reflected bases which aren't quaternions."""
    b = np.asarray(basis, dtype=float)
    if b.shape != (3, 3) or not np.isfinite(b).all() or not np.allclose(b @ b.T, np.eye(3)):
        raise ValueError("basis must be an orthogonal finite 3x3 matrix")
    return Pose(tuple(b @ pose.position_m), _quaternion(b @ pose.rotation @ b.T))


def unity_to_operator(position_m, quaternion_xyzw) -> Pose:
    """Unity LH tracking-space global pose to canonical RH operator pose."""
    x, y, z, w = _vector(quaternion_xyzw, 4, "Unity xyzw quaternion")
    return change_basis(Pose(position_m, (w, x, y, z)), UNITY_TO_OPERATOR)


def ovr_to_operator(position_m, quaternion_xyzw) -> Pose:
    """Native OVR RH GLOBAL tracking pose via SDK FromFlippedZ conversion.

    Never call for legacy hand bone-local rotations (their special SDK basis
    requires acquisition-side forward kinematics before this conversion).
    """
    x, y, z = _vector(position_m, 3, "OVR position_m")
    qx, qy, qz, qw = _vector(quaternion_xyzw, 4, "OVR xyzw quaternion")
    return unity_to_operator((x, y, -z), (-qx, -qy, qz, qw))


def pose_from_tracking(value: Mapping) -> Pose:
    """Require both measured components; unavailable data never becomes identity."""
    if value.get("position_valid") is not True or value.get("orientation_valid") is not True:
        raise ValueError("both position and orientation must be valid")
    return Pose(value["position_m"], value["quaternion_wxyz"])


@dataclass(frozen=True)
class TargetSpec:
    """Measured robot neutral plus configured wrist-local palm offset and scale.

    No universal wrist/palm offset or robot neutral is assumed. The caller must
    supply native root-relative measured robot poses; world spawn yaw is excluded.
    """

    robot_neutral: Pose
    wrist_to_palm: Pose = IDENTITY
    translation_scale: float = 1.0
    max_translation_m: float = 0.5
    max_rotation_rad: float = np.pi

    def __post_init__(self):
        if not isinstance(self.robot_neutral, Pose) or not isinstance(self.wrist_to_palm, Pose):
            raise ValueError("robot neutral and palm offset must be validated poses")
        for name in ("translation_scale", "max_translation_m", "max_rotation_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_rotation_rad > np.pi:
            raise ValueError("max_rotation_rad must not exceed pi")


@dataclass(frozen=True)
class CalibrationArtifact:
    session_id: str
    generation: int
    neutral_operator: Mapping[str, Pose]
    specs: Mapping[str, TargetSpec]
    operator_to_robot_wxyz: tuple[float, float, float, float]

    def __post_init__(self):
        if not self.session_id:
            raise ValueError("session_id is required")
        _generation(self.generation)
        if not REQUIRED_POSES <= self.neutral_operator.keys():
            raise ValueError("head and both wrists are required")
        if set(self.neutral_operator) != set(self.specs):
            raise ValueError("every neutral pose needs exactly one target specification")
        if not set(self.specs) <= REQUIRED_POSES | OPTIONAL_POSES:
            raise ValueError("unknown or swapped source names are not supported")
        if not all(isinstance(p, Pose) for p in self.neutral_operator.values()):
            raise ValueError("neutral inputs must be validated poses")
        if not all(isinstance(s, TargetSpec) for s in self.specs.values()):
            raise ValueError("specifications must be validated TargetSpec values")
        object.__setattr__(self, "neutral_operator", MappingProxyType(dict(self.neutral_operator)))
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))
        object.__setattr__(
            self, "operator_to_robot_wxyz", _quaternion(_rotation(self.operator_to_robot_wxyz))
        )

    def targets(self, current: Mapping[str, Pose]) -> dict[str, Pose]:
        """Relative deltas around each captured neutral; all-or-nothing bounds.

        Missing optional body inputs disable their proposals rather than retaining
        stale values. Missing head/either wrist rejects the complete proposal.
        """
        if not REQUIRED_POSES <= current.keys():
            raise ValueError("head and both wrists are required")
        if not set(current) <= self.specs.keys():
            raise ValueError("input has no captured neutral")
        alignment = _rotation(self.operator_to_robot_wxyz)
        result = {}
        for name, pose in current.items():
            if not isinstance(pose, Pose):
                raise ValueError("inputs must be validated poses")
            spec = self.specs[name]
            neutral = self.neutral_operator[name].compose(spec.wrist_to_palm)
            palm = pose.compose(spec.wrist_to_palm)
            delta = alignment @ (np.asarray(palm.position_m) - neutral.position_m)
            delta *= spec.translation_scale
            rotation = alignment @ palm.rotation @ neutral.rotation.T @ alignment.T
            angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
            if np.linalg.norm(delta) > spec.max_translation_m or angle > spec.max_rotation_rad:
                raise ValueError(f"{name} exceeds calibrated relative-motion bounds")
            result[name] = Pose(
                tuple(np.asarray(spec.robot_neutral.position_m) + delta),
                _quaternion(rotation @ spec.robot_neutral.rotation),
            )
        return result

    def to_dict(self):
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "generation": self.generation,
            "frame": "mabel_chassis_native",
            "input_frame": "operator_forward_left_up",
            "position_unit": "m",
            "quaternion_order": "wxyz",
            "unity_to_operator_basis": UNITY_TO_OPERATOR.tolist(),
            "operator_to_robot_wxyz": list(self.operator_to_robot_wxyz),
            "neutral_operator": {k: v.to_dict() for k, v in self.neutral_operator.items()},
            "targets": {
                k: {
                    "robot_neutral": s.robot_neutral.to_dict(),
                    "wrist_to_palm": s.wrist_to_palm.to_dict(),
                    "translation_scale": s.translation_scale,
                    "max_translation_m": s.max_translation_m,
                    "max_rotation_rad": s.max_rotation_rad,
                }
                for k, s in self.specs.items()
            },
        }


def neutral_heading_alignment(head: Pose) -> tuple[float, float, float, float]:
    """Align horizontal neutral head-forward (+X) to native robot-forward (-X).

    Preserve gravity; never inject world spawn yaw or neutral head roll/pitch.
    A near-vertical forward vector cannot define a reliable horizontal heading.
    """
    forward = head.rotation @ np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(forward[:2]) < 0.1:
        raise ValueError("neutral head heading is near vertical")
    yaw = np.pi - np.arctan2(forward[1], forward[0])
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def _generation(value):
    if type(value) is not int or not 0 <= value < 2**53:
        raise ValueError("generation must be an exact nonnegative integer below 2^53")


class Calibration:
    """Alignment lifecycle only. #20 must additionally gate all live commands.

    Host owns generations. Every invalidation/capture advances generation and
    clears deliberate-engage eligibility, including after a failed capture.
    """

    def __init__(self, session_id: str, generation: int = 0):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        _generation(generation)
        self.session_id = session_id
        self.generation = generation
        self.artifact = None
        self.engagement_acknowledged = False
        self.reason = "uncalibrated"

    def invalidate(self, reason: str) -> int:
        if self.generation == 2**53 - 1:
            raise ValueError("generation exhausted; establish a new session")
        self.generation += 1
        self.artifact = None
        self.engagement_acknowledged = False
        self.reason = reason
        return self.generation

    def capture(
        self,
        neutral: Mapping[str, Pose],
        specs: Mapping[str, TargetSpec],
        *,
        ready_or_paused: bool,
        inputs_usable: bool,
        operator_to_robot_wxyz=None,
    ) -> CalibrationArtifact:
        if ready_or_paused is not True:
            raise ValueError("pause before calibration")
        self.invalidate("calibration pending")
        if inputs_usable is not True:
            raise ValueError("freshness/lifecycle/profile inputs must be usable")
        if operator_to_robot_wxyz is None:
            if "head" not in neutral or not isinstance(neutral["head"], Pose):
                raise ValueError("a validated neutral head is required")
            operator_to_robot_wxyz = neutral_heading_alignment(neutral["head"])
        artifact = CalibrationArtifact(
            self.session_id, self.generation, neutral, specs, operator_to_robot_wxyz
        )
        # Verify all captured neutrals yield their exact robot reference before publication.
        artifact.targets(neutral)
        self.artifact = artifact
        self.reason = "deliberate engagement required"
        return artifact

    def acknowledge_engagement(
        self, *, session_id: str, generation: int, deliberate: bool, inputs_usable: bool
    ):
        self.engagement_acknowledged = False
        if (
            self.artifact is None
            or session_id != self.session_id
            or type(generation) is not int
            or generation != self.generation
            or deliberate is not True
            or inputs_usable is not True
        ):
            raise ValueError("current calibrated alignment and deliberate engagement required")
        self.engagement_acknowledged = True

    def propose(
        self, current: Mapping[str, Pose], *, session_id: str, generation: int, inputs_usable: bool
    ) -> dict[str, Pose]:
        if (
            session_id != self.session_id
            or type(generation) is not int
            or generation != self.generation
        ):
            raise ValueError("obsolete session/generation")
        if not self.engagement_acknowledged or self.artifact is None:
            raise ValueError("deliberate engagement required")
        if inputs_usable is not True:
            self.engagement_acknowledged = False
            self.reason = "input unavailable; deliberate re-engagement required"
            raise ValueError(self.reason)
        try:
            return self.artifact.targets(current)
        except ValueError:
            self.engagement_acknowledged = False
            self.reason = "invalid targets; deliberate re-engagement required"
            raise
