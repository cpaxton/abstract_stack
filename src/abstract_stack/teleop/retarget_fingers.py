"""OVR24 global hand positions → bounded native MABEL finger proposals.

This module owns no simulator and never writes live model/data. It does not
establish freshness, calibration, engagement, grasp or contact safety; #20 owns
those gates. Missing input invalidates both-hand proposals rather than reusing a
cached posture. The pinned MuJoCo hand has direct joint servos, not finger tendons.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

HAND_NAMES = tuple(
    "Hand_" + x
    for x in (
        "WristRoot ForearmStub Thumb0 Thumb1 Thumb2 Thumb3 Index1 Index2 Index3 "
        "Middle1 Middle2 Middle3 Ring1 Ring2 Ring3 Pinky0 Pinky1 Pinky2 Pinky3 "
        "ThumbTip IndexTip MiddleTip RingTip PinkyTip"
    ).split()
)
SUFFIXES = ("Thumb MCP", "Thumb Abd", "Thumb PIP", "Thumb DIP") + tuple(
    f"{finger} {joint}"
    for finger in ("Index", "Middle", "Ring", "Pinky")
    for joint in ("Abd", "MCP", "PIP")
)
FINGER_ACTUATORS = tuple(f"{side} {suffix}" for side in ("Left", "Right") for suffix in SUFFIXES)
_REQUIRED = set(HAND_NAMES) - {"Hand_ForearmStub", "Hand_Pinky0"}


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (float, int, np.integer, np.floating))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-5:
        raise ValueError("degenerate hand segment or palm frame")
    return vector / length


def _angle(a, b):
    return math.acos(float(np.clip(np.dot(_unit(a), _unit(b)), -1, 1)))


@dataclass(frozen=True)
class FingerMappingConfig:
    """Declared geometric projection/settings, not a fitted human/robot calibration."""

    filter_tau_s: float = 0.08
    max_velocity_rad_s: float = 2.0
    max_acceleration_rad_s2: float = 8.0
    max_dt_s: float = 0.05
    human_mcp_full_rad: float = 1.4
    human_pip_full_rad: float = 1.5
    human_thumb_full_rad: float = 1.3
    human_opposition_full_rad: float = 1.2
    # Conservative native endpoints. All are validated against the actual model.
    finger_closed_mcp_rad: float = 1.4
    finger_closed_pip_rad: float = 1.6
    thumb_closed_mcp_rad: float = 0.55
    thumb_opposed_abd_rad: float = -0.1
    thumb_closed_pip_rad: float = 0.8
    thumb_closed_dip_rad: float = 1.1
    abduction_gain: float = 0.7

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "thumb_opposed_abd_rad":
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError("thumb opposition endpoint must be finite")
            else:
                _positive(value, key)


@dataclass(frozen=True)
class FingerProposal:
    valid: bool
    targets: dict[str, float] = field(default_factory=dict)
    nominal: dict[str, float] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    reason: str = ""


def hand_features(source: Mapping, side: str) -> dict[str, float]:
    """Extract features in an anatomical palm basis, invariant to rigid motion.

    Only global positions of named joints are used. Tip orientations may be
    unavailable; local rotations cannot be substituted for these global points.
    ``side`` is supplied by the trusted left/right source slot, not inferred.
    """
    if side not in ("Left", "Right") or not isinstance(source, Mapping):
        raise ValueError("named Left/Right hand source required")
    if source.get("skeleton_profile") != "ovr_hand_24":
        raise ValueError("expected canonical ovr_hand_24 global positions")
    root = source.get("pose", {})
    if (
        not isinstance(root, Mapping)
        or root.get("position_valid") is not True
        or root.get("orientation_valid") is not True
    ):
        raise ValueError("hand root pose unavailable")
    joints = source.get("joints")
    if not isinstance(joints, list) or len(joints) != 24:
        raise ValueError("complete named OVR24 joint list required")
    positions = {}
    seen = set()
    for joint in joints:
        if not isinstance(joint, Mapping):
            raise ValueError("invalid named joint")
        name = joint.get("name")
        if not isinstance(name, str) or name not in HAND_NAMES or name in seen:
            raise ValueError("unknown or duplicate joint name")
        seen.add(name)
        if name in _REQUIRED:
            if joint.get("position_valid") is not True:
                raise ValueError(f"{name} position unavailable")
            raw = np.asarray(joint.get("position_m"))
            if raw.shape != (3,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
                raise ValueError(f"{name} requires finite global metres")
            positions[name] = np.asarray(raw, dtype=float)
    wrist = positions["Hand_WristRoot"]
    forward = _unit(positions["Hand_Middle1"] - wrist)
    radial = positions["Hand_Index1"] - positions["Hand_Pinky1"]
    radial = _unit(radial - forward * np.dot(radial, forward))
    sign = 1 if side == "Left" else -1
    normal = sign * _unit(np.cross(radial, forward))
    hinge_axis = sign * radial
    result = {}
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        p = [positions[f"Hand_{finger}{suffix}"] for suffix in ("1", "2", "3", "Tip")]
        proximal, intermediate, distal = (_unit(p[i + 1] - p[i]) for i in range(3))
        mcp = math.atan2(-float(np.dot(proximal, normal)), float(np.dot(proximal, forward)))
        if math.hypot(np.dot(proximal, normal), np.dot(proximal, forward)) < 0.1:
            raise ValueError(f"{finger} has unusable flexion projection")
        pip = math.atan2(
            -float(np.dot(np.cross(proximal, intermediate), hinge_axis)),
            float(np.dot(proximal, intermediate)),
        )
        dip = math.atan2(
            -float(np.dot(np.cross(intermediate, distal), hinge_axis)),
            float(np.dot(intermediate, distal)),
        )
        result[f"{finger} MCP"] = mcp
        # Native fingers have one PIP/IP servo and no independent DIP actuator.
        result[f"{finger} PIP"] = 0.7 * max(0, pip) + 0.3 * max(0, dip)
        result[f"{finger} Abd"] = math.atan2(
            float(np.dot(proximal, radial)), float(np.dot(proximal, forward))
        )
    thumb = [positions[f"Hand_Thumb{x}"] for x in ("0", "1", "2", "3", "Tip")]
    segments = [thumb[i + 1] - thumb[i] for i in range(4)]
    for key, index in (("Thumb MCP", 0), ("Thumb PIP", 1), ("Thumb DIP", 2)):
        result[key] = _angle(segments[index], segments[index + 1])
    base = _unit(segments[0])
    if math.hypot(np.dot(base, radial), np.dot(base, forward)) < 0.1:
        raise ValueError("thumb opposition projection is unavailable")
    result["Thumb Abd"] = math.atan2(float(np.dot(base, forward)), float(np.dot(base, radial)))
    return result


class FingerRetargeter:
    """Copy native model descriptors, then produce detached complete finger maps.

    ``reset`` requires caller-approved applied references after calibration or a
    pause. No measured out-of-range posture is silently clipped into an initial
    command. Missing data clears the filter and requires another explicit reset.
    """

    def __init__(self, model: mujoco.MjModel, config: FingerMappingConfig | None = None):
        self.config = config or FingerMappingConfig()
        self.names = FINGER_ACTUATORS
        limits, opened = [], []
        for name in self.names:
            try:
                actuator = model.actuator(name)
            except KeyError as error:
                raise ValueError(f"missing native finger actuator {name}") from error
            aid = actuator.id
            if (
                model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_biasprm[aid, 1] >= 0
                or not model.actuator_ctrllimited[aid]
            ):
                raise ValueError("finger mapping requires limited native joint position servos")
            jid = model.actuator_trnid[aid, 0]
            if (
                model.joint(jid).name != name
                or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                or not np.array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError("unsupported native finger transmission")
            low, high = model.actuator_ctrlrange[aid]
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("invalid native finger limits")
            limits.append((low, high))
            opened.append(model.qpos0[model.jnt_qposadr[jid]])
        self.limits = np.array(limits)
        self.opened = np.array(opened)
        self._check(self.opened)
        self.closed = self.opened.copy()
        c = self.config
        for i, name in enumerate(self.names):
            suffix = " ".join(name.split()[1:])
            if suffix == "Thumb MCP":
                self.closed[i] = c.thumb_closed_mcp_rad
            elif suffix == "Thumb Abd":
                self.closed[i] = c.thumb_opposed_abd_rad
            elif suffix == "Thumb PIP":
                self.closed[i] = c.thumb_closed_pip_rad
            elif suffix == "Thumb DIP":
                self.closed[i] = c.thumb_closed_dip_rad
            elif suffix.endswith(" MCP"):
                self.closed[i] = c.finger_closed_mcp_rad
            elif suffix.endswith(" PIP"):
                self.closed[i] = c.finger_closed_pip_rad
        self._check(self.closed)
        for descriptor in (self.limits, self.opened, self.closed):
            descriptor.setflags(write=False)
        self._previous = None
        self._velocity = np.zeros(32)
        self._feedback_position = None
        self._feedback_velocity = np.zeros(32)

    def describe(self):
        return {
            "actuators": list(self.names),
            "config": asdict(self.config),
            "native_ctrlrange_rad": self.limits.tolist(),
            "open_reference_rad": self.opened.tolist(),
            "closed_reference_rad": self.closed.tolist(),
            "transmission": "32 direct joint position servos; no finger tendon synthesis",
        }

    def _check(self, vector):
        if (
            vector.shape != (32,)
            or not np.isfinite(vector).all()
            or np.any(vector < self.limits[:, 0])
            or np.any(vector > self.limits[:, 1])
        ):
            raise ValueError("finger references outside native finite control ranges")

    def reset(self, applied: Mapping[str, float]):
        self.invalidate()
        if not isinstance(applied, Mapping) or not set(self.names) <= applied.keys():
            raise ValueError("all32 applied finger references are required")
        raw = [applied[name] for name in self.names]
        if any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (float, int, np.integer, np.floating))
            or not math.isfinite(v)
            for v in raw
        ):
            raise ValueError("applied finger references must be finite numbers")
        values = np.array(raw, dtype=float)
        self._check(values)
        self._previous, self._velocity = values.copy(), np.zeros(32)
        self._feedback_position, self._feedback_velocity = values.copy(), np.zeros(32)

    def invalidate(self):
        self._previous = None
        self._velocity[:] = 0
        self._feedback_position = None
        self._feedback_velocity[:] = 0

    def propose(
        self,
        left: Mapping,
        right: Mapping,
        dt_s: float,
        *,
        applied_references: Mapping[str, float] | None = None,
    ) -> FingerProposal:
        try:
            dt = _positive(dt_s, "dt_s")
            if dt > self.config.max_dt_s:
                raise ValueError("control gap exceeds finger filter max_dt_s")
            if self._previous is None:
                raise ValueError("explicit finger reference reset required")
            if applied_references is not None:
                if (
                    not isinstance(applied_references, Mapping)
                    or not set(self.names) <= applied_references.keys()
                ):
                    raise ValueError("all applied finger references are required for feedback")
                raw = [applied_references[name] for name in self.names]
                if any(
                    isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (float, int, np.integer, np.floating))
                    for v in raw
                ):
                    raise ValueError("applied finger feedback must be numeric")
                actual = np.asarray(raw, dtype=float)
                self._check(actual)
                velocity = (actual - self._feedback_position) / dt
                if np.any(np.abs(velocity) > self.config.max_velocity_rad_s + 1e-9) or np.any(
                    np.abs(velocity - self._feedback_velocity)
                    > self.config.max_acceleration_rad_s2 * dt + 1e-9
                ):
                    raise ValueError(
                        "applied references changed outside finger rate envelope; reset required"
                    )
                self._previous, self._velocity = actual.copy(), velocity.copy()
                self._feedback_position, self._feedback_velocity = actual.copy(), velocity.copy()
            features = {
                side: hand_features(source, side)
                for side, source in (("Left", left), ("Right", right))
            }
            c = self.config
            nominal = self.opened.copy()
            clipped_features = []
            for i, name in enumerate(self.names):
                side, suffix = name.split(" ", 1)
                value = features[side][suffix]
                if suffix.endswith(" Abd") and not suffix.startswith("Thumb"):
                    nominal[i] += c.abduction_gain * value
                else:
                    full = (
                        c.human_opposition_full_rad
                        if suffix == "Thumb Abd"
                        else c.human_thumb_full_rad
                        if suffix.startswith("Thumb")
                        else c.human_mcp_full_rad
                        if suffix.endswith("MCP")
                        else c.human_pip_full_rad
                    )
                    normalized = value / full
                    clipped = float(np.clip(normalized, 0, 1))
                    if clipped != normalized:
                        clipped_features.append(name)
                    nominal[i] += clipped * (self.closed[i] - self.opened[i])
            bounded = np.clip(nominal, self.limits[:, 0], self.limits[:, 1])
            saturated = [
                name for name, a, b in zip(self.names, nominal, bounded, strict=True) if a != b
            ]
            alpha = -math.expm1(-dt / c.filter_tau_s)
            requested_velocity = alpha * (bounded - self._previous) / dt
            acceleration = c.max_acceleration_rad_s2
            maximum = c.max_velocity_rad_s
            # Reserve braking distance to each native bound. Unlike target-space
            # clipping after filtering, this preserves the acceleration contract.
            brake_positive = (
                np.sqrt(
                    (acceleration * dt) ** 2
                    + 2 * acceleration * np.maximum(0, self.limits[:, 1] - self._previous)
                )
                - acceleration * dt
            )
            brake_negative = (
                np.sqrt(
                    (acceleration * dt) ** 2
                    + 2 * acceleration * np.maximum(0, self._previous - self.limits[:, 0])
                )
                - acceleration * dt
            )
            low = np.maximum.reduce(
                (np.full(32, -maximum), self._velocity - acceleration * dt, -brake_negative)
            )
            high = np.minimum.reduce(
                (np.full(32, maximum), self._velocity + acceleration * dt, brake_positive)
            )
            if np.any(low > high + 1e-12):
                raise ValueError("finger filter has no bounded acceleration transition")
            velocity = np.clip(requested_velocity, low, high)
            applied = self._previous + velocity * dt
            self._check(applied)
            diagnostics = {
                "features_rad": features,
                "feature_saturation": clipped_features,
                "native_saturation": saturated,
                "velocity_rad_s": velocity.tolist(),
                "acceleration_rad_s2": ((velocity - self._velocity) / dt).tolist(),
                "dt_s": dt,
                "input_basis": "canonical global operator positions",
            }
            self._previous, self._velocity = applied.copy(), velocity.copy()
            return FingerProposal(
                True,
                dict(zip(self.names, map(float, applied), strict=True)),
                dict(zip(self.names, map(float, bounded), strict=True)),
                diagnostics,
            )
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.invalidate()
            return FingerProposal(False, reason=str(error))
