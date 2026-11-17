"""Small reproducible synthetic global OVR24 hand fixtures; no device evidence."""

import math

import numpy as np

from abstract_stack.teleop.retarget_fingers import HAND_NAMES


def synthetic_hand(side, curl=0.0, *, pinch=False, abduction_rad=0.0):
    """Illustrative mirrored landmarks, with independent finger curves.

    ``curl`` may be one scalar or a mapping of Index/Middle/Ring/Pinky/Thumb.
    Pinch places the two tip landmarks together; it is not an anatomical model
    or a demonstration of physical contact, tracking accuracy or grasp success.
    """
    if side not in ("Left", "Right"):
        raise ValueError("Left/Right fixture side required")
    if not math.isfinite(abduction_rad):
        raise ValueError("finite abduction required")
    values = {
        finger: float(curl.get(finger, 0)) if isinstance(curl, dict) else float(curl)
        for finger in ("Index", "Middle", "Ring", "Pinky", "Thumb")
    }
    if any(not 0 <= x <= 1 for x in values.values()):
        raise ValueError("synthetic curl must be in[0,1]")
    points = {
        "WristRoot": np.zeros(3),
        "ForearmStub": np.array([0, -0.04, 0]),
        "Pinky0": np.array([-0.035, 0.035, 0]),
    }
    for finger, base in (
        ("Index", [0.025, 0.075, 0]),
        ("Middle", [0, 0.08, 0]),
        ("Ring", [-0.02, 0.073, 0]),
        ("Pinky", [-0.035, 0.06, 0]),
    ):
        amount = 0.75 if pinch and finger == "Index" else values[finger]
        p = np.array(base)
        points[finger + "1"] = p
        angles = (1.4 * amount, 2.9 * amount, 3.7 * amount)
        for name, length, angle in zip(("2", "3", "Tip"), (0.04, 0.025, 0.02), angles, strict=True):
            direction = np.array(
                [
                    math.sin(abduction_rad) * math.cos(angle),
                    math.cos(abduction_rad) * math.cos(angle),
                    -math.sin(angle),
                ]
            )
            p = p + length * direction
            points[finger + name] = p
    amount = 0.8 if pinch else values["Thumb"]
    opposition = 1.1 if pinch else 0.1 + amount
    direction = np.array([math.cos(opposition), math.sin(opposition), 0])
    p = np.array([0.035, 0.025, 0])
    points["Thumb0"] = p
    for suffix, length, angle in zip(
        ("1", "2", "3", "Tip"),
        (0.025, 0.025, 0.02, 0.018),
        (0, 1.0 * amount, 2.0 * amount, 2.8 * amount),
        strict=True,
    ):
        p = p + length * (math.cos(angle) * direction + np.array([0, 0, -math.sin(angle)]))
        points["Thumb" + suffix] = p
    if pinch:
        points["ThumbTip"] = points["IndexTip"].copy()
    mirror = np.array([1 if side == "Left" else -1, 1, 1])
    joints = [
        {
            "name": name,
            "position_m": (points[name.removeprefix("Hand_")] * mirror).tolist(),
            "position_valid": True,
            "orientation_valid": False,
            "quaternion_wxyz": None,
            "position_tracked": None,
            "orientation_tracked": None,
            "confidence": None,
        }
        for name in HAND_NAMES
    ]
    return {
        "skeleton_profile": "ovr_hand_24",
        "pose": {
            "position_valid": True,
            "orientation_valid": True,
            "position_m": [0, 0, 0],
            "quaternion_wxyz": [1, 0, 0, 0],
        },
        "joints": joints,
    }
