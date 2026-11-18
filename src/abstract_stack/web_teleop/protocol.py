"""Wire protocol for ``web_teleop_v1``: strict JSON validation and binary state frames.

JSON control messages reuse the Quest v1 envelope (``schema_version``, ``type``,
``session_id``). Unknown fields, unknown types, non-finite numbers and oversize
payloads are rejected before they reach the simulation owner.
"""

from __future__ import annotations

import json
import math
import struct
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
PROFILE = "web_teleop_v1"
MAX_JSON_BYTES = 16 * 1024
MAX_SAFE_INTEGER = 2**53 - 1
SIDES = ("left", "right")
CONTROL_ACTIONS = ("engage", "pause", "reset", "ready")
ROLES = ("operator", "spectator")

# Binary state frame: little-endian header followed by float32 body poses.
STATE_MAGIC = b"WTS1"
STATE_HEADER = struct.Struct("<4sIddHH")  # magic, tick, sim_time, host_time, nbody, flags
FLAG_ENGAGED = 1


class ProtocolError(ValueError):
    """A message violated the v1 contract; the connection reports it and continues or closes."""


def _require_keys(message: Mapping, required: set[str], optional: set[str] = frozenset()) -> None:
    keys = set(message)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ProtocolError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ProtocolError(f"unexpected fields: {', '.join(sorted(unknown))}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError(f"{name} must be finite")
    return value


def _counter(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ProtocolError(f"{name} must be an integer in [0, 2^53)")
    return value


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{name} must be a list of {length} numbers")
    return np.array([_finite(v, f"{name}[{i}]") for i, v in enumerate(value)])


def _quaternion(value: Any, name: str) -> np.ndarray:
    quat = _vector(value, 4, name)
    if abs(float(np.linalg.norm(quat)) - 1.0) > 1e-3:
        raise ProtocolError(f"{name} must be a unit quaternion (norm within 1e-3)")
    return quat / np.linalg.norm(quat)


def decode_json(raw: str | bytes) -> dict:
    size = len(raw.encode() if isinstance(raw, str) else raw)
    if size > MAX_JSON_BYTES:
        raise ProtocolError(f"message of {size} bytes exceeds {MAX_JSON_BYTES}")
    try:
        message = json.loads(raw, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid JSON: {error.msg}") from error
    if not isinstance(message, dict):
        raise ProtocolError("message must be a JSON object")
    if message.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError(f"unsupported schema_version {message.get('schema_version')!r}")
    if not isinstance(message.get("type"), str):
        raise ProtocolError("type must be a string")
    return message


def _reject_constant(name: str):
    raise ProtocolError(f"non-finite JSON value {name}")


def _session(value: Any) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError as error:
        raise ProtocolError("session_id must be a UUID") from error


def parse_hello(message: Mapping) -> dict:
    _require_keys(message, {"schema_version", "type", "session_id", "profile", "role"})
    if message["profile"] != PROFILE:
        raise ProtocolError(f"unsupported profile {message['profile']!r}; expected {PROFILE}")
    if message["role"] not in ROLES:
        raise ProtocolError(f"role must be one of {', '.join(ROLES)}")
    return {"session_id": _session(message["session_id"]), "role": message["role"]}


def parse_control(message: Mapping) -> dict:
    _require_keys(
        message,
        {"schema_version", "type", "session_id", "event_id", "calibration_generation", "action"},
    )
    if message["action"] not in CONTROL_ACTIONS:
        raise ProtocolError(f"action must be one of {', '.join(CONTROL_ACTIONS)}")
    return {
        "session_id": _session(message["session_id"]),
        "event_id": _counter(message["event_id"], "event_id"),
        "generation": _counter(message["calibration_generation"], "calibration_generation"),
        "action": message["action"],
    }


def parse_ping(message: Mapping) -> dict:
    _require_keys(message, {"schema_version", "type", "session_id", "t0"})
    return {"session_id": _session(message["session_id"]), "t0": _finite(message["t0"], "t0")}


def _parse_arm(value: Any, side: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError(f"arms.{side} must be an object or null")
    _require_keys(value, {"position_m", "quaternion_wxyz", "grip"})
    grip = _finite(value["grip"], f"arms.{side}.grip")
    if not 0.0 <= grip <= 1.0:
        raise ProtocolError(f"arms.{side}.grip must be in [0, 1]")
    return {
        "position": _vector(value["position_m"], 3, f"arms.{side}.position_m"),
        "quaternion": _quaternion(value["quaternion_wxyz"], f"arms.{side}.quaternion_wxyz"),
        "grip": grip,
    }


MAX_BASE_LINEAR = 2.0  # m/s accepted on the wire; the controller clamps further
MAX_BASE_ANGULAR = 3.0  # rad/s


def _parse_base(value: Any) -> dict | None:
    """Velocity intent in the robot's own frame; null (or omitted) means stop."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("base must be an object or null")
    _require_keys(value, {"forward_mps", "left_mps", "turn_radps"})
    out = {key: _finite(value[key], f"base.{key}") for key in value}
    if max(abs(out["forward_mps"]), abs(out["left_mps"])) > MAX_BASE_LINEAR:
        raise ProtocolError(f"base speed must be within {MAX_BASE_LINEAR} m/s")
    if abs(out["turn_radps"]) > MAX_BASE_ANGULAR:
        raise ProtocolError(f"base.turn_radps must be within {MAX_BASE_ANGULAR} rad/s")
    return out


def parse_command(message: Mapping) -> dict:
    """Absolute operator intents in the ``mabel_chassis`` root frame.

    ``arms.<side>`` null holds that arm; ``head``/``lift_m``/``torso_rad`` null hold
    those channels. Optional ``base`` is a velocity intent (forward/left m/s,
    turn-left rad/s) that the host ramps and stops whenever input is stale or control
    pauses. Absolute pose targets make each message self-contained, so a dropped
    message never accumulates as drift.
    """
    _require_keys(
        message,
        {
            "schema_version",
            "type",
            "session_id",
            "sequence",
            "calibration_generation",
            "target_epoch",
            "sent_monotonic_s",
            "arms",
            "head",
            "lift_m",
            "torso_rad",
        },
        {"base"},
    )
    arms = message["arms"]
    if not isinstance(arms, dict):
        raise ProtocolError("arms must be an object")
    _require_keys(arms, set(), set(SIDES))
    head = message["head"]
    if head is not None:
        if not isinstance(head, dict):
            raise ProtocolError("head must be an object or null")
        if "look_at_m" in head:
            # Gaze target in the chassis frame; the host solves neck angles so the head
            # camera points at it (follow-hand and click-to-look).
            _require_keys(head, {"look_at_m"})
            head = {"look_at": _vector(head["look_at_m"], 3, "head.look_at_m")}
        else:
            _require_keys(head, {"yaw_rad", "pitch_rad"})
            head = {
                "Neck Yaw": _finite(head["yaw_rad"], "head.yaw_rad"),
                "Neck Pitch": _finite(head["pitch_rad"], "head.pitch_rad"),
            }
    return {
        "session_id": _session(message["session_id"]),
        "sequence": _counter(message["sequence"], "sequence"),
        "generation": _counter(message["calibration_generation"], "calibration_generation"),
        "target_epoch": _counter(message["target_epoch"], "target_epoch"),
        "sent_monotonic_s": _finite(message["sent_monotonic_s"], "sent_monotonic_s"),
        "arms": {side: _parse_arm(arms.get(side), side) for side in SIDES},
        "head": head,
        "base": _parse_base(message.get("base")),
        "lift": None if message["lift_m"] is None else _finite(message["lift_m"], "lift_m"),
        "torso": None
        if message["torso_rad"] is None
        else _finite(message["torso_rad"], "torso_rad"),
    }


def pack_state(
    tick: int, sim_time: float, host_time: float, xpos: np.ndarray, xquat: np.ndarray, flags: int
) -> bytes:
    nbody = xpos.shape[0]
    header = STATE_HEADER.pack(STATE_MAGIC, tick & 0xFFFFFFFF, sim_time, host_time, nbody, flags)
    poses = np.concatenate((xpos.astype("<f4").ravel(), xquat.astype("<f4").ravel()))
    return header + poses.tobytes()


def unpack_state(frame: bytes) -> dict:
    magic, tick, sim_time, host_time, nbody, flags = STATE_HEADER.unpack_from(frame)
    if magic != STATE_MAGIC:
        raise ProtocolError("bad state frame magic")
    expected = STATE_HEADER.size + nbody * 7 * 4
    if len(frame) != expected:
        raise ProtocolError(f"state frame has {len(frame)} bytes, expected {expected}")
    poses = np.frombuffer(frame, dtype="<f4", offset=STATE_HEADER.size)
    return {
        "tick": tick,
        "sim_time": sim_time,
        "host_time": host_time,
        "flags": flags,
        "xpos": poses[: nbody * 3].reshape(nbody, 3),
        "xquat": poses[nbody * 3 :].reshape(nbody, 4),
    }


def envelope(message_type: str, session_id: str | None, **fields) -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "type": message_type,
            "session_id": session_id,
            **fields,
        },
        allow_nan=False,
        separators=(",", ":"),
    )
