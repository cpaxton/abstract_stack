"""Bounded Quest v1 WebSocket transport. This module never commands actuators."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

MAX_JSON = 262144
BASE = {"schema_version", "type", "session_id"}
HAND_NAMES = (
    "WristRoot ForearmStub Thumb0 Thumb1 Thumb2 Thumb3 Index1 Index2 Index3 "
    "Middle1 Middle2 Middle3 Ring1 Ring2 Ring3 Pinky0 Pinky1 Pinky2 Pinky3 "
    "ThumbTip IndexTip MiddleTip RingTip PinkyTip"
).split()
BODY_NAMES = (
    "Root Hips SpineLower SpineMiddle SpineUpper Chest Neck Head LeftShoulder "
    "LeftScapula LeftArmUpper LeftArmLower LeftHandWristTwist RightShoulder "
    "RightScapula RightArmUpper RightArmLower RightHandWristTwist"
).split()
PROFILES = {
    "ovr_hand_24": ["Hand_" + n for n in HAND_NAMES],
    "ovr_body_upper_18": ["Body_" + n for n in BODY_NAMES],
}


class ProtocolError(ValueError):
    """The peer violated the versioned wire contract."""


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ProtocolError("missing fields or expected object")
    if value.keys() - set(required) - set(optional):
        raise ProtocolError("unexpected fields")


def number(value):
    if type(value) not in (int, float):
        raise ProtocolError("expected finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ProtocolError("expected finite number")
    return value


def integer(value):
    if type(value) is not int or not 0 <= value < 2**53:
        raise ProtocolError("expected exact nonnegative integer below 2^53")
    return value


def boolean(value, nullable=False):
    if type(value) is not bool and not (nullable and value is None):
        raise ProtocolError("expected boolean")


def decode_message(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_JSON:
        raise ProtocolError("expected bounded UTF-8 JSON text")

    def pairs(items):
        result = {}
        for key, val in items:
            if key in result:
                raise ProtocolError("duplicate JSON field")
            result[key] = val
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("expected object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ProtocolError("unsupported schema")
    if not isinstance(value.get("session_id"), str):
        raise ProtocolError("missing session")
    return value


def validate_pose(pose, joint=False):
    fields(
        pose,
        {
            "position_m",
            "quaternion_wxyz",
            "position_valid",
            "orientation_valid",
            "position_tracked",
            "orientation_tracked",
            "confidence",
        }
        | ({"name"} if joint else set()),
    )
    for key in ("position_valid", "orientation_valid"):
        boolean(pose[key])
    for key in ("position_tracked", "orientation_tracked"):
        boolean(pose[key], True)
    confidence = pose["confidence"]
    if confidence is not None and not 0 <= number(confidence) <= 1:
        raise ProtocolError("confidence outside [0,1]")
    for key, valid, size in (
        ("position_m", "position_valid", 3),
        ("quaternion_wxyz", "orientation_valid", 4),
    ):
        vector = pose[key]
        if not pose[valid]:
            if vector is not None:
                raise ProtocolError("invalid component must be null")
        elif not isinstance(vector, list) or len(vector) != size:
            raise ProtocolError("invalid pose vector")
        else:
            for val in vector:
                number(val)
            if size == 4 and abs(sum(v * v for v in vector) - 1) > 0.002001:
                raise ProtocolError("quaternion must be unit length")


def validate_tracking(message):
    fields(
        message,
        BASE
        | {
            "sequence",
            "calibration_generation",
            "sent_monotonic_s",
            "frame",
            "worn",
            "focused",
            "head",
            "left_hand",
            "right_hand",
            "body",
        },
    )
    integer(message["sequence"])
    integer(message["calibration_generation"])
    number(message["sent_monotonic_s"])
    if message["frame"] != "operator_forward_left_up":
        raise ProtocolError("unsupported frame")
    boolean(message["worn"])
    boolean(message["focused"])
    common = {"source_time_s", "source_clock", "timestamp_kind", "observed_monotonic_s", "fresh"}
    for name in ("head", "left_hand", "right_hand", "body"):
        source = message[name]
        required = common | ({"pose"} if name != "body" else {"confidence"})
        if name != "head":
            required |= {"skeleton_profile", "joints"}
        fields(source, required, {"state_advanced"} if name == "body" else ())
        observed = number(source["observed_monotonic_s"])
        if observed > message["sent_monotonic_s"]:
            raise ProtocolError("observation follows send time")
        boolean(source["fresh"], True)
        kind = source["timestamp_kind"]
        if not isinstance(kind, str) or kind not in {
            "sensor_sample",
            "state_target",
            "predicted_target",
            "unavailable",
        }:
            raise ProtocolError("unknown timestamp kind")
        if kind == "unavailable":
            if source["source_time_s"] is not None or source["source_clock"] != "unavailable":
                raise ProtocolError("unavailable timestamp must be null")
        elif source["source_clock"] != "quest_runtime" or number(source["source_time_s"]) < 0:
            raise ProtocolError("invalid runtime timestamp")
        if kind != "sensor_sample" and source["fresh"] is not None:
            raise ProtocolError("state/prediction timestamps cannot assert freshness")
        if name != "body":
            validate_pose(source["pose"])
        else:
            if source["confidence"] is not None and not 0 <= number(source["confidence"]) <= 1:
                raise ProtocolError("invalid body confidence")
            if "state_advanced" in source:
                boolean(source["state_advanced"])
        if name != "head":
            profile = "ovr_body_upper_18" if name == "body" else "ovr_hand_24"
            if source["skeleton_profile"] != profile:
                raise ProtocolError("unsupported skeleton profile")
            joints = source["joints"]
            if not isinstance(joints, list) or len(joints) != len(PROFILES[profile]):
                raise ProtocolError("incomplete skeleton")
            for joint, expected in zip(joints, PROFILES[profile], strict=True):
                validate_pose(joint, joint=True)
                if joint["name"] != expected:
                    raise ProtocolError("wrong joint names/order")


async def invoke(callback, *args):
    if callback is not None:
        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
    return None


@dataclass
class ClockEstimate:
    offset_s: float
    uncertainty_s: float
    established_monotonic_s: float


@dataclass
class Session:
    session_id: str
    connection: Any
    calibration_generation: int = 0
    latest_tracking: dict | None = None
    last_tracking_sequence: int = -1
    clock: ClockEstimate | None = None
    video_connection: Any = None
    observations: dict = field(default_factory=dict)
    source_highwater: dict = field(default_factory=dict)
    exchanges: OrderedDict = field(default_factory=OrderedDict)
    acknowledgements: OrderedDict = field(default_factory=OrderedDict)
    last_event_id: int = -1
    last_exchange_id: int = -1
    active: bool = True
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, message):
        """Reliable status/event send, independent of latest_tracking slot."""
        payload = {"schema_version": 1, "session_id": self.session_id, **message}
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_JSON:
            raise ProtocolError("outbound JSON exceeds budget")
        async with self._send_lock:
            await asyncio.wait_for(self.connection.send(encoded), 2)

    def observation_ages(self, now=None):
        now = time.monotonic() if now is None else now
        if self.clock is None or now - self.clock.established_monotonic_s > 10:
            return {name: None for name in ("head", "left_hand", "right_hand", "body")}
        return {
            name: now - observed + self.clock.uncertainty_s
            for name, observed in self.observations.items()
        }

    def set_calibration_generation(self, generation):
        integer(generation)
        self.calibration_generation = generation
        self.latest_tracking = None
        self.observations.clear()
        self.source_highwater.clear()


class SessionRegistry:
    """Event-loop-owned session registry shared with optional video route."""

    def __init__(self):
        self.sessions = {}
        self._seen = set()
        self._disconnect_callbacks = []

    def get(self, session_id):
        return self.sessions.get(session_id)

    def add_disconnect_callback(self, callback):
        self._disconnect_callbacks.append(callback)

    def create(self, session_id, connection):
        try:
            if str(UUID(session_id)) != session_id:
                raise ValueError()
        except ValueError as exc:
            raise ProtocolError("session_id must be canonical UUID") from exc
        if session_id in self._seen:
            raise ProtocolError("fresh UUID required on reconnect")
        # Bound replay protection; fail closed instead of forgetting old IDs.
        if len(self._seen) >= 10000:
            raise ProtocolError("session lifetime budget exhausted; restart host")
        self._seen.add(session_id)
        session = Session(session_id, connection)
        self.sessions[session_id] = session
        return session

    def bind_video(self, session_id, connection):
        session = self.get(session_id)
        if session is None or not session.active or session.video_connection is not None:
            raise ProtocolError("orphan or duplicate video binding")
        session.video_connection = connection
        return session

    def unbind_video(self, session_id, connection):
        session = self.get(session_id)
        if session is not None and session.video_connection is connection:
            session.video_connection = None

    async def disconnect(self, session):
        session.active = False
        self.sessions.pop(session.session_id, None)
        session.latest_tracking = None
        session.observations.clear()
        if session.video_connection is not None:
            await session.video_connection.close(code=1001, reason="control disconnected")
            session.video_connection = None
        for callback in self._disconnect_callbacks:
            await invoke(callback, session.session_id)


class ControlServer:
    """Real WS host with bounded receive buffers and injectable control/video routes."""

    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        *,
        registry=None,
        extra_routes=None,
        on_control=None,
        on_video_status=None,
        hello_metadata=None,
    ):
        self.host, self.port = host, port
        self.registry = registry if registry is not None else SessionRegistry()
        self.extra_routes = extra_routes or {}
        self.on_control, self.on_video_status = on_control, on_video_status
        self.hello_metadata = hello_metadata or {}
        self._server = None

    @property
    def bound_port(self):
        return self._server.sockets[0].getsockname()[1]

    async def start(self):
        self._server = await serve(
            self._route,
            self.host,
            self.port,
            max_size=MAX_JSON,
            max_queue=4,
            compression=None,
            close_timeout=2,
        )
        return self

    async def close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _route(self, connection):
        path = connection.request.path
        if path in self.extra_routes:
            await self.extra_routes[path](connection, self.registry)
        elif path == "/quest/control":
            await self._control(connection)
        else:
            await connection.close(code=1008, reason="unknown route")

    async def _control(self, connection):
        session = None
        status_task = None
        try:
            hello = decode_message(await asyncio.wait_for(connection.recv(), 5))
            fields(hello, BASE | {"profile"})
            if hello["type"] != "hello" or hello["profile"] != "quest_teleop_v1":
                raise ProtocolError("expected v1 hello/profile")
            session = self.registry.create(hello["session_id"], connection)
            await session.send(
                {
                    "type": "hello_ack",
                    "profile": "quest_teleop_v1",
                    "calibration_generation": 0,
                    "actuation_enabled": False,
                    "baseline": {
                        "main_commit": "c74837384559479ca1b7522e446801e313dd026a",
                        "mabel_revision": "d0181f04025c06e1790b7ab41fe44d602cc2ee5b",
                    },
                    "video_codec": "jpeg",
                    "video_profile": "QTP1",
                    "budgets": {
                        "max_control_json_bytes": MAX_JSON,
                        "required_input_age_limit_s": 0.15,
                        "clock_sync_max_uncertainty_s": 0.025,
                    },
                    **self.hello_metadata,
                }
            )
            status_task = asyncio.create_task(self._status_loop(session))
            async for raw in connection:
                message = decode_message(raw)
                if message["session_id"] != session.session_id:
                    raise ProtocolError("session mismatch")
                await self.dispatch(session, message)
        except (ProtocolError, asyncio.TimeoutError) as exc:
            await connection.close(code=1008, reason=str(exc)[:100])
        except ConnectionClosed:
            pass
        finally:
            if status_task is not None:
                status_task.cancel()
                await asyncio.gather(status_task, return_exceptions=True)
            if session is not None:
                await self.registry.disconnect(session)

    async def _status_loop(self, session):
        try:
            while session.active:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                ages = session.observation_ages(now)
                synchronized = (
                    session.clock is not None
                    and now - session.clock.established_monotonic_s <= 10
                    and session.clock.uncertainty_s <= 0.025
                )
                stale = any(
                    ages.get(name) is None or ages[name] > 0.15
                    for name in ("head", "left_hand", "right_hand", "body")
                )
                reason = (
                    "tracking_unsynchronized"
                    if not synchronized
                    else "required_input_stale"
                    if stale
                    else "controller_unavailable"
                )
                await session.send(
                    {
                        "type": "status",
                        "host_monotonic_s": now,
                        "state": "Paused",
                        "reason": reason,
                        "calibration_generation": session.calibration_generation,
                        "last_tracking_sequence": (
                            session.last_tracking_sequence
                            if session.last_tracking_sequence >= 0
                            else None
                        ),
                        "sim_time_s": None,
                        "command_sequence": None,
                        "video_pair_id": None,
                        "measured_tracking_error": None,
                        "ik_residual": None,
                        "observation_age_s": ages,
                        "clock": None
                        if session.clock is None
                        else {
                            "offset_s": session.clock.offset_s,
                            "uncertainty_s": session.clock.uncertainty_s,
                            "mapping_age_s": now - session.clock.established_monotonic_s,
                        },
                    }
                )
        except (ConnectionClosed, asyncio.TimeoutError):
            await session.connection.close(code=1001, reason="status consumer unavailable")

    async def dispatch(self, session, message):
        kind = message.get("type")
        if kind == "tracking":
            validate_tracking(message)
            if message["sequence"] <= session.last_tracking_sequence:
                return
            session.last_tracking_sequence = message["sequence"]
            if message["calibration_generation"] != session.calibration_generation:
                return
            now = time.monotonic()
            estimate = session.clock
            if estimate is None or now - estimate.established_monotonic_s > 10:
                return
            if estimate.uncertainty_s > 0.025:
                return
            for name in ("head", "left_hand", "right_hand", "body"):
                source = message[name]
                mapped = source["observed_monotonic_s"] + estimate.offset_s
                if mapped > now + estimate.uncertainty_s:
                    raise ProtocolError("future observation")
                valid = (
                    source["pose"]["position_valid"] and source["pose"]["orientation_valid"]
                    if name != "body"
                    else any(
                        j["position_valid"] and j["orientation_valid"] for j in source["joints"]
                    )
                )
                if not valid or now - mapped + estimate.uncertainty_s > 0.15:
                    continue
                if name in ("left_hand", "right_hand"):
                    stamp = source["source_time_s"]
                    if (
                        source["timestamp_kind"] != "sensor_sample"
                        or stamp is None
                        or stamp <= session.source_highwater.get(name, 0)
                    ):
                        continue
                    session.source_highwater[name] = stamp
                if mapped > session.observations.get(name, -math.inf):
                    session.observations[name] = mapped
            session.latest_tracking = message
        elif kind == "clock_ping":
            fields(message, BASE | {"exchange_id", "t0"})
            exchange_id = integer(message["exchange_id"])
            t0 = number(message["t0"])
            if exchange_id <= session.last_exchange_id:
                raise ProtocolError("old or duplicate clock exchange")
            session.last_exchange_id = exchange_id
            t1 = time.monotonic()
            t2 = time.monotonic()
            session.exchanges[exchange_id] = (t0, t1, t2)
            if len(session.exchanges) > 8:
                session.exchanges.popitem(last=False)
            await session.send(
                {"type": "clock_pong", "exchange_id": exchange_id, "t0": t0, "t1": t1, "t2": t2}
            )
        elif kind == "clock_result":
            fields(message, BASE | {"exchange_id", "t0", "t1", "t2", "t3"})
            exchange_id = integer(message["exchange_id"])
            times = tuple(number(message[k]) for k in ("t0", "t1", "t2", "t3"))
            saved = session.exchanges.pop(exchange_id, None)
            if saved is None or saved != times[:3]:
                raise ProtocolError("unknown or altered clock exchange")
            t0, t1, t2, t3 = times
            rtt = (t3 - t0) - (t2 - t1)
            if t3 < t0 or rtt < 0 or rtt > 1 or time.monotonic() - t1 > 2:
                raise ProtocolError("invalid clock round trip")
            session.clock = ClockEstimate(((t1 - t0) + (t2 - t3)) / 2, rtt / 2, time.monotonic())
            await session.send(
                {
                    "type": "clock_ack",
                    "exchange_id": exchange_id,
                    "offset_s": session.clock.offset_s,
                    "uncertainty_s": session.clock.uncertainty_s,
                }
            )
        elif kind == "control":
            fields(message, BASE | {"event_id", "calibration_generation", "action"})
            event_id = integer(message["event_id"])
            generation = integer(message["calibration_generation"])
            action = message["action"]
            if not isinstance(action, str) or action not in {
                "calibrate",
                "engage",
                "pause",
                "resume",
                "reset",
                "reference_space_changed",
            }:
                raise ProtocolError("unknown control action")
            cached = session.acknowledgements.get(event_id)
            if cached is not None:
                original, ack = cached
                if original != message:
                    raise ProtocolError("event ID reused for different action")
                await session.send(ack)
                return
            if event_id <= session.last_event_id:
                raise ProtocolError("expired or reordered event ID")
            session.last_event_id = event_id
            allowed = generation == session.calibration_generation or action in {
                "pause",
                "reference_space_changed",
            }
            if action == "reference_space_changed":
                session.set_calibration_generation(session.calibration_generation + 1)
            result = await invoke(self.on_control, session, message) if allowed else None
            ack = {
                "type": "control_ack",
                "event_id": event_id,
                "calibration_generation": session.calibration_generation,
                "accepted": result is not None,
                "reason": "controller_unavailable"
                if allowed
                else "calibration_generation_mismatch",
            }
            if result is not None:
                ack.update(result)
            session.acknowledgements[event_id] = (dict(message), ack)
            if len(session.acknowledgements) > 64:
                session.acknowledgements.popitem(last=False)
            await session.send(ack)
        elif kind == "video_status":
            fields(
                message,
                BASE
                | {
                    "sequence",
                    "calibration_generation",
                    "presented_pair_id",
                    "presented_monotonic_s",
                    "reported_monotonic_s",
                    "stale",
                },
            )
            integer(message["sequence"])
            integer(message["calibration_generation"])
            number(message["reported_monotonic_s"])
            boolean(message["stale"])
            if message["presented_pair_id"] is not None:
                integer(message["presented_pair_id"])
                number(message["presented_monotonic_s"])
            elif message["presented_monotonic_s"] is not None or not message["stale"]:
                raise ProtocolError("no pair must mean stale with null presentation")
            await invoke(self.on_video_status, session, message)
        else:
            raise ProtocolError("unknown message type")
