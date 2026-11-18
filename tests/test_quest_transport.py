"""Real loopback transport and hostile canonical input checks."""

import asyncio
import copy
import json
import time
from pathlib import Path
from uuid import uuid4

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from abstract_stack.teleop.transport import (
    ClockEstimate,
    ControlServer,
    ProtocolError,
    Session,
    SessionRegistry,
    decode_message,
    validate_tracking,
)

FIXTURE = Path(__file__).parents[1] / "docs/quest_teleop/examples/tracking.json"


async def receive_kind(ws, kind):
    while True:
        message = json.loads(await ws.recv())
        if message["type"] == kind:
            return message


async def receive_until_closed(ws):
    while True:
        await ws.recv()


def tracking(session=None):
    message = json.loads(FIXTURE.read_text())
    if session:
        message["session_id"] = session.session_id
        message["calibration_generation"] = session.calibration_generation
    now = time.monotonic()
    message["sent_monotonic_s"] = now
    for name in ("head", "body", "left_hand", "right_hand"):
        message[name]["observed_monotonic_s"] = now - 0.001
    return message


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(extra=True),
        lambda m: m.update(sequence=True),
        lambda m: m["head"]["pose"].update(position_m=[float("nan"), 0, 0]),
        lambda m: m["head"]["pose"].update(quaternion_wxyz=[2, 0, 0, 0]),
        lambda m: m["head"]["pose"].update(position_valid=False),
        lambda m: m["left_hand"].update(fresh=True, timestamp_kind="state_target"),
        lambda m: m["left_hand"]["joints"][0].update(name="unknown"),
        lambda m: m["body"].update(skeleton_profile="ovr_body_full"),
        lambda m: m["right_hand"]["joints"].pop(),
    ],
)
def test_strict_tracking(mutation):
    message = tracking()
    validate_tracking(message)
    mutation(message)
    with pytest.raises(ProtocolError):
        validate_tracking(message)


@pytest.mark.parametrize(
    "raw", ['{"schema_version":1,"schema_version":1}', "{}", "[]", b"{}", " " * 262145]
)
def test_decode_rejects(raw):
    with pytest.raises(ProtocolError):
        decode_message(raw)


class Sink:
    def __init__(self):
        self.messages = []
        self.closed = False

    async def send(self, payload):
        self.messages.append(json.loads(payload))

    async def close(self, **kwargs):
        self.closed = True


def test_latest_only_and_nonadvancing_samples():
    async def run():
        server = ControlServer()
        session = Session(str(uuid4()), Sink())
        session.clock = ClockEstimate(0, 0.00001, time.monotonic())
        first = tracking(session)
        await server.dispatch(session, first)
        initial = session.observations["left_hand"]
        second = tracking(session)
        second["sequence"] += 1
        await server.dispatch(session, second)
        assert session.latest_tracking is second
        assert session.observations["left_hand"] == initial
        third = copy.deepcopy(second)
        third["sequence"] += 1
        third["left_hand"]["source_time_s"] += 1
        await server.dispatch(session, third)
        assert session.observations["left_hand"] > initial
        await server.dispatch(session, first)
        assert session.latest_tracking is third
        session.set_calibration_generation(1)
        assert session.latest_tracking is None and not session.observations
        assert not session.observation_ages()

    asyncio.run(run())


def test_reliable_control_idempotence_and_recenter():
    async def run():
        calls = []
        server = ControlServer(on_control=lambda s, m: calls.append(m) or {"accepted": True})
        sink = Sink()
        session = Session(str(uuid4()), sink)
        message = {
            "schema_version": 1,
            "type": "control",
            "session_id": session.session_id,
            "event_id": 0,
            "calibration_generation": 0,
            "action": "reference_space_changed",
        }
        await server.dispatch(session, message)
        await server.dispatch(session, message)
        assert len(calls) == 1
        assert session.calibration_generation == 1
        assert sink.messages[0] == sink.messages[1]
        with pytest.raises(ProtocolError):
            await server.dispatch(session, {**message, "action": "reset"})

    asyncio.run(run())


def test_real_websocket_clock_tracking_reconnect_and_video_lifecycle():
    async def run():
        registry = SessionRegistry()
        closed = []
        registry.add_disconnect_callback(lambda sid: closed.append(sid))
        server = await ControlServer(port=0, registry=registry).start()
        uri = f"ws://127.0.0.1:{server.bound_port}/quest/control"
        sid = str(uuid4())
        base = {"schema_version": 1, "session_id": sid}
        try:
            async with connect(uri) as ws:
                await ws.send(json.dumps({**base, "type": "hello", "profile": "quest_teleop_v1"}))
                assert (await receive_kind(ws, "hello_ack"))["profile"] == "quest_teleop_v1"
                session = registry.get(sid)
                video = Sink()
                registry.bind_video(sid, video)
                with pytest.raises(ProtocolError):
                    registry.bind_video(sid, Sink())
                await ws.send(
                    json.dumps(
                        {**base, "type": "clock_ping", "exchange_id": 1, "t0": time.monotonic()}
                    )
                )
                pong = await receive_kind(ws, "clock_pong")
                await ws.send(
                    json.dumps(
                        {
                            **base,
                            "type": "clock_result",
                            "exchange_id": 1,
                            **{key: pong[key] for key in ("t0", "t1", "t2")},
                            "t3": time.monotonic(),
                        }
                    )
                )
                assert (await receive_kind(ws, "clock_ack"))["uncertainty_s"] < 0.025
                message = tracking(session)
                await ws.send(json.dumps(message))
                await ws.send(
                    json.dumps(
                        {
                            **base,
                            "type": "control",
                            "event_id": 1,
                            "calibration_generation": 0,
                            "action": "engage",
                        }
                    )
                )
                assert (await receive_kind(ws, "control_ack"))["accepted"] is False
                assert session.latest_tracking["sequence"] == 42
            for _ in range(100):
                if registry.get(sid) is None:
                    break
                await asyncio.sleep(0.001)
            assert video.closed and closed == [sid]
            async with connect(uri) as ws:
                await ws.send(json.dumps({**base, "type": "hello", "profile": "quest_teleop_v1"}))
                with pytest.raises(ConnectionClosed):
                    await receive_until_closed(ws)
        finally:
            await server.close()

    asyncio.run(run())


def test_altered_clock_and_future_observation_fail():
    async def run():
        server = ControlServer()
        session = Session(str(uuid4()), Sink())
        base = {"schema_version": 1, "session_id": session.session_id}
        await server.dispatch(session, {**base, "type": "clock_ping", "exchange_id": 1, "t0": 1})
        with pytest.raises(ProtocolError):
            await server.dispatch(
                session,
                {
                    **base,
                    "type": "clock_result",
                    "exchange_id": 1,
                    "t0": 1,
                    "t1": 0,
                    "t2": 0,
                    "t3": 2,
                },
            )
        session.clock = ClockEstimate(10, 0.001, time.monotonic())
        with pytest.raises(ProtocolError):
            await server.dispatch(session, tracking(session))

    asyncio.run(run())


def test_clock_exchange_budget_and_old_replay():
    async def run():
        server = ControlServer()
        session = Session(str(uuid4()), Sink())
        base = {"schema_version": 1, "session_id": session.session_id, "type": "clock_ping"}
        for exchange_id in range(20):
            await server.dispatch(
                session, {**base, "exchange_id": exchange_id, "t0": time.monotonic()}
            )
        assert len(session.exchanges) == 8
        with pytest.raises(ProtocolError):
            await server.dispatch(session, {**base, "exchange_id": 0, "t0": time.monotonic()})

    asyncio.run(run())


def test_stale_unsynchronized_generation_and_sample_regression():
    async def run():
        server = ControlServer()
        session = Session(str(uuid4()), Sink())
        message = tracking(session)
        await server.dispatch(session, message)
        assert session.latest_tracking is None
        session.clock = ClockEstimate(0, 0.001, time.monotonic())
        message["sequence"] += 1
        message["calibration_generation"] = 99
        await server.dispatch(session, message)
        assert session.latest_tracking is None
        message["sequence"] += 1
        message["calibration_generation"] = 0
        for name in ("head", "left_hand", "right_hand", "body"):
            message[name]["observed_monotonic_s"] -= 1
        await server.dispatch(session, message)
        assert not session.observations
        valid = tracking(session)
        valid["sequence"] = 100
        await server.dispatch(session, valid)
        original = dict(session.observations)
        regressed = tracking(session)
        regressed["sequence"] = 101
        for name in ("left_hand", "right_hand"):
            regressed[name]["source_time_s"] -= 1
        await server.dispatch(session, regressed)
        assert session.observations["left_hand"] == original["left_hand"]
        session.clock.established_monotonic_s -= 11
        assert all(age is None for age in session.observation_ages().values())

    asyncio.run(run())


def test_fragmented_and_binary_unknown_oversized_wire_inputs():
    async def run():
        server = await ControlServer(port=0).start()
        uri = f"ws://127.0.0.1:{server.bound_port}/quest/control"
        try:
            for bad in (
                b"binary",
                "{" + " " * 262144,
                '{"schema_version":2}',
                json.dumps({"schema_version": 1, "type": "surprise", "session_id": "x"}),
            ):
                async with connect(uri) as ws:
                    sid = str(uuid4())
                    hello = json.dumps(
                        {
                            "schema_version": 1,
                            "type": "hello",
                            "session_id": sid,
                            "profile": "quest_teleop_v1",
                        }
                    )
                    await ws.send([hello[:10], hello[10:]])
                    assert (await receive_kind(ws, "hello_ack"))["profile"] == "quest_teleop_v1"
                    await ws.send(bad)
                    with pytest.raises(ConnectionClosed):
                        await receive_until_closed(ws)
        finally:
            await server.close()

    asyncio.run(run())


def test_wire_overload_latest_slot_and_independent_status():
    async def run():
        server = await ControlServer(port=0).start()
        uri = f"ws://127.0.0.1:{server.bound_port}/quest/control"
        sid = str(uuid4())
        try:
            async with connect(uri, max_queue=4) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "type": "hello",
                            "session_id": sid,
                            "profile": "quest_teleop_v1",
                        }
                    )
                )
                await receive_kind(ws, "hello_ack")
                session = server.registry.get(sid)
                session.clock = ClockEstimate(0, 0.00001, time.monotonic())
                for sequence in range(200):
                    message = tracking(session)
                    message["sequence"] = sequence
                    await ws.send(json.dumps(message))
                await ws.send(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "type": "control",
                            "session_id": sid,
                            "event_id": 0,
                            "calibration_generation": 0,
                            "action": "pause",
                        }
                    )
                )
                await receive_kind(ws, "control_ack")
                assert session.latest_tracking["sequence"] == 199
                assert len(session.source_highwater) == 2
                status = await receive_kind(ws, "status")
                assert status["state"] == "Paused" and status["command_sequence"] is None
                assert status["clock"]["uncertainty_s"] < 0.025
        finally:
            await server.close()

    asyncio.run(run())


def test_video_feedback_validation_and_separate_callback():
    async def run():
        callbacks = []
        server = ControlServer(on_video_status=lambda session, message: callbacks.append(message))
        session = Session(str(uuid4()), Sink())
        feedback = {
            "schema_version": 1,
            "type": "video_status",
            "session_id": session.session_id,
            "sequence": 0,
            "calibration_generation": 0,
            "presented_pair_id": None,
            "presented_monotonic_s": None,
            "reported_monotonic_s": 0,
            "stale": True,
        }
        await server.dispatch(session, feedback)
        assert callbacks == [feedback] and session.latest_tracking is None
        for mutation in (
            {"presented_monotonic_s": 1},
            {"stale": False},
            {"sequence": True},
            {"reported_monotonic_s": float("nan")},
            {"extra": 1},
        ):
            with pytest.raises(ProtocolError):
                await server.dispatch(session, {**feedback, **mutation})

    asyncio.run(run())


def test_non_numeric_timestamps_and_unknown_action_types():
    async def run():
        server = ControlServer()
        session = Session(str(uuid4()), Sink())
        base = {
            "schema_version": 1,
            "type": "control",
            "session_id": session.session_id,
            "event_id": 0,
            "calibration_generation": 0,
        }
        for action in ([], {}, None, "unknown"):
            with pytest.raises(ProtocolError):
                await server.dispatch(session, {**base, "action": action})
        message = tracking(session)
        message["body"]["observed_monotonic_s"] = 10**1000
        with pytest.raises(ProtocolError):
            validate_tracking(message)

    asyncio.run(run())
