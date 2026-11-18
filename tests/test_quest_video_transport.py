"""Live #15/#21 seam; runs when the independent transport module is present."""

import asyncio
import json
import time
import uuid

import pytest

from abstract_stack.teleop.video import (
    LatestSlot,
    VideoService,
    encode_pair,
    synthetic_pair,
    unpack_pair,
)

transport = pytest.importorskip(
    "abstract_stack.teleop.transport", reason="requires issue #15 module"
)
client = pytest.importorskip("websockets.asyncio.client")
exceptions = pytest.importorskip("websockets.exceptions")


async def receive_kind(socket, kind):
    while True:
        message = json.loads(await asyncio.wait_for(socket.recv(), 2))
        if message["type"] == kind:
            return message


def test_actual_websocket_pair_feedback_generation_and_control_disconnect():
    async def run():
        slot = LatestSlot()
        service = VideoService(slot)
        registry = transport.SessionRegistry()
        registry.add_disconnect_callback(service.disconnect)
        server = transport.ControlServer(
            port=0,
            registry=registry,
            extra_routes={"/quest/video": service.handle},
            on_video_status=service.on_video_status,
        )
        await server.start()
        try:
            sid = str(uuid.uuid4())
            root = f"ws://127.0.0.1:{server.bound_port}"
            async with client.connect(root + "/quest/control") as control:

                def envelope(kind, **kw):
                    return json.dumps(dict(schema_version=1, session_id=sid, type=kind, **kw))

                await control.send(envelope("hello", profile="quest_teleop_v1"))
                await receive_kind(control, "hello_ack")
                t0 = time.monotonic() - 5
                await control.send(envelope("clock_ping", exchange_id=0, t0=t0))
                pong = await receive_kind(control, "clock_pong")
                t3 = time.monotonic() - 5
                await control.send(
                    envelope(
                        "clock_result", exchange_id=0, t0=t0, t1=pong["t1"], t2=pong["t2"], t3=t3
                    )
                )
                await receive_kind(control, "clock_ack")
                async with client.connect(root + "/quest/video", max_size=5_000_000) as video:
                    await video.send(envelope("video_bind"))
                    await receive_kind(video, "video_bind_ack")
                    slot.put(encode_pair(synthetic_pair(3, width=160, height=120)))
                    metadata, _, _ = unpack_pair(
                        await asyncio.wait_for(video.recv(), 2), decode=True
                    )
                    assert metadata["pair_id"] == 3 and metadata["session_id"] == sid
                    await control.send(
                        envelope(
                            "video_status",
                            sequence=0,
                            calibration_generation=0,
                            presented_pair_id=3,
                            presented_monotonic_s=time.monotonic() - 5,
                            reported_monotonic_s=time.monotonic() - 5,
                            stale=False,
                        )
                    )
                    await asyncio.sleep(0.02)
                    session = registry.get(sid)
                    assert service.health(session).fresh
                    # Reference changes invalidate presentation immediately; the
                    # already encoded image is bound to the new generation at send.
                    session.set_calibration_generation(1)
                    assert not service.health(session).fresh
                    slot.put(encode_pair(synthetic_pair(4, width=160, height=120)))
                    newer, _, _ = unpack_pair(await asyncio.wait_for(video.recv(), 2))
                    assert newer["calibration_generation"] == 1
                    await control.close()
                    with pytest.raises(exceptions.ConnectionClosed):
                        await asyncio.wait_for(video.recv(), 2)
        finally:
            await server.close()
        assert not service._sent and not service._feedback

    asyncio.run(run())


def test_actual_websocket_rejects_orphan_video_and_does_not_create_session():
    async def run():
        service = VideoService(LatestSlot())
        registry = transport.SessionRegistry()
        server = transport.ControlServer(
            port=0, registry=registry, extra_routes={"/quest/video": service.handle}
        )
        await server.start()
        try:
            async with client.connect(f"ws://127.0.0.1:{server.bound_port}/quest/video") as video:
                await video.send(
                    json.dumps(dict(schema_version=1, type="video_bind", session_id="orphan"))
                )
                with pytest.raises(exceptions.ConnectionClosed) as error:
                    await video.recv()
                assert error.value.rcvd.code == 1008
                assert registry.get("orphan") is None
        finally:
            await server.close()

    asyncio.run(run())
