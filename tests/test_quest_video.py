"""Paired framing, bounded work, scratch replay and feedback-age regressions."""

import asyncio
import json
import struct
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.sim import MujocoTeleopAdapter
from abstract_stack.teleop.video import (
    HEADER,
    MAX_EYE,
    LatestSlot,
    RenderSnapshot,
    ScratchStereoRenderer,
    VideoService,
    VideoWorker,
    encode_pair,
    synthetic_pair,
    unpack_pair,
)


def test_synthetic_pair_has_order_identity_motion_and_atomic_roundtrip():
    pair = synthetic_pair(123, width=320, height=240, captured=42.5)
    metadata, left, right = unpack_pair(encode_pair(pair).packet("test-session", 7), decode=True)
    assert metadata["pair_id"] == 123
    assert metadata["capture_host_monotonic_s"] == 42.5
    assert metadata["calibration_generation"] == 7
    assert metadata["eye_order"] == ["left", "right"]
    assert left.shape == right.shape == (240, 320, 3)
    assert left.dtype == right.dtype == np.uint8
    assert not np.array_equal(left, right)
    assert not np.array_equal(pair.left.rgb, synthetic_pair(133, width=320, height=240).left.rgb)


@pytest.fixture
def packet():
    return encode_pair(synthetic_pair(0, width=160, height=120)).packet("test", 0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b[:15],
        lambda b: b + b"x",
        lambda b: b[:-1],
        lambda b: b"FAIL" + b[4:],
        lambda b: b[:8] + struct.pack(">I", MAX_EYE + 1) + b[12:],
        lambda b: b[:4] + struct.pack(">I", 0) + b[8:],
    ],
)
def test_invalid_framing_is_rejected(packet, mutation):
    with pytest.raises(ValueError):
        unpack_pair(mutation(packet))


def rewrite_metadata(packet, change):
    magic, size, left, right = HEADER.unpack_from(packet)
    metadata = json.loads(packet[16 : 16 + size])
    change(metadata)
    raw = json.dumps(metadata).encode()
    return HEADER.pack(magic, len(raw), left, right) + raw + packet[16 + size :]


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 2049),
        ("width", True),
        ("width", 161),
        ("schema_version", True),
        ("pair_id", -1),
        ("pair_id", 2**53),
        ("sim_time_s", float("nan")),
        ("eye_order", ["right", "left"]),
        ("codec", "h264"),
        ("extra", 1),
    ],
)
def test_bad_metadata_or_jpeg_dimensions_are_rejected(packet, field, value):
    with pytest.raises(ValueError):
        unpack_pair(rewrite_metadata(packet, lambda m: m.update({field: value})))


def test_bad_jpeg_and_duplicate_json_are_rejected(packet):
    _, size, left, _ = HEADER.unpack_from(packet)
    broken = packet[: 16 + size] + bytes(left) + packet[16 + size + left :]
    with pytest.raises((ValueError, OSError)):
        unpack_pair(broken)
    raw = packet[16 : 16 + size]
    duplicate = b'{"pair_id":999,' + raw[1:]
    _, _, ls, rs = HEADER.unpack_from(packet)
    with pytest.raises(ValueError, match="Duplicate"):
        unpack_pair(HEADER.pack(b"QTP1", len(duplicate), ls, rs) + duplicate + packet[16 + size :])


def test_encoder_rejects_mixed_eyes_bad_quality_and_nonfinite_geometry():
    pair = synthetic_pair(0, width=160, height=120)
    with pytest.raises(ValueError):
        encode_pair(replace(pair, right=replace(pair.right, name="zed_left")))
    with pytest.raises(ValueError):
        encode_pair(pair, quality=True)
    with pytest.raises(ValueError):
        encode_pair(replace(pair, left=replace(pair.left, position_m=np.full(3, np.nan))))


def test_latest_slot_retains_one_value_with_no_backlog():
    slot = LatestSlot()
    for i in range(10000):
        slot.put(i)
    version, value = slot.read()
    assert version == 10000 and value == 9999 and slot.replaced == 9999
    assert slot.read(version)[1] is None
    slot.clear()
    assert slot.read()[1] is None


@pytest.fixture
def sim():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets unavailable")
    with MujocoTeleopAdapter(ShelfConfig(product_count=0)) as adapter:
        yield adapter


@pytest.mark.mabel
def test_slow_render_does_not_block_live_sim_and_reset_discards_pair(sim):
    started, release = threading.Event(), threading.Event()

    class SlowRenderer:
        def render(self, snapshot, pair_id):
            started.set()
            assert release.wait(5)
            return replace(
                synthetic_pair(pair_id, width=160, height=120),
                reset_generation=snapshot.state.reset_generation,
            )

        def close(self):
            pass

    worker = VideoWorker(SlowRenderer)
    worker.start()
    initial = sim.read_state()
    worker.submit(initial)
    try:
        assert started.wait(5)
        advanced = sim.step()
        for _ in range(5000):
            worker.submit(advanced)
        assert worker.snapshots.read()[1].state.step_index == advanced.step_index
        assert advanced.sim_time_s > initial.sim_time_s
        worker.submit(sim.reset())
        worker._stop.set()
        release.set()
        worker.close()
        assert worker.encoded.read()[1] is None
    finally:
        release.set()
        worker.close()


@pytest.mark.mabel
@pytest.mark.rendering
def test_scratch_replay_matches_live_pair_without_mutating_or_advancing_live_sim(sim):
    state = sim.step()
    provenance = sim.describe()
    scratch = ScratchStereoRenderer(provenance, width=160, height=120)
    snapshot = RenderSnapshot.capture(state)
    state.qpos[:] = 999
    try:
        replay = scratch.render(snapshot, 11)
        live = sim.render_stereo_pair(width=160, height=120)
        after = sim.read_state()
        assert replay.sim_time_s == live.sim_time_s == after.sim_time_s
        assert replay.capture_host_monotonic_s == snapshot.capture_host_monotonic_s
        for side in ("left", "right"):
            a, b = getattr(replay, side), getattr(live, side)
            np.testing.assert_array_equal(a.rgb, b.rgb)
            np.testing.assert_allclose(a.position_m, b.position_m)
            np.testing.assert_allclose(a.quaternion_wxyz, b.quaternion_wxyz)
        np.testing.assert_array_equal(after.qpos, snapshot.state.qpos)
        assert np.linalg.norm(replay.right.position_m - replay.left.position_m) == pytest.approx(
            0.063
        )
    finally:
        scratch.close()


@pytest.mark.mabel
def test_scratch_model_mismatch_fails_before_replay(sim):
    provenance = sim.describe()
    provenance["model_xml_sha256"] = "bad"
    with pytest.raises(ValueError, match="mismatch"):
        ScratchStereoRenderer(provenance)


def session(now):
    return SimpleNamespace(
        session_id="session",
        calibration_generation=0,
        clock=SimpleNamespace(offset_s=5, uncertainty_s=0.001, established_monotonic_s=now),
    )


def feedback(now, *, sequence=0, pair_id=3, generation=0):
    return dict(
        sequence=sequence,
        calibration_generation=generation,
        stale=False,
        presented_pair_id=pair_id,
        presented_monotonic_s=now - 5 - 0.02,
        reported_monotonic_s=now - 5 - 0.01,
    )


def test_feedback_checks_capture_age_sequence_clock_generation_and_missing_reports(monkeypatch):
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    service = VideoService(LatestSlot())
    client = session(now)
    service._generation["session"] = 0
    service._sent["session"] = {3: (now - 0.05, now - 0.04)}
    asyncio.run(service.on_video_status(client, feedback(now)))
    assert service.health(client).fresh
    assert not service.health(client, now=now + 0.26).fresh
    assert not service.health(client, now=now + 0.31).fresh
    asyncio.run(service.on_video_status(client, dict(feedback(now), stale=True)))
    assert service.health(client).fresh
    now += 0.4
    asyncio.run(service.on_video_status(client, feedback(now, sequence=1)))
    assert not service.health(client).fresh
    client.calibration_generation = 1
    assert not service.health(client).fresh


@pytest.mark.parametrize("fault", ["future", "uncertain", "expired", "unknown", "old_generation"])
def test_untrusted_presentation_is_never_fresh(monkeypatch, fault):
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    client = session(now)
    service = VideoService(LatestSlot())
    service._generation["session"] = 0
    service._sent["session"] = {3: (99.95, 99.96)}
    message = feedback(now)
    if fault == "future":
        message["presented_monotonic_s"] = now + 100
    elif fault == "uncertain":
        client.clock.uncertainty_s = 0.1
    elif fault == "expired":
        client.clock.established_monotonic_s = 0
    elif fault == "unknown":
        message["presented_pair_id"] = 99
    else:
        message["calibration_generation"] = 7
    asyncio.run(service.on_video_status(client, message))
    assert not service.health(client).fresh


def test_slow_video_socket_times_out_with_no_unbounded_send_queue():
    async def check():
        service = VideoService(LatestSlot(), send_timeout=0.02)
        client = session(time.monotonic())
        service.encoded.put(encode_pair(synthetic_pair(0, width=160, height=120)))
        registry = SimpleNamespace(get=lambda sid: client)
        receiver = asyncio.create_task(asyncio.sleep(2))
        calls = []

        async def slow_send(packet):
            calls.append(packet)
            await asyncio.sleep(2)

        try:
            with pytest.raises(TimeoutError):
                await service._send_loop(
                    SimpleNamespace(send=slow_send), registry, client, receiver
                )
            assert len(calls) == 1
            assert service.metrics["send_timeouts"] == 1
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.mabel
@pytest.mark.rendering
def test_production_process_renders_detached_snapshot_while_owner_steps(sim):
    from functools import partial

    from abstract_stack.teleop.video import ProcessVideoWorker

    worker = ProcessVideoWorker(
        partial(ScratchStereoRenderer, sim.describe(), width=160, height=120)
    )
    before = sim.read_state()
    worker.start()
    worker.submit(before)
    try:
        deadline = time.monotonic() + 15
        version, pair = worker.encoded.read()
        while pair is None and time.monotonic() < deadline:
            sim.step()
            time.sleep(0.01)
            version, pair = worker.encoded.read()
            if worker.error:
                raise worker.error
        assert pair is not None
        assert pair.sim_time_s == before.sim_time_s
        assert sim.read_state().sim_time_s > pair.sim_time_s
        metadata, _, _ = unpack_pair(pair.packet("process-test", 0), decode=True)
        assert metadata["width"] == 160 and metadata["eye_order"] == ["left", "right"]
        worker.submit(sim.reset())
        assert worker.encoded.read()[1] is None
    finally:
        worker.close()


@pytest.mark.mabel
def test_production_process_failure_is_reported_not_silently_empty(sim):
    from functools import partial

    from abstract_stack.teleop.video import ProcessVideoWorker

    provenance = dict(sim.describe(), model_xml_sha256="bad")
    worker = ProcessVideoWorker(partial(ScratchStereoRenderer, provenance))
    worker.start()
    deadline = time.monotonic() + 15
    while worker.error is None and time.monotonic() < deadline:
        time.sleep(0.02)
    with pytest.raises(RuntimeError, match="mismatch|exited"):
        worker.close()
    assert worker.encoded.read()[1] is None
