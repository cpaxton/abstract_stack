"""Owner gate, idempotent lifecycle, hostile streams and native pause evidence."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.calibration import Pose, TargetSpec
from abstract_stack.teleop.session import (
    GROUPS,
    CommandProposal,
    EngagementGate,
    GateSettings,
    LifecycleEvidence,
    SessionState,
    VideoEvidence,
)
from abstract_stack.teleop.sim import MujocoTeleopAdapter
from abstract_stack.teleop.transport import ClockEstimate

FIXTURE = Path(__file__).parents[1] / "docs/quest_teleop/examples/tracking.json"
pytestmark = pytest.mark.mabel


@pytest.fixture(scope="module")
def native():
    with MujocoTeleopAdapter(
        ShelfConfig(product_count=0, front_x=-1.4, terminate_on_success=False)
    ) as sim:
        yield sim


class FixtureAdapter:
    """Detached deterministic policy fixture; not native tracking evidence."""

    def __init__(self, sim):
        self.provenance = sim.describe()
        self.current = sim.read_state()
        self.writes = []
        self.resets = 0
        self.control_dt = sim.control_dt

    def describe(self):
        return copy.deepcopy(self.provenance)

    def read_state(self):
        return copy.deepcopy(self.current)

    def apply_actuator_targets(self, targets):
        self.writes.append(targets.copy())
        self.current = replace(self.current, applied_targets=targets.copy())

    def step(self):
        self.current = replace(
            self.current,
            step_index=self.current.step_index + 1,
            sim_time_s=self.current.sim_time_s + self.control_dt,
            actuator_lengths=self.current.applied_targets.copy(),
        )

    def reset(self, seed=None):
        self.resets += 1
        self.current = replace(
            self.current,
            reset_generation=self.current.reset_generation + 1,
            step_index=0,
            sim_time_s=0,
            terminated=False,
            truncated=False,
        )
        return self.read_state()


@pytest.fixture
def gate(native):
    native.reset(seed=0)
    adapter = FixtureAdapter(native)
    # Fixture guard deliberately certifies no geometry. Production must inject
    # #17's real full-command sampled-path guard; integration tests exercise it.
    return EngagementGate(
        adapter,
        settings=GateSettings(enabled_groups=("head",)),
        composed_guard=lambda state, target: {"valid": True, "scope": "fixture"},
    )


def observe(
    gate,
    now,
    *,
    stamp=None,
    qualified=True,
    worn=True,
    focused=True,
    video_age=0.01,
    lifecycle_age=0,
    clock_age=0,
    uncertainty=0.001,
    sequence=None,
    body_valid=True,
):
    packet = json.loads(FIXTURE.read_text())
    packet.update(
        session_id=gate.session_id,
        calibration_generation=gate.generation,
        sequence=gate._last_tracking + 1 if sequence is None else sequence,
        sent_monotonic_s=now,
        worn=worn,
        focused=focused,
    )
    for name in ("head", "left_hand", "right_hand", "body"):
        packet[name]["observed_monotonic_s"] = now
        if name in ("left_hand", "right_hand"):
            packet[name]["source_time_s"] = now if stamp is None else stamp
        if name == "body" and not body_valid:
            for joint in packet[name]["joints"]:
                joint.update(
                    position_valid=False,
                    orientation_valid=False,
                    position_m=None,
                    quaternion_wxyz=None,
                )
    life = LifecycleEvidence(gate.session_id, now - lifecycle_age, True, focused, worn, qualified)
    video = VideoEvidence(
        gate.session_id,
        gate.generation,
        max(0, packet["sequence"]),
        now - video_age,
        now,
        0.001,
        True,
    )
    clock = ClockEstimate(0, uncertainty, now - clock_age)
    gate.observe(packet, clock, life, video, now=now)
    return packet, clock, life, video


def neutral_specs():
    poses = {
        "head": Pose((0, 0, 1.6), (1, 0, 0, 0)),
        "left_hand": Pose((0.3, 0.2, 1), (1, 0, 0, 0)),
        "right_hand": Pose((0.3, -0.2, 1), (1, 0, 0, 0)),
    }
    return poses, {name: TargetSpec(pose) for name, pose in poses.items()}


def control(gate, action, now, *, event=None, generation=None):
    neutral, specs = neutral_specs()
    return gate.control(
        gate.session_id,
        action,
        gate._last_event + 1 if event is None else event,
        gate.generation if generation is None else generation,
        now=now,
        neutral=neutral,
        specs=specs,
    )


def engaged(gate):
    gate.connect(str(uuid4()), now=10)
    observe(gate, 10)
    assert control(gate, "calibrate", 10)["accepted"]
    observe(gate, 10.02)
    assert control(gate, "engage", 10.02)["accepted"]
    assert gate.state == SessionState.ENGAGED


def proposal(gate, now, yaw_delta=0.1):
    refs = dict(zip(gate.names, gate._last_applied, strict=True))
    targets = {name: refs[name] for name in GROUPS["head"]}
    targets["Neck Yaw"] += yaw_delta
    return {
        "head": CommandProposal(gate.session_id, gate.generation, gate._last_tracking, now, targets)
    }


def test_no_actuation_before_qualified_current_calibration_and_deliberate_action(gate):
    gate.connect(str(uuid4()), now=10)
    observe(gate, 10, qualified=False)
    assert control(gate, "calibrate", 10)["reason"] == "removal_detection_unqualified"
    assert not control(gate, "engage", 10)["accepted"]
    observe(gate, 10.02)
    assert control(gate, "calibrate", 10.02)["accepted"]
    assert gate.generation == 1
    assert not control(gate, "engage", 10.02)["accepted"]  # old generation observations cleared
    observe(gate, 10.04)
    assert control(gate, "engage", 10.04)["accepted"]
    assert not gate.sim.writes


def test_rate_limits_complete_atomic_commands_and_reserved_base(gate):
    engaged(gate)
    last = gate._last_applied.copy()
    previous_velocity = np.zeros(len(last))
    for index in range(15):
        now = 10.04 + index * 0.02
        observe(gate, now)
        result = gate.tick(proposal(gate, now), now=now)
        assert result is not None
        assert gate.state == SessionState.ENGAGED
        velocity = (result - last) / gate.sim.control_dt
        assert np.max(np.abs(velocity)) <= 1 + 1e-10
        assert np.max(np.abs(velocity - previous_velocity)) <= 5 * 0.02 + 1e-9
        np.testing.assert_array_equal(result[gate.base_ids], gate.base_reference)
        assert len(gate.sim.writes) == index + 1
        gate.sim.step()
        last, previous_velocity = result, velocity
    assert result[gate.indices["Neck Yaw"]] > gate.sim.writes[0][gate.indices["Neck Yaw"]]


@pytest.mark.parametrize("failure", ["worn", "focus", "clock", "video", "lifecycle", "qualified"])
def test_required_failure_pauses_invalidates_and_never_automatically_resumes(gate, failure):
    engaged(gate)
    generation = gate.generation
    kwargs = {
        "worn": failure != "worn",
        "focused": failure != "focus",
        "clock_age": 11 if failure == "clock" else 0,
        "video_age": 0.4 if failure == "video" else 0.01,
        "lifecycle_age": 0.2 if failure == "lifecycle" else 0,
        "qualified": failure != "qualified",
    }
    observe(gate, 10.04, **kwargs)
    assert gate.state == SessionState.PAUSED
    assert gate.generation == generation + 1
    assert gate.calibration.artifact is None
    observe(gate, 10.06)
    assert gate.state == SessionState.PAUSED
    assert not control(gate, "resume", 10.06)["accepted"]
    assert control(gate, "calibrate", 10.06)["accepted"]
    observe(gate, 10.08)
    assert not control(gate, "engage", 10.08)["accepted"]
    assert control(gate, "resume", 10.08)["accepted"]


def test_repeated_hand_sdk_timestamp_does_not_refresh_observation(gate):
    engaged(gate)
    observe(gate, 10.04, stamp=11)
    for now in (10.08, 10.12, 10.16, 10.20):
        observe(gate, now, stamp=11)
    assert gate.state == SessionState.PAUSED
    assert "hand_observation_stale" in gate.reason


def test_repeated_tracking_sequence_does_not_refresh_lifecycle_or_video(gate):
    engaged(gate)
    for now in (10.06, 10.10, 10.22):
        observe(gate, now, sequence=gate._last_tracking)
    gate.tick(proposal(gate, 10.22), now=10.22)
    assert gate.state == SessionState.PAUSED


def test_expired_input_without_any_new_packets_pauses_on_control_tick(gate):
    engaged(gate)
    gate.tick(proposal(gate, 10.22), now=10.22)
    assert gate.state == SessionState.PAUSED
    assert gate.calibration.artifact is None


def test_optional_body_loss_is_explicit_and_body_mode_requires_it(gate):
    engaged(gate)
    observe(gate, 10.04, body_valid=False)
    assert gate.state == SessionState.ENGAGED
    with pytest.raises(ValueError):
        GateSettings(enabled_groups=("body",))


def test_old_generation_proposal_pauses_without_applying_its_target(gate):
    engaged(gate)
    p = proposal(gate, 10.04)
    p["head"] = replace(p["head"], calibration_generation=0)
    before = gate._last_applied.copy()
    result = gate.tick(p, now=10.04)
    assert gate.state == SessionState.PAUSED
    np.testing.assert_array_equal(result, before)


@pytest.mark.parametrize("fault", ["range", "unknown", "nan", "stale", "sequence", "invalid"])
def test_invalid_proposals_are_rejected_as_a_complete_transaction(gate, fault):
    engaged(gate)
    p = proposal(gate, 10.04)
    item = p["head"]
    if fault == "range":
        p["head"] = replace(item, targets=dict(item.targets, **{"Neck Yaw": 999}))
    elif fault == "unknown":
        p["head"] = replace(item, targets=dict(item.targets, **{"FL Drive": 1}))
    elif fault == "nan":
        p["head"] = replace(item, targets=dict(item.targets, **{"Neck Yaw": np.nan}))
    elif fault == "stale":
        p["head"] = replace(item, observed_host_monotonic_s=1)
    elif fault == "sequence":
        p["head"] = replace(item, tracking_sequence=999)
    else:
        p["head"] = replace(item, valid=False)
    before = gate._last_applied.copy()
    result = gate.tick(p, now=10.04)
    assert gate.state == SessionState.PAUSED
    np.testing.assert_array_equal(result, before)


def test_geometry_guard_sees_final_full_command_including_fingers(gate):
    engaged(gate)
    calls = []

    def reject(state, command):
        calls.append(command)
        return {"valid": False, "reason": "fixture_collision"}

    gate.composed_guard = reject
    assert gate.tick(proposal(gate, 10.04), now=10.04) is None
    assert gate.state == SessionState.FAULT
    assert not gate.sim.writes
    assert len(calls[0]) == 59 and "Left Thumb MCP" in calls[0]
    assert calls[0]["Neck Yaw"] != proposal(gate, 10.04)["head"].targets["Neck Yaw"]


def test_missing_geometry_guard_cannot_engage(gate):
    gate.composed_guard = None
    gate.connect(str(uuid4()), now=10)
    observe(gate, 10)
    control(gate, "calibrate", 10)
    observe(gate, 10.02)
    assert control(gate, "engage", 10.02)["reason"] == "composed_geometry_guard_required"


def test_pause_reference_change_reset_are_idempotent_and_replay_bound(gate):
    engaged(gate)
    old_generation = gate.generation
    result = control(gate, "reference_space_changed", 10.04, event=5, generation=0)
    assert result["accepted"] and gate.generation == old_generation + 1
    repeated = control(gate, "reference_space_changed", 10.04, event=5, generation=0)
    assert repeated == result and gate.generation == old_generation + 1
    with pytest.raises(ValueError):
        control(gate, "reset", 10.04, event=5)
    reset = control(gate, "reset", 10.05, event=6)
    assert reset["accepted"] and gate.sim.resets == 1
    assert (
        control(gate, "reset", 10.06, event=6, generation=result["calibration_generation"]) == reset
    )
    assert gate.sim.resets == 1 and gate.calibration.artifact is None
    assert not control(gate, "engage", 10.06)["accepted"]


def test_disconnect_reconnect_cannot_reuse_calibration_or_uuid(gate):
    engaged(gate)
    sid = gate.session_id
    gate.disconnect(sid, now=10.04)
    assert gate.state == SessionState.DISCONNECTED
    with pytest.raises(ValueError):
        gate.connect(sid, now=10.06)
    gate.connect(str(uuid4()), now=10.06)
    assert gate.calibration.artifact is None and gate.generation == 0
    assert not control(gate, "engage", 10.06)["accepted"]


def test_external_writes_and_duplicate_ticks_fault_without_new_write(gate):
    engaged(gate)
    assert gate.tick(proposal(gate, 10.04), now=10.04) is not None
    count = len(gate.sim.writes)
    assert gate.tick(proposal(gate, 10.06), now=10.06) is None
    assert gate.state == SessionState.FAULT and len(gate.sim.writes) == count


def test_gate_rejects_wrong_thread(gate):
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="owner"):
            pool.submit(gate.host_pause, now=10).result()


def test_pause_telemetry_separates_latched_hold_from_rate_limited_command(gate):
    engaged(gate)
    index = gate.indices["Neck Yaw"]
    measured = gate.sim.current.actuator_lengths.copy()
    measured[index] += 0.1
    gate.sim.current = replace(gate.sim.current, actuator_lengths=measured)
    gate.host_pause(now=10.04)
    applied = gate.tick(now=10.04)
    assert applied is not None and applied[index] < measured[index]
    telemetry = gate.telemetry(now=10.04)
    assert telemetry.nominal_targets["Neck Yaw"] == measured[index]
    assert telemetry.applied_targets["Neck Yaw"] == applied[index]


def test_native_measured_hold_failure_keeps_native_limits_and_last_reference():
    # Reproduce #14's native measured-reference counterexample using ctrl+step,
    # before installing the sole final gate; never edit live poses or limits.
    with MujocoTeleopAdapter(ShelfConfig(product_count=0)) as native:
        fields = (
            "body_mass",
            "body_inertia",
            "geom_contype",
            "geom_conaffinity",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "jnt_range",
            "eq_data",
        )
        original = {name: getattr(native._env.model, name).copy() for name in fields}
        for _ in range(10):
            try:
                targets = native.stationary_command()
            except ValueError:
                break
            native.apply_actuator_targets(targets)
            native.step()
        with pytest.raises(ValueError, match="outside"):
            native.stationary_command()
        last = native.read_state().applied_targets.copy()
        gate = EngagementGate(native, settings=GateSettings(enabled_groups=("head",)))
        gate.connect(str(uuid4()), now=10)
        assert gate.state == SessionState.FAULT
        assert "measured_hold_unavailable" in gate.reason
        np.testing.assert_array_equal(native.read_state().applied_targets, last)
        for name in fields:
            np.testing.assert_array_equal(getattr(native._env.model, name), original[name])
        assert gate.host_reset(now=11)["accepted"]
        assert gate.state == SessionState.PAUSED


def test_independent_removal_pauses_when_pose_sequence_is_frozen(gate):
    engaged(gate)
    observe(gate, 10.04, sequence=gate._last_tracking, worn=False)
    assert gate.state == SessionState.PAUSED
    assert gate.calibration.artifact is None


def test_mailbox_bounds_cancelled_controls_actual_drain_time_and_urgent_disconnect(gate):
    from abstract_stack.teleop.session_bridge import OwnerMailbox

    mailbox = OwnerMailbox(capacity=1)
    sid = str(uuid4())
    cancelled = mailbox.submit("connect", sid)
    assert cancelled.cancel()
    mailbox.drain(gate, now=10)
    assert gate.session_id is None
    connected = mailbox.submit("connect", sid)
    mailbox.drain(gate, now=10)
    assert connected.result() == 0
    observe(gate, 10)
    neutral, specs = neutral_specs()
    pending = mailbox.submit(
        "control", sid, "calibrate", 0, 0, now=10, neutral=neutral, specs=specs
    )
    mailbox.drain(gate, now=11)
    assert not pending.result()["accepted"]  # enqueue time cannot revive input
    first = mailbox.submit("host_pause")
    overflow = mailbox.submit("host_pause")
    with pytest.raises(RuntimeError, match="full"):
        overflow.result()
    mailbox.disconnect(sid)
    mailbox.drain(gate, now=11.02)
    assert first.result() is None
    assert gate.session_id is None
    assert mailbox.status()["state"] == "Disconnected"


def test_mailbox_keeps_one_detached_latest_input(gate):
    from abstract_stack.teleop.session_bridge import OwnerMailbox

    gate.connect(str(uuid4()), now=10)
    mailbox = OwnerMailbox()
    values = observe(gate, 10)
    for _ in range(1000):
        mailbox.observe(*values)
    assert mailbox.replaced_inputs == 999
    values[0]["head"]["pose"]["position_m"][0] = 999
    mailbox.drain(gate, now=10.01)
    assert gate._tracking["head"]["pose"]["position_m"][0] != 999


def test_real_websocket_status_and_actions_use_owner_mailbox(gate):
    import asyncio
    import threading
    import time
    from concurrent.futures import Future

    from websockets.asyncio.client import connect

    from abstract_stack.teleop.session_bridge import GateControlServer, OwnerMailbox

    mailbox = OwnerMailbox(calibration_provider=lambda gate: neutral_specs())
    finished = Future()
    stop = threading.Event()

    def provider(session):
        now = time.monotonic()
        return (
            LifecycleEvidence(session.session_id, now, True, True, True, True),
            VideoEvidence(
                session.session_id, session.calibration_generation, 1, now - 0.01, now, 0.001, True
            ),
        )  # trusted synthetic providers

    async def receive(socket, kind, state=None):
        while True:
            message = json.loads(await asyncio.wait_for(socket.recv(), 3))
            if message["type"] == kind and (state is None or message["state"] == state):
                return message

    async def run():
        server = GateControlServer(mailbox, observation_provider=provider, port=0)
        await server.start()
        try:
            async with connect(f"ws://127.0.0.1:{server.bound_port}/quest/control") as socket:
                sid = str(uuid4())

                def envelope(kind, **kwargs):
                    return json.dumps(dict(schema_version=1, session_id=sid, type=kind, **kwargs))

                await socket.send(envelope("hello", profile="quest_teleop_v1"))
                await receive(socket, "hello_ack")
                t0 = time.monotonic()
                await socket.send(envelope("clock_ping", exchange_id=0, t0=t0))
                pong = await receive(socket, "clock_pong")
                await socket.send(
                    envelope(
                        "clock_result",
                        exchange_id=0,
                        t0=t0,
                        t1=pong["t1"],
                        t2=pong["t2"],
                        t3=time.monotonic(),
                    )
                )
                await receive(socket, "clock_ack")

                async def tracking(sequence, generation):
                    packet = json.loads(FIXTURE.read_text())
                    now = time.monotonic()
                    packet.update(
                        session_id=sid,
                        sequence=sequence,
                        calibration_generation=generation,
                        sent_monotonic_s=now,
                    )
                    for name in ("head", "left_hand", "right_hand", "body"):
                        packet[name]["observed_monotonic_s"] = now
                        if "hand" in name:
                            packet[name]["source_time_s"] = now
                    await socket.send(json.dumps(packet))

                await tracking(0, 0)
                await receive(socket, "status", "Ready")
                await socket.send(
                    envelope("control", action="calibrate", event_id=0, calibration_generation=0)
                )
                ack = await receive(socket, "control_ack")
                assert ack["accepted"] and ack["calibration_generation"] == 1
                await tracking(1, 1)
                # Wait for the owner to consume fresh post-calibration input.
                await receive(socket, "status", "Ready")
                await asyncio.sleep(0.04)
                await socket.send(
                    envelope("control", action="engage", event_id=1, calibration_generation=1)
                )
                ack = await receive(socket, "control_ack")
                assert ack["accepted"] and ack["state"] == "Engaged"
                await receive(socket, "status", "Engaged")
                await socket.send(
                    envelope("control", action="pause", event_id=2, calibration_generation=0)
                )
                ack = await receive(socket, "control_ack")
                assert ack["accepted"] and ack["state"] == "Paused"
                await receive(socket, "status", "Paused")
        finally:
            await server.close()

    def background():
        try:
            asyncio.run(run())
            finished.set_result(True)
        except BaseException as error:
            finished.set_exception(error)
        finally:
            stop.set()

    thread = threading.Thread(target=background, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not stop.is_set() and time.monotonic() < deadline:
        mailbox.drain(gate, now=time.monotonic())
        time.sleep(0.001)
    thread.join(1)
    assert not thread.is_alive()
    assert finished.result()
    mailbox.drain(gate, now=time.monotonic())
    assert gate.session_id is None
    assert not gate.sim.writes  # actual network states, no fake physics success


def test_unexpected_guard_exception_faults_before_any_actuator_write(gate):
    engaged(gate)

    def broken_guard(state, targets):
        raise OverflowError("fixture module failure")

    gate.composed_guard = broken_guard
    assert gate.tick(proposal(gate, 10.04), now=10.04) is None
    assert gate.state == SessionState.FAULT and "fixture module failure" in gate.reason
    assert not gate.sim.writes


def test_nonfinite_physics_cannot_be_hidden_by_reset(gate):
    from dataclasses import asdict

    gate.sim.current.actuator_forces[0] = float("nan")
    assert gate.tick(now=10) is None
    assert gate.state == SessionState.FAULT
    assert not gate.host_reset(now=11)["accepted"]
    assert gate.sim.resets == 0
    telemetry = asdict(gate.telemetry(now=11))
    assert telemetry["measured_forces"][gate.names[0]] is None
    json.dumps(telemetry, allow_nan=False)


def test_repeated_engage_cannot_reset_command_acceleration_history(gate):
    engaged(gate)
    gate.tick(proposal(gate, 10.04), now=10.04)
    velocity = gate._previous_velocity.copy()
    result = control(gate, "engage", 10.04)
    assert not result["accepted"] and result["reason"] == "already_engaged"
    np.testing.assert_array_equal(gate._previous_velocity, velocity)


def test_delayed_control_ack_cannot_roll_back_transport_generation():
    import asyncio
    from concurrent.futures import Future
    from types import SimpleNamespace

    from abstract_stack.teleop.session_bridge import OwnerMailbox

    mailbox = OwnerMailbox()
    completed = Future()
    completed.set_result({"accepted": True, "calibration_generation": 1})
    mailbox.submit = lambda *args, **kwargs: completed
    changed = []
    session = SimpleNamespace(
        session_id="fixture",
        calibration_generation=3,
        set_calibration_generation=lambda value: changed.append(value),
    )
    result = asyncio.run(
        mailbox.on_control(session, {"action": "pause", "event_id": 0, "calibration_generation": 0})
    )
    assert result["accepted"]
    assert not changed and session.calibration_generation == 3
