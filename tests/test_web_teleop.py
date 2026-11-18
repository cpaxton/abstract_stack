"""Web teleop protocol, session state machine, retargeting and transport checks."""

import json
import time
import uuid

import mujoco
import numpy as np
import pytest

from abstract_stack.web_teleop import protocol

SESSION = str(uuid.uuid4())


def _command(**overrides):
    message = {
        "schema_version": 1,
        "type": "command",
        "session_id": SESSION,
        "sequence": 1,
        "calibration_generation": 0,
        "target_epoch": 1,
        "sent_monotonic_s": 12.5,
        "arms": {
            "left": {
                "position_m": [-0.4, -0.15, 0.85],
                "quaternion_wxyz": [1, 0, 0, 0],
                "grip": 0.5,
            },
            "right": None,
        },
        "head": {"yaw_rad": 0.1, "pitch_rad": -0.2},
        "lift_m": 0.33,
        "torso_rad": None,
    }
    message.update(overrides)
    return message


# -- protocol (no MuJoCo model needed) ---------------------------------------------------


def test_command_parses_absolute_intents():
    parsed = protocol.parse_command(protocol.decode_json(json.dumps(_command())))
    assert parsed["arms"]["right"] is None
    np.testing.assert_allclose(parsed["arms"]["left"]["position"], [-0.4, -0.15, 0.85])
    assert parsed["head"] == {"Neck Yaw": 0.1, "Neck Pitch": -0.2}
    assert parsed["lift"] == 0.33 and parsed["torso"] is None


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m.update(extra=1), "unexpected fields"),
        (lambda m: m.pop("target_epoch"), "missing fields"),
        (lambda m: m["arms"]["left"].update(grip=1.5), "grip"),
        (lambda m: m["arms"]["left"].update(quaternion_wxyz=[1, 1, 0, 0]), "unit quaternion"),
        (lambda m: m["arms"].update(middle=None), "unexpected fields"),
        (lambda m: m.update(sequence=-1), "sequence"),
        (lambda m: m.update(sequence=True), "sequence"),
        (lambda m: m.update(session_id="nope"), "UUID"),
        (lambda m: m.update(lift_m="high"), "lift_m"),
    ],
)
def test_command_rejects_contract_violations(mutate, match):
    message = _command()
    mutate(message)
    with pytest.raises(protocol.ProtocolError, match=match):
        protocol.parse_command(protocol.decode_json(json.dumps(message)))


def test_decode_rejects_non_finite_oversize_and_wrong_version():
    with pytest.raises(protocol.ProtocolError, match="non-finite"):
        protocol.decode_json('{"schema_version":1,"type":"ping","t0":NaN}')
    with pytest.raises(protocol.ProtocolError, match="exceeds"):
        protocol.decode_json("[" + " " * protocol.MAX_JSON_BYTES + "]")
    with pytest.raises(protocol.ProtocolError, match="schema_version"):
        protocol.decode_json('{"schema_version":2,"type":"hello"}')


def test_state_frame_round_trip():
    xpos = np.arange(9, dtype=float).reshape(3, 3)
    xquat = np.tile([1.0, 0, 0, 0], (3, 1))
    frame = protocol.pack_state(7, 1.25, 99.0, xpos, xquat, protocol.FLAG_ENGAGED)
    assert len(frame) == protocol.STATE_HEADER.size + 3 * 7 * 4
    decoded = protocol.unpack_state(frame)
    assert decoded["tick"] == 7 and decoded["flags"] == protocol.FLAG_ENGAGED
    np.testing.assert_allclose(decoded["xpos"], xpos)
    with pytest.raises(protocol.ProtocolError):
        protocol.unpack_state(frame[:-4])


# -- native MABEL ------------------------------------------------------------------------


@pytest.fixture
def clear_env():
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.scene import ShelfConfig

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    # Same isolation as the controller tests: the default bench touches the fingers.
    env = ShelfStockingEnv(
        ShelfConfig(
            product_count=0,
            center_y=-1.5,
            table_center_y=-1.6,
            terminate_on_success=False,
            time_limit=600.0,
        )
    )
    yield env
    env.close()


def _core(env, **config):
    from abstract_stack.web_teleop.core import CoreConfig, TeleopCore

    return TeleopCore(env, config=CoreConfig(**config), realtime=False)


def _event(core, session, action, event_id, generation=None):
    gen = core.status()["calibration_generation"] if generation is None else generation
    future = core.control(
        session,
        {"session_id": session, "event_id": event_id, "generation": gen, "action": action},
    )
    # reset/ready resolve on the sim thread; drive one tick when not yet resolved.
    if not future.done():
        core._tick_once(core.env.dt)
    return future.result(timeout=1)


def _arm_command(core, sequence, offset, grip=0.0):
    targets = core.retargeter.arm_targets()
    status = core.status()
    return {
        "session_id": None,
        "sequence": sequence,
        "generation": status["calibration_generation"],
        "target_epoch": core.retargeter.target_epoch,
        "sent_monotonic_s": time.monotonic(),
        "arms": {
            side: {
                "position": np.array(targets[side]["position_m"]) + offset,
                "quaternion": np.array(targets[side]["quaternion_wxyz"]),
                "grip": grip,
            }
            for side in ("left", "right")
        },
        "head": None,
        "base": None,
        "lift": None,
        "torso": None,
    }


@pytest.mark.mabel
def test_operator_lease_and_engage_preconditions(clear_env):
    core = _core(clear_env)
    operator, other = str(uuid.uuid4()), str(uuid.uuid4())
    assert core.status()["state"] == "disconnected"
    assert core.open_session(operator, "operator") == "operator"
    assert core.open_session(other, "operator") == "spectator"
    core._tick_once(clear_env.dt)
    assert core.status()["state"] == "ready"
    denied = _event(core, other, "engage", 1)
    assert not denied["accepted"] and denied["error"] == "not the operator"
    stale = _event(core, operator, "engage", 1, generation=5)
    assert not stale["accepted"] and stale["error"] == "obsolete generation"
    ack = _event(core, operator, "engage", 2)
    assert ack["accepted"] and ack["state"] == "engaged"
    assert _event(core, operator, "engage", 2) == ack  # idempotent retry
    core.close_session(operator)
    assert core._state.value == "disconnected"


@pytest.mark.mabel
def test_missing_input_pauses_without_auto_resume(clear_env):
    core = _core(clear_env, input_timeout_s=0.03)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "engage", 1)["accepted"]
    core.submit_command(operator, _arm_command(core, 1, np.zeros(3)) | {"session_id": operator})
    core._tick_once(clear_env.dt)
    assert core._state.value == "engaged"
    time.sleep(0.05)
    core._tick_once(clear_env.dt)
    assert core._state.value == "paused" and "timed out" in core._reason
    core.submit_command(operator, _arm_command(core, 2, np.zeros(3)) | {"session_id": operator})
    core._tick_once(clear_env.dt)
    assert core._state.value == "paused"  # fresh input never resumes by itself


@pytest.mark.mabel
def test_both_arms_track_cartesian_targets_from_ready(clear_env):
    core = _core(clear_env, input_timeout_s=10.0)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "ready", 1)["accepted"]
    for _ in range(100):
        core._tick_once(clear_env.dt)
    assert _event(core, operator, "engage", 2)["accepted"]
    core._tick_once(clear_env.dt)  # applies the engage resync (new target epoch)
    offset = np.array([-0.08, 0.0, -0.08])  # forward and down, clear of the body
    goal = {s: np.array(t["position_m"]) + offset for s, t in core.retargeter.arm_targets().items()}
    for sequence in range(1, 101):
        command = _arm_command(core, sequence, np.zeros(3))
        for side in ("left", "right"):
            command["arms"][side]["position"] = goal[side]
        core.submit_command(operator, command | {"session_id": operator})
        core._tick_once(clear_env.dt)
        lower, upper = clear_env.model.actuator_ctrlrange.T
        limited = clear_env.model.actuator_ctrllimited.astype(bool)
        ctrl = clear_env.data.ctrl
        assert np.all(ctrl[limited] >= lower[limited]) and np.all(ctrl[limited] <= upper[limited])
    status = core._publish_status()
    assert status["state"] == "engaged"
    for side in ("left", "right"):
        assert status["arms"][side]["ik"]["reachable"]
        measured = np.array(status["arms"][side]["palm"]["position_m"])
        assert np.linalg.norm(measured - goal[side]) < 0.01, side


@pytest.mark.mabel
def test_stale_target_epoch_and_reset_generation(clear_env):
    core = _core(clear_env, input_timeout_s=10.0)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "engage", 1)["accepted"]
    core._tick_once(clear_env.dt)
    stale = _arm_command(core, 1, np.array([0.0, 0.0, -0.1]))
    stale["target_epoch"] -= 1
    before = dict(core.retargeter.reference)
    core.submit_command(operator, stale | {"session_id": operator})
    core._tick_once(clear_env.dt)
    assert core.retargeter.reference["Left Arm 1"] == before["Left Arm 1"]
    ack = _event(core, operator, "reset", 2)
    assert ack["accepted"] and ack["state"] == "ready" and ack["calibration_generation"] == 1
    assert clear_env.data.time <= clear_env.dt + 1e-9  # reset, then one tick
    old = _arm_command(core, 99, np.zeros(3)) | {"session_id": operator, "generation": 0}
    core.submit_command(operator, old)
    assert core._command is None  # previous-generation commands are dropped


@pytest.mark.mabel
def test_model_export_is_consistent(clear_env, tmp_path, monkeypatch):
    from abstract_stack.web_teleop.geometry import load_model_export

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    manifest, blob = load_model_export(clear_env.model)
    assert manifest["blob_bytes"] == len(blob)
    assert manifest["face_count"] == sum(m["index_count"] for m in manifest["meshes"]) // 3
    for mesh in manifest["meshes"]:
        n = mesh["vertex_count"]
        dtype = "<u2" if mesh["index_type"] == "uint16" else "<u4"
        assert all(mesh[k] % 4 == 0 for k in ("position_offset", "normal_offset", "index_offset"))
        indices = np.frombuffer(
            blob, dtype=dtype, count=mesh["index_count"], offset=mesh["index_offset"]
        )
        assert indices.max() < n
        normals = np.frombuffer(blob, dtype=np.int8, count=3 * n, offset=mesh["normal_offset"])
        lengths = np.linalg.norm(normals.reshape(-1, 3) / 127.0, axis=1)
        assert np.all(np.abs(lengths - 1) < 0.03)
    # Full resolution keeps every compiled visual face.
    visual = {
        int(clear_env.model.geom_dataid[g])
        for g in range(clear_env.model.ngeom)
        if clear_env.model.geom_group[g] in (0, 2, 5) and clear_env.model.geom_type[g] == 7
    }
    assert manifest["face_count"] == int(sum(clear_env.model.mesh_facenum[m] for m in visual))
    assert {g["group"] for g in manifest["geoms"]} <= {0, 2, 5}
    assert all(-1 <= g["mesh"] < len(manifest["meshes"]) for g in manifest["geoms"])
    cached, cached_blob = load_model_export(clear_env.model)
    assert cached == manifest and cached_blob == blob


@pytest.mark.mabel
def test_websocket_handshake_auth_and_state_stream(clear_env, tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from abstract_stack.web_teleop.core import TeleopCore
    from abstract_stack.web_teleop.server import TeleopServer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    server = TeleopServer(TeleopCore(clear_env), token="secret", face_budget=200)
    with TestClient(server.app) as client:
        assert client.get("/api/model").status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/model?token=SECRET").status_code == 200  # case-insensitive
        manifest = client.get("/api/model?token=secret").json()
        assert manifest["format"] == "web_teleop_model_v2"
        with client.websocket_connect("/teleop/control?token=wrong") as rejected:
            with pytest.raises(WebSocketDisconnect) as closed:
                rejected.receive_text()
            assert closed.value.code == 4401  # explicit: client stops retrying
        session = str(uuid.uuid4())
        with client.websocket_connect("/teleop/control?token=secret") as control:
            hello = {
                "schema_version": 1,
                "type": "hello",
                "session_id": session,
                "profile": "web_teleop_v1",
                "role": "operator",
            }
            control.send_text(json.dumps(hello))
            ack = control.receive_json()
            assert ack["type"] == "hello_ack" and ack["role"] == "operator"
            assert ack["model_hash"] == manifest["model_hash"]
            with client.websocket_connect("/teleop/state?token=secret") as state:
                bind = {"schema_version": 1, "type": "state_bind", "session_id": session}
                state.send_text(json.dumps(bind))
                frame = protocol.unpack_state(state.receive_bytes())
                assert frame["xpos"].shape == (clear_env.model.nbody, 3)
            control.send_text(
                json.dumps({"schema_version": 1, "type": "bogus", "session_id": session})
            )
            while (reply := control.receive_json())["type"] == "status":
                pass
            assert reply["type"] == "error" and "unknown message type" in reply["message"]
            ping = {"schema_version": 1, "type": "ping", "session_id": session, "t0": 1.0}
            control.send_text(json.dumps(ping))
            while (reply := control.receive_json())["type"] == "status":
                pass
            assert reply["type"] == "pong" and reply["t0"] == 1.0


def test_stereo_pair_framing_round_trip():
    from abstract_stack.web_teleop.video import pack_pair, unpack_pair

    message = pack_pair({"pair_id": 3}, b"\xff\xd8left", b"\xff\xd8right!")
    meta, left, right = unpack_pair(message)
    assert meta == {"pair_id": 3} and left == b"\xff\xd8left" and right == b"\xff\xd8right!"
    assert message[:4] == b"QTP1"
    with pytest.raises(ValueError):
        unpack_pair(message + b"x")


@pytest.mark.mabel
@pytest.mark.rendering
def test_video_socket_streams_stereo_head_camera_pairs(clear_env, tmp_path, monkeypatch):
    import io

    from PIL import Image
    from starlette.testclient import TestClient

    from abstract_stack.web_teleop.core import TeleopCore
    from abstract_stack.web_teleop.server import TeleopServer
    from abstract_stack.web_teleop.video import StereoVideo, unpack_pair

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    video = StereoVideo(clear_env.model, width=160, height=120, fps=10)
    server = TeleopServer(TeleopCore(clear_env, video=video), token=None, face_budget=200)
    session = str(uuid.uuid4())
    with TestClient(server.app) as client, client.websocket_connect("/teleop/control") as control:
        hello = {
            "schema_version": 1,
            "type": "hello",
            "session_id": session,
            "profile": "web_teleop_v1",
            "role": "operator",
        }
        control.send_text(json.dumps(hello))
        ack = control.receive_json()
        assert ack["video"]["cameras"] == ["zed_left", "zed_right"]
        with client.websocket_connect("/teleop/video") as stream:
            bind = {"schema_version": 1, "type": "video_bind", "session_id": session}
            stream.send_text(json.dumps(bind))
            meta, left, right = unpack_pair(stream.receive_bytes())
    if video.error:
        pytest.skip(video.error)
    assert meta["eye_order"] == ["left", "right"] and meta["width"] == 160
    eyes = [np.asarray(Image.open(io.BytesIO(eye))) for eye in (left, right)]
    assert all(eye.shape == (120, 160, 3) for eye in eyes)
    # 63 mm stereo baseline: the two eyes see measurably different images.
    assert np.abs(eyes[0].astype(int) - eyes[1].astype(int)).mean() > 1.0


def test_persistent_token_survives_restarts_and_rotates(tmp_path):
    from abstract_stack.web_teleop.server import persistent_token

    path = tmp_path / "token"
    path.write_text("kmkz88iR9IYR-9O9soiVPQwA\n")  # legacy token with look-alikes
    first = persistent_token(path)
    assert first != "kmkz88iR9IYR-9O9soiVPQwA"
    assert persistent_token(path) == first
    assert not set(first) & set("01ilo") and len(first) == 14
    assert path.stat().st_mode & 0o077 == 0  # private to the user
    rotated = persistent_token(path, rotate=True)
    assert rotated != first and persistent_token(path) == rotated


def test_base_velocity_intent_is_validated():
    base = {"forward_mps": 0.2, "left_mps": -0.1, "turn_radps": 0.3}
    parsed = protocol.parse_command(protocol.decode_json(json.dumps(_command(base=base))))
    assert parsed["base"] == base
    assert protocol.parse_command(protocol.decode_json(json.dumps(_command())))["base"] is None
    for bad, match in (
        ({**base, "forward_mps": 5.0}, "base speed"),
        ({**base, "turn_radps": 9.0}, "turn_radps"),
        ({"forward_mps": 0.1}, "missing fields"),
    ):
        with pytest.raises(protocol.ProtocolError, match=match):
            protocol.parse_command(protocol.decode_json(json.dumps(_command(base=bad))))


def _base_command(core, sequence, forward=0.0, left=0.0, turn=0.0):
    command = _arm_command(core, sequence, np.zeros(3))
    command["arms"] = {"left": None, "right": None}
    command["base"] = {"forward_mps": forward, "left_mps": left, "turn_radps": turn}
    return command


def _chassis_speed(env):
    velocity = np.zeros(6)
    chassis = env.model.body("mabel_chassis").id
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, chassis, velocity, 0)
    return float(np.linalg.norm(velocity[3:5]))


@pytest.mark.mabel
def test_engaged_base_drives_forward_and_brakes_on_pause(clear_env):
    core = _core(clear_env, input_timeout_s=10.0)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "engage", 1)["accepted"]
    chassis = clear_env.model.body("mabel_chassis").id
    start = clear_env.data.xpos[chassis].copy()
    heading = clear_env.data.xmat[chassis].reshape(3, 3)[:, 0] * -1  # MABEL faces -X
    for sequence in range(1, 101):  # 2 s at 0.3 m/s forward
        core.submit_command(
            operator, _base_command(core, sequence, forward=0.3) | {"session_id": operator}
        )
        core._tick_once(clear_env.dt)
    moved = clear_env.data.xpos[chassis] - start
    assert moved @ heading > 0.35  # measured forward travel, not commanded
    assert abs(moved @ np.cross([0, 0, 1], heading)) < 0.03  # no sideways drift
    assert core.retargeter.tilt() < np.radians(2)
    assert _event(core, operator, "pause", 2)["accepted"]
    for _ in range(50):
        core._tick_once(clear_env.dt)
    assert _chassis_speed(clear_env) < 0.02  # braked to a stop within 1 s


@pytest.mark.mabel
def test_base_stops_when_operator_input_goes_stale(clear_env):
    core = _core(clear_env, input_timeout_s=0.03)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "engage", 1)["accepted"]
    for sequence in range(1, 51):
        core.submit_command(
            operator, _base_command(core, sequence, turn=0.6) | {"session_id": operator}
        )
        core._tick_once(clear_env.dt)
    assert core.retargeter.base.moving
    time.sleep(0.05)  # input goes stale: host pauses and brakes
    for _ in range(50):
        core._tick_once(clear_env.dt)
    assert core._state.value == "paused" and not core.retargeter.base.moving
