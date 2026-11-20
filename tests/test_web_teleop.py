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


def _test_client():
    """Starlette's TestClient needs httpx (``ci`` group), not the runtime ``teleop`` extra."""
    pytest.importorskip(
        "httpx", reason="WebSocket tests need httpx: uv sync --extra teleop --group ci"
    )
    from starlette.testclient import TestClient

    return TestClient


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
    TestClient = _test_client()
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

    TestClient = _test_client()

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


def test_head_look_at_is_validated():
    look = {"look_at_m": [-0.5, 0.1, 0.8]}
    parsed = protocol.parse_command(protocol.decode_json(json.dumps(_command(head=look))))
    np.testing.assert_allclose(parsed["head"]["look_at"], [-0.5, 0.1, 0.8])
    with pytest.raises(protocol.ProtocolError, match="unexpected fields"):
        bad = {"look_at_m": [0, 0, 1], "yaw_rad": 0.1}
        protocol.parse_command(protocol.decode_json(json.dumps(_command(head=bad))))


@pytest.mark.mabel
def test_head_follows_a_gaze_target(clear_env):
    core = _core(clear_env, input_timeout_s=10.0)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "engage", 1)["accepted"]
    target_root = np.array([-0.6, -0.35, 0.85])  # ahead, to the robot's left, below eye level
    for sequence in range(1, 151):
        command = _arm_command(core, sequence, np.zeros(3))
        command["arms"] = {"left": None, "right": None}
        command["head"] = {"look_at": target_root}
        core.submit_command(operator, command | {"session_id": operator})
        core._tick_once(clear_env.dt)
    model, data = clear_env.model, clear_env.data
    camera = model.camera("zed_left").id
    chassis = model.body("mabel_chassis").id
    goal = data.xpos[chassis] + data.xmat[chassis].reshape(3, 3) @ target_root
    forward = -data.cam_xmat[camera].reshape(3, 3)[:, 2]
    direction = (goal - data.cam_xpos[camera]) / np.linalg.norm(goal - data.cam_xpos[camera])
    # Measured camera axis (native physics), not the commanded angles.
    assert np.degrees(np.arccos(np.clip(forward @ direction, -1, 1))) < 6
    assert core._publish_status()["channels"]["head_mode"] == "gaze"


def _engaged_from_ready(env):
    core = _core(env, input_timeout_s=10.0)
    operator = str(uuid.uuid4())
    core.open_session(operator, "operator")
    assert _event(core, operator, "ready", 1)["accepted"]
    for _ in range(150):
        core._tick_once(env.dt)
    assert _event(core, operator, "engage", 2)["accepted"]
    core._tick_once(env.dt)
    return core, operator


class _Operator:
    """One monotonically sequenced command stream relative to the Ready palm poses."""

    def __init__(self, core, operator):
        self.core, self.operator, self.sequence = core, operator, 0
        self.start = _arm_command(core, 0, np.zeros(3))

    def goal(self, side, offset):
        return np.array(self.start["arms"][side]["position"]) + offset

    def hold(self, offsets, ticks, *, base=None, each=None):
        """Command each arm at Ready + ``offsets[side]`` (root frame) for ``ticks``."""
        for _ in range(ticks):
            self.sequence += 1
            command = _arm_command(self.core, self.sequence, np.zeros(3))
            for side, offset in offsets.items():
                arm = command["arms"][side]
                arm["position"] = self.goal(side, offset)
                arm["quaternion"] = self.start["arms"][side]["quaternion"]
            command["base"] = base
            self.core.submit_command(self.operator, command | {"session_id": self.operator})
            self.core._tick_once(self.core.env.dt)
            if each is not None:
                each()


def _deepest(env, bodies_a, bodies_b):
    """Deepest current contact distance (m, negative = penetration) between two body sets."""
    deepest = 0.0
    for contact in env.data.contact[: env.data.ncon]:
        pair = (env.model.geom_bodyid[contact.geom1], env.model.geom_bodyid[contact.geom2])
        if (pair[0] in bodies_a and pair[1] in bodies_b) or (
            pair[1] in bodies_a and pair[0] in bodies_b
        ):
            deepest = min(deepest, float(contact.dist))
    return deepest


def _table_env():
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.scene import ShelfConfig

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    # Supply table directly under the Ready palms; the shelf is out of the way.
    return ShelfStockingEnv(
        ShelfConfig(
            product_count=0,
            center_y=-1.5,
            table_center_x=-0.60,
            table_center_y=0.0,
            table_top=0.70,
            terminate_on_success=False,
            time_limit=600.0,
        )
    )


def _robot_bodies(env):
    root = env.model.body_rootid[env.model.body("mabel_chassis").id]
    return {b for b in range(env.model.nbody) if env.model.body_rootid[b] == root}


def test_out_of_reach_pose_keeps_the_palm_on_target_and_relaxes_orientation(clear_env):
    from abstract_stack.ik import rotation_error
    from abstract_stack.web_teleop.retarget import READY_POSTURE, _quat_to_mat

    core, operator = _engaged_from_ready(clear_env)
    retargeter = core.retargeter
    offset = np.array([-0.25, -0.20, -0.25])  # far forward, outward and down
    stream = _Operator(core, operator)
    stream.hold({"left": offset}, ticks=150)
    start, goal_root = stream.start, stream.goal("left", offset)
    origin, root = retargeter.root_pose()
    goal = origin + root @ goal_root
    rotation = root @ _quat_to_mat(np.array(start["arms"]["left"]["quaternion"]))

    # Holding the full Ready orientation, this point is out of reach for the
    # weighted pose solver even with a generous budget.
    ik = retargeter.ik["left"]
    ready = [READY_POSTURE.get(n, 0.0) for n in retargeter.arm_names["left"]]
    seed = np.r_[ready, retargeter.reference["Lift"]]
    _, point, _ = ik.solve(goal, rotation, seed, iterations=300)
    assert np.linalg.norm(point - goal) > 0.05

    report = core._publish_status()["arms"]["left"]
    assert report["ik"]["orientation_relaxed"] and not report["ik"]["reachable"]
    assert report["ik"]["ik_position_error_m"] < 0.01
    assert not report["ik"]["blocked_by"]
    measured = np.array(report["palm"]["position_m"])
    assert np.linalg.norm(measured - goal_root) < 0.02
    # Orientation is relaxed only as far as needed, not dropped.
    actual = root @ _quat_to_mat(np.array(report["palm"]["quaternion_wxyz"]))
    assert np.degrees(np.linalg.norm(rotation_error(actual, rotation))) < 55


def test_collision_guard_stops_hands_at_a_table():
    env = _table_env()
    try:
        core, operator = _engaged_from_ready(env)
        robot, table = _robot_bodies(env), {env.model.body("supply").id}
        down = np.array([0.0, 0.0, -0.40])  # 0.4 m down: ~0.2 m into the table top
        deepest = [0.0]
        stream = _Operator(core, operator)
        stream.hold(
            {"left": down, "right": down},
            ticks=150,
            each=lambda: deepest.append(_deepest(env, robot, table)),
        )
        status = core._publish_status()
        for side in ("left", "right"):
            assert status["arms"][side]["ik"]["blocked_by"] == ["supply"], side
            assert not status["arms"][side]["ik"]["reachable"]
        assert min(deepest) > -0.005  # resting contact at every tick, not the commanded 0.2 m
        assert status["base"]["tilt_deg"] < 2.0
    finally:
        env.close()


def test_collision_guard_checks_the_path_not_just_the_endpoint():
    env = _table_env()
    try:
        core, operator = _engaged_from_ready(env)
        retargeter = core.retargeter
        robot, table = _robot_bodies(env), {env.model.body("supply").id}
        # Both endpoints clear the table top, but the joint path between them dips
        # into it (review repro: ~7 mm at 60 % of the move in scratch data).
        offset = np.array([0.0, -0.20, -0.35])
        stream = _Operator(core, operator)
        deepest, blocked = [0.0], []
        stream.hold(
            {"left": offset},
            ticks=150,
            each=lambda: (
                deepest.append(_deepest(env, robot, table)),
                blocked.append(retargeter.arm_report["left"]["blocked_by"]),
            ),
        )
        ik = retargeter.ik["left"]
        goal = retargeter._candidate["left"]["q"]
        reference = env.data.qpos.copy()
        baseline = retargeter._penetrations(ik, ik.measured_q(), reference)
        assert retargeter._collisions(ik, goal, reference, baseline) == []
        assert ["supply"] in blocked  # the path, not the endpoint, was blocked
        assert min(deepest) > -0.005
    finally:
        env.close()


def test_collision_guard_rechecks_a_cached_target_as_the_base_moves():
    env = _table_env()
    try:
        core, operator = _engaged_from_ready(env)
        stream = _Operator(core, operator)
        down = {"left": np.array([0.0, 0.0, -0.40])}
        stream.hold(down, ticks=100)
        assert core.retargeter.arm_report["left"]["blocked_by"] == ["supply"]
        # Same arm target throughout; drive left (−Y) until the hand clears the table.
        stream.hold(down, ticks=250, base={"forward_mps": 0.0, "left_mps": 0.4, "turn_radps": 0.0})
        stream.hold(down, ticks=100, base={"forward_mps": 0.0, "left_mps": 0.0, "turn_radps": 0.0})
        report = core.retargeter.arm_report["left"]
        assert report["blocked_by"] == []
        assert report["ik_position_error_m"] < 0.01
    finally:
        env.close()


def test_collision_guard_checks_both_arms_together(clear_env):
    core, operator = _engaged_from_ready(clear_env)
    retargeter, env = core.retargeter, clear_env
    stream = _Operator(core, operator)
    # Review repro: both palms 5 cm forward (MABEL faces -X) onto the chassis
    # midline, heights and orientations kept. Each arm alone is clear of the other
    # arm's measured pose; together they collide.
    offsets = {
        side: np.array([-0.05, -stream.start["arms"][side]["position"][1], 0.0])
        for side in ("left", "right")
    }
    left = {b for b in _robot_bodies(env) if env.model.body(b).name.startswith(("left_", "l_"))}
    right = {b for b in _robot_bodies(env) if env.model.body(b).name.startswith(("right_", "r_"))}
    deepest = [0.0]
    stream.hold(offsets, ticks=150, each=lambda: deepest.append(_deepest(env, left, right)))
    assert any("self" in retargeter.arm_report[side]["blocked_by"] for side in ("left", "right"))
    # The composed commanded references are clear of each other in scratch data...
    ik, reference = retargeter.ik["left"], env.data.qpos.copy()
    reference[retargeter.ik["right"].qadr] = [
        retargeter.reference[n] for n in retargeter.arm_names["right"]
    ]
    q_left = np.r_[
        [retargeter.reference[n] for n in retargeter.arm_names["left"]],
        retargeter.reference["Lift"],
    ]
    contacts = ik._contacts(q_left, reference, None)
    between = [
        info["distance_m"]
        for pair, info in contacts.items()
        if {env.model.geom_bodyid[g] in left for g in pair} == {True, False}
        and any(env.model.geom_bodyid[g] in right for g in pair)
    ]
    assert min(between, default=0.0) > -0.005
    # ...and the live arms never press into each other.
    assert min(deepest) > -0.005


def test_collision_guard_treats_products_as_manipulable(clear_env):
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.scene import ShelfConfig

    env = ShelfStockingEnv(ShelfConfig(product_count=2))
    try:
        retargeter = _core(env).retargeter
        products = {env.model.body(p.name).id for p in env.products}
        assert products <= retargeter._movable
        assert env.model.body("mabel_chassis").id not in retargeter._movable
        assert env.model.body("shelf").id not in retargeter._movable
    finally:
        env.close()


# -- latency -------------------------------------------------------------------------


def test_status_echoes_when_each_command_was_applied(clear_env):
    core, operator = _engaged_from_ready(clear_env)
    command = _arm_command(core, 1, np.zeros(3)) | {"session_id": operator}
    command["sent_monotonic_s"] = 1234.5  # browser clock, echoed back in ms
    core.submit_command(operator, command)
    before = time.monotonic()
    core._tick_once(clear_env.dt)
    timing = core._publish_status()["command_timing"]
    assert timing["sequence"] == 1
    assert timing["sent_ms"] == pytest.approx(1234500.0)
    assert timing["received_host_s"] <= before <= timing["applied_host_s"] <= time.monotonic()
    # The echo names the first tick that applied a command, not later re-applications.
    core._tick_once(clear_env.dt)
    assert core._publish_status()["command_timing"] == timing


def test_video_render_model_simplifies_only_peripheral_meshes(clear_env):
    from abstract_stack.web_teleop.video import PERIPHERAL_FACE_BUDGET, render_model

    pytest.importorskip("fast_simplification")
    model = clear_env.model
    original = model.mesh_facenum.copy()
    copy, simplified = render_model(model, PERIPHERAL_FACE_BUDGET)
    assert simplified > 0 and copy is not model
    assert np.array_equal(model.mesh_facenum, original)  # the live model is untouched
    changed = {m for m in range(model.nmesh) if copy.mesh_facenum[m] != original[m]}
    assert len(changed) == simplified
    # The simplifier targets the budget but cannot always reach it on every mesh.
    assert all(copy.mesh_facenum[m] < original[m] for m in changed)
    keep = ("l_elbow", "r_forearm", "left_palm", "torso", "l_shoulder_1")
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom]).name
        if body in keep and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            assert int(model.geom_dataid[geom]) not in changed, body
    # Faces actually drawn: shared meshes (three swerve modules) count per geom.
    drawn = [
        int(model.geom_dataid[g])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] == 2
    ]
    assert copy.mesh_facenum[drawn].sum() < 0.75 * original[drawn].sum()  # ~0.70
    assert (copy.nq, copy.nbody, copy.ngeom) == (model.nq, model.nbody, model.ngeom)


@pytest.mark.mabel
@pytest.mark.rendering  # needs a GL context: deselected on hosted macOS CI
def test_simplified_head_camera_render_matches_full_detail(clear_env):
    from abstract_stack.web_teleop.retarget import READY_POSTURE
    from abstract_stack.web_teleop.video import PERIPHERAL_FACE_BUDGET, render_model

    pytest.importorskip("fast_simplification")
    model = clear_env.model
    simplified, _ = render_model(model, PERIPHERAL_FACE_BUDGET)
    # Worst case for the base: arms at Ready, head fully down and turned.
    pose = {**READY_POSTURE, "Neck Pitch": 0.785, "Neck Yaw": 0.9}
    images = []
    for render in (model, simplified):
        data = mujoco.MjData(render)
        data.qpos[:] = clear_env.data.qpos
        for name, value in pose.items():
            data.qpos[render.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(render, data)
        mujoco.mj_camlight(render, data)  # as the render thread does
        with mujoco.Renderer(render, height=240, width=320) as renderer:
            renderer.update_scene(data, camera="zed_left")
            images.append(renderer.render().astype(int))
    differing = (np.abs(images[0] - images[1]).max(axis=2) > 16).mean()
    assert differing < 0.002, f"{differing:.2%} of pixels differ"


def test_websocket_compression_is_opt_in(monkeypatch, clear_env, tmp_path):
    pytest.importorskip("uvicorn")
    import uvicorn

    from abstract_stack.web_teleop import server as server_module
    from abstract_stack.web_teleop.core import TeleopCore

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    core = TeleopCore(clear_env)
    for deflate in (False, True):
        server_module.serve(
            core, host="127.0.0.1", port=0, token=None, static_dir=None, ws_deflate=deflate
        )
    assert [call["ws_per_message_deflate"] for call in calls] == [False, True]


def test_command_timing_is_scoped_to_the_sending_session(clear_env):
    from abstract_stack.web_teleop.server import _status_for

    core, first = _engaged_from_ready(clear_env)
    command = _arm_command(core, 1, np.zeros(3)) | {"session_id": first, "sent_monotonic_s": 10.0}
    core.submit_command(first, command)
    core._tick_once(clear_env.dt)
    status = core._publish_status()
    assert status["command_timing"]["session_id"] == first
    # Only the sending session sees the echo, without the session id.
    own = _status_for(first, status)["command_timing"]
    assert own["sent_ms"] == pytest.approx(10000.0) and "session_id" not in own
    assert _status_for("someone-else", status)["command_timing"] is None

    # Review repro: a reconnecting browser restarts its sequences at 1 on its own clock.
    core.close_session(first)
    assert core._publish_status()["command_timing"] is None
    second = str(uuid.uuid4())
    core.open_session(second, "operator")
    assert _event(core, second, "ready", 1)["accepted"]
    assert _event(core, second, "engage", 2)["accepted"]
    core._tick_once(clear_env.dt)
    command = _arm_command(core, 1, np.zeros(3)) | {"session_id": second, "sent_monotonic_s": 99.0}
    core.submit_command(second, command)
    core._tick_once(clear_env.dt)
    timing = core._publish_status()["command_timing"]
    assert (timing["session_id"], timing["sequence"]) == (second, 1)
    assert timing["sent_ms"] == pytest.approx(99000.0)


# -- efficient transport ---------------------------------------------------------------


def test_compact_state_frames_halve_size_within_quarter_millimetre():
    from abstract_stack.web_teleop.protocol import compact_state, pack_state, unpack_state

    rng = np.random.default_rng(0)
    xpos = rng.uniform(-2.7, 2.7, size=(81, 3))
    xquat = rng.normal(size=(81, 4))
    xquat /= np.linalg.norm(xquat, axis=1, keepdims=True)
    full = pack_state(7, 1.25, 99.5, xpos, xquat, 1)
    compact = compact_state(full)
    assert compact[:4] == b"WTS2" and len(compact) == 28 + 81 * 14
    assert len(compact) < 0.51 * len(full)
    decoded = unpack_state(compact)
    assert (decoded["tick"], decoded["sim_time"], decoded["host_time"]) == (7, 1.25, 99.5)
    assert np.abs(decoded["xpos"] - xpos).max() <= 0.5 / 4096 + 1e-6  # ≤ 0.12 mm
    assert np.abs(decoded["xquat"] - xquat).max() <= 0.5 / 32767 + 1e-6


def test_link_controller_backs_off_in_order_and_recovers():
    from abstract_stack.web_teleop.link import LinkController

    link = LinkController(max_kbps=1500)
    t = 0.0
    while link.video_divider < 3:  # sustained congestion
        link.feedback(300.0, t)
        t += 0.5
    assert (link.kbps, link.pose_divider, link.video_divider) == (250, 3, 3)
    steps = link.decreases
    link.feedback(300.0, t - 0.4)  # rate-limited: no step faster than every 0.5 s
    assert link.decreases == steps
    t += 2.0
    restored = []
    while link.kbps < 1500:
        link.feedback(5.0, t)
        restored.append((link.video_divider, link.pose_divider))
        t += 1.0
    # Video frame rate comes back first, then pose rate, then bitrate.
    assert restored[0] == (2, 3) and (1, 1) in restored
    assert link.kbps == 1500


def test_in_flight_window_bounds_queueing_to_rtt_plus_slack():
    from abstract_stack.web_teleop.link import InFlightWindow

    window = InFlightWindow(slack_s=0.06, max_window_s=0.3)
    interval = 0.02
    # A client that never acks is not windowed (older clients keep working).
    for tick in range(20):
        assert window.can_send(tick * interval, interval)
        window.sent(tick, tick * interval)
    window.ack(19, 0.43)  # rtt 50 ms
    assert window.allowed(0.43, interval) == 6  # (50 + 60) ms / 20 ms
    for tick in range(20, 26):
        assert window.can_send(0.44, interval)
        window.sent(tick, 0.44)
    assert not window.can_send(0.44, interval) and window.skipped == 1
    window.ack(25, 0.50)
    assert window.can_send(0.50, interval)
    # A slow page acks late: the 90th percentile, not the minimum, sizes the window,
    # and the absolute cap still bounds it.
    for k, rtt in enumerate([0.05] * 5 + [0.2] * 5):
        window._rtts.append((0.6 + k * 0.01, rtt))
    assert window.allowed(0.7, interval) == 13  # (200 + 60) ms / 20 ms
    window._rtts.append((0.7, 5.0))
    window._rtts.append((0.7, 5.0))
    assert window.allowed(0.7, interval) == 15  # capped at 0.3 s
    # Frames the client never acks (dropped) expire, so the window cannot deadlock.
    for tick in range(26, 60):
        window.sent(tick, 0.71)
    assert not window.can_send(0.72, interval)
    assert window.can_send(2.3, interval)


def test_h264_encoder_keyframes_on_request_and_decodes():
    av = pytest.importorskip("av")
    from abstract_stack.web_teleop.video import H264Encoder

    encoder = H264Encoder(320, 120, 20.0, 600)
    decoder = av.CodecContext.create("h264", "r")
    rng = np.random.default_rng(1)
    base = (rng.random((120, 320, 3)) * 255).astype(np.uint8)
    keyframes, decoded = [], 0
    for i in range(12):
        if i == 6:
            encoder.request_keyframe()
        if i == 8:
            encoder.set_kbps(300, now=0.0)  # within [cap/4, cap]: live change, no keyframe
        data, key = encoder.encode(np.roll(base, i * 3, axis=1), i * 50)
        assert data[:4] == b"\x00\x00\x00\x01"  # Annex B
        keyframes.append(key)
        for packet in decoder.parse(data):
            decoded += len(decoder.decode(packet))
    assert keyframes[0] and keyframes[6] and not keyframes[1] and not keyframes[8]
    assert encoder.kbps == 300
    assert decoded >= 10


@pytest.mark.mabel
@pytest.mark.rendering  # renders through the video thread
def test_video_socket_negotiates_h264_and_flow_controls(clear_env, tmp_path, monkeypatch):
    pytest.importorskip("av")
    TestClient = _test_client()
    from abstract_stack.web_teleop.core import TeleopCore
    from abstract_stack.web_teleop.server import TeleopServer
    from abstract_stack.web_teleop.video import StereoVideo, unpack_h264

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    video = StereoVideo(clear_env.model, width=160, height=120, fps=10)
    if "h264" not in video.codecs:
        pytest.skip("PyAV with libx264 not installed")
    server = TeleopServer(TeleopCore(clear_env, video=video), token=None, face_budget=200)
    session = str(uuid.uuid4())
    with TestClient(server.app) as client, client.websocket_connect("/teleop/control") as control:
        control.send_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "hello",
                    "session_id": session,
                    "profile": "web_teleop_v1",
                    "role": "operator",
                }
            )
        )
        ack = json.loads(control.receive_text())
        assert ack["video"]["codecs"] == ["h264", "jpeg"]
        with client.websocket_connect("/teleop/video") as stream:
            stream.send_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "type": "video_bind",
                        "session_id": session,
                        "codecs": ["h264", "jpeg"],
                    }
                )
            )
            first, data = unpack_h264(stream.receive_bytes())
            assert first["codec"] == "h264" and first["keyframe"]  # a viewer starts on a keyframe
            assert first["layout"] == "side_by_side" and first["width"] == 160
            assert b"\x00\x00\x00\x01\x67" in data[:64]  # SPS in-band
            stream.send_text(
                json.dumps({"schema_version": 1, "type": "video_ack", "pair_id": first["pair_id"]})
            )
            second, _ = unpack_h264(stream.receive_bytes())
            assert second["pair_id"] > first["pair_id"]
        # Operator feedback drives the link controller; status reports it.
        control.send_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "link_feedback",
                    "session_id": session,
                    "queue_ms": 500,
                }
            )
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            message = json.loads(control.receive_text())
            if message["type"] == "status" and message["link"]["decreases"]:
                assert message["link"]["video_kbps"] < video.h264_max_kbps
                break
        else:
            pytest.fail("link feedback never reached the controller")


def test_link_feedback_and_bind_formats_are_validated():
    from abstract_stack.web_teleop.protocol import ProtocolError, parse_link_feedback

    ok = {"schema_version": 1, "type": "link_feedback", "session_id": "s", "queue_ms": 12.5}
    assert parse_link_feedback(ok)["queue_ms"] == 12.5
    for bad in ({**ok, "queue_ms": -1}, {**ok, "queue_ms": float("nan")}, {**ok, "extra": 1}):
        with pytest.raises(ProtocolError):
            parse_link_feedback(bad)
