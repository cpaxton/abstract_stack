"""Bounded paired-JPEG pipeline, with rendering isolated from live physics.

Only the scratch renderer writes replay qpos. It owns a separate model/data;
these writes never command or change the live simulator.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import multiprocessing
import queue
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.sim import CameraFrame, SimState, StereoPair

MAX_METADATA = 16384
MAX_EYE = 2097152
MAX_PACKET = 16 + MAX_METADATA + 2 * MAX_EYE
MAX_DIMENSION = 2048
MAX_PIXELS = 2048 * 1536
HEADER = struct.Struct(">4sIII")
T = TypeVar("T")


def _integer(value, name, minimum=0, maximum=2**53 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")
    return value


def _number(value, name, minimum=0):
    if type(value) not in (int, float):
        raise ValueError(f"Invalid {name}")
    try:
        valid = math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _dimensions(width, height):
    _integer(width, "width", 1, MAX_DIMENSION)
    _integer(height, "height", 1, MAX_DIMENSION)
    if width * height > MAX_PIXELS:
        raise ValueError("Image pixel budget exceeded")


def _keys(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("Unexpected or missing fields")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_unique_object)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid bounded JSON") from error


def validate_metadata(metadata):
    """Reject v1 extensions and unbounded/malformed geometry before decode."""
    _keys(
        metadata,
        "schema_version type session_id calibration_generation pair_id sim_time_s "
        "capture_host_monotonic_s codec pixel_format frame width height eye_order cameras",
    )
    if (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
        or metadata["type"] != "stereo_pair"
        or metadata["codec"] != "jpeg"
        or metadata["pixel_format"] != "rgb8"
        or metadata["frame"] != "mabel_chassis_native"
        or metadata["eye_order"] != ["left", "right"]
    ):
        raise ValueError("Unsupported video profile")
    sid = metadata["session_id"]
    if not isinstance(sid, str) or not 1 <= len(sid) <= 128:
        raise ValueError("Invalid session ID")
    for name in ("calibration_generation", "pair_id"):
        _integer(metadata[name], name)
    for name in ("sim_time_s", "capture_host_monotonic_s"):
        _number(metadata[name], name)
    _dimensions(metadata["width"], metadata["height"])
    _keys(metadata["cameras"], "left right")
    for side in ("left", "right"):
        camera = metadata["cameras"][side]
        _keys(camera, "name position_m quaternion_wxyz vertical_fov_deg")
        if camera["name"] != f"zed_{side}":
            raise ValueError("Incorrect eye name")
        for name, length in (("position_m", 3), ("quaternion_wxyz", 4)):
            value = camera[name]
            if not isinstance(value, list) or len(value) != length:
                raise ValueError("Invalid camera pose")
            for component in value:
                _number(component, name, -math.inf)
        if abs(np.linalg.norm(camera["quaternion_wxyz"]) - 1) > 1e-3:
            raise ValueError("Camera quaternion is not normalized")
        if not 0 < _number(camera["vertical_fov_deg"], "fovy") < 180:
            raise ValueError("Invalid FOV")


@dataclass(frozen=True)
class EncodedPair:
    """Session-neutral immutable bytes; both eyes originate from one snapshot."""

    pair_id: int
    sim_time_s: float
    capture_host_monotonic_s: float
    reset_generation: int
    width: int
    height: int
    cameras_json: bytes
    left: bytes
    right: bytes
    encode_seconds: float

    def packet(self, session_id: str, calibration_generation: int) -> bytes:
        metadata = {
            "schema_version": 1,
            "type": "stereo_pair",
            "session_id": session_id,
            "calibration_generation": calibration_generation,
            "pair_id": self.pair_id,
            "sim_time_s": self.sim_time_s,
            "capture_host_monotonic_s": self.capture_host_monotonic_s,
            "codec": "jpeg",
            "pixel_format": "rgb8",
            "frame": "mabel_chassis_native",
            "width": self.width,
            "height": self.height,
            "eye_order": ["left", "right"],
            "cameras": _json(self.cameras_json),
        }
        validate_metadata(metadata)
        header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        if len(header) > MAX_METADATA or any(
            not 0 < len(x) <= MAX_EYE for x in (self.left, self.right)
        ):
            raise ValueError("Encoded pair budget exceeded")
        return (
            HEADER.pack(b"QTP1", len(header), len(self.left), len(self.right))
            + header
            + self.left
            + self.right
        )


def encode_pair(pair: StereoPair, *, quality: int = 75) -> EncodedPair:
    _integer(quality, "JPEG quality", 1, 95)
    start = time.monotonic()
    height, width = pair.left.rgb.shape[:2]
    _dimensions(width, height)
    eyes, cameras = [], {}
    for side in ("left", "right"):
        eye = getattr(pair, side)
        if (
            eye.name != f"zed_{side}"
            or eye.rgb.shape != (height, width, 3)
            or eye.rgb.dtype != np.uint8
        ):
            raise ValueError("Eyes must be matching RGB8 zed_left/zed_right frames")
        stream = io.BytesIO()
        Image.fromarray(eye.rgb).save(stream, format="JPEG", quality=quality, subsampling=2)
        payload = stream.getvalue()
        if len(payload) > MAX_EYE:
            raise ValueError("JPEG eye budget exceeded")
        eyes.append(payload)
        cameras[side] = {
            "name": eye.name,
            "position_m": eye.position_m.tolist(),
            "quaternion_wxyz": eye.quaternion_wxyz.tolist(),
            "vertical_fov_deg": eye.vertical_fov_deg,
        }
    return EncodedPair(
        pair.pair_id,
        pair.sim_time_s,
        pair.capture_host_monotonic_s,
        pair.reset_generation,
        width,
        height,
        json.dumps(cameras, allow_nan=False).encode(),
        *eyes,
        time.monotonic() - start,
    )


def unpack_pair(packet: bytes, *, decode=False):
    """Diagnostic receiver; verify bounded lengths, JSON and JPEG dimensions."""
    if not isinstance(packet, bytes) or not HEADER.size <= len(packet) <= MAX_PACKET:
        raise ValueError("Invalid packet size")
    magic, meta_size, left_size, right_size = HEADER.unpack_from(packet)
    if (
        magic != b"QTP1"
        or not 0 < meta_size <= MAX_METADATA
        or any(not 0 < n <= MAX_EYE for n in (left_size, right_size))
        or len(packet) != HEADER.size + meta_size + left_size + right_size
    ):
        raise ValueError("Invalid packet framing")
    index = HEADER.size + meta_size
    metadata = _json(packet[HEADER.size : index])
    validate_metadata(metadata)
    left, right = packet[index : index + left_size], packet[index + left_size :]
    decoded = []
    for eye in (left, right):
        with Image.open(io.BytesIO(eye)) as image:
            if image.format != "JPEG" or image.size != (metadata["width"], metadata["height"]):
                raise ValueError("JPEG dimensions/profile differ from metadata")
            image.verify()
        if decode:
            with Image.open(io.BytesIO(eye)) as image:
                decoded.append(np.asarray(image.convert("RGB")).copy())
    return metadata, *(decoded if decode else (left, right))


class LatestSlot(Generic[T]):
    """One value, no accumulated work. Replacing never waits for a consumer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._version = 0
        self.replaced = 0

    def put(self, value: T):
        with self._lock:
            if self._value is not None:
                self.replaced += 1
            self._version += 1
            self._value = value
            return self._version

    def read(self, after=0) -> tuple[int, T | None]:
        with self._lock:
            return self._version, self._value if self._version > after else None

    def clear(self):
        with self._lock:
            self._version += 1
            self._value = None


@dataclass(frozen=True)
class RenderSnapshot:
    """Detached render-only state. Its clock is capture, not completion time."""

    state: SimState
    capture_host_monotonic_s: float

    @classmethod
    def capture(cls, state: SimState):
        # read_state already detaches; copy again so callers can reuse/mutate it.
        from dataclasses import fields, replace

        values = {
            field.name: getattr(state, field.name).copy()
            for field in fields(state)
            if isinstance(getattr(state, field.name), np.ndarray)
        }
        return cls(replace(state, **values), time.monotonic())


class ScratchStereoRenderer:
    """Private replay environment constructed AND used on the video worker.

    A matching generated XML hash is mandatory. No reference to live MjData or
    MjModel crosses the ownership boundary. No live model is modified for video.
    """

    def __init__(self, provenance: dict, *, mabel_root=None, width=640, height=480):
        _dimensions(width, height)
        self.owner = threading.current_thread()
        self.env = ShelfStockingEnv(
            ShelfConfig(**provenance["config"]), seed=provenance["seed"], mabel_root=mabel_root
        )
        actual = hashlib.sha256(self.env.xml.encode()).hexdigest()
        if actual != provenance["model_xml_sha256"] or self.env.model.na != 0:
            self.env.close()
            raise ValueError("Scratch scene mismatch or unsupported actuator activation state")
        self.width, self.height = width, height
        self.root_id = self.env.model.body("mabel_chassis").id

    def _check(self):
        if threading.current_thread() is not self.owner:
            raise RuntimeError("Scratch renderer belongs to its creating thread")

    def render(self, snapshot: RenderSnapshot, pair_id: int) -> StereoPair:
        self._check()
        model, data, state = self.env.model, self.env.data, snapshot.state
        for value, shape in (
            (state.qpos, (model.nq,)),
            (state.qvel, (model.nv,)),
            (state.applied_targets, (model.nu,)),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid render snapshot")
        # Replay into scratch data only. Never step here, even between eyes.
        data.qpos[:] = state.qpos
        data.qvel[:] = state.qvel
        data.ctrl[:] = state.applied_targets
        data.time = state.sim_time_s
        mujoco.mj_forward(model, data)
        root_rotation = data.xmat[self.root_id].reshape(3, 3)
        eyes = []
        for side in ("left", "right"):
            name = f"zed_{side}"
            cid = model.camera(name).id
            position = root_rotation.T @ (data.cam_xpos[cid] - data.xpos[self.root_id])
            rotation = root_rotation.T @ data.cam_xmat[cid].reshape(3, 3)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            fovy = float(model.cam_fovy[cid])
            focal = self.height / (2 * math.tan(math.radians(fovy) / 2))
            eyes.append(
                CameraFrame(
                    name,
                    self.env.render(name, self.width, self.height),
                    position.copy(),
                    quaternion,
                    fovy,
                    {
                        "fx": focal,
                        "fy": focal,
                        "cx": (self.width - 1) / 2,
                        "cy": (self.height - 1) / 2,
                    },
                )
            )
        return StereoPair(
            pair_id,
            state.reset_generation,
            state.sim_time_s,
            snapshot.capture_host_monotonic_s,
            *eyes,
        )

    def close(self):
        self._check()
        self.env.close()


class VideoWorker:
    """Latest snapshot → render → encode → latest encoded pair, independently.

    The render factory runs on the worker; its model/context never cross threads.
    A slow consumer cannot block submit(), physics or control sockets. This is
    best-effort Python scheduling, not a hard real-time guarantee.
    """

    def __init__(self, render_factory: Callable, *, fps=30, quality=75):
        if not 0 < _number(fps, "fps") <= 120:
            raise ValueError("Invalid video rate")
        _integer(quality, "quality", 1, 95)
        self.snapshots: LatestSlot[RenderSnapshot] = LatestSlot()
        self.encoded: LatestSlot[EncodedPair] = LatestSlot()
        self.render_factory, self.fps, self.quality = render_factory, fps, quality
        self._stop = threading.Event()
        self._generation_lock = threading.Lock()
        self._reset_generation = None
        self.thread = None
        self.error = None
        self.stats = {
            "pairs": 0,
            "render_seconds_total": 0.0,
            "encode_seconds_total": 0.0,
            "last_capture_to_encode_s": None,
        }

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Video worker already started")
        self.thread = threading.Thread(target=self._run, name="quest-video", daemon=True)
        self.thread.start()

    def submit(self, state: SimState):
        snapshot = RenderSnapshot.capture(state)
        with self._generation_lock:
            if state.reset_generation != self._reset_generation:
                self.encoded.clear()
                self._reset_generation = state.reset_generation
            self.snapshots.put(snapshot)

    def _run(self):
        renderer = None
        try:
            renderer = self.render_factory()
            version, pair_id, next_render = 0, 0, 0.0
            while not self._stop.is_set():
                if self._stop.wait(max(0, next_render - time.monotonic())):
                    break
                current, snapshot = self.snapshots.read(version)
                if snapshot is None:
                    self._stop.wait(0.002)
                    continue
                version = current
                start = time.monotonic()
                pair = renderer.render(snapshot, pair_id)
                rendered = time.monotonic()
                encoded = encode_pair(pair, quality=self.quality)
                # A reset during rendering invalidates the completed old pair.
                with self._generation_lock:
                    if self._reset_generation == pair.reset_generation:
                        self.encoded.put(encoded)
                self.stats["pairs"] += 1
                self.stats["render_seconds_total"] += rendered - start
                self.stats["encode_seconds_total"] += encoded.encode_seconds
                self.stats["last_capture_to_encode_s"] = (
                    time.monotonic() - pair.capture_host_monotonic_s
                )
                pair_id += 1
                next_render = start + 1 / self.fps
        except Exception as error:
            self.error = error
            self.encoded.clear()
        finally:
            if renderer is not None:
                renderer.close()

    def close(self, timeout=5):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise TimeoutError("Video worker has not stopped; no live simulator data is shared")


def _replace_queue(slot, value):
    """Bounded IPC; an in-flight feeder race may drop this update, never wait."""
    try:
        slot.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        slot.get_nowait()
    except queue.Empty:
        pass
    try:
        slot.put_nowait(value)
    except queue.Full:
        pass


def _render_process(factory, snapshots, results, stop, fps, quality):
    renderer = None
    stats = {
        "pairs": 0,
        "render_seconds_total": 0.0,
        "encode_seconds_total": 0.0,
        "last_capture_to_encode_s": None,
        "render_seconds_samples": [],
    }
    try:
        renderer = factory()
        pair_id = 0
        while not stop.is_set():
            try:
                snapshot = snapshots.get(timeout=0.1)
            except queue.Empty:
                continue
            start = time.monotonic()
            pair = renderer.render(snapshot, pair_id)
            rendered = time.monotonic()
            encoded = encode_pair(pair, quality=quality)
            stats["pairs"] += 1
            stats["render_seconds_total"] += rendered - start
            stats["encode_seconds_total"] += encoded.encode_seconds
            stats["last_capture_to_encode_s"] = time.monotonic() - pair.capture_host_monotonic_s
            stats["render_seconds_samples"].append(rendered - start)
            if len(stats["render_seconds_samples"]) > 7200:
                del stats["render_seconds_samples"][:3600]
            _replace_queue(results, (encoded, stats, None))
            pair_id += 1
            stop.wait(max(0, start + 1 / fps - time.monotonic()))
    except Exception as error:
        _replace_queue(results, (None, stats, f"{type(error).__name__}: {error}"))
        raise
    finally:
        if renderer is not None:
            renderer.close()


class ProcessVideoWorker:
    """Production worker: spawn isolates graphics/GIL from physics and control.

    Both IPC queues have capacity one, in addition to the final latest slot.
    There may also be one item in rendering/sending. No queue ever grows with
    lag. Submit copies detached state but performs no rendering or encoding.
    """

    def __init__(self, render_factory: Callable, *, fps=30, quality=75):
        if not 0 < _number(fps, "fps") <= 120:
            raise ValueError("Invalid video rate")
        _integer(quality, "quality", 1, 95)
        context = multiprocessing.get_context("spawn")
        self._snapshots, self._results = context.Queue(1), context.Queue(1)
        self._stop = context.Event()
        self._process = context.Process(
            target=_render_process,
            args=(render_factory, self._snapshots, self._results, self._stop, fps, quality),
            name="quest-video-render",
            daemon=True,
        )
        self.encoded: LatestSlot[EncodedPair] = LatestSlot()
        self._generation_lock = threading.Lock()
        self._reset_generation = None
        self._collector = threading.Thread(
            target=self._collect, name="quest-video-ipc", daemon=True
        )
        self.error, self.stats = None, {}

    def start(self):
        self._process.start()
        self._collector.start()

    def submit(self, state: SimState):
        snapshot = RenderSnapshot.capture(state)
        with self._generation_lock:
            if self._reset_generation != state.reset_generation:
                self._reset_generation = state.reset_generation
                self.encoded.clear()
        _replace_queue(self._snapshots, snapshot)

    def _collect(self):
        while True:
            try:
                pair, stats, error = self._results.get(timeout=0.1)
            except queue.Empty:
                if not self._process.is_alive():
                    if self._process.exitcode not in (0, None) and self.error is None:
                        self.error = RuntimeError(f"Video process exited {self._process.exitcode}")
                        self.encoded.clear()
                    return
                continue
            self.stats = stats
            if error is not None:
                self.error = RuntimeError(error)
                self.encoded.clear()
            elif pair is not None:
                with self._generation_lock:
                    if pair.reset_generation == self._reset_generation:
                        self.encoded.put(pair)

    def close(self, timeout=5):
        self._stop.set()
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()  # only this private scratch process
            self._process.join(2)
            raise TimeoutError("Private video process failed to stop")
        self._collector.join(1)
        # Stop feeder threads; no queued snapshot is needed after shutdown.
        self._snapshots.cancel_join_thread()
        self._snapshots.close()
        self._results.close()
        if self.error is not None:
            raise self.error


def synthetic_pair(pair_id: int, *, width=640, height=480, captured=None) -> StereoPair:
    """Labelled diagnostic, with common pair ID/time, motion and stereo disparity.

    Synthetic geometry is illustrative (neutral optical orientation), not a
    rendered MABEL pose or a proof of calibrated depth. Real mode uses MuJoCo.
    """
    _integer(pair_id, "pair ID")
    _dimensions(width, height)
    captured = time.monotonic() if captured is None else captured
    eyes = []
    colors = [
        (240, 240, 240),
        (235, 220, 20),
        (20, 220, 220),
        (20, 190, 20),
        (210, 20, 210),
        (210, 20, 20),
        (20, 20, 210),
    ]
    for side, sign in (("left", -1), ("right", 1)):
        image = Image.new("RGB", (width, height), (15, 15, 15))
        draw = ImageDraw.Draw(image)
        for i, color in enumerate(colors):
            draw.rectangle((i * width // 7, 0, (i + 1) * width // 7, height // 3), fill=color)
        cell = max(8, width // 32)
        for row in range(height // 3, height, cell):
            for col in range(0, width, cell):
                level = 90 if (row // cell + col // cell) % 2 else 40
                draw.rectangle((col, row, col + cell - 1, row + cell - 1), fill=(level,) * 3)
        # Both eyes share motion; opposite six-pixel shifts diagnose eye routing.
        cx = int(width * (0.5 + 0.2 * math.sin(pair_id / 15))) - sign * max(2, width // 100)
        cy, radius = height * 2 // 3, max(8, height // 12)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(0, 240, 90))
        scale = max(1, width // 320)
        label = Image.new("RGB", (width // scale, 45), (0, 0, 0))
        text = ImageDraw.Draw(label)
        text.text((5, 3), f"{side.upper()}  PAIR {pair_id}", fill=(255, 255, 255))
        text.text((5, 20), f"host capture {captured:.3f} s", fill=(255, 255, 255))
        image.paste(label.resize((width, 45 * scale), Image.Resampling.NEAREST), (0, height // 3))
        eyes.append(
            CameraFrame(
                f"zed_{side}",
                np.array(image),
                np.array([0.0, sign * 0.0315, 1.2]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                83.3,
                {},
            )
        )
    return StereoPair(pair_id, 0, pair_id / 30, captured, *eyes)


@dataclass(frozen=True)
class PresentationHealth:
    fresh: bool
    reason: str
    capture_age_s: float | None = None
    capture_to_present_s: float | None = None
    uncertainty_s: float | None = None


class VideoService:
    """Inject into #15 ControlServer; no competing control socket/session.

    ``extra_routes={'/quest/video': service.handle}``,
    ``on_video_status=service.on_video_status``. Registry is duck typed:
    bind_video/unbind_video/get; session has session_id, calibration_generation,
    and clock with offset_s/uncertainty_s. No actuation is implemented here.
    """

    def __init__(self, encoded: LatestSlot[EncodedPair], *, max_age=0.3, send_timeout=0.25):
        self.encoded, self.max_age, self.send_timeout = encoded, max_age, send_timeout
        self._sent = {}
        self._feedback = {}
        self._generation = {}
        self.metrics = {
            "sent": 0,
            "send_timeouts": 0,
            "last_send_seconds": None,
            "last_capture_to_send_s": None,
        }

    def disconnect(self, session_id):
        self._sent.pop(session_id, None)
        self._feedback.pop(session_id, None)
        self._generation.pop(session_id, None)

    async def handle(self, connection, registry):
        # Installed explicitly by #15; codec/render-only use has no network dep.
        from websockets.exceptions import ConnectionClosed

        session = None
        try:
            raw = await asyncio.wait_for(connection.recv(), 5)
            if not isinstance(raw, str) or len(raw.encode()) > 1024:
                raise ValueError("Expected bounded video bind text")
            bind = _json(raw)
            _keys(bind, "schema_version type session_id")
            if (
                type(bind["schema_version"]) is not int
                or bind["schema_version"] != 1
                or bind["type"] != "video_bind"
                or not isinstance(bind["session_id"], str)
            ):
                raise ValueError("Invalid video bind")
            session = registry.bind_video(bind["session_id"], connection)
            self.disconnect(session.session_id)
            await asyncio.wait_for(
                connection.send(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "type": "video_bind_ack",
                            "session_id": session.session_id,
                        }
                    )
                ),
                self.send_timeout,
            )
            # Incoming messages after bind are forbidden; monitor disconnect even
            # when there are no frames. Never build a growing queue of send tasks.
            receiver = asyncio.create_task(connection.recv())
            try:
                await self._send_loop(connection, registry, session, receiver)
            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
        except (ValueError, asyncio.TimeoutError, KeyError, TypeError):
            await connection.close(code=1008, reason="Invalid video bind or video timeout")
        except ConnectionClosed:
            pass
        finally:
            if session is not None:
                registry.unbind_video(session.session_id, connection)
                self.disconnect(session.session_id)

    async def _send_loop(self, connection, registry, session, receiver):
        version, previous_id = 0, -1
        while registry.get(session.session_id) is session:
            if receiver.done():
                # recv propagates disconnect; even valid post-bind data is forbidden.
                receiver.result()
                raise ValueError("Unexpected client video message")
            current, pair = self.encoded.read(version)
            if pair is None:
                await asyncio.sleep(0.003)
                continue
            version = current
            now = time.monotonic()
            if (
                pair.pair_id <= previous_id
                or not 0 <= now - pair.capture_host_monotonic_s <= self.max_age
            ):
                continue
            sid, generation = session.session_id, session.calibration_generation
            if self._generation.get(sid) != generation:
                self._sent[sid] = OrderedDict()
                self._feedback.pop(sid, None)
                self._generation[sid] = generation
            packet = pair.packet(sid, generation)
            # Record BEFORE send: localhost/device feedback can arrive as soon as
            # send yields. All access to these maps is on the same event loop.
            sent = self._sent[sid]
            sent[pair.pair_id] = (pair.capture_host_monotonic_s, now)
            while len(sent) > 128:
                sent.popitem(last=False)
            try:
                await asyncio.wait_for(connection.send(packet), self.send_timeout)
            except asyncio.TimeoutError:
                self.metrics["send_timeouts"] += 1
                raise
            previous_id = pair.pair_id
            self.metrics["sent"] += 1
            self.metrics["last_send_seconds"] = time.monotonic() - now
            self.metrics["last_capture_to_send_s"] = (
                time.monotonic() - pair.capture_host_monotonic_s
            )

    async def on_video_status(self, session, message):
        """Transport validates generic schema; this checks pair provenance/age.

        #20 can query health() and must pause on loss. A fresh result alone does
        not establish tracking/calibration safety or enable robot commands.
        """
        sid, now = session.session_id, time.monotonic()
        sequence = _integer(message["sequence"], "video status sequence")
        previous = self._feedback.get(sid)
        if previous is not None and sequence <= previous[0]:
            return
        health = PresentationHealth(False, "no presented pair")
        generation = message["calibration_generation"]
        if generation != session.calibration_generation or self._generation.get(sid) != generation:
            health = PresentationHealth(False, "generation mismatch")
        elif message["stale"]:
            health = PresentationHealth(False, "client stale")
        elif message["presented_pair_id"] is not None:
            item = self._sent.get(sid, {}).get(message["presented_pair_id"])
            clock = session.clock
            if item is None:
                health = PresentationHealth(False, "unknown/expired presented pair")
            elif (
                clock is None
                or clock.uncertainty_s > 0.025
                or now - clock.established_monotonic_s > 10
            ):
                health = PresentationHealth(False, "clock unavailable/uncertain")
            else:
                capture, sent_at = item
                presented = (
                    _number(message["presented_monotonic_s"], "presented time") + clock.offset_s
                )
                reported = (
                    _number(message["reported_monotonic_s"], "reported time") + clock.offset_s
                )
                uncertainty = clock.uncertainty_s
                age, latency = now - capture, presented - capture
                valid = (
                    sent_at - uncertainty <= presented <= reported <= now + uncertainty
                    and latency >= -uncertainty
                    and 0 <= age + uncertainty <= self.max_age
                    and latency + uncertainty <= self.max_age
                    and now - reported + uncertainty <= self.max_age
                )
                health = PresentationHealth(
                    valid,
                    "fresh" if valid else "old/future presentation",
                    age,
                    latency,
                    uncertainty,
                )
        self._feedback[sid] = (sequence, now, generation, health)

    def health(self, session, *, now=None) -> PresentationHealth:
        now = time.monotonic() if now is None else now
        clock = session.clock
        if clock is None or clock.uncertainty_s > 0.025 or now - clock.established_monotonic_s > 10:
            return PresentationHealth(False, "clock unavailable/expired")
        feedback = self._feedback.get(session.session_id)
        if feedback is None or feedback[2] != session.calibration_generation:
            return PresentationHealth(False, "no current generation feedback")
        _, received, _, health = feedback
        if now - received > self.max_age:
            return PresentationHealth(False, "feedback timed out")
        # Repeated reports cannot refresh capture age, even if a client invents
        # a new presentation timestamp for the same old pair.
        if (
            health.capture_age_s is not None
            and health.capture_age_s + (now - received) + (health.uncertainty_s or 0) > self.max_age
        ):
            return PresentationHealth(False, "capture timed out")
        return health
