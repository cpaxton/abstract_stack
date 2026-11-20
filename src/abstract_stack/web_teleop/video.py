"""Stereo head-camera video from the simulation for remote operators.

A dedicated render thread owns its own GL context, ``mujoco.Renderer`` and scratch
``MjData``. The simulation owner only copies ``qpos``/mocap state into a snapshot
after each tick; the render thread forwards kinematics on its copy and renders both
eyes from that single state (no physics step between eyes), so a slow encoder or
client can never block physics.

Two codecs, encoded only while someone watches them:

* ``jpeg``: one binary message per pair in the Quest v1 ``QTP1`` framing
  (``docs/quest_teleop/README.md``): ASCII ``QTP1``, three big-endian uint32
  lengths (metadata, left JPEG, right JPEG), then exactly those byte regions.
* ``h264``: both eyes side by side in one H.264 stream (libx264, constrained
  baseline, zero latency), one access unit per message in ``QTH1`` framing:
  ASCII ``QTH1``, big-endian uint32 metadata and payload lengths, JSON metadata,
  then the Annex-B access unit (SPS/PPS in-band on keyframes). About 7x less
  bandwidth than JPEG for similar quality; browsers decode it with WebCodecs.
  The bitrate is set by the link controller (``link.py``) from the operator's
  measured queueing delay; senders that fall behind report congestion to it.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import threading
import time
from collections.abc import Callable

import mujoco
import numpy as np

log = logging.getLogger(__name__)

PAIR_MAGIC = b"QTP1"
PAIR_HEADER = struct.Struct(">4sIII")
MAX_JPEG_EYE_BYTES = 2 * 1024 * 1024  # contract max_jpeg_eye_bytes


H264_MAGIC = b"QTH1"
H264_HEADER = struct.Struct(">4sII")
# Constrained Baseline, level 3.1: decodable by every WebCodecs implementation.
H264_CODEC_STRING = "avc1.42E01F"
H264_DEFAULT_KBPS = 1200
H264_MIN_KBPS = 250


def h264_available() -> bool:
    """Whether PyAV with libx264 is installed (the ``teleop`` extra)."""
    try:
        import av
    except ImportError:
        return False
    return "libx264" in av.codecs_available


def pack_h264(metadata: dict, access_unit: bytes) -> bytes:
    meta = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    return H264_HEADER.pack(H264_MAGIC, len(meta), len(access_unit)) + meta + access_unit


def unpack_h264(message: bytes) -> tuple[dict, bytes]:
    magic, meta_n, data_n = H264_HEADER.unpack_from(message)
    if magic != H264_MAGIC:
        raise ValueError("bad H.264 message magic")
    start = H264_HEADER.size
    if len(message) != start + meta_n + data_n:
        raise ValueError("H.264 message length mismatch")
    return json.loads(message[start : start + meta_n]), message[start + meta_n :]


class H264Encoder:
    """Low-latency libx264 encoder for side-by-side stereo frames (any OS).

    ``tune=zerolatency`` disables lookahead and B-frames, so every input frame
    leaves as one access unit straight away. (Apple's VideoToolbox buffered four
    frames, about 200 ms at 20 fps, in testing, and is not used.) Keyframes come
    on request (new viewer, decoder error, resynchronisation) and otherwise every
    four seconds. The average bitrate changes between frames without a keyframe.
    The burst cap (VBV: 2x the bitrate over two frames) keeps keyframes from
    flooding a slow link. When the bitrate leaves [cap/4, cap], the encoder reopens
    with a new cap (one keyframe), at most every two seconds, so the controller's
    normal sawtooth never forces keyframes.
    """

    def __init__(self, width: int, height: int, fps: float, kbps: int):
        import av

        self._av = av
        self.width, self.height, self.fps = width, height, fps
        self.keyframes = 0
        self._reopened = float("-inf")
        self._open(kbps)

    def _open(self, kbps: int) -> None:
        import fractions

        context = self._av.CodecContext.create("libx264", "w")
        context.width, context.height = self.width, self.height
        context.pix_fmt = "yuv420p"
        context.time_base = fractions.Fraction(1, 1000)
        context.framerate = fractions.Fraction(round(self.fps * 1000), 1000)
        context.bit_rate = kbps * 1000
        context.gop_size = max(1, round(self.fps * 4))
        context.max_b_frames = 0
        self.cap_kbps = round(kbps * 2)
        context.options = {
            "preset": "veryfast",
            "tune": "zerolatency",
            "profile": "baseline",
            "level": "3.1",
            "maxrate": f"{self.cap_kbps}k",
            "bufsize": f"{max(1, round(2 * self.cap_kbps / self.fps))}k",
        }
        context.open()
        self._context = context
        self._keyframe = True
        self.kbps = kbps

    def request_keyframe(self) -> None:
        self._keyframe = True  # read by the render thread before the next frame

    def set_kbps(self, kbps: int, now: float) -> None:
        if kbps == self.kbps:
            return
        far = kbps > self.cap_kbps or kbps * 4 < self.cap_kbps
        if far and now - self._reopened >= 2.0:
            self._reopened = now
            self._open(kbps)
        else:
            self.kbps = kbps
            self._context.bit_rate = kbps * 1000

    def encode(self, rgb: np.ndarray, pts_ms: int) -> tuple[bytes, bool]:
        frame = self._av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        frame.pts = pts_ms
        if self._keyframe:
            frame.pict_type = self._av.video.frame.PictureType.I
            self._keyframe = False
        packets = self._context.encode(frame)
        keyframe = any(packet.is_keyframe for packet in packets)
        self.keyframes += keyframe
        return b"".join(bytes(packet) for packet in packets), keyframe


def pack_pair(metadata: dict, left: bytes, right: bytes) -> bytes:
    meta = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    return PAIR_HEADER.pack(PAIR_MAGIC, len(meta), len(left), len(right)) + meta + left + right


def unpack_pair(message: bytes) -> tuple[dict, bytes, bytes]:
    magic, meta_n, left_n, right_n = PAIR_HEADER.unpack_from(message)
    if magic != PAIR_MAGIC:
        raise ValueError("bad stereo pair magic")
    start = PAIR_HEADER.size
    if len(message) != start + meta_n + left_n + right_n:
        raise ValueError("stereo pair length mismatch")
    meta = json.loads(message[start : start + meta_n])
    left = message[start + meta_n : start + meta_n + left_n]
    right = message[start + meta_n + left_n :]
    return meta, left, right


# Per-mesh face budget for robot parts the head cameras barely see (see render_model).
PERIPHERAL_FACE_BUDGET = 8000


def _subtree(model, name: str) -> set[int]:
    root = model.body(name).id
    result = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent == root:
                result.add(body)
                break
            parent = int(model.body_parentid[parent])
    return result


def render_model(model, face_budget: int):
    """Render-only copy of ``model`` with MABEL's peripheral meshes simplified.

    Head-camera rendering is triangle-bound. Half of MABEL's visual faces are on the
    base (swerve housings, wheels, steering, chassis, lift mount) and the head
    shell, which the head cameras see small, dark and far away or not at all. Meshes
    used only by bodies outside the torso subtree, or inside the head subtree, are
    simplified to ``face_budget`` faces; the arms, hands and torso stay at full
    resolution. The copy is used only by the render thread: physics, collision
    and kinematics on the live model are untouched. Returns (model, meshes simplified).
    """
    try:
        import fast_simplification  # noqa: F401  (teleop extra)

        from abstract_stack.web_teleop.geometry import _simplified

        robot = _subtree(model, "mabel_chassis")
        peripheral = (robot - _subtree(model, "torso")) | _subtree(model, "head")
    except (ImportError, KeyError):
        return model, 0  # no simplifier, or not the MABEL body layout
    users: dict[int, set[int]] = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            users.setdefault(int(model.geom_dataid[geom]), set()).add(int(model.geom_bodyid[geom]))
    copy = None
    simplified = 0
    for mesh, bodies in sorted(users.items()):
        faces = int(model.mesh_facenum[mesh])
        if faces <= face_budget or not bodies <= peripheral or model.mesh_texcoordadr[mesh] >= 0:
            continue
        if copy is None:
            import copy as copying

            copy = copying.deepcopy(model)
        va, vn, fa = copy.mesh_vertadr[mesh], copy.mesh_vertnum[mesh], copy.mesh_faceadr[mesh]
        na = copy.mesh_normaladr[mesh]
        v, n, f = _simplified(
            copy.mesh_vert[va : va + vn], copy.mesh_face[fa : fa + faces], face_budget
        )
        # Write the simplified mesh into its own slot and shrink the counts; the
        # renderer uploads meshes from these arrays when its context is created.
        copy.mesh_vert[va : va + len(v)] = v
        copy.mesh_normal[na : na + len(n)] = n
        copy.mesh_face[fa : fa + len(f)] = f
        copy.mesh_facenormal[fa : fa + len(f)] = f
        copy.mesh_vertnum[mesh] = len(v)
        copy.mesh_normalnum[mesh] = len(n)
        copy.mesh_facenum[mesh] = len(f)
        simplified += 1
    return (copy if copy is not None else model), simplified


class StereoVideo:
    """Render ``cameras`` at up to ``fps`` while at least one viewer is subscribed."""

    def __init__(
        self,
        model,
        *,
        cameras: tuple[str, str] = ("zed_left", "zed_right"),
        width: int = 640,
        height: int = 480,
        fps: float = 20.0,
        quality: int = 80,
        cutaway_cameras: tuple[str, ...] = (),
        cutaway_group: int | None = None,
        peripheral_face_budget: int = PERIPHERAL_FACE_BUDGET,
        h264_kbps: int = H264_DEFAULT_KBPS,
    ):
        if not 1 <= fps <= 60:
            raise ValueError("video fps must be within [1, 60]")
        if not (64 <= width <= 1920 and 64 <= height <= 1440):
            raise ValueError("video size must be within 64x64 .. 1920x1440")
        for name in cameras:
            model.camera(name)  # raises KeyError for unknown cameras
        self.model = model
        self.peripheral_face_budget = peripheral_face_budget
        self.cameras = cameras
        self.width, self.height, self.fps, self.quality = width, height, fps, quality
        self.fovy = [float(np.ravel(model.camera(c).fovy)[0]) for c in cameras]
        self._options = mujoco.MjvOption()
        if cutaway_group is not None:
            # Same convention as ShelfStockingEnv.render: hide cutaway walls only for
            # the overview cameras; robot cameras see the complete house.
            self._options.geomgroup[cutaway_group] = int(
                not any(c in cutaway_cameras for c in cameras)
            )
        # Offscreen buffer must fit the requested size (shared, write-once vis setting).
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        self._lock = threading.Lock()
        self._qpos = np.zeros(model.nq)
        self._mocap_pos = np.zeros((model.nmocap, 3))
        self._mocap_quat = np.zeros((model.nmocap, 4))
        self._sim_time = 0.0
        self._generation = 0
        self._fresh = threading.Event()
        self._active = threading.Event()
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener: Callable[[str, bytes, bool, int], None] | None = None
        self._demand = {"jpeg": 0, "h264": 0}
        self.codecs = ("h264", "jpeg") if h264_available() else ("jpeg",)
        self.h264_max_kbps = h264_kbps
        self._h264_kbps = h264_kbps
        self._encoder: H264Encoder | None = None
        self._keyframe_wanted = False
        self.on_congestion: Callable[[], None] | None = None
        # Render every Nth period only (link controller, slow links): fewer, better frames.
        self.frame_divider = 1
        self.pair_id = 0
        self.stats = {
            "pairs": 0,
            "render_ms": 0.0,
            "encode_ms": 0.0,
            "bytes": 0,
            "simplified_meshes": 0,
            "h264_kbps": 0,
            "h264_keyframes": 0,
            "h264_bytes": 0,
            "h264_encode_ms": 0.0,
            "congestion_events": 0,
        }
        self.error: str | None = None

    # -- sim-owner side ----------------------------------------------------------------
    def snapshot(self, data, generation: int) -> None:
        """Copy the state needed for rendering (called on the simulation thread)."""
        with self._lock:
            self._qpos[:] = data.qpos
            if self.model.nmocap:
                self._mocap_pos[:] = data.mocap_pos
                self._mocap_quat[:] = data.mocap_quat
            self._sim_time = float(data.time)
            self._generation = generation
        self._fresh.set()

    # -- control -----------------------------------------------------------------------
    def set_listener(self, listener: Callable[[str, bytes, bool, int], None]) -> None:
        """``listener(codec, message, keyframe, pair_id)`` runs on the render thread."""
        self._listener = listener

    def set_demand(self, codec: str, viewers: int) -> None:
        """Render while anyone watches; encode each codec only for its viewers."""
        if codec not in self.codecs:
            raise ValueError(f"codec {codec!r} not available")
        if codec == "h264" and viewers > self._demand["h264"]:
            self.request_keyframe()  # a new viewer starts from a keyframe
        self._demand[codec] = viewers
        (self._active.set if any(self._demand.values()) else self._active.clear)()

    def set_active(self, active: bool) -> None:
        """JPEG demand on/off (kept for single-codec callers)."""
        self.set_demand("jpeg", 1 if active else 0)

    def request_keyframe(self) -> None:
        self._keyframe_wanted = True

    def report_congestion(self) -> None:
        """An H.264 sender had to drop frames; the link controller backs off."""
        self.stats["congestion_events"] += 1
        if self.on_congestion is not None:
            self.on_congestion()

    def set_bitrate_kbps(self, kbps: int) -> None:
        """Target H.264 bitrate (the link controller's choice), capped at the ceiling."""
        self._h264_kbps = max(H264_MIN_KBPS, min(self.h264_max_kbps, int(kbps)))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="web-teleop-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self._active.set()  # wake the thread so it can exit
        self._fresh.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- render thread -----------------------------------------------------------------
    def _emit_h264(self, side_by_side: np.ndarray, metadata: dict, capture: float) -> None:
        started = time.perf_counter()
        kbps = self._h264_kbps
        if self._encoder is None:
            self._encoder = H264Encoder(2 * self.width, self.height, self.fps, kbps)
        encoder = self._encoder
        encoder.set_kbps(kbps, time.monotonic())
        if self._keyframe_wanted:
            self._keyframe_wanted = False
            encoder.request_keyframe()
        data, keyframe = encoder.encode(side_by_side, round(capture * 1000))
        if not data:
            return
        stats = self.stats
        elapsed = (time.perf_counter() - started) * 1e3
        stats["h264_encode_ms"] = 0.8 * stats["h264_encode_ms"] + 0.2 * elapsed
        stats["h264_kbps"], stats["h264_keyframes"] = kbps, encoder.keyframes
        message = pack_h264(
            {
                **metadata,
                "codec": "h264",
                "codec_string": H264_CODEC_STRING,
                "layout": "side_by_side",
                "keyframe": keyframe,
                "bitrate_kbps": kbps,
            },
            data,
        )
        stats["h264_bytes"] = len(message)
        if self._listener is not None:
            self._listener("h264", message, keyframe, metadata["pair_id"])

    def _encode(self, image: np.ndarray) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=self.quality, optimize=False)
        data = buffer.getvalue()
        if len(data) > MAX_JPEG_EYE_BYTES:
            raise ValueError(f"JPEG eye of {len(data)} bytes exceeds the contract limit")
        return data

    def _run(self) -> None:
        model = self.model
        if self.peripheral_face_budget:
            model, simplified = render_model(self.model, self.peripheral_face_budget)
            self.stats["simplified_meshes"] = simplified
        try:
            renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        except Exception as error:  # no GL backend: report instead of crashing the host
            self.error = f"video disabled: {error}"
            log.error(self.error)
            return
        scratch = mujoco.MjData(model)
        period = 1.0 / self.fps
        next_time = time.perf_counter()
        try:
            while self._running.is_set():
                self._active.wait()
                if not self._running.is_set():
                    break
                if not self._fresh.wait(timeout=0.5):
                    continue
                self._fresh.clear()
                with self._lock:
                    scratch.qpos[:] = self._qpos
                    if model.nmocap:
                        scratch.mocap_pos[:] = self._mocap_pos
                        scratch.mocap_quat[:] = self._mocap_quat
                    sim_time, generation = self._sim_time, self._generation
                capture = time.monotonic()
                started = time.perf_counter()
                mujoco.mj_kinematics(model, scratch)
                mujoco.mj_camlight(model, scratch)
                images = []
                for camera in self.cameras:
                    renderer.update_scene(scratch, camera=camera, scene_option=self._options)
                    images.append(renderer.render())
                rendered = time.perf_counter()
                self.pair_id += 1
                metadata = {
                    "pair_id": self.pair_id,
                    "calibration_generation": generation,
                    "capture_host_monotonic_s": capture,
                    "sim_time_s": sim_time,
                    "pixel_format": "rgb8",
                    "width": self.width,
                    "height": self.height,
                    "eye_order": ["left", "right"],
                    "cameras": list(self.cameras),
                    "vertical_fov_deg": self.fovy,
                }
                stats = self.stats
                stats["pairs"] += 1
                stats["render_ms"] = 0.8 * stats["render_ms"] + 0.2 * (rendered - started) * 1e3
                if self._demand["jpeg"]:
                    left, right = (self._encode(image) for image in images)
                    encoded = time.perf_counter()
                    message = pack_pair({**metadata, "codec": "jpeg"}, left, right)
                    stats["encode_ms"] = 0.8 * stats["encode_ms"] + 0.2 * (encoded - rendered) * 1e3
                    stats["bytes"] = len(message)
                    if self._listener is not None:
                        self._listener("jpeg", message, True, self.pair_id)
                if self._demand["h264"]:
                    self._emit_h264(np.hstack(images), metadata, capture)
                spacing = period * max(1, self.frame_divider)
                next_time = max(next_time + spacing, time.perf_counter() - spacing)
                delay = next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        finally:
            renderer.close()
