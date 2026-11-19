"""Stereo head-camera video from the simulation for remote operators.

A dedicated render thread owns its own GL context, ``mujoco.Renderer`` and scratch
``MjData``. The simulation owner only copies ``qpos``/mocap state into a snapshot
after each tick; the render thread forwards kinematics on its copy and renders both
eyes from that single state (no physics step between eyes), so a slow encoder or
client can never block physics.

Each pair is one binary message in the Quest v1 ``QTP1`` framing
(``docs/quest_teleop/README.md``): ASCII ``QTP1``, three big-endian uint32 lengths
(metadata, left JPEG, right JPEG), then exactly those byte regions.
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
    ):
        if not 1 <= fps <= 60:
            raise ValueError("video fps must be within [1, 60]")
        if not (64 <= width <= 1920 and 64 <= height <= 1440):
            raise ValueError("video size must be within 64x64 .. 1920x1440")
        for name in cameras:
            model.camera(name)  # raises KeyError for unknown cameras
        self.model = model
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
        self._listener: Callable[[bytes], None] | None = None
        self.pair_id = 0
        self.stats = {"pairs": 0, "render_ms": 0.0, "encode_ms": 0.0, "bytes": 0}
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
    def set_listener(self, listener: Callable[[bytes], None]) -> None:
        self._listener = listener

    def set_active(self, active: bool) -> None:
        """Render only while someone watches (saves GPU/CPU otherwise)."""
        (self._active.set if active else self._active.clear)()

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
    def _encode(self, image: np.ndarray) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=self.quality, optimize=False)
        data = buffer.getvalue()
        if len(data) > MAX_JPEG_EYE_BYTES:
            raise ValueError(f"JPEG eye of {len(data)} bytes exceeds the contract limit")
        return data

    def _run(self) -> None:
        try:
            renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        except Exception as error:  # no GL backend: report instead of crashing the host
            self.error = f"video disabled: {error}"
            log.error(self.error)
            return
        scratch = mujoco.MjData(self.model)
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
                    if self.model.nmocap:
                        scratch.mocap_pos[:] = self._mocap_pos
                        scratch.mocap_quat[:] = self._mocap_quat
                    sim_time, generation = self._sim_time, self._generation
                capture = time.monotonic()
                started = time.perf_counter()
                mujoco.mj_kinematics(self.model, scratch)
                mujoco.mj_camlight(self.model, scratch)
                images = []
                for camera in self.cameras:
                    renderer.update_scene(scratch, camera=camera, scene_option=self._options)
                    images.append(renderer.render())
                rendered = time.perf_counter()
                left, right = (self._encode(image) for image in images)
                encoded = time.perf_counter()
                self.pair_id += 1
                metadata = {
                    "pair_id": self.pair_id,
                    "calibration_generation": generation,
                    "capture_host_monotonic_s": capture,
                    "sim_time_s": sim_time,
                    "codec": "jpeg",
                    "pixel_format": "rgb8",
                    "width": self.width,
                    "height": self.height,
                    "eye_order": ["left", "right"],
                    "cameras": list(self.cameras),
                    "vertical_fov_deg": self.fovy,
                }
                message = pack_pair(metadata, left, right)
                stats = self.stats
                stats["pairs"] += 1
                stats["render_ms"] = 0.8 * stats["render_ms"] + 0.2 * (rendered - started) * 1e3
                stats["encode_ms"] = 0.8 * stats["encode_ms"] + 0.2 * (encoded - rendered) * 1e3
                stats["bytes"] = len(message)
                if self._listener is not None:
                    self._listener(message)
                next_time = max(next_time + period, time.perf_counter() - period)
                delay = next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        finally:
            renderer.close()
