#!/usr/bin/env python3
"""Reproducible synthetic/native head-video benchmark and #15 server integration."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import platform
import subprocess
import threading
import time
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import PIL

from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop import video as video_module
from abstract_stack.teleop.sim import MujocoTeleopAdapter
from abstract_stack.teleop.video import (
    LatestSlot,
    ProcessVideoWorker,
    ScratchStereoRenderer,
    VideoService,
    encode_pair,
    synthetic_pair,
    unpack_pair,
)


def percentiles(values):
    if not values:
        return None
    return {
        "count": len(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(max(values)),
    }


class Source:
    def __init__(self, args):
        self.args = args
        self.encoded = LatestSlot()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.worker = None
        self.provenance = {"source": args.source, "seed": args.seed}
        # Pin at startup: another worktree/commit operation during a long test
        # must not relabel the code that this process already imported.
        self.revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        files = [
            Path(__file__),
            Path(video_module.__file__),
            Path(inspect.getfile(MujocoTeleopAdapter)),
            Path(inspect.getfile(ShelfConfig)),
        ]
        self.fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.step_times = []
        self.encode_times = []
        self.decode_times = []
        self.pair_bytes = []
        self.render_stats = {}
        self.steps = 0
        self.pairs = 0
        self.thread = threading.Thread(target=self.run, name="video-source", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(60):
            raise TimeoutError("Source startup timed out")
        if self.error:
            raise self.error

    def run(self):
        try:
            if self.args.source == "synthetic":
                self.run_synthetic()
            else:
                self.run_mujoco()
        except Exception as error:
            self.error = error
            self.ready.set()
            self.stop.set()

    def run_synthetic(self):
        self.ready.set()
        pair_id = 0
        while not self.stop.is_set():
            start = time.monotonic()
            pair = synthetic_pair(pair_id, width=self.args.width, height=self.args.height)
            encoded = encode_pair(pair, quality=self.args.quality)
            self.encoded.put(encoded)
            self.measure(encoded)
            pair_id += 1
            self.stop.wait(max(0, 1 / self.args.fps - (time.monotonic() - start)))

    def measure(self, encoded):
        self.pairs += 1
        packet = encoded.packet("local-diagnostic", 0)
        self.encode_times.append(encoded.encode_seconds)
        self.pair_bytes.append(len(packet))
        # Local decode is a Mac codec measurement, never claimed as Quest decode.
        start = time.monotonic()
        unpack_pair(packet, decode=True)
        self.decode_times.append(time.monotonic() - start)
        # Keep optional benchmark samples bounded (up to one minute at120Hz).
        for series in (self.encode_times, self.decode_times, self.pair_bytes):
            if len(series) > 7200:
                del series[:3600]
        if self.args.save_pair and self.pairs == 1:
            path = Path(self.args.save_pair)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(packet)

    def run_mujoco(self):
        config = ShelfConfig(
            product_count=self.args.products,
            time_limit=max(120, self.args.duration + 30),
            terminate_on_success=False,
        )
        with MujocoTeleopAdapter(config, seed=self.args.seed) as sim:
            self.provenance.update(sim.describe())
            self.worker = ProcessVideoWorker(
                partial(
                    ScratchStereoRenderer,
                    sim.describe(),
                    width=self.args.width,
                    height=self.args.height,
                ),
                fps=self.args.fps,
                quality=self.args.quality,
            )
            self.encoded = self.worker.encoded
            self.worker.start()
            self.worker.submit(sim.read_state())
            self.ready.set()
            try:
                while not self.stop.is_set():
                    start = time.monotonic()
                    state = sim.step()  # untouched reset references, zero wheel drive
                    self.steps += 1
                    step_end = time.monotonic()
                    self.step_times.append(step_end - start)
                    if len(self.step_times) > 7200:
                        del self.step_times[:3600]
                    self.worker.submit(state)
                    if self.worker.error is not None:
                        raise self.worker.error
                    self.stop.wait(max(0, sim.control_dt - (time.monotonic() - start)))
            finally:
                self.worker.close()
                self.render_stats = dict(self.worker.stats)

    def close(self):
        self.stop.set()
        self.thread.join(15)
        if self.thread.is_alive():
            raise TimeoutError("Source still stopping")
        if self.error:
            raise self.error


async def collect_latest(source, duration):
    deadline = time.monotonic() + duration
    version = 0
    while time.monotonic() < deadline:
        if source.error is not None:
            raise source.error
        if source.args.source == "mujoco":
            version, encoded = source.encoded.read(version)
            if encoded is not None:
                # to_thread prevents diagnostic Pillow decode blocking control.
                await asyncio.to_thread(source.measure, encoded)
        await asyncio.sleep(0.01)


async def main_async(args):
    source = Source(args)
    # Startup imports/model compilation occur before accepting control sessions.
    await asyncio.to_thread(source.start)
    service = VideoService(source.encoded)
    server = None
    presentation = []

    async def on_status(session, message):
        await service.on_video_status(session, message)
        health = service.health(session)
        presentation.append({"host_monotonic_s": time.monotonic(), **health.__dict__})
        if len(presentation) > 7200:
            del presentation[:3600]

    try:
        if args.mode == "serve":
            try:
                from abstract_stack.teleop.transport import ControlServer, SessionRegistry
            except ImportError as error:
                raise RuntimeError("Serving requires the #15 transport branch/module") from error
            registry = SessionRegistry()
            registry.add_disconnect_callback(service.disconnect)
            server = ControlServer(
                host=args.host,
                port=args.port,
                registry=registry,
                extra_routes={"/quest/video": service.handle},
                on_video_status=on_status,
            )
            await server.start()
            print(json.dumps({"listening": server.bound_port, "source": args.source}), flush=True)
        await collect_latest(source, args.duration)
    finally:
        if server is not None:
            await server.close()
        await asyncio.to_thread(source.close)
    output = {
        "git_revision_at_start": source.revision,
        "worktree_dirty_at_start": source.dirty,
        "source_sha256_at_start": source.fingerprints,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "settings": vars(args),
        "provenance": source.provenance,
        "native_steps": source.steps,
        "pairs_measured": source.pairs,
        "step_seconds": percentiles(source.step_times),
        "encode_seconds": percentiles(source.encode_times),
        "mac_decode_seconds": percentiles(source.decode_times),
        "packet_bytes": percentiles(source.pair_bytes),
        "render_worker": source.render_stats,
        "video_service": service.metrics,
        "presentation_feedback": presentation,
        "timing_limit": "App render submission feedback is not optical motion-to-photon evidence",
    }
    if source.pairs == 0:
        raise RuntimeError("No complete pair was produced; benchmark is incomplete")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(path),
                "pairs": source.pairs,
                "native_steps": source.steps,
                "sent": service.metrics["sent"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "serve"), default="benchmark")
    parser.add_argument("--source", choices=("synthetic", "mujoco"), default="synthetic")
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--products", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", default="outputs/quest-video/report.json")
    parser.add_argument("--save-pair")
    args = parser.parse_args()
    if not 0 < args.duration <= 3600 or not 0 < args.fps <= 120:
        parser.error("duration must be (0,3600], fps (0,120]")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
