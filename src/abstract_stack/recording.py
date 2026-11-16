"""Synchronized native-camera MP4s and actuator-driven head gaze."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


class HeadGaze:
    """Aim the existing head camera using bounded native neck servo references."""

    def __init__(self, model, data, camera="zed_left"):
        self.model, self.data = model, data
        self.camera = model.camera(camera).id
        self.names = ("Neck Yaw", "Neck Pitch")
        self.adr = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.lower = np.array(
            [max(model.joint(n).range[0], model.actuator(n).ctrlrange[0]) for n in self.names]
        )
        self.upper = np.array(
            [min(model.joint(n).range[1], model.actuator(n).ctrlrange[1]) for n in self.names]
        )
        self.command = np.clip(data.qpos[self.adr], self.lower, self.upper)
        self.desired = self.command.copy()
        self.scratch = mujoco.MjData(model)
        self.next_update = 0.0

    def _error(self, q, goal):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.adr] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_camlight(self.model, self.scratch)
        direction = goal - self.scratch.cam_xpos[self.camera]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        forward = -self.scratch.cam_xmat[self.camera].reshape(3, 3)[:, 2]
        return forward - direction

    def targets(self, goal, dt):
        if self.data.time >= self.next_update:
            q = self.data.qpos[self.adr].copy()
            for _ in range(10):
                error = self._error(q, goal)
                jac = np.column_stack(
                    [(self._error(q + np.eye(2)[i] * 1e-4, goal) - error) / 1e-4 for i in range(2)]
                )
                delta = -np.linalg.solve(jac.T @ jac + np.eye(2) * 1e-4, jac.T @ error)
                q = np.clip(q + np.clip(delta, -0.15, 0.15), self.lower, self.upper)
            self.desired = q
            self.next_update = float(self.data.time) + 0.1
        self.command += np.clip(self.desired - self.command, -0.8 * dt, 0.8 * dt)
        self.command = np.clip(self.command, self.lower, self.upper)
        return dict(zip(self.names, self.command.tolist(), strict=True))


class DualVideoRecorder:
    def __init__(
        self,
        env,
        directory,
        *,
        fps=15,
        width=640,
        height=480,
        first_person="zed_left",
        third_person="stocking_third_person",
    ):
        if not isinstance(fps, int) or not 1 <= fps <= 60:
            raise ValueError("video fps must be an integer between 1 and 60")
        if min(width, height) < 64 or width % 2 or height % 2:
            raise ValueError("MP4 dimensions must be even and at least 64 pixels")
        encoder = shutil.which("ffmpeg")
        if not encoder:
            raise RuntimeError("MP4 recording requires ffmpeg on PATH")
        first_camera = env.model.camera(first_person).id
        if not env.robot or env.model.cam_bodyid[first_camera] == 0:
            raise ValueError("first-person recording requires a robot-mounted camera")
        if env.model.cam_bodyid[env.model.camera(third_person).id] != 0:
            raise ValueError("third-person recording requires a world-mounted camera")
        self.env, self.directory = env, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps, self.width, self.height = fps, width, height
        self.cameras = {"third_person": third_person, "first_person": first_person}
        self.encoders, self.logs = {}, {}
        self.times = []
        self.next_time = float(env.data.time)
        self.closed = False
        self.paused_end_frames = 0
        self.encoder_version = subprocess.check_output(
            [encoder, "-version"], text=True, timeout=5
        ).splitlines()[0]
        try:
            for name in self.cameras:
                self.logs[name] = (self.directory / f"{name}-ffmpeg.log").open("wb")
                self.encoders[name] = subprocess.Popen(
                    [
                        encoder,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        str(fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(self.directory / f"{name}.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                    stderr=self.logs[name],
                )
        except BaseException:
            self.close()
            raise

    def _frames(self, label):
        frames = {}
        for name, camera in self.cameras.items():
            frame = Image.fromarray(self.env.render(camera, width=self.width, height=self.height))
            draw = ImageDraw.Draw(frame)
            lines = textwrap.wrap(label, width=max(20, self.width // 6))
            draw.rectangle((0, 0, self.width, 13 * len(lines) + 8), fill="black")
            draw.multiline_text((5, 4), "\n".join(lines), fill="white", spacing=2)
            frames[name] = np.asarray(frame).tobytes()
        return frames

    def _write(self, frames):
        try:
            for name, pixels in frames.items():
                self.encoders[name].stdin.write(pixels)
        except BrokenPipeError as error:
            raise RuntimeError("MP4 encoder stopped; inspect *-ffmpeg.log") from error

    def capture(self, *, phase, product, forces):
        time = float(self.env.data.time)
        if time + 1e-9 < self.next_time:
            return
        label = (
            f"RUNNING | seed={self.env.info()['seed']} t={time:.2f}s | {phase} {product} | "
            f"hand={forces.get('hand', 0):.1f}N back={forces.get('back', 0):.1f}N"
        )
        self._write(self._frames(label))
        self.times.append(time)
        self.next_time += 1 / self.fps
        if len(self.times) % self.fps == 0:
            progress = {
                "status": "recording",
                "sim_time": time,
                "physics_frames": len(self.times),
                "phase": phase,
                "product": product,
            }
            pending = self.directory / "video-progress.tmp"
            pending.write_text(json.dumps(progress) + "\n")
            pending.replace(self.directory / "video-progress.json")

    def finish(self, result):
        task_success = result.get("task_success", result["physical_success"])
        verdict = "SUCCESS" if task_success else "FAILED / PARTIAL"
        reason = result.get("failure") or "All physical gates passed"
        label = f"FINAL STATE (paused) | {verdict} | {reason}"
        try:
            frames = self._frames(label)
            for _ in range(self.fps):
                self._write(frames)
                self.paused_end_frames += 1
        finally:
            self.close()
        metadata = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "cameras": self.cameras,
            "physics_frame_times": self.times,
            "physics_frames": len(self.times),
            "paused_end_frames": self.paused_end_frames,
            "encoded_frames": len(self.times) + self.paused_end_frames,
            "physical_success": result["physical_success"],
            "task_success": task_success,
            "failure": result.get("failure"),
            "synchronization": "both views rendered from the same physics state at each timestamp",
            "first_person_scope": "native robot-mounted camera; neck motion uses actuators",
            "encoder": self.encoder_version,
        }
        (self.directory / "video.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.directory / "video-progress.json").write_text(
            json.dumps({"status": "complete", "physical_success": result["physical_success"]})
            + "\n"
        )
        return metadata

    def close(self):
        if self.closed:
            return
        self.closed = True
        failures = []
        for name, proc in self.encoders.items():
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    failures.append(name)
            try:
                code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                code = -1
            if code:
                failures.append(name)
        for stream in self.logs.values():
            stream.close()
        if failures:
            raise RuntimeError(
                f"MP4 encoding failed for {', '.join(failures)}; inspect encoder logs"
            )
