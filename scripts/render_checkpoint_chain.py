#!/usr/bin/env python3
"""Render saved actuator continuations; reject discontinuities and replay mismatches."""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from abstract_stack.checkpoints import load_checkpoint, model_signature, native_state

    inputs = args.output / "inputs"
    restored = load_checkpoint(inputs / "checkpoint")
    model, data = restored.model, restored.data
    physical_model = model_signature(model)
    active = restored.context["source_summary"]["events"][-1]["product"]
    body = model.body(active).id
    debug = mujoco.MjvCamera()
    debug.type = mujoco.mjtCamera.mjCAMERA_FREE
    debug.lookat[:] = data.xpos[body]
    product = next(p for p in restored.scene["products"] if p["name"] == active)
    debug.distance = product["half_size"][2] + 0.07
    debug.azimuth, debug.elevation = 90, -89
    cameras = dict(
        third_person="stocking_third_person", first_person="zed_left", overhead_debug=debug
    )
    if model.cam_bodyid[model.camera("zed_left").id] == 0:
        raise ValueError("First-person camera must remain robot-mounted")
    encoder = shutil.which("ffmpeg")
    if not encoder:
        raise RuntimeError("Install ffmpeg to encode MP4")
    processes, logs, reports, times = {}, [], [], []
    start = float(data.time)
    next_frame = start
    segment_dirs = sorted(inputs.glob("segment-*"))
    last_trace = None
    withdrawal_time = None
    ending_summary = {}
    try:
        for name in [*cameras, "overview"]:
            width, height = (960, 720) if name == "overview" else (480, 360)
            log = (args.output / f"{name}-ffmpeg.log").open("wb")
            logs.append(log)
            processes[name] = subprocess.Popen(
                [
                    encoder,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(args.fps),
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
                    str(args.output / f"{name}.mp4"),
                ],
                stdin=subprocess.PIPE,
                stderr=log,
            )
        with mujoco.Renderer(model, width=480, height=360) as renderer:

            def capture(trace):
                before = native_state(model, data)
                rotation = data.xmat[body].reshape(3, 3)
                yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
                elapsed = float(data.time) - start
                frames = {}
                for name, camera in cameras.items():
                    renderer.update_scene(data, camera=camera)
                    if name == "overhead_debug":
                        for eye in renderer.scene.camera:
                            eye.frustum_top *= 2
                            eye.frustum_bottom *= 2
                            eye.frustum_center *= 2
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 480, 30), fill="black")
                    draw.text((5, 4), f"{name} | {elapsed:.2f}s | yaw {yaw:.3f} deg", fill="white")
                    draw.text(
                        (5, 17),
                        "Native actuator replay | reset fixture | zero task credit",
                        fill="white",
                    )
                    frames[name] = frame
                overview = Image.new("RGB", (960, 720), "#10151b")
                for name, xy in (
                    ("third_person", (0, 0)),
                    ("overhead_debug", (480, 0)),
                    ("first_person", (0, 360)),
                ):
                    overview.paste(frames[name], xy)
                lines = [
                    "SUPPORTED SQUARING DEVELOPMENT",
                    "",
                    f"Elapsed: {elapsed:.3f} s",
                    f"Active carton yaw: {yaw:.4f} deg",
                    "Yaw gate: 0.15 deg for 0.12 s",
                    "",
                    "One dynamic neighbor prefilled",
                    "Eight cartons remain on table",
                    "No fresh pick / full-stock credit",
                    "Overhead is a debug view",
                ]
                lines += [
                    "",
                    "Phase: unload / retreat"
                    if withdrawal_time is not None and data.time > withdrawal_time
                    else "Phase: supported squaring",
                ]
                if trace is not None:
                    lines += [
                        "",
                        f"Hand {trace[5]:.3f} N | Back {trace[6]:.3f} N",
                        f"Pair {trace[7]:.3f} N | Shelf {trace[8]:.3f} N",
                    ]
                ImageDraw.Draw(overview).multiline_text(
                    (496, 380), "\n".join(lines), fill="white", spacing=6
                )
                frames["overview"] = overview
                if not np.array_equal(before, native_state(model, data)):
                    raise ValueError("Rendering changed native integration state")
                for name, frame in frames.items():
                    processes[name].stdin.write(np.asarray(frame).tobytes())
                times.append(float(data.time))

            capture(None)
            next_frame += 1 / args.fps
            for index, directory in enumerate(segment_dirs):
                source_summary = json.loads((directory / "summary.json").read_text())
                ending_summary = source_summary["continuation"]
                if ending_summary.get("withdrawal"):
                    withdrawal_time = ending_summary["backstop_dwell"]["last_time_s"]
                if index:
                    boundary = load_checkpoint(directory / "boundary-checkpoint")
                    if model_signature(boundary.model) != physical_model or not np.array_equal(
                        native_state(model, data), native_state(boundary.model, boundary.data)
                    ):
                        raise ValueError("Chain changes model or resets the native boundary state")
                with np.load(directory / "continuation.npz") as archive:
                    controls = archive["controls"]
                    terminal = archive["terminal_state"]
                    traces = archive["contact_trace"]
                if controls.shape != (len(traces), model.nu) or not np.isfinite(controls).all():
                    raise ValueError("Invalid controls or missing step-aligned force trace")
                for control, trace in zip(controls, traces, strict=True):
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if not np.isclose(data.time, trace[0], atol=1e-8, rtol=0):
                        raise ValueError("Force trace clock differs from actuator replay")
                    last_trace = trace
                    if data.time + 1e-9 >= next_frame:
                        capture(trace)
                        next_frame += 1 / args.fps
                matched = np.array_equal(native_state(model, data), terminal)
                reports.append(
                    dict(
                        segment=index,
                        terminal_integration_bit_equal=matched,
                        physics_steps=len(controls),
                    )
                )
                write(
                    args.output / "replay-progress.json", dict(segments=reports, frames=len(times))
                )
                if not matched:
                    raise ValueError("Actuator replay differs from the recorded terminal state")
            if not times or times[-1] < data.time - 1e-9:
                capture(last_trace)
    finally:
        failures = []
        for name, process in processes.items():
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -1
            if code:
                failures.append(name)
        for log in logs:
            log.close()
        if failures:
            raise RuntimeError(f"Video encoding failed: {failures}")
    write(
        args.output / "result.json",
        dict(
            kind="verified_actuator_replay_video",
            task_success=False,
            fresh_task_credit=0,
            source_fixture=restored.scene.get("reset_fixture"),
            segments=reports,
            fps=args.fps,
            frame_times_s=times,
            simulated_seconds=float(data.time) - start,
            playback="real-time simulation timestamps; no reset or duplicate segment frames",
            cameras=[*cameras],
            overhead_scope="debug render only; not a student observation",
            render_integration_state_unchanged=True,
            release_retreat_verified=bool(
                (ending_summary.get("withdrawal") or {}).get("release_retreat_verified", False)
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.fps <= 30
        or not 1 <= len(args.segments) <= 10
        or not 0 < args.wall_seconds <= 600
    ):
        parser.error("Bound frames, segment count and render wall time")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    first = args.checkpoint or args.segments[0] / "fixture-checkpoint"
    shutil.copytree(first, inputs / "checkpoint")
    for index, segment in enumerate(args.segments):
        directory = inputs / f"segment-{index:02}"
        directory.mkdir()
        for name in ("continuation.npz", "summary.json", "manifest.json"):
            shutil.copy2(segment / name, directory / name)
        if index:
            shutil.copytree(segment / "inputs/checkpoint", directory / "boundary-checkpoint")
    snapshot = args.output / "source_snapshot"
    shutil.copytree(
        root / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, snapshot / "render.py")
    shutil.copy2(root / "uv.lock", snapshot / "uv.lock")
    shutil.copy2(root / "scripts/run_frozen_replanning.py", snapshot / "supervisor.py")
    write(
        args.output / "manifest.json",
        dict(
            command=sys.argv,
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            sha256={
                p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for parent in (inputs, snapshot)
                for p in parent.rglob("*")
                if p.is_file()
            },
            fresh_task_credit=0,
            task_success=False,
        ),
    )
    write(args.output / "process.json", dict(outcome="unknown", fresh_task_credit=0))
    command = [
        sys.executable,
        str(snapshot / "render.py"),
        "--worker",
        "--segments",
        ".",
        "--output",
        str(args.output),
        "--fps",
        str(args.fps),
    ]
    supervise = runpy.run_path(str(snapshot / "supervisor.py"))["supervise"]
    result = supervise(
        command,
        cwd=root,
        environment=dict(os.environ, PYTHONPATH=str(snapshot / "src"), PYTHONFAULTHANDLER="1"),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    result.update(
        outcome="verified_replay"
        if result["returncode"] == 0 and (args.output / "result.json").exists()
        else "unknown",
        fresh_task_credit=0,
        task_success=False,
    )
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["outcome"] == "verified_replay" else 2


if __name__ == "__main__":
    raise SystemExit(main())
