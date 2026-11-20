#!/usr/bin/env python3
"""Render native stocking traces, with exact gates at terminal states and branch points."""

import argparse
import hashlib
import json
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from abstract_stack.checkpoints import load_checkpoint, native_state
from abstract_stack.native_views import retain_data_views, retain_model_views
from abstract_stack.source_store import freeze_source


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def episode_start(directory):
    model = mujoco.MjModel.from_xml_string((directory / "initial-scene/scene.xml").read_text())
    data = mujoco.MjData(model)
    with np.load(directory / "rollout_start.npz") as archive:
        mujoco.mj_setState(model, data, archive["state"], int(archive["state_spec"]))
    mujoco.mj_forward(model, data)
    return model, data


def create_encoder(output, fps):
    return subprocess.Popen(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "960x360",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-threads",
            "2",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def render_sequence(recipe, output, fps, speed):
    segments = recipe["segments"]
    if not segments or ("episode" in segments[0]) == ("checkpoint" in recipe):
        raise ValueError("Start from one recorded episode or one checkpoint")
    if any("episode" in segment for segment in segments[1:]):
        raise ValueError("Only the first segment can start at a reset")
    sources = {}
    if "checkpoint" in recipe:
        directory = Path(recipe["checkpoint"])
        checkpoint = load_checkpoint(directory)
        manifest = directory / "manifest.json"
        sources[str(manifest)] = sha256(manifest)
        model, data = checkpoint.model, checkpoint.data
    else:
        directory = Path(segments[0]["episode"])
        model, data = episode_start(directory)
        for path in (directory / "initial-scene/scene.xml", directory / "rollout_start.npz"):
            sources[str(path)] = sha256(path)
    retained = retain_model_views(model), retain_data_views(model, data)
    start = float(data.time)
    next_frame = start
    frame_count = 0
    last_frame = None
    completed = int(recipe.get("starting_completed", 0))
    segment_reports = []

    with create_encoder(output / "stocking-timeline.mp4", fps) as encoder:
        with mujoco.Renderer(model, width=480, height=360) as renderer:

            def frame(event, label):
                before = native_state(model, data)
                canvas = Image.new("RGB", (960, 360))
                for index, camera in enumerate(("stocking_third_person", "zed_left")):
                    renderer.update_scene(data, camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()), (480 * index, 0))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 960, 40), fill="#15212c")
                draw.text((10, 4), f"Third person | {recipe['title']}", fill="white")
                draw.text((490, 4), f"Robot camera | {speed:g}x simulation", fill="white")
                caption = label
                if "scheduled" in recipe:
                    caption = f"{completed}/{recipe['scheduled']} withdrawals | {label}"
                draw.text((10, 22), caption, fill="#bde7f4")
                product = event.get("product", "")
                phase = event.get("phase", "recorded start").replace("_", " ")
                draw.text((490, 22), f"{data.time:.1f} s sim | {product} | {phase}", fill="#bde7f4")
                if not np.array_equal(before, native_state(model, data)):
                    raise ValueError("Rendering changed the native state")
                return np.asarray(canvas).tobytes()

            for index, segment in enumerate(segments):
                episode = "episode" in segment
                directory = Path(segment["episode"] if episode else segment["continuation"])
                summary = directory / "summary.json"
                if not episode and not summary.is_file():
                    summary = directory / "result.json"
                trace = directory / ("trace.npz" if episode else "continuation.npz")
                sources.update({str(path): sha256(path) for path in (summary, trace)})
                events = sorted(read_json(summary)["events"], key=lambda event: event["sim_time"])
                event_times = [event["sim_time"] for event in events]
                initial = native_state(model, data)
                with np.load(trace) as archive:
                    controls = archive["controls"]
                    times = archive["samples"][:, 0]
                    if episode:
                        terminal_path = directory / "rollout_terminal.npz"
                        sources[str(terminal_path)] = sha256(terminal_path)
                        with np.load(terminal_path) as terminal_file:
                            terminal = terminal_file["state"]
                    else:
                        terminal = archive["terminal_state"]
                        if not np.array_equal(initial, archive["initial_state"]):
                            raise ValueError(f"Native boundary mismatch entering {directory}")
                stop = len(controls)
                if "stop_at_initial_of" in segment:
                    boundary = Path(segment["stop_at_initial_of"]) / "continuation.npz"
                    sources[str(boundary)] = sha256(boundary)
                    with np.load(boundary) as archive:
                        terminal = archive["initial_state"]
                    stop = int(np.searchsorted(times, terminal[0] + 1e-8, side="right"))
                if controls.shape != (len(times), model.nu) or not stop:
                    raise ValueError(f"Missing per-physics-step controls for {directory}")
                event_index = bisect_right(event_times, data.time + 1e-8)
                event = events[event_index - 1] if event_index else {}
                first_frame = frame_count
                for controls_at_step, expected_time in zip(
                    controls[:stop], times[:stop], strict=True
                ):
                    if data.time + 1e-8 >= next_frame:
                        encoder.stdin.write(frame(event, segment["label"]))
                        frame_count += 1
                        next_frame += speed / fps
                    data.ctrl[:] = controls_at_step
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if abs(data.time - expected_time) > 1e-8:
                        raise ValueError(f"Replay clock disagrees with {directory}")
                    while (
                        event_index < len(events) and event_times[event_index] <= data.time + 1e-8
                    ):
                        event = events[event_index]
                        completed = int(event.get("completed_count", completed))
                        event_index += 1
                if not np.array_equal(native_state(model, data), terminal):
                    raise ValueError(f"Native terminal state mismatch for {directory}")
                last_frame = frame(event, segment["label"])
                segment_reports.append(
                    dict(
                        index=index,
                        path=str(directory),
                        start_sim_s=float(initial[0]),
                        end_sim_s=float(data.time),
                        frames=frame_count - first_frame,
                        physics_steps=stop,
                        bit_exact=True,
                        completed_withdrawals=completed,
                    )
                )
                write_json(output / "progress.json", segment_reports)
            # Keep the final measured result visible for one second.
            for _ in range(fps):
                encoder.stdin.write(last_frame)
            frame_count += fps
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg failed to encode the stocking timeline")
    write_json(
        output / "result.json",
        dict(
            replay_verified=True,
            fresh_task_credit=0,
            frames=frame_count,
            playback_speed=speed,
            fps=fps,
            simulated_seconds=float(data.time) - start,
            segments=segment_reports,
            input_file_sha256=sources,
            final_completed_withdrawals=completed,
        ),
    )
    assert retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--speed", type=float, default=16)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or not 1 <= args.speed <= 32:
        parser.error("Choose 1–30 fps and 1–32x playback speed")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/render_stocking_timeline.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    recipe = read_json(args.recipe)
    write_json(args.output / "recipe.json", recipe)
    render_sequence(recipe, args.output, args.fps, args.speed)


if __name__ == "__main__":
    main()
