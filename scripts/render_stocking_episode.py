#!/usr/bin/env python3
"""Render archived native actuator trajectories with an exact terminal-state gate."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    from types import SimpleNamespace

    import mujoco
    import numpy as np

    from abstract_stack.checkpoints import native_state
    from abstract_stack.native_views import retain_data_views, retain_model_views
    from abstract_stack.recording import DualVideoRecorder

    inputs = json.loads((args.output / "input_ref.json").read_text())
    episode = Path(inputs["episode"])
    for name, digest in inputs["sha256"].items():
        if hashlib.sha256((episode / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Archived episode changed")
    summary = json.loads((episode / "summary.json").read_text())
    model = mujoco.MjModel.from_xml_string((episode / "initial-scene/scene.xml").read_text())
    views = retain_model_views(model)
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    with np.load(episode / "rollout_start.npz") as archive:
        mujoco.mj_setState(model, data, archive["state"], int(archive["state_spec"]))
    mujoco.mj_forward(model, data)
    start = float(data.time)
    events = summary["events"]
    recorder = None
    matched = False
    with mujoco.Renderer(model, width=640, height=480) as renderer:

        def render(camera, *, width, height):
            if (width, height) != (640, 480):
                raise ValueError("Renderer dimensions changed")
            before = native_state(model, data)
            renderer.update_scene(data, camera=camera)
            image = renderer.render().copy()
            if not np.array_equal(before, native_state(model, data)):
                raise ValueError("Rendering changed native integration state")
            return image

        env = SimpleNamespace(
            model=model,
            data=data,
            robot=True,
            render=render,
            info=lambda: dict(seed=summary["metrics"]["seed"]),
        )
        recorder = DualVideoRecorder(env, args.output / "videos", fps=args.fps)
        recorder.capture(phase="initial recorded state", product="", forces={})
        try:
            with np.load(episode / "trace.npz") as archive:
                controls, samples = archive["controls"], archive["samples"]
                if controls.shape != (len(samples), model.nu):
                    raise ValueError("Missing per-physics-step controls/forces")
                for control, row in zip(controls, samples, strict=True):
                    data.ctrl[:] = control
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    if not np.isclose(data.time, row[0], rtol=0, atol=1e-8):
                        raise ValueError("Recorded trace clock differs from native replay")
                    phase = next(
                        (e for e in reversed(events) if e["sim_time"] <= data.time + 1e-9), None
                    )
                    recorder.capture(
                        phase="recorded " + (phase["phase"] if phase else "initial"),
                        product=phase["product"] if phase else "",
                        forces=dict(hand=float(row[7]), back=float(row[9])),
                    )
            with np.load(episode / "rollout_terminal.npz") as archive:
                matched = bool(np.array_equal(native_state(model, data), archive["state"]))
            if not matched:
                raise ValueError("Native actuator replay differs from recorded terminal state")
            video = recorder.finish(summary)
        finally:
            recorder.close()
    write(
        args.output / "result.json",
        dict(
            kind="verified_recorded_stocking_video",
            full_terminal_integration_bit_equal=matched,
            simulated_seconds=float(data.time) - start,
            original_task_success=summary["task_success"],
            original_failure=summary["failure"],
            replay_fresh_task_credit=0,
            render_native_state_unchanged=True,
            video=video,
        ),
    )
    assert views and data_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or not 0 < args.wall_seconds <= 600:
        parser.error("Bound rendering frequency and wall time")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.episode is None:
        parser.error("Require an archived episode")
    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    args.episode = args.episode.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    names = [
        "initial-scene/scene.xml",
        "initial-scene/manifest.json",
        "summary.json",
        "trace.npz",
        "rollout_start.npz",
        "rollout_terminal.npz",
    ]
    write(
        args.output / "input_ref.json",
        dict(
            episode=str(args.episode),
            sha256={
                name: hashlib.sha256((args.episode / name).read_bytes()).hexdigest()
                for name in names
            },
        ),
    )
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/render_stocking_episode.py",
            "scripts/run_frozen_replanning.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    write(
        args.output / "manifest.json",
        dict(
            command=sys.argv,
            renderer_source_content_id=frozen.content_id,
            archived_policy=json.loads((args.episode / "initial-scene/manifest.json").read_text())[
                "policy_source"
            ],
            wall_budget_seconds=args.wall_seconds,
            replay_fresh_task_credit=0,
        ),
    )
    supervise = runpy.run_path(str(frozen.runtime_root / "scripts/run_frozen_replanning.py"))[
        "supervise"
    ]
    result = supervise(
        [
            sys.executable,
            str(frozen.runtime_root / "scripts/render_stocking_episode.py"),
            "--worker",
            "--output",
            str(args.output),
            "--fps",
            str(args.fps),
        ],
        cwd=root,
        environment=dict(
            os.environ,
            PYTHONPATH=str(frozen.runtime_root / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONFAULTHANDLER="1",
            ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
            ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
        ),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["returncode"] == 0 and not result["timeout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
