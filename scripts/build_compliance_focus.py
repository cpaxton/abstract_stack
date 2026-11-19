#!/usr/bin/env python3
"""Build measured last-millimeter checkpoint frames from recorded actuator replay."""

import argparse
import hashlib
import html
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def nominal_at(summary, controller, step, interval):
    """Recover only recorded nominal arm/lift references, never applied labels."""
    known = dict(controller["nominal_subset"])
    cumulative, integral = 0, 0.0
    for record in summary["continuation"]["replans"]:
        executed = record.get("executed_steps", 0)
        if step <= cumulative:
            break
        used = min(executed, step - cumulative)
        metadata = next(
            (
                m
                for m in record["candidate_metadata"]
                if m.get("name") == record["evaluation"]["selected_name"]
            ),
            None,
        )
        if metadata and metadata.get("nominal_references") and used:
            names, trace = metadata["nominal_names"], metadata["nominal_references"]
            tick, partial = divmod(used - 1, interval)
            previous = [known[n] for n in names] if tick == 0 else trace[tick - 1]
            known.update(
                {
                    n: float(a + (b - a) * (partial + 1) / interval)
                    for n, a, b in zip(names, previous, trace[tick], strict=True)
                }
            )
        integral = record.get("yaw_integral_lever_m", 0.0)
        cumulative += executed
        if step <= cumulative:
            break
    return known, integral


def worker(args):
    import mujoco
    import numpy as np
    from PIL import Image

    from abstract_stack.checkpoints import (
        load_checkpoint,
        native_state,
        save_checkpoint,
        verify_checkpoint,
    )
    from abstract_stack.contact_finish import BackstopDwell
    from abstract_stack.contact_rollout import ContactMonitor, RolloutScene
    from abstract_stack.native_views import retain_data_views

    started = time.monotonic()
    recipe = json.loads((args.output / "recipe.json").read_text())
    rows, failures = [], []
    for task in recipe["trajectories"]:
        try:
            checkpoint = load_checkpoint(args.output / "inputs/checkpoint")
            model, data = checkpoint.model, checkpoint.data
            source = args.output / "inputs" / task["name"]
            process = json.loads((source / "process.json").read_text())
            if process.get("returncode") != 0 or process.get("timeout"):
                raise ValueError("Source trajectory has an unknown/failed process outcome")
            summary = json.loads((source / "summary.json").read_text())
            with np.load(source / "candidates.npz", allow_pickle=False) as archive:
                initial = archive["initial_state"].copy()
            if not np.array_equal(initial, native_state(model, data)):
                raise ValueError("Source trajectory initial state differs from the checkpoint")
            with np.load(source / "continuation.npz", allow_pickle=False) as archive:
                controls, terminal = archive["controls"].copy(), archive["terminal_state"].copy()
            if (
                controls.shape != (summary["continuation"]["executed_physics_steps"], model.nu)
                or not np.isfinite(controls).all()
            ):
                raise ValueError("Source controls are incomplete")
            manifest, original = checkpoint.scene, checkpoint.context["source_summary"]
            active = summary["active"]
            product = next(p for p in manifest["products"] if p["name"] == active)
            level = manifest["config"]["levels"][product["level"]]
            scene = RolloutScene(
                tuple(p["name"] for p in manifest["products"]),
                active,
                "mabel_chassis",
                "left_palm",
                ("supply_table", f"shelf_board_{product['level']}"),
                tuple(product["half_size"]),
                level,
                back_geom="shelf_back",
                base_root="mabel_chassis",
            )
            monitor = ContactMonitor(model, data, scene)
            yaw_limit = next(
                e["requested_backstop_yaw_tolerance_deg"]
                for e in reversed(original["events"])
                if "requested_backstop_yaw_tolerance_deg" in e
            )
            dwell = BackstopDwell(
                float(model.opt.timestep),
                original["controller"]["contact_threshold"],
                original["controller"]["contact_dwell"],
                original["controller"]["payload_upright_degrees"],
                yaw_limit,
            )
            selected = {f["step"]: f["name"] for f in task["frames"]}
            if any(type(s) is not int or not 0 <= s <= len(controls) for s in selected):
                raise ValueError("Focus frame is outside the retained trajectory")
            snapshots = []
            for step in range(len(controls) + 1):
                if step:
                    data.ctrl[:] = controls[step - 1]
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                if np.any(data.warning.number) or not np.isfinite(native_state(model, data)).all():
                    raise ValueError("Native replay became invalid")
                failure = monitor.sample(
                    data, elapsed=0.0 if step == 0 else float(model.opt.timestep)
                )
                if failure:
                    raise ValueError(f"Source replay guard: {failure}")
                dwell.update(data.time, monitor.last, data.xmat[monitor.active].reshape(3, 3))
                if step in selected:
                    snapshot = mujoco.MjData(model)
                    mujoco.mj_copyData(snapshot, model, data)
                    snapshots.append(
                        (
                            step,
                            selected[step],
                            snapshot,
                            retain_data_views(model, snapshot),
                            monitor.state_dict(),
                            dwell.report(),
                        )
                    )
            if native_state(model, data).tobytes() != terminal.tobytes():
                raise ValueError(
                    "Actuator replay does not match the retained terminal state bit for bit"
                )
            for step, name, snapshot, views, monitor_state, dwell_state in snapshots:
                directory = args.output / "frames" / name
                directory.mkdir(parents=True)
                controller = dict(checkpoint.controller)
                nominal, integral = nominal_at(summary, controller, step, 10)
                origin = summary["generation"]["nominal_origin_m"]
                controller.update(
                    nominal_subset=nominal,
                    applied_controls=snapshot.ctrl.tolist(),
                    control_dt_s=0.02,
                    yaw_integral_lever_m=integral,
                    motion_bounds=dict(
                        lower_m=[origin[0] - 0.03, product["target"][1] - 0.02, level + 0.023],
                        upper_m=[origin[0] + 0.03, product["target"][1] + 0.02, level + 0.053],
                    ),
                )
                save_checkpoint(
                    directory / "checkpoint",
                    model,
                    snapshot,
                    scene=manifest,
                    controller=controller,
                    context=dict(
                        source_summary=original,
                        source_trajectory=task["directory"],
                        source_step=step,
                        original_outcome=original.get("failure"),
                        source_trajectory_task_success=False,
                        contact_monitor=monitor_state,
                        backstop_dwell=dwell_state,
                        control_phase_step=step % 10,
                        source_replay_terminal_bit_equal=True,
                    ),
                    scene_xml=(checkpoint.directory / "scene.xml").read_text(),
                    validation_controls=np.tile(snapshot.ctrl, (20, 1)),
                )
                verification = verify_checkpoint(directory / "checkpoint")
                if not verification["verified"]:
                    raise ValueError("Focus checkpoint's private replay witness differs")
                write(directory / "verification.json", verification)
                cameras = {}
                options = mujoco.MjvOption()
                options.sitegroup[:] = 0
                before = native_state(model, snapshot)
                debug_camera = mujoco.MjvCamera()
                debug_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                debug_camera.lookat[:] = snapshot.xpos[monitor.active]
                debug_camera.distance = float(product["half_size"][2] + 0.07)
                debug_camera.azimuth = 90
                debug_camera.elevation = -89
                with mujoco.Renderer(model, width=320, height=240) as renderer:
                    for camera in ("stocking_third_person", "zed_left", "dense_front"):
                        renderer.update_scene(snapshot, camera=camera, scene_option=options)
                        Image.fromarray(renderer.render()).save(directory / f"{camera}.png")
                        cameras[camera] = f"frames/{name}/{camera}.png"
                    renderer.update_scene(snapshot, camera=debug_camera, scene_option=options)
                    # Widen only the debug render projection while keeping
                    # the camera below the next shelf board. No model/data
                    # parameter or student camera is changed.
                    for eye in renderer.scene.camera:
                        eye.frustum_top *= 2
                        eye.frustum_bottom *= 2
                        eye.frustum_center *= 2
                    Image.fromarray(renderer.render()).save(directory / "focus_overhead.png")
                    cameras["debug_overhead_not_student"] = f"frames/{name}/focus_overhead.png"
                if not np.array_equal(before, native_state(model, snapshot)):
                    raise ValueError("Rendering changed native integration state")
                rotation = snapshot.xmat[monitor.active].reshape(3, 3)
                extent = np.abs(rotation) @ product["half_size"]
                back = model.geom("shelf_back").id
                gap = (
                    snapshot.xpos[monitor.active, 0]
                    - extent[0]
                    - (snapshot.geom_xpos[back, 0] + model.geom_size[back, 0])
                )
                row = dict(
                    name=name,
                    source_trajectory=task["directory"],
                    source_step=step,
                    simulation_time_s=float(snapshot.time),
                    nominal_back_plane_gap_m=float(gap),
                    yaw_degrees=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                    forces=monitor_state["last"],
                    incidents=monitor_state["incidents"],
                    checkpoint=f"frames/{name}/checkpoint",
                    cameras=cameras,
                    replay_verified=True,
                    debug_camera=dict(
                        lookat=debug_camera.lookat.tolist(),
                        distance=debug_camera.distance,
                        azimuth=90,
                        elevation=-89,
                        render_frustum_scale=2,
                        role="debug_only_not_student_observation",
                    ),
                    fresh_task_credit=0,
                    task_success=False,
                )
                write(directory / "frame.json", row)
                rows.append(row)
                assert views
        except (OSError, ValueError, RuntimeError, StopIteration) as error:
            failures.append(dict(trajectory=task["name"], error=str(error), fresh_task_credit=0))
        write(
            args.output / "progress.json", dict(frames=rows, failures=failures, fresh_task_credit=0)
        )
    result = dict(
        kind="development_compliance_focus_bank",
        scheduled_frames=sum(len(t["frames"]) for t in recipe["trajectories"]),
        frames=rows,
        failures=failures,
        wall_seconds=time.monotonic() - started,
        fresh_task_credit=0,
        task_success=False,
    )
    write(args.output / "bank.json", result)
    cards = []
    for row in rows:
        images = "".join(
            f'<figure><img width="320" src="{html.escape(path)}">'
            f"<figcaption>{html.escape(camera)}</figcaption></figure>"
            for camera, path in row["cameras"].items()
        )
        cards.append(
            f"<section><h2>{html.escape(row['name'])}</h2><p>Yaw {row['yaw_degrees']:.3f}°; "
            f"back gap {1000 * row['nominal_back_plane_gap_m']:.3f} mm; "
            f"back force {row['forces']['back_N']:.3f} N. Zero new task credit.</p>"
            f"<div>{images}</div></section>"
        )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Compliance focus checkpoints</title>'
        "<style>body{font:16px system-ui;margin:32px;background:#fafafa}"
        "section{margin:24px 0}div{display:flex;flex-wrap:wrap}figure{margin:8px}"
        "img{border:1px solid #ddd}</style><h1>Compliance focus checkpoints</h1>"
        "<p>Native actuator replay; failed development continuations. "
        "These images do not establish manipulation success.</p>" + "".join(cards)
    )
    return 0 if len(rows) == result["scheduled_frames"] and not failures else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path("configs/compliance_focus.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=120)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 180:
        parser.error("Keep wall budget within180 seconds")
    args.output = args.output.resolve()
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.worker:
        return worker(args)
    from abstract_stack.checkpoint_manifest import inspect_checkpoint

    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    directory, _ = inspect_checkpoint(args.checkpoint)
    shutil.copytree(directory, inputs / "checkpoint")
    recipe = json.loads(args.recipe.read_text())
    write(args.output / "recipe.json", recipe)
    for task in recipe["trajectories"]:
        destination = inputs / task["name"]
        destination.mkdir()
        for name in (
            "candidates.npz",
            "continuation.npz",
            "summary.json",
            "process.json",
            "manifest.json",
        ):
            shutil.copy2(Path(task["directory"]) / name, destination / name)
    snapshot = args.output / "source_snapshot"
    shutil.copytree(
        root / "src/abstract_stack",
        snapshot / "src/abstract_stack",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(Path(__file__), snapshot / "build.py")
    shutil.copy2(root / "scripts/run_frozen_replanning.py", snapshot / "supervisor.py")
    write(
        args.output / "manifest.json",
        dict(
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            hashes={
                p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for parent in (inputs, snapshot)
                for p in parent.rglob("*")
                if p.is_file()
            },
            wall_budget_seconds=args.wall_seconds,
        ),
    )
    command = [
        sys.executable,
        str(snapshot / "build.py"),
        "--worker",
        "--checkpoint",
        str(inputs / "checkpoint"),
        "--output",
        str(args.output),
    ]
    write(
        args.output / "process.json", dict(returncode=None, outcome="unknown", fresh_task_credit=0)
    )
    supervise = runpy.run_path(str(snapshot / "supervisor.py"))["supervise"]
    result = supervise(
        command,
        cwd=root,
        environment=dict(os.environ, PYTHONPATH=str(snapshot / "src"), PYTHONFAULTHANDLER="1"),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    result.update(fresh_task_credit=0, task_success=False)
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
