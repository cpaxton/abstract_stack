"""Commands for asset setup, scene export, viewer and headless simulation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

from abstract_stack.config import SCENES


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(prog="abstract-stack", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-mabel", help="Download the pinned MABEL model and meshes")
    fetch.add_argument("--destination", type=Path)
    random_scene = commands.add_parser(
        "random-scene", help="Generate a seeded carton scene configuration"
    )
    random_scene.add_argument("--seed", type=int, default=0)
    random_scene.add_argument("--objects", type=_positive_int, default=2)
    random_scene.add_argument("--output", type=Path, default=Path("outputs/random-scene.json"))
    batch = commands.add_parser(
        "batch", help="Generate and solve scene seeds serially, preserving every outcome"
    )
    batch.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    batch.add_argument("--objects", type=_positive_int, default=2)
    batch.add_argument("--video", action="store_true")
    batch.add_argument("--fps", type=_positive_int, default=15)
    batch.add_argument("--output-dir", type=Path, default=Path("outputs/stocking-campaign"))
    for command in ("sim", "export", "inspect", "serve", "build-scene", "script", "solve"):
        sub = commands.add_parser(command)
        sub.add_argument("--mabel-root", type=Path, help="Existing open_mabel checkout")
        sub.add_argument("--seed", type=int, default=0)
        selection = sub.add_mutually_exclusive_group()
        selection.add_argument(
            "--scene",
            choices=SCENES,
            default="pick_place" if command in ("script", "solve") else "stocking",
        )
        selection.add_argument("--config", type=Path, help="Custom JSON scene configuration")
        sub.add_argument("--fetch-mabel", action="store_true", help="Fetch MABEL if missing")
        if command not in ("serve", "script", "solve"):
            sub.add_argument("--robot", choices=("mabel", "none"), default="mabel")
        sub.add_argument(
            "--enable-sensors", action="store_true", help="Enable onboard sensors/LiDAR"
        )
        if command == "export":
            sub.add_argument("--output", type=Path, default=Path("outputs/shelf.xml"))
        if command == "build-scene":
            sub.add_argument("--output-dir", type=Path, default=Path("outputs/scene"))
            sub.add_argument("--render", action="store_true", help="Render scene preview images")
        if command in ("script", "solve"):
            selection.add_argument(
                "--random-scene", action="store_true", help="Generate the scene from --seed"
            )
            sub.add_argument(
                "--objects", type=_positive_int, help="Product count for a random scene (default 2)"
            )
            sub.add_argument("--output-dir", type=Path, default=Path("outputs/scripted-stocking"))
            sub.add_argument("--render", action="store_true", help="Save phase previews")
            sub.add_argument(
                "--video", action="store_true", help="Record synchronized third/first-person MP4s"
            )
            sub.add_argument("--fps", type=_positive_int, default=15)
            sub.add_argument("--video-width", type=_positive_int, default=640)
            sub.add_argument("--video-height", type=_positive_int, default=480)
            sub.add_argument(
                "--no-head-tracking", action="store_true", help="Hold the head for ablations"
            )
            sub.add_argument(
                "--first-person-camera",
                default="zed_left",
                choices=("zed_left", "zed_right", "l_wrist_cam", "body_cam"),
            )
            sub.add_argument(
                "--max-force", type=float, default=25.0, help="Contact abort limit in N"
            )
            sub.add_argument(
                "--grip", type=float, default=0.8, help="Finger closure fraction (0, 1]"
            )
            sub.add_argument(
                "--no-best-effort",
                action="store_true",
                help="Hold instead of taking a bounded safe staging step",
            )
            sub.add_argument(
                "--fixed-grasps",
                action="store_true",
                help="Use the original product order and centre grasp for ablations",
            )
            sub.add_argument(
                "--no-gravity-compensation",
                action="store_true",
                help="Disable robot-gravity feedforward for controller ablations",
            )
            sub.add_argument(
                "--ik-orientation-span",
                type=float,
                default=40.0,
                help="Maximum sampled orientation offset from the preferred pose (deg)",
            )
        if command in ("sim", "serve"):
            sub.add_argument("--headless", action="store_true")
            sub.add_argument(
                "--steps",
                type=_positive_int,
                help="Maximum control ticks (sim) or physics steps (serve)",
            )
        if command == "sim":
            sub.add_argument("--screenshot", type=Path)
            sub.add_argument("--camera", default="overview")
            sub.add_argument("--json", type=Path, help="Write episode summary")
            sub.add_argument("--actuator", action="append", default=[], metavar="NAME=VALUE")
        if command == "serve":
            sub.add_argument("--port-offset", type=int, default=0)
            sub.add_argument("--no-cameras", action="store_true", help="Publish state only")
    return parser


def _simulate(env, args):
    action = {}
    for item in args.actuator:
        name, separator, value = item.rpartition("=")
        if not separator:
            raise ValueError(f"Expected actuator NAME=VALUE, got {item!r}")
        action[name] = float(value)
    if args.headless:
        for _ in range(args.steps or 500):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    else:
        import mujoco
        import mujoco.viewer

        reset_requested = False

        def on_key(key):
            nonlocal reset_requested
            if key in (ord("r"), ord("R")):
                reset_requested = True

        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env.model.camera(args.camera).id
            if env.config.environment == "small_house":
                from abstract_stack.house import CUTAWAY_CAMERAS, CUTAWAY_GROUP

                viewer.opt.geomgroup[CUTAWAY_GROUP] = int(args.camera not in CUTAWAY_CAMERAS)
            steps = 0
            while viewer.is_running() and (args.steps is None or steps < args.steps):
                started = time.monotonic()
                with viewer.lock():
                    if reset_requested:
                        env.reset()
                        reset_requested = False
                    _, _, terminated, truncated, _ = env.step(action)
                    steps += 1
                viewer.sync()
                if terminated or truncated:
                    print(json.dumps(env.info(), indent=2))
                    break
                time.sleep(max(0, env.dt - (time.monotonic() - started)))
    if args.screenshot:
        from PIL import Image

        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render(args.camera)).save(args.screenshot)
    summary = json.dumps(env.info(), indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(summary + "\n")
    print(summary)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "solve":
        args.command = "script"
    if args.command == "batch":
        from abstract_stack.campaign import run_campaign

        try:
            report = run_campaign(
                args.output_dir, args.seeds, objects=args.objects, video=args.video, fps=args.fps
            )
        except (OSError, ValueError) as error:
            parser.exit(1, f"Campaign failed: {error}\n")
        if report["successful_runs"] != len(report["runs"]):
            parser.exit(
                2,
                "Campaign retained failed/unverified episodes; inspect its JSON manifest.\n",
            )
        return
    if args.command == "random-scene":
        from abstract_stack.random_scenes import random_shelf_config, scene_manifest

        try:
            config = random_shelf_config(args.seed, product_count=args.objects)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(asdict(config), indent=2) + "\n")
            args.output.with_suffix(".manifest.json").write_text(
                json.dumps(scene_manifest(config, args.seed), indent=2) + "\n"
            )
            print(args.output)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Scene generation failed: {error}\n")
        return
    if args.command == "fetch-mabel":
        from abstract_stack.assets import fetch_mabel

        try:
            print(fetch_mabel(args.destination))
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            parser.exit(1, f"MABEL download failed: {error}\n")
        return
    # Choose the GL backend before importing MuJoCo. Physics-only headless runs
    # do not create a renderer or require a display server.
    if getattr(args, "headless", False) or args.command in ("build-scene", "script"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv

    try:
        if getattr(args, "random_scene", False):
            from abstract_stack.random_scenes import random_shelf_config, scene_manifest

            config = random_shelf_config(args.seed, product_count=args.objects or 2)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "scene.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
            (args.output_dir / "scene-manifest.json").write_text(
                json.dumps(scene_manifest(config, args.seed), indent=2) + "\n"
            )
        else:
            if getattr(args, "objects", None) is not None:
                raise ValueError("--objects requires --random-scene")
            config = load_scene_config(args.config, scene=args.scene)
        if args.command == "script":
            # Keep checking geometric success while the physical protocol
            # verifies final contact/release and retreat; retain the time cap.
            config = replace(config, terminate_on_success=False)
        if args.enable_sensors:
            config = replace(config, enable_sensors=True)
        if args.fetch_mabel and getattr(args, "robot", "mabel") == "mabel":
            from abstract_stack.assets import fetch_mabel, mabel_model_path

            try:
                mabel_model_path(args.mabel_root)
            except FileNotFoundError:
                fetch_mabel(args.mabel_root)
        env = ShelfStockingEnv(
            config,
            robot=getattr(args, "robot", "mabel") == "mabel",
            mabel_root=args.mabel_root,
            seed=args.seed,
        )
        try:
            if args.command == "script":
                from abstract_stack.interactions import ScriptConfig, StockingScript

                result = StockingScript(
                    env,
                    args.output_dir,
                    ScriptConfig(
                        max_force=args.max_force,
                        grip=args.grip,
                        best_effort_staging=not args.no_best_effort,
                        ik_orientation_span_deg=args.ik_orientation_span,
                        gravity_compensation=not args.no_gravity_compensation,
                        adaptive_grasps=not args.fixed_grasps,
                        head_tracking=not args.no_head_tracking,
                    ),
                    render=args.render,
                    video=args.video,
                    video_fps=args.fps,
                    video_width=args.video_width,
                    video_height=args.video_height,
                    first_person=args.first_person_camera,
                ).run()
                print(json.dumps(result, indent=2))
                if result.get("video_error"):
                    parser.exit(1, f"Recording failed: {result['video_error']}\n")
                if not result["physical_success"]:
                    parser.exit(2, f"Interaction stopped: {result['failure']}\n")
            elif args.command == "build-scene":
                from abstract_stack.bundle import build_scene_bundle

                print(build_scene_bundle(env, args.output_dir, render=args.render))
                print(json.dumps(env.info(), indent=2))
            elif args.command == "export":
                print(env.export(args.output))
            elif args.command == "inspect":
                print(f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}")
                for i, name in enumerate(env.actuator_names):
                    limits = (
                        env.model.actuator_ctrlrange[i]
                        if env.model.actuator_ctrllimited[i]
                        else "unlimited"
                    )
                    print(f"{i:2d}  {name:20s} target={env.data.ctrl[i]:.5f} range={limits}")
                print("Products:", ", ".join(p.name for p in env.products))
                print(
                    "Cameras:", ", ".join(env.model.camera(i).name for i in range(env.model.ncam))
                )
            elif args.command == "serve":
                from abstract_stack.robots.mabel import serve_environment

                serve_environment(
                    env,
                    headless=args.headless,
                    port_offset=args.port_offset,
                    steps=args.steps,
                    no_cameras=args.no_cameras,
                )
            else:
                _simulate(env, args)
        finally:
            env.close()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
