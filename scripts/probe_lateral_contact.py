#!/usr/bin/env python3
"""Native-MABEL exposed-side push diagnostic; reset fixture earns no pick credit."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.evaluation import StageObserver
from abstract_stack.front_tangential import FrontTangentialStockingScript
from abstract_stack.interactions import CartesianHand, InteractionFailure
from abstract_stack.lateral_contact import LateralPushConfig, LateralStockingScript
from abstract_stack.mabel_controller import MabelController, TrackingConfig
from abstract_stack.packing import dense_shelf_config, row_geometry
from abstract_stack.provenance import policy_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("exposed-side", "front-tangential"), default="exposed-side"
    )
    parser.add_argument("--normal-force", type=float, default=8.0)
    parser.add_argument("--base-x", type=float, help="Explicit reset fixture base X; no live move")
    parser.add_argument("--base-y", type=float, help="Explicit reset fixture base Y; no live move")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--objects", type=int, default=10)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--contact-height", type=float, default=0.028)
    parser.add_argument("--force-target", type=float, default=1.5)
    parser.add_argument("--push-speed", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = dense_shelf_config(
        args.seed,
        product_count=args.objects,
        initially_stocked=args.objects - 1,
        total_free_space=0.01,
    )
    if args.mode == "front-tangential":
        config = replace(config, terminate_on_success=False)
    if args.base_x is not None or args.base_y is not None:
        config = replace(
            config,
            robot_xy=(
                config.robot_xy[0] if args.base_x is None else args.base_x,
                config.robot_xy[1] if args.base_y is None else args.base_y,
            ),
        )
    env = ShelfStockingEnv(config, seed=args.seed)
    script = None
    result = {
        "experiment": args.mode + " push reset fixture",
        "full_task_success": False,
        "pick_credit": 0,
        "seed": args.seed,
        "source": policy_provenance(),
        "scene": asdict(config),
        "model_revision": subprocess.run(
            ["git", "-C", str(mabel_model_path().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "model_xml_sha256": hashlib.sha256(env.xml.encode()).hexdigest(),
    }
    try:
        active = next(p for p in env.products if not p.initially_stocked)
        # Explicit reset construction, before the rollout: active carton at shelf
        # front with an exposed side, neighbors remain in their original bays.
        adr = int(env.model.joint(f"{active.name}_joint").qposadr[0])
        fixture = np.array(
            [
                config.front_x - 0.07,
                active.target[1],
                config.levels[active.level] + active.half_size[2] + 0.001,
            ]
        )
        if args.mode == "front-tangential":
            fixture = np.array(active.target) + [0, 0, 0.001]
        env.data.qpos[adr : adr + 3] = fixture
        mujoco.mj_forward(env.model, env.data)
        result["reset_fixture"] = {
            "product": active.name,
            "position_m": fixture.tolist(),
            "role": "isolated reset skill, not table-to-shelf insertion",
        }
        for _ in range(int(0.5 / env.dt)):
            env.step()
        kind = (
            FrontTangentialStockingScript
            if args.mode == "front-tangential"
            else LateralStockingScript
        )
        script = kind(env, args.output)
        script.product = active
        script.controller = CartesianHand(env, script.config)
        script.tracker = MabelController(env.model, env.data, TrackingConfig())
        script.observer = StageObserver(env)
        np.savez_compressed(
            args.output / "rollout_start.npz",
            qpos=env.data.qpos,
            qvel=env.data.qvel,
            ctrl=env.data.ctrl,
        )
        result["start_pose"] = script.observer.snapshot()
        settings = LateralPushConfig(
            height_m=args.contact_height,
            force_target_N=args.force_target,
            speed_m_s=args.push_speed,
            timeout_s=args.timeout,
            position_tolerance_m=(
                args.tolerance
                if args.tolerance is not None
                else (0.0002 if args.mode == "front-tangential" else 0.001)
            ),
        )
        if args.mode == "front-tangential":
            script.push_front_tangentially(
                direction=args.direction,
                distance_m=args.distance,
                normal_force_N=args.normal_force,
                settings=settings,
            )
        else:
            script.push_laterally(
                direction=args.direction, distance_m=args.distance, settings=settings
            )
        result["primitive_success"] = True
    except InteractionFailure as error:
        result.update(primitive_success=False, failure=str(error))
    finally:
        if script is not None:
            result.update(
                terminal_source=policy_provenance(),
                events=script.events,
                incidents=script.observer.incidents,
                final_pose=script.observer.snapshot(),
                row_geometry=row_geometry(env),
                physics_seconds=float(env.data.time),
                contact=script.episode_evidence(),
            )
            np.savez_compressed(
                args.output / "trace.npz",
                rows=np.asarray(script.rows),
                controls=np.asarray(script.controls),
            )
        np.savez_compressed(
            args.output / "rollout_terminal.npz",
            qpos=env.data.qpos,
            qvel=env.data.qvel,
            ctrl=env.data.ctrl,
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        env.close()
    print(json.dumps({k: result[k] for k in ("primitive_success", "failure") if k in result}))


if __name__ == "__main__":
    main()
