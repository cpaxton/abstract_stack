#!/usr/bin/env python3
"""Actuator-only MABEL base reposition diagnostic; no stocking credit."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.assets import MABEL_REVISION, mabel_model_path
from abstract_stack.base_motion import (
    BaseContactGuard,
    BaseMotionConfig,
    BaseMotionFailure,
    NativeSwerve,
    screen_base_translation,
)
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.evaluation import StageObserver
from abstract_stack.mabel_controller import MabelController
from abstract_stack.packing import dense_shelf_config
from abstract_stack.provenance import policy_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dy", type=float, default=0.1)
    parser.add_argument("--initially-stocked", type=int, default=9)
    args = parser.parse_args()
    if not np.isfinite(args.dy) or not 0.08 <= abs(args.dy) <= 0.12:
        parser.error("dy must request 0.08–0.12 metres")
    args.output.mkdir(parents=True, exist_ok=False)
    config = BaseMotionConfig()
    scene = dense_shelf_config(
        args.seed, product_count=10, initially_stocked=args.initially_stocked
    )
    env = ShelfStockingEnv(scene, seed=args.seed)
    tracker = MabelController(env.model, env.data)
    observer = StageObserver(env)
    base = env.model.body("mabel_chassis").id
    guard = BaseContactGuard(env, config)
    phase = "settle"
    rows, controls = [], []
    result = dict(
        experiment="native actuator-only base translation",
        base_motion_success=False,
        full_task_success=False,
        pick_credit=0,
        scene=asdict(scene),
        config=asdict(config),
        requested_model_pin=MABEL_REVISION,
        model_revision=subprocess.run(
            ["git", "-C", str(mabel_model_path().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        model_xml_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
        source=policy_provenance(),
        seed=args.seed,
    )
    velocity = np.zeros(6)

    def sample(current):
        observer.sample(phase, "base_motion")
        observation = guard.step(current)
        velocity[:] = observation["velocity_world"]
        rows.append(
            [
                observation["sim_time"],
                *observation["base_xyz_m"],
                *observation["base_wxyz"],
                *velocity,
                observation["robot_fixture_N"],
                observation["robot_product_N"],
                observation["carton_pair_N"],
            ]
        )

    def state(path):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(env.model, spec))
        mujoco.mj_getState(env.model, env.data, values, spec)
        np.savez_compressed(path, state=values, state_spec=int(spec))

    try:
        for _ in range(int(1.2 / env.dt)):
            action = tracker.targets({}, dt=env.dt)
            env.step(action, on_physics_step=sample)
            controls.append([float(env.data.time), *action])
        result["preparation_incidents"] = observer.incidents.copy()
        result["preparation_guard"] = guard.report()
        guard = BaseContactGuard(env, config, max_product_displacement_m=0.002)
        swerve = NativeSwerve(env.model, env.data, config)
        initial = env.data.xpos[base].copy()
        rotation = env.data.xmat[base].reshape(3, 3)
        initial_yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        goal = initial[:2] + [0, args.dy]
        result.update(
            initial_base_xyz_m=initial.tolist(),
            desired_base_xy_m=goal.tolist(),
            initial_product_poses=observer.snapshot(),
            wheel_interface=[
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in module.items()}
                for module in swerve.modules
            ],
        )
        state(args.output / "rollout_start.npz")
        phase = "planning"
        screening = screen_base_translation(env, [0, args.dy], config=config)
        result["path_screening"] = screening
        if not screening["accepted"]:
            raise BaseMotionFailure("base_motion: whole-robot path rejected before wheel motion")
        phase, held = "base_translation", 0.0
        for _ in range(int(config.motion_timeout_s / env.dt)):
            rotation = env.data.xmat[base].reshape(3, 3)
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            yaw_error = float(np.arctan2(np.sin(initial_yaw - yaw), np.cos(initial_yaw - yaw)))
            if abs(yaw_error) > config.path_yaw_margin_rad:
                raise BaseMotionFailure("base_motion: yaw left the screened envelope")
            error = goal - env.data.xpos[base, :2]
            mujoco.mj_objectVelocity(
                env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, base, velocity, 0
            )
            twist = np.r_[1.2 * error - 0.3 * velocity[3:5], 2 * yaw_error - 0.4 * velocity[2]]
            arrived = (
                np.linalg.norm(error) <= config.position_tolerance_m
                and abs(yaw_error) <= config.yaw_tolerance_rad
            )
            if arrived:
                twist[:] = 0
            action = tracker.targets(swerve.targets(twist, dt=env.dt), dt=env.dt)
            env.step(action, on_physics_step=sample)
            controls.append([float(env.data.time), *action])
            stopped = np.linalg.norm(velocity[3:5]) < 0.003 and abs(velocity[2]) < 0.01
            held = held + env.dt if arrived and stopped else 0.0
            if held >= 0.3:
                break
        else:
            raise BaseMotionFailure("base_motion: measured pose/stop dwell timeout")
        phase = "braked_hold"
        for _ in range(int(0.5 / env.dt)):
            action = tracker.targets(swerve.targets([0, 0, 0], dt=env.dt), dt=env.dt)
            env.step(action, on_physics_step=sample)
            controls.append([float(env.data.time), *action])
        final_error = float(np.linalg.norm(goal - env.data.xpos[base, :2]))
        final_rotation = env.data.xmat[base].reshape(3, 3)
        final_yaw = float(np.arctan2(final_rotation[1, 0], final_rotation[0, 0]))
        final_yaw_error = abs(
            float(np.arctan2(np.sin(final_yaw - initial_yaw), np.cos(final_yaw - initial_yaw)))
        )
        result.update(
            final_position_error_m=final_error,
            final_yaw_error_rad=final_yaw_error,
            final_speed_m_s=float(np.linalg.norm(velocity[3:5])),
        )
        if (
            final_error > config.position_tolerance_m
            or final_yaw_error > config.yaw_tolerance_rad
            or np.linalg.norm(velocity[3:5]) >= 0.003
            or abs(velocity[2]) >= 0.01
            or abs(env.data.xpos[base, 2] - initial[2]) > 0.005
            or max(guard.maximum_product_displacement_m.values()) > 0.002
        ):
            raise BaseMotionFailure("base_motion: final pose/product disturbance guard")
        if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied):
            raise BaseMotionFailure("base_motion: unexpected external force channels")
        result["base_motion_success"] = True
    except BaseMotionFailure as error:
        result["failure"] = str(error)
    finally:
        result.update(
            terminal_source=policy_provenance(),
            final_base_xyz_m=env.data.xpos[base].tolist(),
            final_base_wxyz=env.data.xquat[base].tolist(),
            final_product_poses=observer.snapshot(),
            product_incidents=observer.incidents,
            product_maximum_displacement_m=observer.maximum_displacements,
            peak_contact_normal_N=guard.peaks,
            base_contact_guard=guard.report(),
            physics_seconds=float(env.data.time),
            interventions=env.interventions,
            external_force_channels_zero=bool(
                not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
            ),
        )
        state(args.output / "rollout_terminal.npz")
        np.savez_compressed(
            args.output / "trace.npz",
            physics_samples=np.asarray(rows),
            timed_actuator_targets=np.asarray(controls),
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        env.close()
    print(
        json.dumps(
            {key: result[key] for key in ("base_motion_success", "failure") if key in result}
        )
    )


if __name__ == "__main__":
    main()
