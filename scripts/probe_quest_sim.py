#!/usr/bin/env python3
"""Measure the native stationary-command baseline; no success thresholds or teleop."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from abstract_stack.assets import MABEL_REVISION, mabel_model_path
from abstract_stack.config import SCENES, load_scene_config
from abstract_stack.provenance import policy_provenance
from abstract_stack.teleop.sim import BASE_DRIVE, MujocoTeleopAdapter


def record(state):
    return {
        "sim_time_s": state.sim_time_s,
        "step_index": state.step_index,
        "qpos": state.qpos.tolist(),
        "qvel": state.qvel.tolist(),
        "applied_targets": state.applied_targets.tolist(),
        "actuator_lengths": state.actuator_lengths.tolist(),
        "actuator_velocities": state.actuator_velocities.tolist(),
        "actuator_forces": state.actuator_forces.tolist(),
        "root_position_m": state.root_position_m.tolist(),
        "root_quaternion_wxyz": state.root_quaternion_wxyz.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="empty")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mabel-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quest-sim-baseline"))
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
        parser.error("--seconds must be finite and in (0,120]")
    config = load_scene_config(args.config, scene=args.scene)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with MujocoTeleopAdapter(config, mabel_root=args.mabel_root, seed=args.seed) as sim:
        initial = sim.read_state()
        names = initial.actuator_names
        drive_ids = [names.index(name) for name in BASE_DRIVE]
        peaks = np.zeros(len(names))
        base_translation = base_angle = 0.0
        drive_command = 0.0
        records = []
        frames = []

        def sample(state):
            nonlocal base_translation, base_angle, drive_command
            measured = state.actuator_lengths.copy()
            measured[drive_ids] = state.actuator_velocities[drive_ids]
            np.maximum(peaks, np.abs(measured - state.applied_targets), out=peaks)
            base_translation = max(
                base_translation,
                float(np.linalg.norm(state.root_position_m - initial.root_position_m)),
            )
            dot = abs(float(np.dot(state.root_quaternion_wxyz, initial.root_quaternion_wxyz)))
            base_angle = max(base_angle, 2 * math.acos(min(1.0, dot)))
            drive_command = max(
                drive_command, float(np.max(np.abs(state.applied_targets[drive_ids])))
            )
            records.append(record(state))

        def capture(label):
            pair = sim.render_stereo_pair(width=160, height=120)
            for side, eye in (("left", pair.left), ("right", pair.right)):
                Image.fromarray(eye.rgb).save(args.output_dir / f"{label}_{side}.png")
            frames.append(
                {
                    "label": label,
                    "pair_id": pair.pair_id,
                    "reset_generation": pair.reset_generation,
                    "sim_time_s": pair.sim_time_s,
                    "left_camera": pair.left.name,
                    "right_camera": pair.right.name,
                    "baseline_m": float(
                        np.linalg.norm(pair.right.position_m - pair.left.position_m)
                    ),
                }
            )

        sample(initial)
        if args.render:
            capture("initial")
        started = time.monotonic()
        state = initial
        for _ in range(math.ceil(args.seconds / sim.control_dt)):
            state = sim.step()
            sample(state)
            if state.terminated or state.truncated:
                break
        wall_seconds = time.monotonic() - started
        if args.render:
            capture("final")
        source_root = mabel_model_path(args.mabel_root).parents[3]
        actual_revision = None
        if (source_root / ".git").exists():
            git = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=False,
            )
            if git.returncode == 0:
                actual_revision = git.stdout.strip()
        report = {
            "schema_version": 1,
            "purpose": "Stationary-command native physics baseline; no acceptance verdict",
            "source": policy_provenance(),
            "model_reference_revision": MABEL_REVISION,
            "actual_model_checkout_revision": actual_revision,
            "scene": args.scene,
            "model": sim.describe(),
            "effective_mode": "reset position holds; zero wheel commands; no compensation",
            "requested_duration_s": args.seconds,
            "elapsed_sim_time_s": state.sim_time_s,
            "loop_wall_time_s": wall_seconds,
            "real_time_factor_excluding_render": state.sim_time_s / wall_seconds,
            "control_samples": len(records),
            "peak_base_displacement_m_at_control_samples": base_translation,
            "peak_base_rotation_rad_at_control_samples": base_angle,
            "peak_abs_drive_command_rad_s": drive_command,
            "peak_actuator_tracking_errors": {
                name: {
                    "value": float(peaks[i]),
                    "unit": "rad/s" if name in BASE_DRIVE else "m" if name == "Lift" else "rad",
                }
                for i, name in enumerate(names)
            },
            "terminated": state.terminated,
            "truncated": state.truncated,
            "stereo_checks": frames,
            "limitations": [
                "Zero wheel commands do not weld or immobilize the free base.",
                "Control-sample peaks do not bound unsampled physics-substep extrema.",
                "Raw position holds do not establish loaded holds or manipulation success.",
                "Two rendered stereo pairs do not establish a streaming frame rate.",
            ],
        }
    (args.output_dir / "baseline.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "trace.jsonl").open("w") as trace:
        for state_record in records:
            trace.write(json.dumps(state_record, allow_nan=False) + "\n")
    print(args.output_dir / "baseline.json")


if __name__ == "__main__":
    main()
