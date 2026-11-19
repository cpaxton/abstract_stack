#!/usr/bin/env python3
"""Collect a tiny native MABEL controller fixture, not a successful stocking demo."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from abstract_stack.assets import mabel_model_path
from abstract_stack.base_motion import BaseContactGuard
from abstract_stack.config import load_scene_config
from abstract_stack.control import stationary_targets
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.lfd import EpisodeRecorder, ObservationField, SensorSample, Snapshot
from abstract_stack.lfd.cameras import RawCameraCollector
from abstract_stack.lfd.fixtures import MOTION_PROFILES, fixture_scene_context, motion_reference
from abstract_stack.lfd.native import action_contract, proprioception
from abstract_stack.mabel_controller import MabelController, TrackingConfig
from abstract_stack.packing import dense_shelf_config
from abstract_stack.provenance import policy_provenance
from abstract_stack.sim_sensors import ContactSensorSuite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reset-seed",
        type=int,
        default=0,
        help="Explicit environment jitter seed; 0 preserves historical pilot behavior",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="Explicit scene configuration, including a domain-randomized case",
    )
    parser.add_argument(
        "--domain-context",
        type=Path,
        help="Case identity JSON; stored only in collection provenance",
    )
    parser.add_argument("--motion-profile", choices=tuple(MOTION_PROFILES), default="hold")
    parser.add_argument(
        "--cameras", action="store_true", help="Capture raw first-/third-person RGB"
    )
    parser.add_argument("--camera-width", type=int, default=160)
    parser.add_argument("--camera-height", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Fixture supports one through 100 control steps")
    context = fixture_scene_context(
        scene_seed=args.seed,
        reset_seed=args.reset_seed,
        scene_config=args.scene_config,
        domain_context=args.domain_context,
    )
    scene_config = (
        load_scene_config(args.scene_config)
        if args.scene_config is not None
        else dense_shelf_config(args.seed, product_count=2)
    )
    env = ShelfStockingEnv(scene_config, seed=args.reset_seed)
    recorder = None
    cameras = None
    guard = None
    physics_steps = 0
    control_reports = []
    collection_complete = False
    try:
        tracking = TrackingConfig()
        controller = MabelController(env.model, env.data, tracking)
        guard = BaseContactGuard(env)

        def guarded_step(current_env):
            nonlocal physics_steps
            physics_steps += 1
            guard.step(current_env)

        for _ in range(50):
            env.step(controller.targets({}, dt=env.dt), on_physics_step=guarded_step)
        suite = ContactSensorSuite.mabel(env.model)
        sensor_metadata = suite.metadata()
        n = env.model.nu
        fields = (
            ObservationField(
                "observation.state",
                (3 * n,),
                "native transmission position(rad,m), velocity(rad/s,m/s), effort(Nm,N)",
                "native actuator transmission coordinates",
                "native measured actuator arrays",
            ),
            ObservationField(
                "observation.wrist_wrench",
                (2, 6),
                "N,Nm",
                "native wrist-child frames",
                sensor_metadata["wrist_wrench_provenance"],
            ),
            ObservationField(
                "observation.contact_wrench",
                (2, 6),
                "N,Nm",
                "native wrist-child frames",
                sensor_metadata["contact_wrench_provenance"],
            ),
            ObservationField(
                "observation.tactile",
                (len(suite.geoms), 3),
                "N",
                "native geom frames",
                sensor_metadata["tactile_provenance"],
            ),
        )
        if args.cameras:
            cameras = RawCameraCollector(
                env.model, width=args.camera_width, height=args.camera_height
            )
            camera_fields = cameras.fields
            camera_metadata = cameras.metadata()
            camera_metadata["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "platform default")
        else:
            camera_fields = (
                ObservationField(
                    "observation.images.zed_left",
                    (64, 64, 3),
                    "uint8 RGB",
                    "zed_left",
                    "raw native camera channel; unavailable in this state-only fixture",
                    dtype="uint8",
                ),
            )
            camera_metadata = dict(enabled=False, missing_frame_policy="camera not requested")
        fields += camera_fields
        contract = action_contract(env.model)
        model_revision = subprocess.check_output(
            ["git", "-C", str(mabel_model_path(env.mabel_root).parent), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        base_units = [u.removesuffix("/s") for u in contract.units]
        provenance = dict(
            model_revision=model_revision,
            controller_source=policy_provenance(),
            controller_settings=asdict(tracking),
            scene_seed=args.seed,
            **context,
            sensor_contract=sensor_metadata,
            sensor_mask_scope="whole modality unavailable if any native entry unavailable",
            scene_configuration=asdict(env.config),
            scene_xml_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
            collector_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            camera_contract=camera_metadata,
            fixture_motion=dict(
                profile=args.motion_profile,
                amplitudes=MOTION_PROFILES[args.motion_profile],
                envelope="half sine from clean settled nominal hold; returns to baseline",
                scheduled_transitions=args.steps,
                joint_amplitude_units="rad; Lift amplitude in m",
                scope="controller/sensing fixture; no manipulation attempted",
            ),
            guard_contract=dict(
                name="BaseContactGuard",
                configuration=asdict(guard.config),
                cadence_s=env.config.timestep,
                scope="all products tip/drop; whole robot contact normals; native carton-pair sums",
            ),
            proprioception_order=[
                kind + ":" + name
                for kind in ("position", "velocity", "effort")
                for name in contract.names
            ],
            proprioception_units=base_units
            + [u + "/s" for u in base_units]
            + ["Nm" if u == "rad" else "N" for u in base_units],
        )
        recorder = EpisodeRecorder(
            args.output,
            contract,
            fields,
            fps=round(1 / env.dt),
            physics_dt_s=env.config.timestep,
            provenance=provenance,
            task=f"Unloaded native MABEL {args.motion_profile}; controller/sensing fixture only",
            purpose="pipeline_fixture",
        )

        def observe():
            measured = proprioception(env.data)
            sample = suite.sample(env.data)
            measured.sensors.update(
                {
                    "observation.wrist_wrench": SensorSample(
                        sample["wrist_wrench"],
                        sample["timestamp_s"],
                        bool(np.all(sample["wrist_available"])),
                    ),
                    "observation.contact_wrench": SensorSample(
                        sample["contact_wrench"],
                        sample["timestamp_s"],
                        bool(np.all(sample["wrist_available"])),
                    ),
                    "observation.tactile": SensorSample(
                        sample["tactile"],
                        sample["timestamp_s"],
                        bool(np.all(sample["tactile_available"])),
                    ),
                }
            )
            if cameras is not None:
                measured.sensors.update(cameras.sample(env.data))
            return Snapshot(measured.time_s, measured.sensors)

        observation = observe()
        baseline = stationary_targets(env.model, env.data)
        baseline[controller.ids] = controller.nominal
        limited = env.model.actuator_ctrllimited != 0
        baseline[limited] = np.clip(
            baseline[limited],
            env.model.actuator_ctrlrange[limited, 0],
            env.model.actuator_ctrlrange[limited, 1],
        )
        for index in range(args.steps):
            nominal = motion_reference(
                baseline, contract, args.motion_profile, index=index, steps=args.steps
            )
            privileged = {
                "product_poses": {p.name: env.data.body(p.name).xpos.tolist() for p in env.products}
            }
            recorder.begin(observation, nominal, teacher=privileged)
            # This vector is the clean expert reference; compensation is recorded separately.
            applied = controller.targets(
                dict(zip(contract.names, nominal.tolist(), strict=True)), dt=env.dt
            )
            control_reports.append(controller.last_report)
            env.step(applied, on_physics_step=guarded_step)
            observation = observe()
            recorder.end(observation, applied)
        recorder.finish(
            status="failure",
            task_success=False,
            reason="Completed unloaded collection fixture; no stocking task attempted",
        )
        collection_complete = True
    except BaseException as error:
        if recorder is not None and not recorder.closed:
            recorder.finish(
                status="unknown",
                task_success=None,
                reason=f"Interrupted collector: {type(error).__name__}: {error}",
            )
        raise
    finally:
        if recorder is not None:
            (args.output / "fixture_execution.json").write_text(
                json.dumps(
                    dict(
                        collection_complete=collection_complete,
                        profile=args.motion_profile,
                        scheduled_transitions=args.steps,
                        completed_transitions=len(recorder.nominal),
                        guarded_physics_steps=physics_steps,
                        settling_control_steps=50,
                        control_dt_s=env.dt,
                        physics_dt_s=env.config.timestep,
                        guard=guard.report(),
                        controller_reports=control_reports,
                        fresh_task_credit=0,
                    ),
                    indent=2,
                )
                + "\n"
            )
        if cameras is not None:
            if recorder is not None:
                (args.output / "camera_captures.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in cameras.events)
                )
            cameras.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
