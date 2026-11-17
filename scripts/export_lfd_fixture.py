#!/usr/bin/env python3
"""Collect a tiny native MABEL hold fixture, not a successful stocking demo."""

import argparse
import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from abstract_stack.assets import mabel_model_path
from abstract_stack.base_motion import BaseContactGuard
from abstract_stack.control import stationary_targets
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.lfd import EpisodeRecorder, ObservationField, SensorSample, Snapshot
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
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error("Fixture supports one through 100 control steps")
    env = ShelfStockingEnv(dense_shelf_config(args.seed, product_count=2))
    recorder = None
    try:
        tracking = TrackingConfig()
        controller = MabelController(env.model, env.data, tracking)
        guard = BaseContactGuard(env)
        for _ in range(50):
            env.step(controller.targets({}, dt=env.dt), on_physics_step=guard.step)
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
            ObservationField(
                "observation.images.zed_left",
                (64, 64, 3),
                "uint8 RGB",
                "zed_left",
                "raw native camera channel; unavailable in this state-only fixture",
                dtype="uint8",
            ),
        )
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
            sensor_contract=sensor_metadata,
            sensor_mask_scope="whole modality unavailable if any native entry unavailable",
            scene_configuration=asdict(env.config),
            scene_xml_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
            collector_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            task="Unloaded native MABEL hold; pipeline validation only",
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
            return Snapshot(measured.time_s, measured.sensors)

        observation = observe()
        for _ in range(args.steps):
            nominal = stationary_targets(env.model, env.data)
            nominal[controller.ids] = controller.nominal
            limited = env.model.actuator_ctrllimited != 0
            nominal[limited] = np.clip(
                nominal[limited],
                env.model.actuator_ctrlrange[limited, 0],
                env.model.actuator_ctrlrange[limited, 1],
            )
            privileged = {
                "product_poses": {p.name: env.data.body(p.name).xpos.tolist() for p in env.products}
            }
            recorder.begin(observation, nominal, teacher=privileged)
            # This vector is the clean expert reference; compensation is recorded separately.
            applied = controller.targets(
                dict(zip(contract.names, nominal.tolist(), strict=True)), dt=env.dt
            )
            env.step(applied, on_physics_step=guard.step)
            observation = observe()
            recorder.end(observation, applied)
        recorder.finish(
            status="failure",
            task_success=False,
            reason="Completed unloaded collection fixture; no stocking task attempted",
        )
    except BaseException as error:
        if recorder is not None and not recorder.closed:
            recorder.finish(
                status="unknown",
                task_success=None,
                reason=f"Interrupted collector: {type(error).__name__}: {error}",
            )
        raise
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
