"""MABEL RobotBackend and stationary-control adapter for home_robot_v3 / Emet."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

from abstract_stack.assets import mabel_model_path
from abstract_stack.control import stationary_targets


class MabelStationaryControl:
    """Implement Emet's stationary-control protocol, including the tendon lift."""

    def write_ctrl_with_spec_hold(self, model, data, spec, hold):
        data.ctrl[:] = stationary_targets(model, data)
        if hold is not None:
            for name, target in zip(spec.actuator_names, hold, strict=True):
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    data.ctrl[aid] = target

    def sync_ctrl_and_spec_hold(self, model, data, spec, hold):
        self.write_ctrl_with_spec_hold(model, data, spec, None)
        for i, name in enumerate(spec.actuator_names):
            hold[i] = data.actuator(name).ctrl[0]


class MabelBackend(RobotBackend):
    """59 command channels; Lift reports the sum of both physical lift stages."""

    def __init__(self, mabel_root=None):
        self._mabel_root = mabel_root

    def get_spec(self) -> RobotSpec:
        path = mabel_model_path(self._mabel_root)
        root = ET.parse(path).getroot()
        actuators = [a.get("name") for a in root.find("actuator")]
        joints = ["Lift Lower" if name == "Lift" else name for name in actuators]
        return RobotSpec(
            name="mabel",
            dof=len(actuators),
            joint_names=joints,
            actuator_names=actuators,
            camera_names=["zed_left", "l_wrist_cam", "r_wrist_cam"],
            urdf_path=None,
            mjcf_path=str(path),
            base_link_name="mabel_chassis",
            footprint=Footprint(length=0.50, width=0.56),
            arm_chains={
                side: ArmChain(
                    joint_names=tuple(f"{side.title()} Arm {i}" for i in range(1, 8)),
                    actuator_names=tuple(f"{side.title()} Arm {i}" for i in range(1, 8)),
                    ee_body=f"{side}_palm",
                    base_freejoint_name="base_root",
                )
                for side in ("left", "right")
            },
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        kwargs.setdefault("start_immediately", False)
        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        raise NotImplementedError("Use the shelf environment's MuJoCo model for MABEL kinematics")

    def create_mujoco_stationary_control(self):
        return MabelStationaryControl()


def register_mabel() -> None:
    """Register only in the current process; do not patch the upstream package."""
    from emet.robots import ROBOT_REGISTRY

    ROBOT_REGISTRY["mabel"] = __name__


def serve_environment(env, *, headless=True, port_offset=0, steps=None, no_cameras=False):
    """Reuse Emet's ZMQ transport with the composed shelf model.

    Emet's generic SE(2) navigation sets base velocities directly. This initial
    adapter is for joint/control integration; use wheel actuator targets in the
    local environment for physical swerve experiments.
    """
    import emet.core.server as core_server
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    for name, rate in (("STATE", "50"), ("FULL", "15"), ("SERVO", "15")):
        os.environ.setdefault(f"EMET_ZMQ_{name}_HZ", rate)

    class ShelfServer(RobosuiteZmqServer):
        def spin_send(self):
            # Upstream retries None observations without sleeping. With cameras
            # disabled, skip these publisher loops to avoid starving physics.
            if not no_cameras:
                super().spin_send()

        def spin_send_servo(self):
            if not no_cameras:
                super().spin_send_servo()

        def _load_model(self):
            super()._load_model()
            env.data = self._mjdata
            env.reset()
            self._sync_actuator_ctrl_from_joint_positions()

        def _mj_step_once(self):
            super()._mj_step_once()
            mujoco.mj_forward(self._mjmodel, self._mjdata)
            env.update_task_state(env.config.timestep)

        def get_state_message(self):
            with self._mj_lock:
                message = super().get_state_message()
                if message is not None:
                    message["shelf_stocking"] = env.info()
                return message

        def get_joint_state(self):
            with self._mj_lock:
                positions, velocities, efforts = super().get_joint_state()
                if self._mjdata is not None:
                    lift = self._spec.actuator_names.index("Lift")
                    positions[lift] = self._mjdata.actuator("Lift").length[0]
                    velocities[lift] = self._mjdata.actuator("Lift").velocity[0]
                return positions, velocities, efforts

        def handle_action(self, action):
            with self._mj_lock:
                for key in (
                    "sim_set_body_pose",
                    "sim_set_joint_qpos",
                    "sim_attach_body",
                    "sim_detach_body",
                ):
                    if key in action:
                        env.record_intervention(f"Emet simulation helper: {key}")
                super().handle_action(action)

    register_mabel()
    # Use the environment model's actuator names rather than assuming fixed IDs.
    spec = MabelBackend(env.mabel_root).get_spec()
    if tuple(spec.actuator_names) != env.actuator_names:
        raise ValueError("MABEL spec and environment actuators do not match")
    if no_cameras:
        spec.camera_names = []
    # Emet's base constructor currently creates Piper even when its speech-engine
    # argument is disabled. Suppress that constructor branch for this simulation
    # server so starting a shelf task never downloads unrelated speech models.
    with patch.object(core_server, "imported_text_to_speech", False):
        server = ShelfServer(
            robot_spec=spec,
            scene_model=env.model,
            max_sim_steps=steps,
            mujoco_stationary_control=MabelStationaryControl(),
            send_port=4401 + port_offset,
            recv_port=4402 + port_offset,
            send_state_port=4403 + port_offset,
            send_servo_port=4404 + port_offset,
            use_remote_computer=False,
            environment={"task": "shelf_stocking", "seed": env.info()["seed"]},
        )
    try:
        server.start(headless=headless, show_viewer_ui=not headless)
    finally:
        server.stop()
