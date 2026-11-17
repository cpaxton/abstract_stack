"""Single-thread-owned MuJoCo adapter; no transport or retargeting."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.assets import mabel_model_path
from abstract_stack.control import stationary_targets, validate_targets
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig

BASE_STEER = ("FL Steer", "FR Steer", "B Steer")
BASE_DRIVE = ("FL Drive", "FR Drive", "B Drive")


@dataclass(frozen=True)
class SimState:
    """Detached native coordinates; actuator length is distinct from joint qpos."""

    reset_generation: int
    step_index: int
    sim_time_s: float
    actuator_names: tuple[str, ...]
    qpos: np.ndarray
    qvel: np.ndarray
    applied_targets: np.ndarray
    actuator_lengths: np.ndarray
    actuator_velocities: np.ndarray
    actuator_forces: np.ndarray
    root_position_m: np.ndarray
    root_quaternion_wxyz: np.ndarray
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class CameraFrame:
    """RGB8 image and optical-frame pose relative to native mabel_chassis."""

    name: str
    rgb: np.ndarray
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    vertical_fov_deg: float
    intrinsics: dict[str, float]


@dataclass(frozen=True)
class StereoPair:
    pair_id: int
    reset_generation: int
    sim_time_s: float
    capture_host_monotonic_s: float
    left: CameraFrame
    right: CameraFrame


class MujocoTeleopAdapter:
    """Own a private ShelfStockingEnv on the creating thread.

    All operations, including rendering/close, stay on that thread. Workers may
    encode or serialize detached snapshots. Each command is a complete native
    actuator vector; base channels are reserved for the reset-measured steer
    hold and zero wheel speed. No live model/data objects are returned.
    """

    def __init__(
        self,
        config: ShelfConfig | None = None,
        *,
        mabel_root: str | Path | None = None,
        seed: int = 0,
    ):
        self._owner = threading.current_thread()
        self._closed = False
        self._env = ShelfStockingEnv(config, mabel_root=mabel_root, seed=seed)
        self._source_model_sha256 = hashlib.sha256(
            mabel_model_path(mabel_root).read_bytes()
        ).hexdigest()
        self._names = self._env.actuator_names
        self._camera_names = tuple(
            self._env.model.camera(i).name for i in range(self._env.model.ncam)
        )
        try:
            self._base_ids = np.array(
                [self._env.model.actuator(n).id for n in (*BASE_STEER, *BASE_DRIVE)]
            )
            self._root_id = self._env.model.body("mabel_chassis").id
            for aid in self._base_ids[:3]:
                if self._env.model.actuator_biasprm[aid, 1] >= 0:
                    raise ValueError("Base steer channels must be native position servos")
            for aid in self._base_ids[3:]:
                bias = self._env.model.actuator_biasprm[aid]
                if bias[1] != 0 or bias[2] >= 0:
                    raise ValueError("Base drive channels must be native velocity servos")
            for name in ("zed_left", "zed_right"):
                self._env.model.camera(name)
            self._base_targets = self._env.data.ctrl[self._base_ids].copy()
            if np.any(self._base_targets[3:] != 0):
                raise ValueError("Stationary policy must command zero wheel speed")
            validate_targets(self._env.model, self._env.data.ctrl, self._env.data.ctrl)
            mujoco.mj_forward(self._env.model, self._env.data)
        except Exception:
            self._env.close()
            raise
        self._generation = 0
        self._step_index = 0
        self._pair_id = 0
        self._terminated = self._truncated = False
        self._seed = seed

    def _check_owner(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("Simulator access must remain on its creating thread")
        if self._closed:
            raise RuntimeError("Simulator adapter is closed")

    @property
    def control_dt(self) -> float:
        self._check_owner()
        return self._env.dt

    @property
    def camera_names(self) -> tuple[str, ...]:
        self._check_owner()
        return self._camera_names

    def describe(self) -> dict:
        """Reproducibility inputs and timing, independent of wall-clock progress."""
        self._check_owner()
        model = self._env.model
        return {
            "config": asdict(self._env.config),
            "seed": self._seed,
            "model_xml_sha256": hashlib.sha256(self._env.xml.encode()).hexdigest(),
            "source_model_sha256": self._source_model_sha256,
            "counts": {"nq": model.nq, "nv": model.nv, "nu": model.nu},
            "physics_timestep_s": float(model.opt.timestep),
            "frame_skip": self._env.config.frame_skip,
            "control_dt_s": self.control_dt,
            "mujoco_version": mujoco.__version__,
            "base_policy": "reset-measured steer position hold; zero drive velocity",
            "camera_names": list(self._camera_names),
        }

    def read_state(self) -> SimState:
        self._check_owner()
        data = self._env.data
        return SimState(
            self._generation,
            self._step_index,
            float(data.time),
            self._names,
            data.qpos.copy(),
            data.qvel.copy(),
            data.ctrl.copy(),
            data.actuator_length.copy(),
            data.actuator_velocity.copy(),
            data.actuator_force.copy(),
            data.xpos[self._root_id].copy(),
            data.xquat[self._root_id].copy(),
            self._terminated,
            self._truncated,
        )

    def apply_actuator_targets(self, targets: Mapping[str, float] | np.ndarray) -> np.ndarray:
        """Validate atomically, then write ctrl only; applying never steps physics."""
        self._check_owner()
        if self._terminated or self._truncated:
            raise RuntimeError("Episode finished; reset before applying commands")
        if isinstance(targets, Mapping) and set(targets) != set(self._names):
            raise ValueError("Named commands must include every actuator exactly once")
        validated = validate_targets(self._env.model, self._env.data.ctrl, targets)
        if not np.array_equal(validated[self._base_ids], self._base_targets):
            raise ValueError("Base channels are reserved for stationary targets")
        self._env.data.ctrl[:] = validated
        mujoco.mj_forward(self._env.model, self._env.data)
        return validated.copy()

    def stationary_command(self) -> np.ndarray:
        """Measured actuator-length hold, preserving the reserved base references.

        This is a raw servo target helper, not a load-retention or pause guarantee.
        Native control-range validation may reject a physically out-of-range state.
        """
        self._check_owner()
        targets = stationary_targets(self._env.model, self._env.data)
        targets[self._base_ids] = self._base_targets
        return validate_targets(self._env.model, self._env.data.ctrl, targets)

    def step(self) -> SimState:
        self._check_owner()
        _, _, self._terminated, self._truncated, _ = self._env.step()
        self._step_index += 1
        return self.read_state()

    def reset(self, *, seed: int | None = None) -> SimState:
        self._check_owner()
        self._env.reset(seed=seed)
        mujoco.mj_forward(self._env.model, self._env.data)
        if seed is not None:
            self._seed = seed
        self._base_targets = self._env.data.ctrl[self._base_ids].copy()
        self._generation += 1
        self._step_index = 0
        self._terminated = self._truncated = False
        return self.read_state()

    @staticmethod
    def _dimensions(width, height):
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (width, height)):
            raise ValueError("Render dimensions must be positive integers")

    def render_camera(self, name: str, *, width: int = 640, height: int = 480) -> CameraFrame:
        self._check_owner()
        self._dimensions(width, height)
        if name not in self._camera_names:
            raise ValueError(f"Unknown camera: {name}")
        model, data = self._env.model, self._env.data
        cid = model.camera(name).id
        root_rotation = data.xmat[self._root_id].reshape(3, 3)
        position = root_rotation.T @ (data.cam_xpos[cid] - data.xpos[self._root_id])
        rotation = root_rotation.T @ data.cam_xmat[cid].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        fovy = float(model.cam_fovy[cid])
        focal = height / (2 * math.tan(math.radians(fovy) / 2))
        rgb = self._env.render(name, width=width, height=height)
        return CameraFrame(
            name,
            rgb,
            position.copy(),
            quaternion,
            fovy,
            {"fx": focal, "fy": focal, "cx": (width - 1) / 2, "cy": (height - 1) / 2},
        )

    def render_stereo_pair(self, *, width: int = 640, height: int = 480) -> StereoPair:
        self._check_owner()
        self._dimensions(width, height)
        # Only this thread owns the environment: no stepping or command update
        # can occur between eyes. Encoding workers receive the detached pair.
        captured = time.monotonic()
        sim_time = float(self._env.data.time)
        left = self.render_camera("zed_left", width=width, height=height)
        right = self.render_camera("zed_right", width=width, height=height)
        pair = StereoPair(self._pair_id, self._generation, sim_time, captured, left, right)
        self._pair_id += 1
        return pair

    def close(self):
        if threading.current_thread() is not self._owner:
            raise RuntimeError("Simulator close must remain on its creating thread")
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self):
        self._check_owner()
        return self

    def __exit__(self, *_):
        self.close()
