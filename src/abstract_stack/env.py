"""Continuous MuJoCo shelf-stocking episodes with state observations and evaluation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.control import stationary_targets, validate_targets
from abstract_stack.scene import ShelfConfig, build_scene_xml, initial_positions, products_for


class ShelfStockingEnv:
    """MuJoCo environment; actions are persistent, absolute actuator targets.

    ``step`` follows the five-value observation/reward/terminated/truncated/info
    convention. This is a small simulator API, not a Gymnasium Env subclass.
    Robot and products remain fully dynamic; there is no grasp latch or teleport
    placement policy. ``model`` and ``data`` are exposed for VR/control adapters.
    """

    def __init__(
        self,
        config: ShelfConfig | None = None,
        *,
        robot: bool = True,
        mabel_root: str | Path | None = None,
        seed: int = 0,
    ):
        self.config = config or ShelfConfig()
        self.mabel_root = mabel_root
        self.products = products_for(self.config)
        self.xml = build_scene_xml(self.config, robot=robot, mabel_root=mabel_root, seed=seed)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.robot = robot
        self._renderer = None
        self._seed = seed
        self._spawn_qpos = self.model.qpos0.copy()
        self._product_bodies = np.array(
            [self.model.body(p.name).id for p in self.products], dtype=int
        )
        self._product_joints = [self.model.joint(f"{p.name}_joint") for p in self.products]
        self._boards = [self.model.geom(f"shelf_board_{p.level}").id for p in self.products]
        self.reset(seed=seed)

    @property
    def dt(self) -> float:
        return self.config.timestep * self.config.frame_skip

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(self.model.actuator(i).name for i in range(self.model.nu))

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._seed = seed
        if self.model.nkey:
            # MABEL's LiDAR <replicate> expands home to home000, home001, ... .
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._spawn_qpos
        positions = initial_positions(self.config, self.products, self._seed)
        for position, joint in zip(positions, self._product_joints, strict=True):
            qadr = int(joint.qposadr[0])
            self.data.qpos[qadr : qadr + 3] = position
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = stationary_targets(self.model, self.data)
        self._stable_time = np.zeros(len(self.products))
        self._stocked = np.zeros(len(self.products), dtype=bool)
        self._done = False
        self._started_wall = time.monotonic()
        self.interventions: list[dict] = []
        return self.observe(), self.info()

    def observe(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "product_poses": np.array(
                [
                    np.concatenate((self.data.xpos[bid], self.data.xquat[bid]))
                    for bid in self._product_bodies
                ]
            ).reshape(-1, 7),
            "targets": np.array([p.target for p in self.products]).reshape(-1, 3),
            "stocked": self._stocked.copy(),
            "sim_time": float(self.data.time),
        }

    def _placement_conditions(self) -> np.ndarray:
        supported = set()
        for contact in self.data.contact:
            if contact.dist > 0.002:
                continue
            for geom, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                supported.add((int(self.model.geom_bodyid[geom]), int(other)))
        good = np.zeros(len(self.products), dtype=bool)
        for i, (product, bid, joint) in enumerate(
            zip(self.products, self._product_bodies, self._product_joints, strict=True)
        ):
            pos = self.data.xpos[bid]
            rotation = self.data.xmat[bid].reshape(3, 3)
            extent = np.abs(rotation) @ np.array(product.half_size)
            vadr = int(joint.dofadr[0])
            velocity = self.data.qvel[vadr : vadr + 6]
            good[i] = (
                np.max(np.abs(pos[:2] - np.array(product.target[:2])))
                <= self.config.target_tolerance
                and pos[0] + extent[0] <= self.config.front_x + 0.002
                and pos[0] - extent[0] >= self.config.front_x - self.config.depth - 0.002
                and abs(pos[1] - self.config.center_y) + extent[1] <= self.config.width / 2 + 0.002
                and abs(pos[2] - extent[2] - self.config.levels[product.level]) < 0.012
                and rotation[2, 2] >= math.cos(math.radians(15))
                and np.linalg.norm(velocity[:3]) < 0.04
                and np.linalg.norm(velocity[3:]) < 0.2
                and (int(bid), self._boards[i]) in supported
            )
        return good

    def step(
        self,
        action: Mapping[str, float] | np.ndarray | None = None,
        *,
        on_physics_step: Callable[[ShelfStockingEnv], None] | None = None,
    ) -> tuple:
        if self._done:
            raise RuntimeError("Episode is finished; call reset() before stepping again")
        if action is not None:
            self.data.ctrl[:] = validate_targets(self.model, self.data.ctrl, action)
        previous = int(self._stocked.sum())
        # Evaluate every physics step: a collision or loss of support must break
        # the stable interval even when it happens between control observations.
        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise FloatingPointError("MuJoCo produced a non-finite physics state")
            mujoco.mj_forward(self.model, self.data)
            self.update_task_state(self.config.timestep)
            if on_physics_step is not None:
                on_physics_step(self)
        terminated = bool(self.products) and bool(self._stocked.all())
        truncated = bool(self.data.time >= self.config.time_limit and not terminated)
        self._done = terminated or truncated
        return (
            self.observe(),
            float(int(self._stocked.sum()) - previous),
            terminated,
            truncated,
            self.info(),
        )

    def update_task_state(self, elapsed: float) -> None:
        """Evaluate the current physics state; also used by the Emet server adapter."""
        good = self._placement_conditions()
        self._stable_time = np.where(good, self._stable_time + elapsed, 0)
        self._stocked = self._stable_time >= self.config.stable_seconds

    def info(self) -> dict:
        tipped, dropped = [], []
        for p, bid in zip(self.products, self._product_bodies, strict=True):
            if self.data.xmat[bid].reshape(3, 3)[2, 2] < math.cos(math.radians(45)):
                tipped.append(p.name)
            if self.data.xpos[bid, 2] < 0.25:
                dropped.append(p.name)
        success = bool(self.products) and bool(self._stocked.all())
        return {
            "seed": self._seed,
            "environment": self.config.environment,
            "success": success,
            "unassisted_success": success and not self.interventions,
            "stocked_count": int(self._stocked.sum()),
            "product_count": len(self.products),
            "initially_stocked_count": self.config.initially_stocked,
            "stocked": [
                p.name for p, good in zip(self.products, self._stocked, strict=True) if good
            ],
            "tipped": tipped,
            "dropped": dropped,
            "elapsed_sim_time": float(self.data.time),
            "elapsed_wall_time": time.monotonic() - self._started_wall,
            "intervention_count": len(self.interventions),
            "interventions": list(self.interventions),
            "time_limit_reached": bool(self.data.time >= self.config.time_limit),
        }

    def record_intervention(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("An intervention must include a reason")
        self.interventions.append({"sim_time": float(self.data.time), "reason": reason})

    def render(self, camera: str = "overview", width: int = 960, height: int = 720) -> np.ndarray:
        if self._renderer is not None and (self._renderer.width, self._renderer.height) != (
            width,
            height,
        ):
            self.close()
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        options = mujoco.MjvOption()
        if self.config.environment == "small_house":
            from abstract_stack.house import CUTAWAY_CAMERAS, CUTAWAY_GROUP

            options.geomgroup[CUTAWAY_GROUP] = int(camera not in CUTAWAY_CAMERAS)
        self._renderer.update_scene(self.data, camera=camera, scene_option=options)
        return self._renderer.render().copy()

    def export(self, path: str | Path) -> Path:
        """Export episode-start MJCF using the most recent reset seed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            build_scene_xml(
                self.config,
                robot=self.robot,
                mabel_root=self.mabel_root,
                seed=self._seed,
            )
        )
        return path

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
