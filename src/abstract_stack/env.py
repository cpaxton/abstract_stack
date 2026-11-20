"""Continuous MuJoCo shelf-stocking episodes with state observations and evaluation."""

from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.control import stationary_targets, validate_targets
from abstract_stack.native_views import retain_data_views, retain_model_views
from abstract_stack.scene import ShelfConfig, build_scene_xml, initial_positions, products_for


@dataclass(frozen=True)
class PlanningState:
    """Independent scratch data and timing, sharing the model read-only.

    Deliberately omits episode bookkeeping, reset, stepping and rendering.
    Planners may modify ``data``; they must not modify the shared ``model``.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    dt: float


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
        # Read-only wrapper retention; preserve native model/state semantics.
        self._native_model_views = retain_model_views(self.model)
        self.data = mujoco.MjData(self.model)
        self._native_data_views = retain_data_views(self.model, self.data)
        self.robot = robot
        self._renderer = None
        self._seed = seed
        self._spawn_qpos = self.model.qpos0.copy()
        if robot:
            for name, value in self.config.initial_joint_positions:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0 or int(self.model.jnt_type[jid]) not in (
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                ):
                    raise ValueError(f"Initial joint {name!r} is not a named scalar joint")
                if self.model.jnt_limited[jid] and not (
                    self.model.jnt_range[jid, 0] <= value <= self.model.jnt_range[jid, 1]
                ):
                    raise ValueError(f"Initial joint {name!r} exceeds native limits")
                self._spawn_qpos[self.model.jnt_qposadr[jid]] = value
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

    def clone_for_planning(self) -> PlanningState:
        """Snapshot MuJoCo state without aliasing mutable episode state.

        The native model is shared under the planner's read-only contract.
        Controller candidates operate on an independent MjData; this snapshot
        cannot be used as a second task environment or claim rollout success.
        """
        data = mujoco.MjData(self.model)
        mujoco.mj_copyData(data, self.model, self.data)
        return PlanningState(self.model, data, self.dt)

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._seed = seed
        if self.model.nkey:
            # MABEL's LiDAR <replicate> expands home to home000, home001, ... .
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._initial_qpos()
        mujoco.mj_forward(self.model, self.data)
        if self.config.row_free_space is not None:
            self.validate_robot_fixture_clearance()
        self.data.ctrl[:] = stationary_targets(self.model, self.data)
        self._stable_time = np.zeros(len(self.products))
        self._stocked = np.zeros(len(self.products), dtype=bool)
        self._packed_time = 0.0
        self._done = False
        self._started_wall = time.monotonic()
        self.interventions: list[dict] = []
        return self.observe(), self.info()

    def _initial_qpos(self):
        state = self._spawn_qpos.copy()
        positions = initial_positions(self.config, self.products, self._seed)
        for position, joint in zip(positions, self._product_joints, strict=True):
            qadr = int(joint.qposadr[0])
            state[qadr : qadr + 3] = position
        return state

    def validate_robot_fixture_clearance(self):
        """Reject an intersecting dense reset, rather than settling the robot out of it."""
        if not self.robot:
            return
        robot = np.zeros(self.model.nbody, dtype=bool)
        root = self.model.body("mabel_chassis").id
        for body in range(1, self.model.nbody):
            robot[body] = body == root or robot[self.model.body_parentid[body]]
        rejected = []
        for contact in self.data.contact:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if robot[b1] == robot[b2] or contact.dist >= -0.001:
                continue
            names = (self.model.geom(g1).name, self.model.geom(g2).name)
            robot_body = self.model.body(b1 if robot[b1] else b2).name
            if "floor" in names and robot_body.endswith("wheel"):
                continue
            rejected.append((names, robot_body, float(contact.dist)))
        if rejected:
            raise ValueError(f"Robot intersects dense-scene fixtures/products: {rejected[:4]}")

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
        terminated = self.config.terminate_on_success and self.info()["success"]
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
        if self.config.row_free_space is not None:
            from abstract_stack.packing import row_geometry

            fits = bool(good.all()) and row_geometry(self)["envelopes_fit"]
            self._packed_time = self._packed_time + elapsed if fits else 0.0

    def info(self) -> dict:
        tipped, dropped = [], []
        for p, bid in zip(self.products, self._product_bodies, strict=True):
            if self.data.xmat[bid].reshape(3, 3)[2, 2] < math.cos(math.radians(45)):
                tipped.append(p.name)
            if self.data.xpos[bid, 2] < 0.25:
                dropped.append(p.name)
        success = bool(self.products) and bool(self._stocked.all())
        packing = None
        if self.config.row_free_space is not None:
            from abstract_stack.packing import row_geometry

            packing = row_geometry(self)
            packing["stable_seconds"] = self._packed_time
            success = success and self._packed_time >= self.config.stable_seconds
        return {
            "seed": self._seed,
            "environment": self.config.environment,
            "success": success,
            "packing": packing,
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
        xml = build_scene_xml(
            self.config,
            robot=self.robot,
            mabel_root=self.mabel_root,
            seed=self._seed,
        )
        if self.robot and self.config.initial_joint_positions:
            root = ET.fromstring(xml)
            keys = root.find("keyframe")
            if keys is None:
                keys = ET.SubElement(root, "keyframe")
            initial = mujoco.MjData(self.model)
            initial.qpos[:] = self._initial_qpos()
            mujoco.mj_forward(self.model, initial)
            key = ET.Element(
                "key",
                name="scene_start",
                qpos=" ".join(map(str, initial.qpos)),
                ctrl=" ".join(map(str, stationary_targets(self.model, initial))),
            )
            keys.insert(0, key)
            xml = ET.tostring(root, encoding="unicode")
        path.write_text(xml)
        return path

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
