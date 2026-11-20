"""Episode stepping around a restored compiled state, without a new reset."""

import copy
import time
from dataclasses import fields, replace

import mujoco
import numpy as np

from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import Product, ShelfConfig


def reassign_empty_slot(env, scene, active, other):
    """Permute identical-carton slots before placement, preserving every target.

    This is an explicit planning action, never a way to credit an already placed
    carton by changing its label. It changes logical targets only; all native
    state, model parameters, shelf dimensions and evaluation tolerances persist.
    """
    first = next(p for p in env.products if p.name == active)
    second = next(p for p in env.products if p.name == other)
    if (
        first.name == second.name
        or first.initially_stocked
        or second.initially_stocked
        or first.kind != second.kind
        or tuple(first.half_size) != tuple(second.half_size)
        or first.mass != second.mass
        or first.level != second.level
    ):
        raise ValueError("Slot permutation requires two interchangeable scheduled cartons")
    active_body = env.model.body(active).id
    board = env.model.geom(f"shelf_board_{first.level}").id
    support = 0.0
    force = np.empty(6)
    for i, c in enumerate(env.data.contact):
        if (
            board in (c.geom1, c.geom2)
            and active_body
            in (int(env.model.geom_bodyid[c.geom1]), int(env.model.geom_bodyid[c.geom2]))
            and c.efc_address >= 0
        ):
            mujoco.mj_contactForce(env.model, env.data, i, force)
            support += max(0.0, float(force[0]))
    if support > 0.2:
        raise ValueError("Cannot reassign an already shelf-supported active carton")
    for p in env.products:
        if p.name == active:
            continue
        body = env.data.body(p.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(p.half_size)
        if np.all(
            np.abs(body.xpos - np.asarray(second.target))
            < extent + np.asarray(second.half_size) + 0.002
        ):
            raise ValueError("Proposed destination slot is occupied in measured state")
    targets = {active: tuple(second.target), other: tuple(first.target)}
    products = tuple(
        replace(p, target=targets[p.name]) if p.name in targets else p for p in env.products
    )
    updated = copy.deepcopy(scene)
    for p in updated["products"]:
        if p["name"] in targets:
            p["target"] = list(targets[p["name"]])
    updated["placement_assignment"] = dict(
        kind="explicit_interchangeable_carton_slot_permutation",
        swapped_products=[active, other],
        original_targets={active: list(first.target), other: list(second.target)},
        physical_slots_unchanged=True,
        model_unchanged=True,
        reason="Measured loaded-transfer replanning; destination verified empty",
    )
    return products, updated


class CheckpointContinuationEnv(ShelfStockingEnv):
    """Use normal callbacks/task checks on a private restored model/data pair.

    Unknown historical task-stability intervals earn no credit. The checkpoint
    and prior monitor evidence remain attached; callers must report their
    continuation separately from the original failed episode and fresh tasks.
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.config = ShelfConfig(**checkpoint.scene["config"])
        product_fields = {field.name for field in fields(Product)}
        self.products = tuple(
            Product(**{k: v for k, v in p.items() if k in product_fields})
            for p in checkpoint.scene["products"]
        )
        self.model, self.data = checkpoint.model, checkpoint.data
        self.robot = bool(checkpoint.scene["robot"])
        self.mabel_root = None
        self._renderer = None
        self._seed = checkpoint.scene["seed"]
        self._product_bodies = np.array([self.model.body(p.name).id for p in self.products])
        self._product_joints = [self.model.joint(f"{p.name}_joint") for p in self.products]
        self._boards = [self.model.geom(f"shelf_board_{p.level}").id for p in self.products]
        self._stable_time = np.zeros(len(self.products))
        self._stocked = np.zeros(len(self.products), dtype=bool)
        self._packed_time = 0.0
        self._done = False
        self._started_wall = time.monotonic()
        self.interventions = []
        self.continuation_started_s = float(self.data.time)
        self.prior_contact_history = checkpoint.context.get("contact_monitor")

    def set_skill_time_budget(self, seconds):
        """Set a skill-local deadline without changing the restored simulation clock."""
        if type(seconds) not in (int, float) or not 0 < seconds <= 240:
            raise ValueError("Skill-local simulation budget must be in (0, 240] seconds")
        timing = dict(
            original_time_limit_s=self.config.time_limit,
            exact_start_time_s=float(self.data.time),
            local_budget_s=seconds,
            local_time_limit_s=float(self.data.time) + seconds,
        )
        self.config = replace(self.config, time_limit=timing["local_time_limit_s"])
        return timing

    def reset(self, **kwargs):
        raise RuntimeError("A checkpoint continuation cannot reset; create a fresh environment")

    def restore_task_memory(self, state):
        """Restore recorded logical stability only when its current pose agrees."""
        stable = np.asarray(state["stable_time"], float)
        stocked = np.asarray(state["stocked"])
        if (
            stable.shape != (len(self.products),)
            or stocked.shape != stable.shape
            or stocked.dtype != np.dtype(bool)
            or not np.isfinite(stable).all()
            or np.any(stable < 0)
            or np.any(stable > self.data.time + 1e-8)
            or not np.array_equal(stocked, stable >= self.config.stable_seconds)
            or np.any(stocked & ~self._placement_conditions())
        ):
            raise ValueError("Continuation task memory disagrees with native state or thresholds")
        self._stable_time, self._stocked = stable.copy(), stocked.copy()
