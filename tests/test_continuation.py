"""Compiled checkpoint continuations keep native state/clock and normal observers."""

from dataclasses import asdict

import numpy as np
import pytest

from abstract_stack.checkpoints import load_checkpoint, native_state, save_checkpoint
from abstract_stack.continuation import CheckpointContinuationEnv, reassign_empty_slot
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig


def test_continuation_preserves_full_state_and_steps_all_callbacks_without_reset(tmp_path):
    original = ShelfStockingEnv(
        ShelfConfig(product_count=1, product_kind="carton"), robot=False, seed=23
    )
    for _ in range(3):
        original.step()
    scene = dict(
        config=asdict(original.config),
        seed=23,
        robot=False,
        products=[
            dict(asdict(p), initial_pose=[*original.data.body(p.name).xpos, 1, 0, 0, 0])
            for p in original.products
        ],
    )
    directory = save_checkpoint(
        tmp_path / "checkpoint",
        original.model,
        original.data,
        scene=scene,
        controller=dict(kind="test"),
        context=dict(original_outcome="failure"),
    )
    restored = load_checkpoint(directory)
    before = native_state(restored.model, restored.data)
    env = CheckpointContinuationEnv(restored)
    np.testing.assert_array_equal(native_state(env.model, env.data), before)
    assert env.data.time > 0 and env.checkpoint.context["original_outcome"] == "failure"
    assert not env._stocked.any()  # Unknown historical stability cannot earn credit.
    times = []
    env.step(on_physics_step=lambda e: times.append(float(e.data.time)))
    assert len(times) == env.config.frame_skip
    assert times[0] == pytest.approx(float(before[0]) + env.config.timestep)
    with pytest.raises(RuntimeError, match="cannot reset"):
        env.reset(seed=0)
    assert env.info()["seed"] == 23
    env.restore_task_memory(dict(stable_time=[0.0], stocked=[False]))
    with pytest.raises(ValueError, match="task memory"):
        env.restore_task_memory(dict(stable_time=[0.0], stocked=[True]))
    original.close()
    env.close()


def test_permutation_cannot_relabel_an_already_shelf_supported_carton():
    import mujoco

    env = ShelfStockingEnv(
        ShelfConfig(product_count=3, product_kind="carton", levels=(0.65,)), robot=False
    )
    address = int(env.model.joint("carton_0_joint").qposadr[0])
    env.data.qpos[address : address + 3] = env.products[0].target  # Reset fixture only.
    mujoco.mj_forward(env.model, env.data)
    for _ in range(50):
        env.step()
    scene = dict(config=asdict(env.config), products=[asdict(p) for p in env.products])
    with pytest.raises(ValueError, match="already shelf-supported"):
        reassign_empty_slot(env, scene, "carton_0", "carton_1")
    env.close()


def test_slot_permutation_preserves_physics_targets_and_success_tolerances():
    env = ShelfStockingEnv(
        ShelfConfig(product_count=3, product_kind="carton", levels=(0.65,)), robot=False
    )
    scene = dict(config=asdict(env.config), products=[asdict(p) for p in env.products])
    before = native_state(env.model, env.data)
    original_targets = sorted(tuple(p.target) for p in env.products)
    products, manifest = reassign_empty_slot(env, scene, "carton_0", "carton_1")
    assert sorted(tuple(p.target) for p in products) == original_targets
    assert tuple(products[0].target) == tuple(env.products[1].target)
    assert manifest["config"] == scene["config"]
    assert scene["products"][0]["target"] == env.products[0].target
    np.testing.assert_array_equal(native_state(env.model, env.data), before)
    env.close()


def test_permutation_rejects_occupied_destination_before_changing_any_target():
    import mujoco

    env = ShelfStockingEnv(
        ShelfConfig(product_count=3, product_kind="carton", levels=(0.65,)), robot=False
    )
    # Reset-only collision fixture: another carton already occupies slot 1.
    address = int(env.model.joint("carton_2_joint").qposadr[0])
    env.data.qpos[address : address + 3] = env.products[1].target
    mujoco.mj_forward(env.model, env.data)
    before = native_state(env.model, env.data)
    scene = dict(config=asdict(env.config), products=[asdict(p) for p in env.products])
    with pytest.raises(ValueError, match="occupied"):
        reassign_empty_slot(env, scene, "carton_0", "carton_1")
    np.testing.assert_array_equal(native_state(env.model, env.data), before)
    env.close()
