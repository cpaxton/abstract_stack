"""Physics and evaluation regressions; no rendering or display server required."""

import math

import mujoco
import numpy as np
import pytest

from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig


@pytest.fixture
def env():
    instance = ShelfStockingEnv(robot=False)
    yield instance
    instance.close()


def move_product(env, product, position, quaternion=(1, 0, 0, 0)):
    """Test fixture setup only; the environment exposes no teleport action."""
    joint = env.model.joint(f"{product.name}_joint")
    qadr, vadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    env.data.qpos[qadr : qadr + 7] = (*position, *quaternion)
    env.data.qvel[vadr : vadr + 6] = 0
    mujoco.mj_forward(env.model, env.data)


def test_seeded_reset_and_supply_contact(env):
    initial, _ = env.reset(seed=13)
    for _ in range(40):
        env.step()
    assert not env.info()["tipped"]
    assert not env.info()["dropped"]
    for product in env.products:
        assert env.data.body(product.name).xpos[2] == pytest.approx(
            env.config.table_top + product.half_size[2], abs=0.002
        )
    repeated, _ = env.reset(seed=13)
    np.testing.assert_array_equal(initial["qpos"], repeated["qpos"])
    different, _ = env.reset(seed=14)
    assert not np.array_equal(initial["qpos"], different["qpos"])
    assert env.data.time == 0


def test_complete_shelf_requires_stable_supported_products(env):
    for product in env.products:
        move_product(env, product, product.target)
    _, _, terminated, _, info = env.step()
    assert not terminated
    assert info["stocked_count"] == 0
    for _ in range(80):
        _, _, terminated, truncated, info = env.step()
        assert not truncated
        if terminated:
            break
    assert terminated
    assert info["stocked_count"] == 6
    assert info["unassisted_success"]


def test_hovering_near_target_does_not_count_as_stocked(env):
    product = env.products[0]
    position = np.array(product.target) + [0, 0, 0.007]
    move_product(env, product, position)
    env.update_task_state(env.config.stable_seconds + 1)
    assert env.info()["stocked_count"] == 0


def test_tipped_product_is_not_success(env):
    product = env.products[0]
    for p in env.products:
        move_product(env, p, p.target)
    move_product(env, product, product.target, (math.sqrt(0.5), math.sqrt(0.5), 0, 0))
    for _ in range(70):
        _, _, terminated, _, info = env.step()
        assert not terminated
    assert product.name in info["tipped"]
    assert product.name not in info["stocked"]


def test_dislodging_stocked_product_removes_progress(env):
    product = env.products[0]
    move_product(env, product, product.target)
    for _ in range(60):
        env.step()
    assert env.info()["stocked_count"] == 1
    move_product(env, product, product.spawn)
    _, reward, terminated, _, info = env.step()
    assert info["stocked_count"] == 0
    assert reward == -1
    assert not terminated


def test_timeout_and_interventions():
    env = ShelfStockingEnv(ShelfConfig(time_limit=0.04), robot=False)
    try:
        env.record_intervention("manual retry")
        assert env.info()["intervention_count"] == 1
        env.step()
        _, _, terminated, truncated, _ = env.step()
        assert truncated and not terminated
        with pytest.raises(RuntimeError, match="finished"):
            env.step()
        _, info = env.reset()
        assert info["intervention_count"] == 0
    finally:
        env.close()


def test_export_loads_in_plain_mujoco(env, tmp_path):
    model = mujoco.MjModel.from_xml_path(str(env.export(tmp_path / "shelf.xml")))
    assert model.nq == 42
    assert model.camera("shelf_front").id >= 0
    assert model.geom("shelf_board_2").id >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"levels": ()},
        {"levels": (0.6, 0.7)},
        {"frame_skip": 0},
        {"width": -1},
        {"timestep": float("nan")},
    ],
)
def test_invalid_scene_configuration(kwargs):
    with pytest.raises(ValueError):
        ShelfConfig(**kwargs)
