"""Physics, passages and scene metadata for the furnished house."""

import json
from dataclasses import asdict

import mujoco
import numpy as np
import pytest

from abstract_stack.bundle import build_scene_bundle
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.house import CUTAWAY_GROUP


@pytest.fixture
def house():
    env = ShelfStockingEnv(load_scene_config(scene="house"), robot=False)
    yield env
    env.close()


def ray(env, point, direction):
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        env.model,
        env.data,
        np.array(point, dtype=float),
        np.array(direction, dtype=float),
        np.ones(6, dtype=np.uint8),
        True,
        -1,
        geom,
    )
    return distance, env.model.geom(geom[0]).name if geom[0] >= 0 else None


def test_doorways_clear_and_cutaway_walls_remain_physical(house):
    distance, name = ray(house, (0, 0, 1.2), (1, 0, 0))
    assert distance == pytest.approx(2.94)
    assert name == "house_east_wall_upper"
    distance, name = ray(house, (0.30, 1.8, 1.2), (1, 0, 0))
    assert distance == pytest.approx(0.19)
    assert name == "house_partition_north_upper"
    wall = house.model.geom(name)
    assert wall.group[0] == CUTAWAY_GROUP
    assert wall.contype[0] == wall.conaffinity[0] == 1
    distance, _ = ray(house, (1.5, -0.8, 0.3), (0, -1, 0))
    assert distance == pytest.approx(1.94)
    distance, _ = ray(house, (-0.2, -2.4, 0.3), (0, -1, 0))
    assert distance == -1  # open entrance, without an invisible door or threshold


def test_movable_product_collides_with_house_wall(house):
    joint = house.model.joint("carton_0_joint")
    qadr, vadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    house.data.qpos[qadr : qadr + 7] = [-2.85, -0.4, 0.4, 1, 0, 0, 0]
    house.data.qvel[vadr] = -1.0
    mujoco.mj_forward(house.model, house.data)
    for _ in range(30):
        house.step()
    assert house.data.body("carton_0").xpos[0] >= -2.94 + 0.045 - 0.003


@pytest.mark.parametrize("scene,prefilled", [("house", 0), ("house_tight", 12)])
def test_house_products_settle_without_falling(scene, prefilled):
    env = ShelfStockingEnv(load_scene_config(scene=scene), robot=False)
    try:
        for _ in range(50):
            _, _, terminated, _, info = env.step()
        assert not terminated
        assert info["stocked_count"] == prefilled
        assert not info["dropped"]
        assert not info["tipped"]
        # Furniture adds no action channels, articulated bodies or free joints.
        assert env.model.nq == 7 * len(env.products)
        assert env.model.nu == 0
        assert env.model.body("house_bed").jntnum[0] == 0
    finally:
        env.close()


def test_house_bundle_records_layout_and_reloadable_physics(house, tmp_path):
    build_scene_bundle(house, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["environment"]["kind"] == "small_house"
    assert len(manifest["environment"]["rooms"]) == 4
    assert len(manifest["environment"]["doorways"]) == 3
    assert "house_overhead" in manifest["cameras"]
    assert "bedroom" in manifest["cameras"]
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    np.testing.assert_allclose(model.qpos0, manifest["initial_qpos"], atol=1e-8)
    assert model.geom("house_ceiling").contype[0] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"environment": "unknown"},
        {"robot_xy": [-2.9, 0]},
        {"table_center_y": -2.7},
        {"center_y": 2.5},
        {"levels": [0.7, 1.5, 2.4]},
    ],
)
def test_house_rejects_invalid_placements(tmp_path, changes):
    config = asdict(load_scene_config(scene="house"))
    config.update(changes)
    path = tmp_path / "bad-house.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_scene_config(path)
