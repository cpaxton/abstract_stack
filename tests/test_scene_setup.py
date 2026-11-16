"""Scene composition, packaged presets, and reproducible exports."""

import json

import mujoco
import numpy as np
import pytest

from abstract_stack.bundle import build_scene_bundle
from abstract_stack.cli import main
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig


@pytest.mark.parametrize(
    "scene,count,prefilled",
    [
        ("empty", 0, 0),
        ("stocking", 6, 0),
        ("tight", 18, 12),
    ],
)
def test_presets_compile_and_settle(scene, count, prefilled):
    env = ShelfStockingEnv(load_scene_config(scene=scene), robot=False)
    try:
        assert len(env.products) == count
        if scene == "tight":
            remaining = [p for p in env.products if not p.initially_stocked]
            assert sum(p.kind == "carton" for p in remaining) == 3
            assert sum(p.kind == "bottle" for p in remaining) == 3
        assert env.data.body("shelf").xpos[0] == pytest.approx(env.config.front_x)
        assert env.observe()["product_poses"].shape == (count, 7)
        for _ in range(50):
            _, _, terminated, _, info = env.step()
            assert not terminated
        assert info["stocked_count"] == prefilled
        assert not info["tipped"]
        assert not info["dropped"]
        table_y = env.data.body("supply").xpos[1]
        assert table_y + env.config.table_size[1] / 2 < (
            env.config.center_y - env.config.width / 2 - env.config.board_thickness
        )
    finally:
        env.close()


def test_moved_shelf_geometry_targets_and_success_agree():
    env = ShelfStockingEnv(ShelfConfig(center_y=0.2, initially_stocked=6), robot=False)
    try:
        for _ in range(60):
            _, _, terminated, _, info = env.step()
            if terminated:
                break
        assert terminated
        assert info["stocked_count"] == 6
        assert env.data.body("shelf").xpos[1] == pytest.approx(0.2)
    finally:
        env.close()


def test_bundle_xml_matches_seed_and_manifest_after_reset(tmp_path):
    env = ShelfStockingEnv(robot=False, seed=1)
    try:
        env.reset(seed=37)
        directory = build_scene_bundle(env, tmp_path)
        manifest = json.loads((directory / "manifest.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
        np.testing.assert_allclose(model.qpos0, manifest["initial_qpos"], atol=1e-8)
        assert manifest["seed"] == 37
        assert manifest["robot"] == "none"
        assert manifest["robot_source"] is None
        assert manifest["products"][0]["initial_pose"][2] > env.config.table_top
        assert manifest["settled_metrics"]["elapsed_sim_time"] == pytest.approx(1.0)
    finally:
        env.close()


def test_cli_custom_config(tmp_path):
    config = tmp_path / "scene.json"
    config.write_text(json.dumps({"width": 1.1, "product_count": 0, "center_y": 0.1}))
    output = tmp_path / "bundle"
    main(["build-scene", "--config", str(config), "--robot", "none", "--output-dir", str(output)])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["width"] == 1.1
    assert not manifest["settled_metrics"]["success"]


@pytest.mark.parametrize(
    "values",
    [
        {"widht": 0.9},
        {"levels": "low"},
        {"robot_xy": [0]},
        {"product_count": 100},
        {"initially_stocked": 7},
        {"enable_sensors": "false"},
        {"product_count": 18, "initially_stocked": 12, "width": 0.5, "target_gap": 0.1},
    ],
)
def test_bad_scene_configs_fail_before_simulation(tmp_path, values):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_scene_config(config)
