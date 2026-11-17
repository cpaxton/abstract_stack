"""Seeded scene geometry, round-trip configuration and physical supply fixtures."""

from dataclasses import asdict

import numpy as np
import pytest

from abstract_stack.cli import main
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.random_scenes import random_shelf_config
from abstract_stack.scene import products_for


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_seeded_scene_fits_products_and_is_repeatable(seed, count):
    config = random_shelf_config(seed, product_count=count)
    assert config == random_shelf_config(seed, product_count=count)
    assert config != random_shelf_config(seed + 1, product_count=count)
    assert len(products_for(config)) == count
    assert 0.09 <= config.center_y + config.width / 2 <= 0.11
    assert not config.terminate_on_success
    assert all(np.isfinite(p.target).all() for p in products_for(config))


def test_generated_config_round_trips_and_has_physical_support(tmp_path):
    path = tmp_path / "scene.json"
    main(["random-scene", "--seed", "7", "--objects", "3", "--output", str(path)])
    config = load_scene_config(path)
    assert asdict(config) == asdict(random_shelf_config(7, product_count=3))
    env = ShelfStockingEnv(config, seed=7, robot=False)
    try:
        for _ in range(40):
            env.step()
        assert not env.info()["tipped"]
        assert not env.info()["dropped"]
        for product in env.products:
            assert env.data.body(product.name).xpos[2] == pytest.approx(
                config.table_top + product.half_size[2], abs=0.002
            )
    finally:
        env.close()


@pytest.mark.parametrize("seed,count", [(-1, 2), (False, 2), (0, 0), (0, 7)])
def test_invalid_generation_requests(seed, count):
    with pytest.raises(ValueError):
        random_shelf_config(seed, product_count=count)
