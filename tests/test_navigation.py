"""Scene occupancy grid and emet A* base routing stay scratch-only and scene-aware."""

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.navigation import (
    BaseNavigationSpace,
    SceneOccupancyGrid,
    plan_base_path,
)


@pytest.fixture
def pick_place_env():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"), seed=0)
    yield env
    env.close()


def test_grid_marks_shelf_table_and_products_and_roundtrips_coordinates(pick_place_env):
    env = pick_place_env
    grid = SceneOccupancyGrid(env, resolution=0.02)
    assert grid.obs.shape == (grid.nx, grid.ny)
    # Shelf center and supply table center are occupied; the open robot start is free.
    assert grid.obs[grid.to_pt((env.config.front_x - env.config.depth / 2, env.config.center_y))]
    assert grid.obs[grid.to_pt(env.config.table_xy)]
    assert not grid.obs[grid.to_pt((0.0, 0.0))]
    # World -> grid -> world round trip returns the cell center, not the corner.
    pt = grid.to_pt((0.37, -0.21))
    x, y = grid.to_xy(pt)
    assert grid.to_pt((x, y)) == pt


def test_plan_route_to_open_goal_is_scene_aware(pick_place_env):
    env = pick_place_env
    path = plan_base_path(env, (0.0, 0.0), (0.0, 0.12))
    assert path is not None and len(path) >= 2
    np.testing.assert_allclose(path[0], (0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(path[-1], (0.0, 0.12), atol=1e-9)
    grid = SceneOccupancyGrid(env, resolution=0.02)
    # Every waypoint (cell center) is free in the obstacle mask.
    for waypoint in path[1:-1]:
        assert not grid.obs[grid.to_pt(waypoint)]


def test_plan_into_shelf_is_blocked(pick_place_env):
    env = pick_place_env
    # The shelf board centre is inside the obstacle mask.
    goal = (env.config.front_x - env.config.depth / 2, env.config.center_y)
    assert plan_base_path(env, (0.0, 0.0), goal) is None


def test_space_is_valid_rejects_obstacle_proximity(pick_place_env):
    env = pick_place_env
    grid = SceneOccupancyGrid(env, resolution=0.02)
    space = BaseNavigationSpace(grid, footprint_radius=0.22)
    assert space.is_valid([0.0, 0.0, 0.0])
    assert not space.is_valid([env.config.front_x - env.config.depth / 2, env.config.center_y, 0.0])


def test_excluded_products_are_not_obstacles(pick_place_env):
    env = pick_place_env
    carried = env.products[0]
    # Place the carried product in open space (not over the supply table) so its
    # own footprint is the only occupant of those cells.
    import mujoco

    adr = int(env.model.joint(f"{carried.name}_joint").qposadr[0])
    env.data.qpos[adr : adr + 3] = [0.25, 0.0, 0.5]
    mujoco.mj_forward(env.model, env.data)
    base_grid = SceneOccupancyGrid(env, resolution=0.02)
    excluded_grid = SceneOccupancyGrid(env, resolution=0.02, exclude_products={carried.name})
    pt = base_grid.to_pt(env.data.body(carried.name).xpos[:2])
    assert base_grid.obs[pt]
    assert not excluded_grid.obs[pt]
