"""Reset geometry and one-decision contracts for isolated stocking skills."""

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.scene import products_for
from abstract_stack.skill_env import ACTIONS, StockingSkillEnv
from abstract_stack.skill_fixtures import sample_skill_fixture


@pytest.mark.parametrize("skill", ACTIONS)
def test_sample_skill_fixture_is_reproducible(skill):
    first = sample_skill_fixture(skill, 0)
    assert first == sample_skill_fixture(skill, 0)
    assert first != sample_skill_fixture(skill, 4)
    products = {p.name for p in products_for(first.config)}
    assert first.active in products
    assert first.poses.keys() <= products
    assert first.config.initially_stocked == 0  # Explicit reset fixtures earn no stocking credit.


@pytest.mark.parametrize("seed,direction", [(0, -1), (4, 1)])
def test_row_press_fixtures_start_with_a_capacity_deficit(seed, direction):
    reset = sample_skill_fixture("row-press", seed)
    assert reset == sample_skill_fixture("row-compaction", seed)
    config = reset.config
    assert reset.direction == direction
    assert reset.active == reset.supported_neighbors[-1]
    width = config.carton_size[1]
    exposed_y = reset.poses[reset.active][1] - direction * width / 2
    open_wall = config.center_y - direction * config.width / 2
    available = direction * (exposed_y - open_wall)
    required = (config.product_count - len(reset.supported_neighbors)) * width
    assert 0.002 <= required - available <= 0.004
    assert config.width - config.product_count * width == pytest.approx(config.row_free_space)


@pytest.mark.mabel
@pytest.mark.parametrize("skill", ACTIONS)
def test_native_fixture_reset_can_repeat_without_inheriting_state(skill, tmp_path):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    isolated = StockingSkillEnv(skill, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Reset"):
            isolated.step(ACTIONS[skill][0])
        obs, info = isolated.reset(seed=4)
        assert info["outcome"] == "ready"
        assert info["fresh_task_credit"] == 0 and not info["training_eligible"]
        with pytest.raises(ValueError, match="Choose"):
            isolated.step("unknown")
        again, _ = isolated.reset(seed=4)
        for name in ("product_poses", "qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(obs[name], again[name])
        assert obs["sim_time"] == again["sim_time"]
    finally:
        isolated.close()
