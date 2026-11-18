"""The per-carton skill plan constructs and plans its first skill on every scene."""

import pytest

from abstract_stack.config import load_scene_config
from abstract_stack.skill_library import build_stocking_plan

# Carton-only studio scenes covering spacious, tight, dense-row, rear-clearance
# and numerical-contact fixtures. ``stocking``/``tight`` are mixed (carton plus
# bottle) multi-level scenes the scripted carton teacher intentionally rejects.
SCENES = [
    "pick_place",
    "configs/scenes/tight_three_cartons.json",
    "configs/scenes/tight_four_cartons.json",
    "configs/scenes/tight_ten_cartons.json",
    "configs/scenes/tight_ten_cartons_20mm.json",
    "configs/scenes/pick_place_rear_clearance.json",
    "configs/scenes/tight_ten_cartons_fixture_contacts.json",
    "configs/scenes/tight_ten_cartons_rigid_contact.json",
]


def _scene_config(name):
    return load_scene_config(name) if name.endswith(".json") else load_scene_config(scene=name)


@pytest.mark.mabel
@pytest.mark.parametrize("scene_name", SCENES)
def test_skill_plan_plans_first_carton_across_scenes(scene_name, tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import ScriptConfig, StockingScript

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")

    env = ShelfStockingEnv(_scene_config(scene_name), robot=True, seed=0)
    try:
        script = StockingScript(
            env,
            tmp_path,
            ScriptConfig(ungrasped_insertion=True, paired_finger_insertion=True),
        )

        plan = build_stocking_plan()
        assert [skill.name for skill in plan] == [
            "pick",
            "place",
            "insert",
            "push_to_contact",
            "retreat",
        ]
        assert all(skill.execute is not None for skill in plan)

        use_finger, neighbors = script._use_finger_contact()
        assert isinstance(use_finger, bool)
        assert isinstance(neighbors, list)

        pending = [product for product in env.products if not product.initially_stocked]
        assert pending, f"{scene_name} has no scheduled cartons"
        # Match the episode loop: select the nearest graspable carton, not the
        # first in product order, since blocked or far cartons can fail IK.
        target, prediction = script._select_grasp(pending)
        assert len(target) == 3
        assert prediction.feasible
    finally:
        env.close()
