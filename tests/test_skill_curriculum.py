import json
from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path


def test_curriculum_tuning_rejects_changes_to_the_force_abort_guard(tmp_path):
    pytest.importorskip("gymnasium")
    from abstract_stack.skill_curriculum import StockingCurriculumEnv

    controls = tmp_path / "controls.json"
    controls.write_text(json.dumps({"max_force": 40}))
    with pytest.raises(ValueError, match="Not a tunable compliant-control setting"):
        StockingCurriculumEnv("push", tmp_path, control_config=controls)


def test_push_action_mask_preserves_paired_dense_protocol(tmp_path):
    pytest.importorskip("gymnasium")
    from abstract_stack.interactions import ScriptConfig
    from abstract_stack.skill_curriculum import StockingCurriculumEnv

    env = StockingCurriculumEnv("push", tmp_path)
    env._case = object()
    env._entry = {"skill": "front_insertion"}
    env.script = SimpleNamespace(
        active_grip=None,
        config=ScriptConfig(
            ungrasped_insertion=True, paired_finger_insertion=True, wall_packing=True
        ),
    )
    assert env._action_mask().tolist() == [1, 1, 0, 0]
    env._entry = {"skill": "side_slide_unloaded"}
    assert env._action_mask().tolist() == [1, 0, 0, 0]


def test_insert_action_mask_distinguishes_held_and_released_entries(tmp_path):
    pytest.importorskip("gymnasium")
    from abstract_stack.skill_curriculum import StockingCurriculumEnv

    env = StockingCurriculumEnv("insert", tmp_path)
    env._case = object()
    env._entry = {"skill": "front_release"}
    assert env._action_mask().tolist() == [1, 0, 0, 0]
    env._entry = {"skill": "free_landing"}
    assert env._action_mask().tolist() == [0, 1, 1, 0]
    env._entry = {"skill": "motion_failure", "case": {"failed_phase": "lateral_plan"}}
    assert env._action_mask().tolist() == [0, 1, 1, 0]
    env._entry = {"skill": "motion_failure", "case": {"failed_phase": "lateral_front_exit"}}
    assert env._action_mask().tolist() == [0, 0, 0, 1]


def test_curriculum_manifest_pins_cases_under_an_explicit_root(tmp_path):
    pytest.importorskip("gymnasium")
    from abstract_stack.skill_curriculum import load_curriculum

    path = tmp_path / "curriculum.json"
    case = dict(
        id="approach-stop",
        skill="push",
        role="recovery",
        lineage="development-9",
        checkpoint="one/checkpoint",
        manifest_sha256="a" * 64,
        original_failure="no admitted path",
    )
    path.write_text(json.dumps(dict(schema_version=1, cases=[case])))
    assert (
        load_curriculum(path, artifact_root=tmp_path)[0].checkpoint == tmp_path / case["checkpoint"]
    )

    case["checkpoint"] = "../checkpoint"
    path.write_text(json.dumps(dict(schema_version=1, cases=[case])))
    with pytest.raises(ValueError, match="artifact root"):
        load_curriculum(path, artifact_root=tmp_path)


def test_sample_cases_balances_lineages_with_unequal_checkpoint_counts():
    pytest.importorskip("gymnasium")
    from abstract_stack.skill_curriculum import _sample_case

    many = [SimpleNamespace(id=f"start-{i}", lineage="many") for i in range(12)]
    choices = [*many, SimpleNamespace(id="isolated", lineage="one")]
    rng = np.random.default_rng(117)
    selected = [_sample_case(rng, choices) for _ in range(300)]
    assert 120 <= sum(case.lineage == "one" for case in selected) <= 180
    assert {case.id for case in selected} == {case.id for case in choices}


@pytest.mark.mabel
def test_gym_reset_is_seeded_and_observes_the_native_lift_transmission(tmp_path):
    gym = pytest.importorskip("gymnasium")
    from abstract_stack.skill_curriculum import StockingCurriculumEnv

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = StockingCurriculumEnv("pick", tmp_path)
    try:
        with pytest.raises(gym.error.ResetNeeded):
            env.step(0)
        first, info = env.reset(seed=9)
        assert env.observation_space.contains(first)
        assert env.action_space.contains(env.action_space.sample(mask=info["action_mask"]))
        lift = env.env.model.actuator("Lift").id
        assert first["joints"][-1, 0] == env.env.data.actuator_length[lift]
        second, _ = env.reset(seed=9)
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
        assert first["active"].sum() == 1
        assert first["present"].sum() == 3
    finally:
        env.close()
