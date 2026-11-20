"""Supported insertion stages preserve task grading and terminal force aborts."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.dense_interactions import DenseStockingScript
from abstract_stack.insertion_skill import SelectedFingerInsertion
from abstract_stack.interactions import InteractionFailure
from abstract_stack.lateral_contact import LateralContact
from abstract_stack.receding_stocking import RecedingStockingScript


@pytest.mark.parametrize(
    ("center_y", "occupied_front_x", "hand_load", "yaw", "expected_fraction"),
    [
        (0.0575, -0.690, 0.0, 0.05, 0.65),
        (0.0575, -0.690, 0.0, -0.05, 0.0),
        (0.052, -0.690, 0.0, 0.05, None),
        (0.0575, -0.647, 0.0, 0.05, None),
        (0.0575, -0.690, 0.3, 0.05, None),
    ],
)
def test_side_recovery_uses_measured_footprint_and_supported_workspace(
    monkeypatch, center_y, occupied_front_x, hand_load, yaw, expected_fraction
):
    moves = []
    script = SimpleNamespace(
        config=SimpleNamespace(dense_recovery=True),
        env=SimpleNamespace(
            data=SimpleNamespace(
                body=lambda name: SimpleNamespace(
                    xpos=np.array(
                        [
                            -0.600,
                            center_y,
                            0.920,
                        ]
                    )
                )
            ),
            config=SimpleNamespace(center_y=-0.15, width=0.52),
        ),
        product=SimpleNamespace(name="carton", half_size=np.array([0.035, 0.025, 0.12])),
        lateral_goal_y=0.060,
        arm_bodies=set(),
        observer=SimpleNamespace(incidents=[]),
        events=[],
        _square_after_side_press=lambda turn, **kw: moves.append(("square", turn, kw)) or True,
    )
    script._stage = lambda phase: script.events.append({"phase": phase})
    monkeypatch.setattr(
        "abstract_stack.interactions.contact_forces",
        lambda *args: {"shelf": 1.0, "hand": hand_load},
    )

    def press(controller, **kwargs):
        moves.append(("press", kwargs))
        return {"lateral_staging_verified": True}

    monkeypatch.setattr(LateralContact, "push_laterally", press)
    assert SelectedFingerInsertion._recover_front_square_clearance(
        script, occupied_front_x, np.array([0.038, 0.028, 0.12]), yaw
    ) is (expected_fraction is not None)
    if expected_fraction is not None:
        _, kwargs = moves[0]
        assert kwargs["direction"] == -1
        assert kwargs["distance_m"] == pytest.approx(0.004)
        assert kwargs["front_contact_fraction"] == expected_fraction
        centered = expected_fraction == 0
        assert kwargs["settings"].standoff_m == (0.01 if centered else 0.015)
        assert moves[1] == ("square", kwargs["base_turn_rad"], {"prefer_coupled": centered})
    else:
        assert not moves


@pytest.fixture
def script(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0))
    value = RecedingStockingScript(
        env,
        tmp_path,
        loaded_route=True,
        base_reposition=True,
        bounded_contact_ik=True,
        payload_load_compensation=True,
        split_insertion=True,
    )
    try:
        yield value
    finally:
        env.close()


@pytest.mark.mabel
@pytest.mark.parametrize("target", [float("nan"), "before_entry", "beyond_final"])
def test_invalid_partial_push_cannot_command_motion_or_claim_a_stage(script, target):
    env, product = script.env, script.product
    if target == "before_entry":
        target = float(env.data.body(product.name).xpos[0]) + 0.01
    elif target == "beyond_final":
        target = env.config.front_x - env.config.depth - 0.01
    before = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time
    with pytest.raises(ValueError, match="strictly between entry and final"):
        DenseStockingScript._insert_from_front(script, partial_stop_x=target)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])
    assert env.data.time == before[2]
    assert not script.events


@pytest.mark.mabel
def test_force_abort_is_not_reinterpreted_as_an_empty_base_workspace_replan(script, monkeypatch):
    monkeypatch.setattr(script, "_move", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "_move_base", lambda *a, **k: pytest.fail("Force abort retried"))

    def abort(*args):
        raise InteractionFailure("finger_push: contact force exceeded 25.0 N")

    monkeypatch.setattr(DenseStockingScript, "_plan_pusher", abort)
    with pytest.raises(InteractionFailure, match="contact force exceeded"):
        script._plan_pusher(np.zeros(3), np.zeros(3))


@pytest.mark.mabel
def test_intermediate_gate_rejects_overshoot_motion_and_loaded_handoff(script):
    # Isolate the gate with scratch state and synthetic force observations.
    # This checks acceptance logic; it does not claim a physical partial push.
    clone = script.env.clone_for_planning()
    script.env = clone
    script.last_forces = dict(shelf=1.0, hand=0.0)
    x = float(clone.data.body(script.product.name).xpos[0])
    assert script._partial_push_state(x, unloaded=True)["accepted"]
    assert not script._partial_push_state(x + 0.006)["accepted"]
    assert not script._partial_push_state(x - 0.006)["accepted"]
    joint = clone.model.joint(f"{script.product.name}_joint")
    clone.data.qvel[int(joint.dofadr[0])] = 0.006
    mujoco.mj_forward(clone.model, clone.data)
    assert not script._partial_push_state(x)["accepted"]
    clone.data.qvel[int(joint.dofadr[0])] = 0
    mujoco.mj_forward(clone.model, clone.data)
    script.last_forces["hand"] = 1
    assert script._partial_push_state(x)["accepted"]
    assert not script._partial_push_state(x, unloaded=True)["accepted"]
    script.last_forces.update(hand=0, shelf=0)
    assert not script._partial_push_state(x)["accepted"]
