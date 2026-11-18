"""Sideways pushing is bounded and remains distinct from insertion/pick credit."""

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import ScriptConfig
from abstract_stack.lateral_contact import (
    LateralPushConfig,
    LateralStockingScript,
    SideFingerPusher,
)
from abstract_stack.mabel_controller import MabelController
from abstract_stack.packing import dense_shelf_config


def test_force_admittance_stops_and_reverses_without_unbounded_windup():
    config = LateralPushConfig()
    assert config.velocity(0) == config.speed_m_s
    assert config.velocity(config.force_target_N) == 0
    assert config.velocity(2 * config.force_target_N) == -config.speed_m_s
    assert config.velocity(1000) == -config.speed_m_s
    assert config.velocity(-1000) == config.speed_m_s
    with pytest.raises(ValueError, match="finite"):
        config.velocity(float("nan"))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(force_target_N=25),
        dict(distance_limit_m=0.051),
        dict(timeout_s=float("nan")),
        dict(speed_m_s=-1),
    ],
)
def test_invalid_lateral_skill_settings(kwargs):
    with pytest.raises(ValueError):
        LateralPushConfig(**kwargs)


@pytest.mark.mabel
@pytest.mark.parametrize("direction", [-1, 1])
def test_native_side_pusher_orientation_and_planning_isolation(direction):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    try:
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        pusher = SideFingerPusher(env.clone_for_planning(), ScriptConfig(), direction=direction)
        # Native local +Z points down the finger: +Y or -Y after yawing.
        assert direction * pusher.ROTATION[1, 2] > 0.9
        np.testing.assert_allclose(pusher.ROTATION.T @ pusher.ROTATION, np.eye(3), atol=1e-12)
        pusher.predict(pusher.position(), carrying=False, alternatives=False)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()


@pytest.mark.mabel
def test_side_tool_switch_preserves_nominal_and_pusher_offset_budget(tmp_path):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    try:
        script = LateralStockingScript(env, tmp_path)
        script.tracker = MabelController(env.model, env.data)
        nominal = script.tracker.nominal.copy()
        before = env.data.ctrl.copy()
        script._change_hand(SideFingerPusher, direction=1)
        assert script.tracker.config.wrist_offset_limit == 0.5
        np.testing.assert_array_equal(script.tracker.nominal, nominal)
        np.testing.assert_array_equal(env.data.ctrl, before)
    finally:
        env.close()


def test_tiny_displacement_cannot_pass_without_a_meaningful_motion():
    script = object.__new__(LateralStockingScript)
    with pytest.raises(ValueError, match="exceed its verification tolerance"):
        script.push_laterally(direction=1, distance_m=0.0005)


@pytest.mark.mabel
def test_selected_finger_compensation_is_bounded_and_does_not_edit_native_state():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    try:
        before_qpos, before_ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        ranges, gains = env.model.jnt_range.copy(), env.model.actuator_gainprm.copy()
        hand = SideFingerPusher(
            env.clone_for_planning(),
            ScriptConfig(ungrasped_insertion=True),
            direction=1,
            wrist_posture=0.2,
            folded_flexion=1.5,
        )
        tracker = hand.finger_load_compensation
        initial = before_ctrl.copy()
        targets, report = tracker.propose(initial, env.products[-1].name)
        for name, state in tracker.joints.items():
            detail = report["joints"][name]
            assert abs(detail["requested_feedforward_offset_rad"]) <= 0.15
            assert abs(detail["bounded_holding_Nm"]) <= 0.15
            assert (
                abs(targets[state["actuator"]] - initial[state["actuator"]]) <= 0.5 * env.dt + 1e-12
            )
            assert state["lower"] <= targets[state["actuator"]] <= state["upper"]
            assert state["previous"] == initial[state["actuator"]]
        tracker.acknowledge(targets)
        assert all(
            state["previous"] == targets[state["actuator"]] for state in tracker.joints.values()
        )
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.data.ctrl, before_ctrl)
        np.testing.assert_array_equal(initial, before_ctrl)
        np.testing.assert_array_equal(env.model.jnt_range, ranges)
        np.testing.assert_array_equal(env.model.actuator_gainprm, gains)
        assert hand.ik.upper[hand.ik.names.index("Left Wrist")] <= 0.324
    finally:
        env.close()


@pytest.mark.mabel
def test_cartesian_route_rejection_never_commands_native_robot(tmp_path):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    try:
        script = LateralStockingScript(env, tmp_path)
        before_qpos, before_ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        from abstract_stack.interactions import InteractionFailure

        with pytest.raises(InteractionFailure, match="rejected before actuation"):
            script._cartesian_move("unreachable_cartesian_fixture", [-100, 0, 1])
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.data.ctrl, before_ctrl)
        assert not script.rows
    finally:
        env.close()
