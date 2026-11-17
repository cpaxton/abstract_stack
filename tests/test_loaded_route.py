"""Retained IK paths and hypothetical carried poses never alter live physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import CartesianHand, ScriptConfig
from abstract_stack.loaded_route import object_tool_goal
from abstract_stack.scene import ShelfConfig


def test_tool_goal_composes_object_relative_offset_and_orientation():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    position = np.array([1.0, 2.0, 3.0])
    local_offset = np.array([0.1, 0.02, 0.03])
    body = SimpleNamespace(xpos=position, xmat=rotation.ravel())
    hand = SimpleNamespace(
        data=SimpleNamespace(body=lambda _: body, xmat=np.array([rotation.ravel()])),
        palm=0,
        position=lambda: position + rotation @ local_offset,
    )
    target, orientation = object_tool_goal(hand, "carton", np.zeros(3), np.eye(3))
    np.testing.assert_allclose(target, local_offset)
    np.testing.assert_allclose(orientation, np.eye(3))


@pytest.mark.mabel
def test_retained_cartesian_path_matches_checked_endpoints_without_changing_world():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    try:
        hand = CartesianHand(env, ScriptConfig())
        before = env.data.qpos.copy(), env.data.ctrl.copy()
        hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
        prediction = hand.predict(
            hand.position() + [0.01, 0, 0], alternatives=False, path_mode="cartesian"
        )
        assert prediction.feasible
        assert len(prediction.joint_path) >= 8
        np.testing.assert_array_equal(prediction.joint_path[0], hand.ik.measured_q())
        assert prediction.initial_cartesian_transition is None
        np.testing.assert_array_equal(prediction.joint_path[-1], prediction.q)
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
        assert prediction.report()["retained_cartesian_joint_waypoints"] == len(
            prediction.joint_path
        )
    finally:
        env.close()


@pytest.mark.mabel
def test_constrained_cartesian_path_retains_and_checks_real_initial_joint_transition():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    try:
        hand = CartesianHand(env, ScriptConfig(ik_position_tolerance=0.001))
        hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
        hand.ik.constrain_posture("Left Wrist", 0.1)
        start = hand.ik.measured_q()
        start_point, start_rotation = hand.ik.forward(start)
        before = env.data.qpos.copy(), env.data.ctrl.copy()
        prediction = hand.predict(
            hand.position() + [0.01, 0, 0], alternatives=False, path_mode="cartesian"
        )
        assert prediction.feasible
        transition = prediction.initial_cartesian_transition
        assert transition["accepted"] and transition["included_in_duration"]
        assert transition["samples"] >= 3
        np.testing.assert_array_equal(prediction.joint_path[0], start)
        wrist = hand.ik.names.index("Left Wrist")
        assert prediction.joint_path[transition["samples"] - 1, wrist] == pytest.approx(0.1)
        for q in prediction.joint_path[: transition["samples"]]:
            point, rotation = hand.ik.forward(q, prediction.reference_qpos)
            assert np.linalg.norm(point - start_point) <= 0.001
            assert np.linalg.norm(rotation - start_rotation) < 0.01
        rates = np.r_[np.ones(8), 0.08]
        segment_times = np.abs(np.diff(prediction.joint_path, axis=0)) / rates
        assert prediction.minimum_seconds >= transition["minimum_seconds"]
        assert (
            prediction.minimum_seconds
            >= 1.875 * (len(prediction.joint_path) - 1) * segment_times.max() - 1e-12
        )
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
    finally:
        env.close()


@pytest.mark.mabel
def test_endpoint_reachability_does_not_hide_unsafe_initial_posture_transition():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    try:
        hand = CartesianHand(env, ScriptConfig(ik_position_tolerance=0.001))
        hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
        hand.ik.constrain_posture("Left Wrist", -0.2)
        goal = hand.position() + [0.01, 0, 0]
        endpoint = hand.predict(goal, alternatives=False, check_path=False)
        assert endpoint.feasible
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        path = hand.predict(goal, alternatives=False, path_mode="cartesian")
        assert not path.feasible
        assert path.joint_path is None
        assert path.path_ik_error["stage"] == "initial_cartesian_transition"
        assert path.initial_cartesian_transition["maximum_hold_position_error_m"] > 0.001
        assert not path.initial_cartesian_transition["accepted"]
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()
