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
        np.testing.assert_allclose(prediction.joint_path[0], hand.ik.measured_q(), atol=0.001)
        np.testing.assert_array_equal(prediction.joint_path[-1], prediction.q)
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
        assert prediction.report()["retained_cartesian_joint_waypoints"] == len(
            prediction.joint_path
        )
    finally:
        env.close()
