"""Object-aligned entry geometry, retained path timing and skill distribution scope."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.grasping import (
    object_aligned_rotation,
    open_hand_clearance,
    timed_joint_path,
)
from abstract_stack.interactions import (
    CartesianHand,
    InteractionFailure,
    ScriptConfig,
    StockingScript,
)
from abstract_stack.skill_entries import SKILL_CONTRACTS, entry_distribution


def yaw_rotation(degrees):
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )


@pytest.mark.parametrize("yaw", [-179, -116.7, -63.3, 0, 63.3, 116.7, 179])
def test_carton_symmetry_keeps_grasp_orientation_bounded(yaw):
    rotation, chosen = object_aligned_rotation(yaw_rotation(yaw), CartesianHand.ROTATION)
    assert -90 <= chosen <= 90
    assert abs(np.sin(np.radians(chosen - yaw))) < 1e-10
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError, match="tilt"):
        object_aligned_rotation(CartesianHand.ROTATION, CartesianHand.ROTATION)


def test_retained_path_timing_limits_each_coordinate_even_with_uneven_segments():
    original = np.array([[0, 0], [0, 0], [0.2, 0.001], [-0.2, 0.0011], [0.0, 0.01]])
    rates = np.array([1.0, 0.08])
    path, knots, duration = timed_joint_path(original, rates)
    times = np.linspace(0, 1, 10001)
    blend = 10 * times**3 - 15 * times**4 + 6 * times**5
    commands = np.array([np.interp(blend, knots, path[:, i]) for i in range(2)]).T
    assert np.all(np.abs(np.diff(commands, axis=0)) <= rates * duration / 10000 + 1e-10)
    np.testing.assert_array_equal(commands[-1], original[-1])
    assert np.min(commands[:, 0]) < -0.19
    with pytest.raises(ValueError):
        timed_joint_path([[0, 1]], rates)


def test_entry_distribution_keeps_unreached_skills_and_reports_only_observed_states():
    entry = dict(
        skill="grasp_lift",
        active_product="carton_0",
        entry_file="entry.json",
        objects={"carton_0": dict(position_world_m=[0.1, -0.2, 0.9], yaw_deg=63.3)},
    )
    report = entry_distribution([entry])
    assert set(report) == set(SKILL_CONTRACTS)
    assert report["grasp_lift"]["active_yaw_degrees"] == [63.3]
    assert report["transfer"]["samples"] == 0
    assert report["transfer"]["active_position_min_m"] is None
    assert not report["grasp_lift"]["robustness_or_training_claim"]


@pytest.fixture
def robot():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    yield env
    env.close()


@pytest.mark.mabel
def test_rotated_carton_rejects_open_hand_overlap_and_aligned_pose_clears(robot):
    # Explicit reset fixture; no live manipulation or success credit.
    joint = int(robot.model.joint("carton_0_joint").qposadr[0])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, yaw_rotation(63.3).ravel())
    robot.data.qpos[joint + 3 : joint + 7] = quat
    mujoco.mj_forward(robot.model, robot.data)
    qpos, ctrl = robot.data.qpos.copy(), robot.data.ctrl.copy()
    hand = CartesianHand(robot, ScriptConfig(grasp_skill=True))
    target = robot.data.body("carton_0").xpos + [0, 0, 0.04]
    old = hand.predict(target, allowed_object="carton_0", alternatives=False, check_path=False)
    assert open_hand_clearance(hand, old, "carton_0")["minimum_distance_m"] < -0.002
    hand.active_rotation, _ = object_aligned_rotation(yaw_rotation(63.3), hand.ROTATION)
    aligned = hand.predict(target, allowed_object="carton_0", alternatives=False, check_path=False)
    assert aligned.feasible
    assert open_hand_clearance(hand, aligned, "carton_0")["minimum_distance_m"] > 0
    # Alignment alone is too tight. Keep the 2 mm entry margin and choose a
    # bounded object-frame centering offset, rather than accepting near contact.
    shifted = target + yaw_rotation(63.3) @ [0.004, -0.004, 0]
    aligned = hand.predict(shifted, allowed_object="carton_0", alternatives=False, check_path=False)
    assert aligned.feasible
    assert open_hand_clearance(hand, aligned, "carton_0")["minimum_distance_m"] >= 0.002
    np.testing.assert_array_equal(robot.data.qpos, qpos)
    np.testing.assert_array_equal(robot.data.ctrl, ctrl)


@pytest.mark.mabel
def test_approach_executes_retained_path_instead_of_endpoint_shortcut(robot, tmp_path, monkeypatch):
    script = StockingScript(robot, tmp_path, ScriptConfig(grasp_skill=True))
    hand = script.controller
    hand.active_rotation = robot.data.xmat[hand.palm].reshape(3, 3).copy()
    proposal = hand.predict(hand.position(), alternatives=False)
    start = hand.ik.measured_q()
    detour = start.copy()
    detour[0] += 0.02
    proposal = replace(proposal, joint_path=np.array([start, detour, start]), feasible=True)
    requested, commands = [], []

    def predict(*args, **kwargs):
        requested.append(kwargs)
        return proposal

    monkeypatch.setattr(hand, "predict", predict)
    monkeypatch.setattr(script, "_apply_tick", lambda action: commands.append(action) or False)
    qpos, ctrl = robot.data.qpos.copy(), robot.data.ctrl.copy()
    script._move("approach", hand.position(), check_tracking=False)
    assert requested[0]["allowed_object"] is None
    assert requested[0]["path_mode"] == "cartesian"
    assert max(command[hand.names[0]] for command in commands) > start[0] + 0.019
    np.testing.assert_array_equal(robot.data.qpos, qpos)
    np.testing.assert_array_equal(robot.data.ctrl, ctrl)


@pytest.mark.mabel
def test_grasp_closure_checks_future_fingers_and_rejects_before_actuation(
    robot, tmp_path, monkeypatch
):
    script = StockingScript(robot, tmp_path, ScriptConfig(grasp_skill=True))
    script._stage("close")
    script.controller.active_rotation = robot.data.xmat[script.controller.palm].reshape(3, 3).copy()
    qpos, ctrl, nominal = (
        robot.data.qpos.copy(),
        robot.data.ctrl.copy(),
        script.controller.command.copy(),
    )
    calls = []
    monkeypatch.setattr(
        script.controller,
        "check_hand_step",
        lambda *args, **kwargs: [dict(geoms=["finger", "fixture"])],
    )
    monkeypatch.setattr(script, "_apply_tick", lambda action: calls.append(action))
    with pytest.raises(InteractionFailure, match="before actuation"):
        script._tick(script.controller.position() + [0.001, 0, 0], 0.3)
    assert not calls
    np.testing.assert_array_equal(script.controller.command, nominal)
    np.testing.assert_array_equal(robot.data.qpos, qpos)
    np.testing.assert_array_equal(robot.data.ctrl, ctrl)
