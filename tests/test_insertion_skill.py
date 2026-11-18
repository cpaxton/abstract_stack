"""Mode selection preserves legacy scenes and reuses the dense skill bodies."""

from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.dense_interactions import DenseStockingScript, FingerPusher
from abstract_stack.finger_pusher import FingerPusher as SharedFingerPusher
from abstract_stack.interactions import ScriptConfig, StockingScript


def script_fixture(gap=0.003, *, completed=True, enabled=True, level=0):
    active = SimpleNamespace(
        name="active", level=0, half_size=(0.035, 0.025, 0.12), target=(-0.72, -0.12, 0.92)
    )
    neighbor = SimpleNamespace(name="neighbor", level=level, half_size=(0.035, 0.025, 0.12))
    body = SimpleNamespace(xpos=np.array([-0.72, -0.12 - 0.05 - gap, 0.92]), xmat=np.eye(3).ravel())
    script = object.__new__(StockingScript)
    script.product = active
    script.env = SimpleNamespace(
        products=[active, neighbor],
        data=SimpleNamespace(body=lambda _: body),
        config=SimpleNamespace(row_free_space=None),
    )
    script.observer = SimpleNamespace(completed={"neighbor"} if completed else set())
    script.config = ScriptConfig(ungrasped_insertion=enabled)
    return script


def test_adjacent_neighbor_is_measured_and_must_be_completed_in_same_row():
    assert script_fixture()._adjacent_placed_neighbors()[0]["product"] == "neighbor"
    assert not script_fixture(gap=0.045)._adjacent_placed_neighbors()
    assert not script_fixture(completed=False)._adjacent_placed_neighbors()
    assert not script_fixture(level=1)._adjacent_placed_neighbors()


def test_neighbor_mode_releases_then_pushes_and_legacy_keeps_wrapping_slide():
    for enabled, expected in ((True, "finger"), (False, "wrapping")):
        script = script_fixture(enabled=enabled)
        calls = []
        script.events = []
        script._stage = lambda phase, script=script: script.events.append(dict(phase=phase))
        script._pick_and_place_front = lambda selected, calls=calls: (
            calls.append("pick_place") or 0.5
        )
        script._release_and_insert_front = lambda grip, calls=calls: calls.append("finger")
        script._push_grasped = lambda grip, calls=calls: calls.append("wrapping")
        script._one_box()
        assert calls == ["pick_place", expected]
        assert bool(script.events) == enabled


def test_dense_controller_inherits_shared_skill_without_duplicating_methods():
    assert FingerPusher is SharedFingerPusher
    for name in ("_insert_from_front", "_precision_release", "_plan_pusher", "_change_hand"):
        assert getattr(DenseStockingScript, name) is getattr(StockingScript, name)


def test_front_reference_is_chosen_inside_the_carry_route_before_seating():
    script = script_fixture()
    canonical = script.product.target[1]
    goal = script._front_reference_y()
    assert canonical < goal <= canonical + 0.02
    assert script.product.target[1] == canonical
    script.config = ScriptConfig(ungrasped_insertion=False)
    assert script._front_reference_y() == canonical


def test_native_continuation_sync_keeps_nominal_separate_from_applied():
    from abstract_stack.contact_execution import sync_tracking_state

    hand = SimpleNamespace(
        names=["Left Arm 1", "Left Wrist"],
        command=np.array([0.2, 0.3, 0.4]),
        ik=SimpleNamespace(forward=lambda q, reference: (np.zeros(3), np.eye(3))),
    )
    tracker = SimpleNamespace(
        NAMES=("Left Arm 1", "Right Arm 1", "Left Wrist", "Lift"),
        nominal=np.array([0.0, 0.8, 0.0, 0.0]),
        ids=np.array([0, 3, 1, 2]),
    )
    data = SimpleNamespace(ctrl=np.array([0.21, 0.42, 0.45, 0.9]), qpos=np.zeros(2))
    script = SimpleNamespace(controller=hand, tracker=tracker, env=SimpleNamespace(data=data))
    sync_tracking_state(script)
    np.testing.assert_array_equal(tracker.nominal, [0.2, 0.8, 0.3, 0.4])
    np.testing.assert_array_equal(tracker.applied, [0.21, 0.9, 0.42, 0.45])


def test_dense_handoff_reserves_whole_row_clearance_and_both_walls():
    script = script_fixture(gap=0.001)
    script.env.config = SimpleNamespace(row_free_space=0.003, center_y=-0.12, width=0.103)
    assert script._front_reference_y() == script.product.target[1]
    # A biased nominal target cannot make the staging corridor leave a wall.
    script.observer.completed.clear()
    script.product.target = (-0.72, -0.15, 0.92)
    goal = script._front_reference_y()
    assert goal >= -0.12 - 0.103 / 2 + 0.025 + 0.0004
    assert goal < -0.14


@pytest.mark.mabel
def test_selected_insertion_reserves_lift_clearance_without_changing_native_model():
    import mujoco

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import CartesianHand

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        initial = env.data.qpos.copy(), env.data.ctrl.copy(), env.model.jnt_range.copy()
        hand = CartesianHand(env, ScriptConfig(ungrasped_insertion=True))
        assert hand.ik.lower[-1] == pytest.approx(0.03)
        geometry = [
            next(
                g
                for g in range(env.model.ngeom)
                if env.model.geom_bodyid[g] == env.model.body(name).id and env.model.geom_contype[g]
            )
            for name in ("lift_base", "body_mount")
        ]
        q = hand.ik.measured_q()
        q[-1] = hand.ik.lower[-1]
        hand.ik.forward(q)
        mujoco.mj_fwdPosition(env.model, hand.ik.scratch)
        assert mujoco.mj_geomDistance(env.model, hand.ik.scratch, *geometry, 0.1, None) > 0.01
        for value, before in zip(
            (env.data.qpos, env.data.ctrl, env.model.jnt_range), initial, strict=True
        ):
            np.testing.assert_array_equal(value, before)
    finally:
        env.close()


@pytest.mark.mabel
def test_paired_native_tool_and_release_forecast_own_their_state(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.contact_rollout import integration_state
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.finger_pusher import PairedFingerPusher
    from abstract_stack.release_planning import ReleaseForecast

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        before = integration_state(env.model, env.data)
        ranges = env.model.jnt_range.copy()
        script = StockingScript(
            env, tmp_path, ScriptConfig(ungrasped_insertion=True, paired_finger_insertion=True)
        )
        script.tracker = script._new_tracker()
        script._change_hand(PairedFingerPusher, finger="Middle", tilt_degrees=20, wrist_posture=0.2)
        hand = script.controller
        targets = hand.finger_targets(0)
        assert targets["Left Index MCP"] == targets["Left Ring MCP"] == -0.2
        assert targets["Left Middle MCP"] == targets["Left Pinky MCP"] == 1.5
        assert targets["Left Index Abd"] == 0.15
        assert targets["Left Ring Abd"] == -0.15
        delta = hand.ROTATION @ (hand.native_tip_offset("Ring") - hand.native_tip_offset("Index"))
        assert abs(delta[0]) < 0.002
        assert abs(delta[1]) > 0.015
        assert abs(delta[1]) < 0.048
        hand.open_posture_overrides = {"Left Thumb PIP": -0.2}
        hand.abduction_targets = {"Left Middle Abd": 0.1}
        forecast = ReleaseForecast(script)
        plane_yaw = hand.contact_plane_yaw()
        joint = env.model.joint("Left Index PIP")
        forecast.env.data.qpos[int(joint.qposadr[0])] += 0.05
        import mujoco

        mujoco.mj_forward(env.model, forecast.env.data)
        assert not np.isclose(forecast.controller.contact_plane_yaw(), plane_yaw, atol=1e-5)
        assert hand.contact_plane_yaw() == plane_yaw
        # Restore the independent fixture before the exact clone-state checks.
        mujoco.mj_copyData(forecast.env.data, env.model, env.data)
        assert forecast.controller.open_posture_overrides == hand.open_posture_overrides
        assert forecast.controller.abduction_targets == hand.abduction_targets
        forecast.controller.open_posture_overrides["Left Thumb PIP"] = -0.1
        forecast.controller.abduction_targets["Left Middle Abd"] = 0.2
        assert hand.open_posture_overrides["Left Thumb PIP"] == -0.2
        assert hand.abduction_targets["Left Middle Abd"] == 0.1
        assert forecast.env.model is env.model
        assert forecast.env.data is not env.data
        np.testing.assert_array_equal(integration_state(env.model, forecast.env.data), before)
        forecast.env.data.ctrl[:] += (
            0.001  # Independent scratch fixture, no native step or success claim.
        )
        forecast.controller.command[:] += 0.001
        forecast.controller.OFFSET[:] += 0.001
        np.testing.assert_array_equal(integration_state(env.model, env.data), before)
        np.testing.assert_array_equal(env.model.jnt_range, ranges)
        assert not np.array_equal(forecast.controller.command, hand.command)
        assert not np.array_equal(forecast.controller.OFFSET, hand.OFFSET)
        script._restore_grasp_hand()
        grasp = script.controller
        assert not isinstance(grasp, SharedFingerPusher)
        for name, value in script.grasp_abduction_targets.items():
            assert grasp.finger_targets(0)[name] == value
            assert grasp.finger_targets(0.5)[name] == value
        assert not grasp.open_posture_overrides
        np.testing.assert_array_equal(integration_state(env.model, env.data), before)
    finally:
        env.close()
