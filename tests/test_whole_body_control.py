"""Native bimanual/shared-lift reference contracts, separate from manipulation."""

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig
from abstract_stack.whole_body_control import (
    PalmGoal,
    StationaryWholeBodyController,
    WholeBodyConfig,
)


@pytest.fixture
def clear_robot():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    for _ in range(50):
        env.step()
    try:
        yield env
    finally:
        env.close()


def test_world_pose_and_configuration_validation():
    with pytest.raises(ValueError, match="proper rotation"):
        PalmGoal(np.zeros(3), -np.eye(3))
    with pytest.raises(ValueError):
        WholeBodyConfig(linear_speed_m_s=float("nan"))
    with pytest.raises(ValueError):
        WholeBodyConfig(reserve_compensation_headroom=1)
    goal = PalmGoal(np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        goal.position_m[0] = 1


@pytest.mark.mabel
def test_bimanual_preview_is_bounded_coupled_and_does_not_touch_live_physics(clear_robot):
    env = clear_robot
    controller = StationaryWholeBodyController(env.model, env.data)
    assert "Left Thumb DIP" in controller.fingers
    assert "Right Thumb DIP" in controller.fingers
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time)
    gains, limits = env.model.actuator_gainprm.copy(), env.model.actuator_ctrlrange.copy()
    nominal_before = controller.nominal.copy()
    poses = controller.nominal_poses()
    goals = {
        side: PalmGoal(position + [0, 0, 0.002], rotation)
        for side, (position, rotation) in poses.items()
    }
    step = controller.predict_step(goals, dt=env.dt, finger_targets={"Left Index MCP": 0.2})
    assert step.report["accepted"], step.report
    assert len(step.nominal) == 17
    assert list(step.report["nominal"]).count("Lift") == 1
    assert (
        np.max(np.abs(step.nominal - nominal_before) / controller.tracker.rates) <= env.dt + 1e-12
    )
    assert step.finger_nominal["Left Index MCP"] - controller.fingers["Left Index MCP"] <= env.dt
    np.testing.assert_array_equal(controller.nominal, nominal_before)
    for actual, old in zip(
        (env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time), before, strict=True
    ):
        np.testing.assert_array_equal(actual, old)
    np.testing.assert_array_equal(env.model.actuator_gainprm, gains)
    np.testing.assert_array_equal(env.model.actuator_ctrlrange, limits)
    action = controller.accept(step)
    np.testing.assert_array_equal(action, step.applied_targets)
    np.testing.assert_array_equal(controller.nominal, step.nominal)
    assert controller.tracker.last_report["gravity_force"]["Lift"] > 200
    with pytest.raises(ValueError, match="stale"):
        controller.accept(step)


@pytest.mark.mabel
def test_shared_lift_compensates_other_palm_hold_and_stale_state_cannot_execute(clear_robot):
    env = clear_robot
    controller = StationaryWholeBodyController(env.model, env.data)
    poses = controller.nominal_poses()
    position, rotation = poses["left"]
    step = controller.predict_step(
        {"left": PalmGoal(position + [0, 0, 0.002], rotation)}, dt=env.dt
    )
    assert step.report["accepted"], step.report
    assert step.report["palms"]["right"]["position_error_m"] < 0.0001
    env.step()
    with pytest.raises(ValueError, match="stale"):
        controller.accept(step)
    with pytest.raises(ValueError, match="native bounds"):
        controller.predict_step({}, dt=env.dt, finger_targets={"Left Index MCP": 20})
    env.data.ctrl[env.model.actuator("FL Drive").id] = 0.01
    with pytest.raises(ValueError, match="stopped"):
        controller.predict_step({}, dt=env.dt)


@pytest.mark.mabel
def test_measured_base_motion_is_rejected_even_with_zero_wheel_commands(clear_robot):
    snapshot = clear_robot.clone_for_planning()
    base_dof = int(snapshot.model.joint("base_root").dofadr[0])
    # Reset-only scratch fixture: zero drive commands do not prove the base is at rest.
    snapshot.data.qvel[base_dof] = 0.01
    mujoco.mj_forward(snapshot.model, snapshot.data)
    assert np.all(
        snapshot.data.ctrl[[snapshot.model.actuator(f"{p} Drive").id for p in ("FL", "FR", "B")]]
        == 0
    )
    with pytest.raises(ValueError, match="measured stationary"):
        StationaryWholeBodyController(snapshot.model, snapshot.data)


@pytest.mark.mabel
def test_coupled_compensated_collision_forecast_is_required(clear_robot, monkeypatch):
    controller = StationaryWholeBodyController(clear_robot.model, clear_robot.data)
    calls = []
    original = controller._contacts

    def applied_block(reference, q, fingers):
        calls.append((q.copy(), fingers.copy()))
        result = original(reference, q, fingers)
        if np.max(np.abs(q - controller.nominal)) > 1e-5:
            result[(10000, 10001)] = dict(geoms=["fixture", "right_hand"], distance_m=-0.01)
        return result

    monkeypatch.setattr(controller, "_contacts", applied_block)
    step = controller.predict_step({}, dt=clear_robot.dt, finger_targets={"Right Middle PIP": 0.3})
    assert not step.report["accepted"]
    assert calls and any(item["mode"] == "applied" for item in step.report["collisions"])
    # Reporting dictionaries are not an execution-admission capability.
    step.report["accepted"] = True
    with pytest.raises(ValueError, match="admission"):
        controller.accept(step)


@pytest.mark.mabel
def test_short_native_bimanual_tracking_retains_zero_external_forces(clear_robot):
    env = clear_robot
    controller = StationaryWholeBodyController(env.model, env.data)
    poses = controller.nominal_poses()
    goals = {
        side: PalmGoal(position + [0, 0, 0.003], rotation)
        for side, (position, rotation) in poses.items()
    }
    for _ in range(60):
        step = controller.predict_step(goals, dt=env.dt)
        assert step.report["accepted"], step.report
        env.step(controller.accept(step))
    for side, goal in goals.items():
        body = env.model.body(f"{side}_palm").id
        assert np.linalg.norm(env.data.xpos[body] - goal.position_m) < 0.006
    assert np.all(env.data.qfrc_applied == 0)
    assert np.all(env.data.xfrc_applied == 0)
    assert env.data.time > 2
    assert not np.any(env.data.warning.number)


@pytest.mark.mabel
def test_corrective_task_adapter_steps_natively_and_bounds_combined_offsets(clear_robot):
    from abstract_stack.whole_body_task import (
        CorrectionLimits,
        PalmDelta,
        WholeBodyTaskAdapter,
        WristCompliance,
    )

    env = clear_robot
    calls = []

    def guard(guard_env):
        assert guard_env is env
        calls.append(float(env.data.time))

    limits = CorrectionLimits(translation_m=0.001, rotation_rad=0.002)
    adapter = WholeBodyTaskAdapter(env, guard=guard, limits=limits)
    position, rotation = adapter.controller.nominal_poses()["left"]
    goals = {"left": PalmGoal(position, rotation)}
    delta = PalmDelta([0, 0, 0.1], [0, 0, 0.1])
    compliance = WristCompliance([0, 0, 0.001], [0, 0, 0.01])
    np.testing.assert_allclose(
        compliance.correction([0, 0, -2, 0, 0, -0.1]).translation_m, [0, 0, -0.002]
    )
    previous_time = env.data.time
    with pytest.raises(ValueError, match="targeted"):
        adapter.advance(goals, deltas={"right": delta})
    assert env.data.time == previous_time
    result = adapter.advance(goals, deltas={"left": delta}, compliance={"left": compliance})
    assert result["advanced"], result
    assert len(calls) == env.config.frame_skip
    assert result["elapsed_s"] == pytest.approx(env.dt)
    assert "left" in result["measured_tracking"]
    assert len(result["contact_wrenches_world"]["left"]) == 6
    assert np.linalg.norm(result["corrections"]["left"]["translation_m"]) == pytest.approx(0.001)
    assert np.linalg.norm(result["corrections"]["left"]["rotation_rad"]) == pytest.approx(0.002)
