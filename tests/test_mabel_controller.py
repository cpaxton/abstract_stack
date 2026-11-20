"""Native-physics tracking, bimanual gravity compensation and reference limits."""

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.mabel_controller import MabelController, TrackingConfig
from abstract_stack.scene import ShelfConfig


@pytest.fixture
def clear_robot():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    # Isolate unloaded tracking: fixtures are explicitly placed away from both
    # hands, unlike the shelf bench where the right thumb initially contacts it.
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    for _ in range(50):
        env.step()
    yield env
    env.close()


@pytest.mark.mabel
def test_gravity_prediction_preserves_physics_and_bounds_references(clear_robot):
    env = clear_robot
    tracker = MabelController(env.model, env.data)
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time)
    gains = env.model.actuator_gainprm.copy()
    initial = tracker.applied.copy()
    action = tracker.targets({"Left Arm 1": 0.2, "Right Arm 1": -0.2, "Lift": 0.4}, dt=env.dt)
    assert np.max(np.abs(action[tracker.ids] - initial) / tracker.rates) <= env.dt + 1e-12
    for actual, old in zip(
        (env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time), before, strict=True
    ):
        np.testing.assert_array_equal(actual, old)
    np.testing.assert_array_equal(env.model.actuator_gainprm, gains)
    assert np.isfinite(action).all()
    assert 200 < tracker.last_report["gravity_force"]["Lift"] < 400
    assert (action[tracker.ids] >= tracker.lower).all()
    assert (action[tracker.ids] <= tracker.upper).all()
    with pytest.raises(ValueError):
        tracker.targets({"Lift": float("nan")}, dt=env.dt)


@pytest.mark.mabel
def test_compensation_tracks_both_arms_and_lift_in_free_space(clear_robot):
    env = clear_robot
    tracker = MabelController(env.model, env.data)
    before_lift = abs(env.data.actuator("Lift").length[0] - tracker.nominal[-1])
    desired = {"Left Arm 1": 0.08, "Right Arm 1": -0.08, "Lift": tracker.nominal[-1] + 0.015}
    for _ in range(200):
        env.step(tracker.targets(desired, dt=env.dt))
    error = env.data.actuator_length[tracker.ids] - tracker.nominal
    assert before_lift > 0.002
    assert abs(error[-1]) < 0.0002
    assert np.max(np.abs(error[:-1])) < 0.006
    assert not tracker.last_report["feedforward_limited"]
    assert not tracker.last_report["target_limited"]
    assert not tracker.last_report["rate_limited"]
    assert np.all(env.data.qfrc_applied == 0)
    assert np.all(env.data.xfrc_applied == 0)
    assert env.data.body("mabel_chassis").xmat.reshape(3, 3)[2, 2] > 0.999


@pytest.mark.mabel
def test_ablation_and_partial_updates_do_not_accumulate_offsets(clear_robot):
    env = clear_robot
    tracker = MabelController(
        env.model, env.data, TrackingConfig(gravity_compensation=False, wrist_damping=0)
    )
    expected = env.data.ctrl[tracker.ids].copy()
    for _ in range(3):
        env.step(tracker.targets({}, dt=env.dt))
    np.testing.assert_array_equal(tracker.nominal, expected)
    np.testing.assert_array_equal(tracker.applied, expected)
    tracker = MabelController(env.model, env.data)
    nominal = tracker.nominal.copy()
    for _ in range(10):
        env.step(tracker.targets({}, dt=env.dt))
    np.testing.assert_array_equal(tracker.nominal, nominal)


@pytest.mark.mabel
def test_offset_caps_are_visible(clear_robot):
    tracker = MabelController(
        clear_robot.model, clear_robot.data, TrackingConfig(lift_offset_limit=0.0001)
    )
    tracker.targets({}, dt=clear_robot.dt)
    assert "Lift" in tracker.last_report["feedforward_limited"]


def test_invalid_tracking_configuration():
    with pytest.raises(ValueError):
        TrackingConfig(lift_speed=0)
    with pytest.raises(ValueError):
        TrackingConfig(wrist_damping=-1)
    with pytest.raises(ValueError):
        TrackingConfig(wrist_offset_limit=float("nan"))


@pytest.mark.mabel
def test_partial_update_validates_retained_nominal_instead_of_live_servo_target(clear_robot):
    env = clear_robot
    tracker = MabelController(env.model, env.data)
    index = tracker.NAMES.index("Left Arm 1")
    nominal = tracker.nominal[index]
    # Synthetic controller-bound fixture: compensated live ctrl may differ
    # from the retained nominal. Native model limits/gains remain untouched.
    tracker.upper[index] = nominal + 0.01
    env.data.ctrl[tracker.ids[index]] = nominal + 0.02
    tracker.targets({"Right Arm 1": -0.08}, dt=env.dt)
    assert tracker.nominal[index] == nominal
    with pytest.raises(ValueError, match="joint bounds"):
        tracker.targets({"Left Arm 1": nominal + 0.02}, dt=env.dt)


@pytest.mark.mabel
def test_load_compensation_is_bounded_and_does_not_reinterpret_nominal(clear_robot):
    env = clear_robot
    config = TrackingConfig(gravity_compensation=False, wrist_damping=0, wrist_offset_limit=0.5)
    tracker = MabelController(env.model, env.data, config)
    wrist = tracker.NAMES.index("Left Wrist")
    nominal = tracker.nominal.copy()
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    gains = env.model.actuator_gainprm.copy()
    for _ in range(100):
        action = tracker.targets({}, dt=env.dt, load_feedforward={"Left Wrist": 0.3})
    assert action[tracker.ids[wrist]] == pytest.approx(nominal[wrist] + 0.3)
    np.testing.assert_array_equal(tracker.nominal, nominal)
    assert tracker.last_report["load_force"]["Left Wrist"] == 0.3
    assert not tracker.last_report["feedforward_limited"]
    tracker.targets({}, dt=env.dt, load_feedforward={"Left Wrist": 2.0})
    assert "Left Wrist" in tracker.last_report["feedforward_limited"]
    for _ in range(100):
        tracker.targets({"Left Wrist": 0.6}, dt=env.dt, load_feedforward={"Left Wrist": 0.3})
    assert "Left Wrist" in tracker.last_report["target_limited"]
    for current, old in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before, strict=True):
        np.testing.assert_array_equal(current, old)
    np.testing.assert_array_equal(env.model.actuator_gainprm, gains)
    for load in ({"not an actuator": 0.1}, {"Left Wrist": float("nan")}):
        with pytest.raises(ValueError, match="finite named"):
            tracker.targets({}, dt=env.dt, load_feedforward=load)
