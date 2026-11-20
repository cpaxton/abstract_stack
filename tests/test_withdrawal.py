"""Intentional contact loss needs a fresh square gate and preserves safety history."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.contact_finish import BackstopDwell
from abstract_stack.contact_rollout import ContactMonitor, RolloutScene, integration_state


def squared_world():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".002" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <geom name="table" type="box" pos="0 0 .3" size="1 1 .3"/>
        <geom name="back" type="box" pos="-.07 0 .65" size=".02 .3 .3"/>
        <body name="robot" pos=".078 0 .65">
          <joint name="drive" type="slide" axis="1 0 0" range="-.1 .1"/>
          <body name="hand"><geom type="sphere" size=".03" mass=".1"/></body>
        </body>
        <body name="active" pos="0 0 .65"><freejoint/>
          <geom type="box" size=".05 .05 .05" mass=".18"/></body>
        <body name="other" pos=".4 .4 .65"><freejoint/>
          <geom type="box" size=".05 .05 .05" mass=".18"/></body>
      </worldbody>
      <actuator><position name="drive" joint="drive" kp="200" ctrlrange="-.1 .1"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.ctrl[:] = -0.01
    mujoco.mj_forward(model, data)
    scene = RolloutScene(
        products=("active", "other"),
        active="active",
        robot_root="robot",
        hand_root="hand",
        support_geoms=("table",),
        active_half_size=(0.05, 0.05, 0.05),
        support_height=0.6,
        back_geom="back",
    )
    monitor = ContactMonitor(model, data, scene)
    dwell = BackstopDwell(0.002, 0.4, 0.12, 5, 0.15)
    dwell.update(data.time, dict(hand_N=0, back_N=0, support_N=0), np.eye(3))
    for _ in range(300):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert monitor.sample(data, elapsed=0.002) is None
        dwell.update(data.time, monitor.last, data.xmat[monitor.active].reshape(3, 3))
    assert dwell.verified
    return model, data, monitor, dwell


def test_native_withdrawal_allows_intentional_hand_loss_and_preserves_history():
    model, data, monitor, dwell = squared_world()
    monitor.begin_withdrawal(data, dwell)
    assert monitor.phase == "withdrawal"
    initial, peak = monitor.initial[monitor.active].copy(), monitor.peaks["robot_product_N"]
    data.ctrl[:] = 0.05  # Native servo retreat, not a pose change.
    for _ in range(400):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        assert monitor.sample(data, elapsed=0.002) is None
    assert monitor.contact_loss > 0.04 and monitor.last["hand_N"] <= 0.2
    assert monitor.peaks["robot_product_N"] >= peak
    restored = ContactMonitor(model, data, monitor.scene)
    restored.restore_state(monitor.state_dict(), data)
    assert restored.phase == "withdrawal" and restored.contact_loss == monitor.contact_loss
    np.testing.assert_array_equal(restored.initial[monitor.active], initial)
    assert restored.sample(data, elapsed=0.002) is None
    # Guard injection fixture: an external input still rejects after hand release.
    data.qfrc_applied[0] = 0.01
    assert restored.sample(data, elapsed=0) == "external_force_input"


def test_stale_or_unverified_square_cannot_enable_contact_loss():
    _, data, monitor, dwell = squared_world()
    dwell.verified = False
    with pytest.raises(ValueError, match="fresh verified"):
        monitor.begin_withdrawal(data, dwell)
    dwell.verified = True
    dwell.last_time_s -= 0.002
    with pytest.raises(ValueError, match="fresh verified"):
        monitor.begin_withdrawal(data, dwell)
    assert monitor.phase == "push"


def test_withdrawal_preserves_all_product_tip_guard():
    model, data, monitor, dwell = squared_world()
    monitor.begin_withdrawal(data, dwell)
    # Deliberate reset-only fault fixture, not a claimed physical manipulation.
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "other")
    address = int(model.jnt_qposadr[model.body_jntadr[body]])
    data.qpos[address + 3 : address + 7] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0, 0]
    mujoco.mj_forward(model, data)
    assert monitor.sample(data, elapsed=0.002) == "product_tip_or_drop"
    assert monitor.incidents[-1]["product"] == "other"


def test_withdrawal_does_not_bypass_force_guard():
    _, data, monitor, dwell = squared_world()
    monitor.begin_withdrawal(data, dwell)
    # Stricter guard injection tests the same measured native contact load.
    monitor.scene = replace(monitor.scene, force_limit_N=0.001)
    assert monitor.sample(data, elapsed=0) == "contact_force_guard"


def test_unload_forecast_checks_each_mode_from_the_same_state(monkeypatch):
    from abstract_stack import release_planning
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.withdrawal_feedback import WithdrawalFeedbackConfig

    model, data, monitor, dwell = squared_world()
    initial = integration_state(model, data)
    product = SimpleNamespace(name="active")
    events = []
    script = SimpleNamespace(
        env=SimpleNamespace(model=model, data=data),
        product=product,
        observer=SimpleNamespace(completed=set()),
        events=events,
        _stage=lambda phase: events.append({"phase": phase}),
    )

    def clone(_):
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, data)
        return SimpleNamespace(
            env=SimpleNamespace(model=model, data=scratch, products=(product,)),
            controller=SimpleNamespace(command=np.zeros(1), data=scratch),
        )

    def forecast(hand, nominal, guarded, square, **kwargs):
        np.testing.assert_array_equal(integration_state(model, hand.data), initial)
        assert kwargs["unload_only"]
        admitted = not kwargs["feedback"].paired_plane_feedback
        mujoco.mj_step(model, hand.data)
        return (
            dict(unloaded_support_verified=admitted, strict_release_yaw_verified=admitted),
            [],
            [],
        )

    monkeypatch.setattr(release_planning, "ReleaseForecast", clone)
    monkeypatch.setattr(release_planning, "supported_withdrawal", forecast)
    monkeypatch.setattr(ShelfStockingEnv, "_placement_conditions", lambda env: [True])
    chosen = release_planning.choose_unload_feedback(
        script,
        monitor,
        dwell,
        WithdrawalFeedbackConfig(yaw_feedback=True, paired_plane_feedback=True),
        clear_distance_m=0.08,
        contact_neighbors=(),
        release_yaw_limit_deg=0.15,
        coarse_loaded_entry=False,
    )
    assert not chosen.paired_plane_feedback
    assert [row["admitted"] for row in events[-1]["candidates"]] == [False, True]
    np.testing.assert_array_equal(integration_state(model, data), initial)
