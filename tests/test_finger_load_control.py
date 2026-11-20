"""Native reference compensation keeps load signs, bounds and geometry responsibility explicit."""

import mujoco
import numpy as np
import pytest

from abstract_stack.finger_load_control import FingerLoadCompensator
from abstract_stack.front_tangential import FrontTangentialStockingScript, TangentialFingerPusher
from abstract_stack.interactions import ScriptConfig


def contact_fixture():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".001"/>
        <worldbody>
          <body name="finger_body" pos="0 0 .4">
            <joint name="finger" axis="0 1 0" range="-1 1" damping=".1"/>
            <geom type="capsule" fromto="0 0 0 .3 0 .1" size=".03" mass=".1"/>
          </body>
          <body name="carton" pos=".35 0 .5"><geom type="box" size=".04 .1 .1"/></body>
        </worldbody>
        <actuator><position name="finger" joint="finger" kp="10" ctrlrange="-1 1"/></actuator>
      </mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def test_measured_contact_load_sign_matches_native_generalized_constraint_force():
    model, data = contact_fixture()
    control = FingerLoadCompensator(model, data, ["finger"], product_name="carton")
    qpos, ctrl = data.qpos.copy(), data.ctrl.copy()
    applied = control.targets({"finger": 0.0}, dt=1.0)
    effort = control.last_report["finger"]["measured_contact_effort_Nm"]
    # No gravity, other contacts or active joint limits in this fixture.
    assert abs(effort) > 0.01
    assert effort == pytest.approx(data.qfrc_constraint[0], abs=1e-8)
    assert applied["finger"] == pytest.approx(-effort / 10, abs=1e-8)
    assert control.last_report["finger"]["nominal_reference_rad"] == 0
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.ctrl, ctrl)


def test_native_bounds_follow_load_rate_and_offset_limits():
    model, data = contact_fixture()
    control = FingerLoadCompensator(
        model, data, ["finger"], product_name="carton", rate_limit=0.1, offset_limit=0.05
    )
    applied = control.targets({"finger": 0.999}, dt=0.02)
    assert -1 <= applied["finger"] <= 1
    assert abs(applied["finger"] - 0.999) <= 0.002 + 1e-12
    assert control.last_report["finger"]["clipped"]
    assert control.last_report["finger"]["applied_offset_rad"] <= 0.001 + 1e-12


@pytest.mark.parametrize("normal", [25, float("nan"), -1])
def test_tangential_preload_cannot_exceed_native_task_guard(normal):
    script = object.__new__(FrontTangentialStockingScript)
    script.config = ScriptConfig()
    with pytest.raises(ValueError, match="unchanged abort guard"):
        script.push_front_tangentially(direction=1, distance_m=0.0015, normal_force_N=normal)


def test_out_of_range_nominal_is_rejected_before_reference_compensation():
    model, data = contact_fixture()
    control = FingerLoadCompensator(model, data, ["finger"], product_name="carton")
    before = data.ctrl.copy()
    with pytest.raises(ValueError, match="Nominal finger reference"):
        control.targets({"finger": 1.01}, dt=0.02)
    np.testing.assert_array_equal(data.ctrl, before)


@pytest.mark.mabel
def test_tangential_abduction_reference_matches_native_command_and_ik_geometry(monkeypatch):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=10, initially_stocked=9))
    try:
        hand = TangentialFingerPusher(
            env,
            ScriptConfig(),
            finger="Middle",
            tilt_degrees=10,
            wrist_posture=0.4,
            abduction_posture=0.0,
        )
        before_qpos, before_ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        captured = []

        def prediction(*args, **kwargs):
            captured.append(kwargs["finger_positions"])
            return None

        monkeypatch.setattr(hand.ik, "predict", prediction)
        hand.predict(hand.position(), alternatives=False, carrying=False)
        nominal = hand.finger_targets(0)
        adr = int(env.model.joint("Left Middle Abd").qposadr[0])
        assert nominal["Left Middle Abd"] == captured[-1][adr] == 0.0
        command = hand.joint_targets(hand.command, 0)
        assert command["Left Middle Abd"] == 0.0
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.data.ctrl, before_ctrl)
    finally:
        env.close()


@pytest.mark.mabel
def test_front_contact_offset_is_bounded_before_any_native_command(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=10, initially_stocked=9))
    try:
        script = FrontTangentialStockingScript(env, tmp_path)
        before_qpos, before_ctrl, before_time = (
            env.data.qpos.copy(),
            env.data.ctrl.copy(),
            env.data.time,
        )
        with pytest.raises(ValueError, match="bounded active face"):
            script.push_front_tangentially(
                direction=1,
                distance_m=0.0015,
                normal_force_N=3,
                force_guided=True,
                contact_offset_m=0.1,
            )
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        np.testing.assert_array_equal(env.data.ctrl, before_ctrl)
        assert env.data.time == before_time
    finally:
        env.close()
