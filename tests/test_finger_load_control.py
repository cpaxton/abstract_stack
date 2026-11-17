"""Native reference compensation keeps load signs, bounds and geometry responsibility explicit."""

import mujoco
import numpy as np
import pytest

from abstract_stack.finger_load_control import FingerLoadCompensator
from abstract_stack.front_tangential import FrontTangentialStockingScript
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
