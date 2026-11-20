"""Wrapper lifetime retention must leave native model/state semantics intact."""

import gc

import mujoco
import numpy as np

from abstract_stack.native_views import retain_data_views, retain_model_views


def test_retained_views_survive_collection_and_preserve_raw_arrays():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body name="moving"><joint name="hinge"/>
      <geom name="ball" size=".02" pos=".1 0 0"/>
      </body></worldbody><actuator><position name="drive" joint="hinge" kp="4"/>
      </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = {
        key: getattr(model, key).copy()
        for key in ("body_pos", "geom_size", "geom_contype", "actuator_gainprm", "jnt_range")
    }
    control, pose = data.ctrl.copy(), data.qpos.copy()
    retained = retain_model_views(model)
    retained_data = retain_data_views(model, data)
    for _ in range(10):
        gc.collect()
        assert retained["body"][1] is model.body("moving")
        assert retained["joint"][0] is model.joint("hinge")
        assert retained["geom"][0] is model.geom("ball")
        assert retained["actuator"][0] is model.actuator("drive")
        assert retained_data["body"][1] is data.body("moving")
        assert retained_data["joint"][0] is data.joint("hinge")
        assert retained_data["actuator"][0] is data.actuator("drive")
        np.testing.assert_array_equal(data.body("moving").xpos, data.xpos[1])
    for key, array in before.items():
        np.testing.assert_array_equal(getattr(model, key), array)
    np.testing.assert_array_equal(data.qpos, pose)
    np.testing.assert_array_equal(data.ctrl, control)
