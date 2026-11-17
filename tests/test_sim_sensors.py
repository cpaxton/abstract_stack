"""Sensor wrench signs/frames match native F/T; rigid proxies are explicit."""

import mujoco
import numpy as np
import pytest

from abstract_stack.contact_rollout import integration_state
from abstract_stack.sim_sensors import ContactSensorSuite, WristFrame


def model_for(*, angle=0, contact=False):
    quaternion = f"{np.cos(angle / 2)} 0 0 {np.sin(angle / 2)}"
    floor = ".881" if contact else "0"
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002"/>
      <worldbody><geom name="floor" type="plane" pos="0 0 {floor}" size="2 2 .1"/>
        <body name="palm" pos="0 0 1" quat="{quaternion}">
          <joint name="wrist" type="hinge" axis="0 0 1" pos=".02 0 0"/>
          <site name="ft" pos=".02 0 0"/>
          <geom name="pad" type="box" pos=".1 0 -.1" size=".02 .02 .02" mass=".2"/>
          <body name="finger" pos=".1 .1 0">
            <joint name="finger_joint" type="hinge" axis="0 0 1"/>
            <geom name="finger_pad" type="sphere" size=".01" mass=".1"/>
          </body>
        </body>
      </worldbody>
      <sensor><force name="native_force" site="ft"/>
        <torque name="native_torque" site="ft"/></sensor>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("angle", [0, np.pi / 2])
@pytest.mark.parametrize("contact", [False, True])
def test_virtual_reaction_matches_native_force_and_torque_and_is_read_only(angle, contact):
    model, data = model_for(angle=angle, contact=contact)
    suite = ContactSensorSuite(model, [WristFrame("left", "wrist")])
    before = integration_state(model, data)
    derived = [v.copy() for v in (data.cfrc_int, data.qacc, data.sensordata)]
    result = suite.sample(data)
    expected = np.r_[data.sensor("native_force").data, data.sensor("native_torque").data]
    np.testing.assert_allclose(result["wrist_wrench"][0], expected, atol=1e-10)
    np.testing.assert_array_equal(integration_state(model, data), before)
    for original, current in zip(derived, (data.cfrc_int, data.qacc, data.sensordata), strict=True):
        np.testing.assert_array_equal(current, original)
    assert result["timestamp_s"] == data.time
    assert result["wrist_available"].tolist() == [True]
    assert result["tactile_available"].all()
    if contact:
        assert result["contact_wrench"][0, 2] > 0
        assert result["contact_wrench"][0, 4] < 0
        assert np.linalg.norm(result["tactile"]) > 0
    else:
        np.testing.assert_array_equal(result["contact_wrench"], 0)
        np.testing.assert_array_equal(result["tactile"], 0)
        assert np.linalg.norm(result["wrist_wrench"]) > 0  # Gravity is not contact.


def test_taxel_and_contact_force_conserve_the_world_contact_resultant():
    model, data = model_for(angle=np.pi / 2, contact=True)
    suite = ContactSensorSuite(model, [WristFrame("left", "wrist")])
    result = suite.sample(data)
    force = sum(
        (data.geom_xmat[g].reshape(3, 3) @ result["tactile"][i] for i, g in enumerate(suite.geoms)),
        start=np.zeros(3),
    )
    wrist_body = suite.bodies[0]
    expected = data.xmat[wrist_body].reshape(3, 3) @ result["contact_wrench"][0, :3]
    np.testing.assert_allclose(force, expected, atol=1e-10)
    meta = suite.metadata()
    assert "coarse" in meta["tactile_provenance"]
    assert len(meta["tactile_order"]) == len(result["tactile"])


def test_sensor_configuration_rejects_overlapping_wrists_and_unrelated_geoms():
    model, _ = model_for()
    with pytest.raises(ValueError, match="overlap"):
        ContactSensorSuite(model, [WristFrame("a", "wrist"), WristFrame("b", "finger_joint")])
    with pytest.raises(ValueError, match="collision geoms"):
        ContactSensorSuite(model, [WristFrame("left", "wrist")], taxel_geoms=["floor"])


def test_disabled_native_sensor_flag_does_not_fabricate_zero_virtual_reaction():
    model, data = model_for()
    expected = ContactSensorSuite(model, [WristFrame("left", "wrist")]).sample(data)["wrist_wrench"]
    # Explicit fixture-only sensor flag, as used by the default stocking scene.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    mujoco.mj_forward(model, data)
    sample = ContactSensorSuite(model, [WristFrame("left", "wrist")]).sample(data)
    np.testing.assert_allclose(sample["wrist_wrench"], expected, atol=1e-10)


def test_internal_hand_contact_cancels_at_wrist_but_remains_in_tactile_proxies():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option><flag filterparent="disable"/></option>
      <worldbody><body name="palm" pos="0 0 1">
        <joint name="wrist" type="hinge" axis="0 0 1"/>
        <geom name="palm_pad" type="sphere" size=".04" mass=".1"/>
        <body name="finger" pos=".05 0 0"><joint name="finger" type="hinge"/>
          <geom name="finger_pad" type="sphere" size=".04" mass=".1"/>
        </body>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    suite = ContactSensorSuite(model, [WristFrame("left", "wrist")])
    sample = suite.sample(data)
    assert data.ncon > 0 and np.linalg.norm(sample["tactile"]) > 0
    np.testing.assert_array_equal(sample["contact_wrench"], 0)
    world = sum(
        (data.geom_xmat[g].reshape(3, 3) @ sample["tactile"][i] for i, g in enumerate(suite.geoms)),
        start=np.zeros(3),
    )
    np.testing.assert_allclose(world, 0, atol=1e-10)


@pytest.mark.mabel
def test_native_mabel_both_wrist_channels_and_collision_proxies():
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=2))
    try:
        initial = integration_state(env.model, env.data)
        suite = ContactSensorSuite.mabel(env.model)
        result = suite.sample(env.data)
        assert result["wrist_wrench"].shape == (2, 6)
        assert result["contact_wrench"].shape == (2, 6)
        assert result["tactile"].shape == (len(suite.geoms), 3)
        assert len(suite.geoms) > 0
        np.testing.assert_array_equal(integration_state(env.model, env.data), initial)
    finally:
        env.close()
