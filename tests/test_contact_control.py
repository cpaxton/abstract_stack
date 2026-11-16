"""Fixture feedback must include wrist mounts and exclude the other arm's loads."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.interactions import ScriptConfig, contact_forces


@pytest.mark.parametrize("fixture_first", [True, False])
def test_fixture_feedback_direction_and_controlled_link_filter(fixture_first):
    fixture = """
      <body name="shelf"><geom name="shelf_board_0" type="box"
          pos="0 0 .9" size="1 1 .05"/>
        <geom name="shelf_back" type="box" pos="-2 0 1" size=".01 1 1"/>
      </body>
      <body name="supply" pos="3 0 0"><geom type="box" size=".1 .1 .1"/></body>
      <body name="carton" pos="4 0 1"><geom type="box" size=".1 .1 .1"/></body>
    """
    links = """
      <body name="left_tower" pos="-.5 0 1.01"><freejoint/>
        <geom type="box" size=".06 .06 .06" mass="2"/></body>
      <body name="right_thumb" pos=".5 0 1.01"><freejoint/>
        <geom type="box" size=".06 .06 .06" mass="1"/></body>
    """
    world = fixture + links if fixture_first else links + fixture
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><option timestep=".002"/><worldbody>{world}</worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    tower, thumb = model.body("left_tower").id, model.body("right_thumb").id
    env = SimpleNamespace(model=model, data=data)
    force = contact_forces(env, "carton", {tower, thumb}, fixture_vector_bodies={tower})
    assert force["arm_fixture"] == pytest.approx(3 * 9.81, abs=0.01)
    np.testing.assert_allclose(force["fixture_normal_world_N"], [0, 0, 2 * 9.81], atol=0.01)
    right = contact_forces(env, "carton", {tower, thumb}, fixture_vector_bodies={thumb})
    np.testing.assert_allclose(right["fixture_normal_world_N"], [0, 0, 9.81], atol=0.01)


def test_soft_contact_targets_cannot_exceed_hard_abort_limit():
    with pytest.raises(ValueError, match="force targets"):
        ScriptConfig(fixture_force_target=26)
    with pytest.raises(ValueError, match="force targets"):
        ScriptConfig(approach_force_target=26)
    with pytest.raises(ValueError, match="contact_aware"):
        ScriptConfig(contact_aware=1)
