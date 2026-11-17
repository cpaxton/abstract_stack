"""Fixture feedback must include wrist mounts and exclude the other arm's loads."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import CartesianHand, ScriptConfig, contact_forces


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


@pytest.mark.mabel
def test_coupled_closing_step_checks_future_fingers_without_live_state_changes():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    # A tiny dynamic carton is a collision-check fixture, not a stocking rollout.
    env = ShelfStockingEnv(
        replace(load_scene_config(scene="pick_place"), carton_size=(0.01, 0.01, 0.01))
    )
    try:
        hand = CartesianHand(env, ScriptConfig())
        q = hand.ik.measured_q()
        reference = env.data.qpos.copy()
        hand.ik.forward(q, reference)
        before = hand.ik.scratch.geom_xpos.copy()
        future = reference.copy()
        for suffix in hand.FLEXION:
            name = f"Left {suffix}"
            lo, hi = env.model.actuator(name).ctrlrange
            future[int(env.model.joint(name).qposadr[0])] = lo + 0.2 * (hi - lo)
        hand.ik.forward(q, future)
        geoms = [
            g
            for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] in hand.ik.hand and env.model.geom_contype[g]
        ]
        geom = max(geoms, key=lambda g: np.linalg.norm(hand.ik.scratch.geom_xpos[g] - before[g]))
        assert np.linalg.norm(hand.ik.scratch.geom_xpos[geom] - before[geom]) > 0.01
        adr = int(env.model.joint("carton_1_joint").qposadr[0])
        env.data.qpos[adr : adr + 3] = hand.ik.scratch.geom_xpos[geom]
        mujoco.mj_forward(env.model, env.data)
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        errors = hand.check_hand_step(SimpleNamespace(q=q), 0.4, "carton_0")
        assert errors
        assert any("carton_1_geom" in e["geoms"] for e in errors)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()
