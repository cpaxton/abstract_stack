"""Reaction torque, reference-frame bookkeeping and native tracker admission."""

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import CartesianHand, ScriptConfig
from abstract_stack.scene import ShelfConfig
from abstract_stack.withdrawal_feedback import (
    WithdrawalFeedbackConfig,
    WithdrawalTracker,
    compensation_translation,
    contact_effort_on_joint,
)


@pytest.mark.parametrize("product_first", [False, True])
def test_contact_torque_matches_native_generalized_constraint_force(product_first):
    robot = """<body name="robot" pos=".08 0 .2">
      <joint name="wrist" axis="0 0 1"/>
      <body name="hand" pos="0 .1 0">
        <geom name="hand_geom" type="sphere" size=".04" mass=".2" condim="6"/>
      </body></body>"""
    product = """<body name="product" pos=".02 .1 .2">
      <geom type="sphere" size=".04" condim="6"/></body>"""
    bodies = product + robot if product_first else robot + product
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody>' + bodies + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[:] = 0.4  # Reset-only force-sign fixture with tangential friction.
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    wrist = model.joint("wrist").id
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    torque = contact_effort_on_joint(
        model,
        data,
        wrist,
        {model.body("robot").id, model.body("hand").id},
        model.body("product").id,
    )
    assert torque < 0
    assert torque == pytest.approx(data.qfrc_constraint[0], rel=1e-9, abs=1e-9)
    assert contact_effort_on_joint(model, data, wrist, set(), model.body("product").id) == 0
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.ctrl, ctrl)


@pytest.mark.parametrize(
    "settings",
    [
        dict(servo_speed_m_s=0.1),
        dict(contact_effort_limit_Nm=0.31),
        dict(wrist_offset_limit_rad=np.nan),
    ],
)
def test_compensation_configuration_cannot_silently_expand_limits(settings):
    with pytest.raises(ValueError):
        WithdrawalFeedbackConfig(**settings)


@pytest.fixture
def native_hand():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    yield env, CartesianHand(env, ScriptConfig())
    env.close()


@pytest.mark.mabel
def test_tracker_holds_unknown_dimensions_and_uses_only_admitted_state(native_hand):
    env, hand = native_hand
    previous = np.clip(hand.command.copy(), hand.ik.lower, hand.ik.upper)
    # Robot body without product contact suffices for a no-load tracker probe.
    tracker = WithdrawalTracker(hand, previous, "mabel_chassis")
    initial = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    desired = tracker.propose(previous)
    ids = tracker.tracker.ids
    other = np.ones(env.model.nu, bool)
    other[ids] = False
    np.testing.assert_array_equal(desired[other], initial[2][other])
    assert tracker.last_report["contact_reaction_effort_Nm"] == 0
    # A rejected proposal must not become the next applied or nominal baseline.
    tracker.acknowledge(previous, initial[2])
    np.testing.assert_array_equal(tracker.tracker.applied, initial[2][ids])
    assert dict(zip(tracker.tracker.NAMES, tracker.tracker.nominal, strict=True)) == dict(
        zip((*hand.names, "Lift"), previous, strict=True)
    )
    for actual, before in zip((env.data.qpos, env.data.qvel, env.data.ctrl), initial, strict=True):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.mabel
def test_virtual_translation_tracks_bias_and_cannot_absorb_nominal_motion(native_hand):
    env, hand = native_hand
    entry = np.clip(hand.command.copy(), hand.ik.lower, hand.ik.upper)
    nominal = entry.copy()
    nominal[0] += 0.01
    bias = np.zeros_like(entry)
    bias[-1] = 0.003
    reference = env.data.qpos.copy()
    shift = compensation_translation(hand.ik, entry, nominal, nominal + bias, reference)
    expected = hand.ik.forward(entry + bias, reference)[0] - hand.ik.forward(entry, reference)[0]
    np.testing.assert_allclose(shift, expected, atol=1e-12)
    moved = nominal.copy()
    moved[0] += 0.01
    np.testing.assert_allclose(
        compensation_translation(hand.ik, entry, moved, moved + bias, reference),
        shift,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compensation_translation(hand.ik, entry, moved, moved, reference), np.zeros(3), atol=1e-12
    )
