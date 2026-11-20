"""Payload reference headroom and wrench signs preserve native actuation contracts."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.cartesian_feedback import bounded_contact_prediction
from abstract_stack.dense_interactions import DenseStockingScript
from abstract_stack.interactions import InteractionFailure, contact_forces
from abstract_stack.loaded_route import planning_hand
from abstract_stack.mabel_controller import MabelController
from abstract_stack.receding_stocking import RecedingStockingScript


@pytest.fixture
def payload_script(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    script = RecedingStockingScript(env, tmp_path, payload_load_compensation=True)
    try:
        yield env, script
    finally:
        env.close()


@pytest.mark.mabel
def test_reference_reserve_is_a_copied_planning_range_not_a_model_change(payload_script):
    env, script = payload_script
    hand = script.controller
    wrist = hand.ik.names.index("Left Wrist")
    ranges, controls = env.model.jnt_range.copy(), env.model.actuator_ctrlrange.copy()
    before = env.data.qpos.copy(), env.data.ctrl.copy()
    assert hand.ik.lower[wrist] == pytest.approx(hand.ik.original_lower[wrist] + 0.4)
    assert hand.ik.upper[wrist] == pytest.approx(hand.ik.original_upper[wrist] - 0.4)
    clone = planning_hand(env.clone_for_planning(), hand, script.config)
    assert clone.ik.reference_range_report() == hand.ik.reference_range_report()
    clone.ik.constrain_reference_range("Left Wrist", -0.2, 0.2)
    assert hand.ik.upper[wrist] > 0.2
    hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
    prediction = hand.predict(hand.position(), alternatives=False, check_path=False)
    assert hand.ik.lower[wrist] <= prediction.q[wrist] <= hand.ik.upper[wrist]
    assert prediction.report()["planning_reference_ranges"] == hand.ik.reference_range_report()
    with pytest.raises(ValueError, match="original bounds"):
        hand.ik.constrain_reference_range("Left Wrist", -2, 0.3)
    hand.ik.constrain_posture("Left Wrist", 0.2)
    with pytest.raises(ValueError, match="contain its posture"):
        hand.ik.constrain_reference_range("Left Wrist", -0.1, 0.1)
    np.testing.assert_array_equal(env.model.jnt_range, ranges)
    np.testing.assert_array_equal(env.model.actuator_ctrlrange, controls)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])


@pytest.mark.mabel
def test_bounded_feedback_rejects_a_cached_wider_reference_envelope(payload_script):
    env, script = payload_script
    hand = script.controller
    hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
    proposal = hand.predict(hand.position(), alternatives=False, check_path=False)
    hand.ik.constrain_reference_range("Left Wrist", -0.3, 0.3)
    with pytest.raises(ValueError, match="stale reference-range"):
        bounded_contact_prediction(hand, proposal, hand.position(), env.dt, np.r_[np.ones(8), 0.08])
    assert hand.staging_pose(proposal) is None


def test_carton_wrench_holding_sign_matches_native_constraint_effort():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".001"/>
        <worldbody>
          <body name="finger_body" pos="0 0 .4">
            <joint name="Left Wrist" axis="0 1 0" range="-1 1" damping=".1"/>
            <geom type="capsule" fromto="0 0 0 .3 0 .1" size=".03" mass=".1"/>
          </body>
          <body name="carton" pos=".35 0 .5"><geom type="box" size=".04 .1 .1"/></body>
          <body name="supply" pos="10 0 0"><geom size=".01"/></body>
          <body name="shelf" pos="20 0 0"><geom name="shelf_back" size=".01"/></body>
        </worldbody>
        <actuator><position name="Left Wrist" joint="Left Wrist" kp="10"/>
        </actuator>
      </mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(model=model, data=data, dt=1.0)
    script = object.__new__(DenseStockingScript)
    script.env, script.product = env, SimpleNamespace(name="carton")
    script.last_forces = contact_forces(env, "carton", {model.body("finger_body").id})
    script.wrist_load = script.wrist_load_peak = script.wrist_load_requested_peak = 0.0
    script.wrist_load_last_requested = script.wrist_load_clipped = 0
    before = data.qpos.copy(), data.ctrl.copy()
    result = script._wrist_contact_load()
    assert abs(data.qfrc_constraint[0]) > 0.01
    assert script.wrist_load_last_requested == pytest.approx(-data.qfrc_constraint[0], abs=1e-8)
    assert result["Left Wrist"] == pytest.approx(np.clip(-data.qfrc_constraint[0], -0.3, 0.3))
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.ctrl, before[1])


@pytest.mark.mabel
def test_actual_compensated_arm_and_all_fingers_are_guarded_before_native_step(
    payload_script, monkeypatch
):
    env, script = payload_script
    script.tracker = MabelController(env.model, env.data, script._tracking_config())
    script.active_grip = 0.4
    script._stage("lift")
    captured = []

    def reject(prediction, grip, allowed_object, *, targets):
        captured.append((prediction.q.copy(), targets.copy(), allowed_object))
        return [{"geoms": ["unit_fixture", "left_palm"], "distance_m": -0.003}]

    monkeypatch.setattr(script.controller, "check_hand_step", reject)
    monkeypatch.setattr(env, "step", lambda *a, **k: pytest.fail("Rejected reference was actuated"))
    before = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time
    with pytest.raises(InteractionFailure, match="compensated arm/finger geometry rejected"):
        script._apply_tick({})
    hand = script.controller
    applied = [script.tracker.last_report["servo_target"][n] for n in [*hand.names, "Lift"]]
    nominal = [script.tracker.last_report["nominal"][n] for n in [*hand.names, "Lift"]]
    np.testing.assert_allclose(captured[0][0], applied)
    assert not np.allclose(applied, nominal)
    assert "Left Middle Abd" in captured[0][1]
    assert captured[0][2] == script.product.name
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])
    assert env.data.time == before[2]
