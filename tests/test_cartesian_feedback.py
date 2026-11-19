"""A Cartesian correction coordinates rate-bounded nominal arm and lift motion."""

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.cartesian_feedback import (
    CartesianFeedbackConfig,
    bounded_cartesian_step,
    bounded_contact_prediction,
    box_damped_least_squares,
)
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import CartesianHand, ScriptConfig
from abstract_stack.scene import ShelfConfig


def test_box_solver_redistributes_motion_instead_of_clipping_an_unbounded_solution():
    matrix = np.array([[1.0, 1.0], [0.0, 0.0]])
    error = np.array([1.0, 0.0])
    lower, upper = np.zeros(2), np.array([0.2, 1.0])
    solution, report = box_damped_least_squares(matrix, error, lower, upper, damping=0.001)
    naive = np.clip(
        np.linalg.solve(matrix.T @ matrix + 1e-6 * np.eye(2), matrix.T @ error), lower, upper
    )
    assert np.linalg.norm(matrix @ solution - error) < 1e-4
    assert np.linalg.norm(matrix @ naive - error) > 0.25
    assert report["projected_gradient_inf"] < 1e-7
    assert np.all(solution >= lower) and np.all(solution <= upper)


def test_box_solver_matches_separable_ridge_solution_and_fixed_coordinates():
    matrix, error = np.eye(3), np.array([2.0, -0.5, 0.4])
    lower, upper = np.array([-1.0, -0.2, 0.3]), np.array([0.2, 1.0, 0.3])
    solution, _ = box_damped_least_squares(matrix, error, lower, upper, damping=0.1)
    np.testing.assert_allclose(solution, np.clip(error / 1.01, lower, upper), atol=1e-12)


@pytest.fixture
def native_hand():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=0, center_y=-1.5, table_center_y=-1.6))
    hand = CartesianHand(env, ScriptConfig())
    yield env, hand
    env.close()


@pytest.mark.mabel
def test_native_step_obeys_one_global_rate_box_and_never_mutates_live_state(native_hand):
    env, hand = native_hand
    previous = np.clip(hand.command.copy(), hand.lower, hand.upper)
    point, rotation = hand.ik.forward(previous)
    qpos, ctrl, qvel = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.qvel.copy()
    rates = np.r_[np.full(8, 0.5), 0.03]
    result = bounded_cartesian_step(
        hand.ik,
        previous,
        point + [0.2, 0.1, 0.1],
        rotation,
        dt=0.02,
        rates=rates,
        config=CartesianFeedbackConfig(linearizations=4),
    )
    assert np.all(np.abs(result.q - previous) <= rates * 0.02 + 1e-12)
    assert np.all(result.q >= hand.lower) and np.all(result.q <= hand.upper)
    independent_point, independent_rotation = hand.ik.forward(result.q)
    np.testing.assert_allclose(result.position, independent_point, atol=1e-12)
    np.testing.assert_allclose(result.rotation, independent_rotation, atol=1e-12)
    assert result.position_error_m > 0.1  # A bounded step does not imply reachability.
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.ctrl, ctrl)
    np.testing.assert_array_equal(env.data.qvel, qvel)


@pytest.mark.mabel
def test_posture_constraint_changes_are_ramped_and_goals_retained(native_hand):
    _, hand = native_hand
    previous = np.clip(hand.command.copy(), hand.lower, hand.upper)
    index = hand.ik.names.index("Left Wrist")
    previous[index] = 0.0
    point, rotation = hand.ik.forward(previous)
    hand.ik.constrain_posture("Left Wrist", 0.6)
    result = bounded_cartesian_step(hand.ik, previous, point, rotation, dt=0.02)
    assert result.q[index] == pytest.approx(0.01)
    assert hand.ik.posture_constraints[index] == 0.6
    assert result.ramped_postures["Left Wrist"]["ramped"] == pytest.approx(0.01)


@pytest.mark.mabel
def test_nominal_hold_returns_the_same_reference_without_accumulating_offsets(native_hand):
    _, hand = native_hand
    previous = np.clip(hand.command.copy(), hand.lower, hand.upper)
    point, rotation = hand.ik.forward(previous)
    result = bounded_cartesian_step(hand.ik, previous, point, rotation, dt=0.02)
    np.testing.assert_allclose(result.q, previous, atol=1e-12)
    assert result.position_error_m < 1e-12


@pytest.mark.mabel
def test_local_nominal_steps_make_progress_without_restarting_from_measured_pose(native_hand):
    env, hand = native_hand
    q = np.clip(hand.command.copy(), hand.lower, hand.upper)
    start, rotation = hand.ik.forward(q)
    goal = start + [0.005, 0, 0.003]
    original = env.data.qpos.copy()
    rates = np.r_[np.ones(8), 0.08]
    for _ in range(20):
        step = bounded_cartesian_step(hand.ik, q, goal, rotation, dt=0.02, rates=rates)
        assert np.all(np.abs(step.q - q) <= rates * 0.02 + 1e-12)
        q = step.q
    assert np.linalg.norm(hand.ik.forward(q)[0] - goal) < 1e-5
    np.testing.assert_array_equal(env.data.qpos, original)


@pytest.mark.mabel
def test_contact_wrapper_is_read_only_and_rejects_stale_baseline(native_hand):
    env, hand = native_hand
    hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
    hand.command = np.clip(hand.command, hand.lower, hand.upper)
    target = hand.ik.forward(hand.command)[0] + [0.0005, 0, 0]
    proposal = hand.predict(target, alternatives=False, check_path=False)
    qpos, ctrl, command = env.data.qpos.copy(), env.data.ctrl.copy(), hand.command.copy()
    result = bounded_contact_prediction(hand, proposal, target, env.dt, np.r_[np.ones(8), 0.08])
    assert result.feasible and result.path_checked
    assert result.allowed_contact_object is None
    assert result.bounded_feedback["allowed_contact_object"] is None
    assert result.joint_path is None and result.initial_cartesian_transition is None
    assert proposal is hand.last_prediction  # The utility leaves caller bookkeeping alone.
    np.testing.assert_array_equal(hand.command, command)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.ctrl, ctrl)
    # Explicit stale-cache metadata fixture; no physical state change.
    proposal.baseline_qpos = proposal.baseline_qpos.copy()
    proposal.baseline_qpos[0] += 0.001
    with pytest.raises(ValueError, match="fresh baseline"):
        bounded_contact_prediction(hand, proposal, target, env.dt, np.r_[np.ones(8), 0.08])


@pytest.mark.mabel
def test_final_cartesian_projection_keeps_the_original_speed_budget(native_hand):
    env, hand = native_hand
    hand.command = np.clip(hand.command, hand.lower, hand.upper)
    point, rotation = hand.ik.forward(hand.command)
    hand.active_rotation = rotation
    target = point + np.array([0.0005, 0, 0])
    proposal = hand.predict(target, alternatives=False, check_path=False)
    before = env.data.qpos.copy(), env.data.ctrl.copy()
    prediction = bounded_contact_prediction(
        hand,
        proposal,
        target,
        0.02,
        np.r_[np.ones(8), 0.08],
        max_translation_step_m=0.00002,
        max_rotation_step_rad=0.006,
    )
    assert prediction.feasible
    assert np.linalg.norm(prediction.position - point) <= 0.00002
    assert prediction.bounded_feedback["cartesian_motion_projection"]["fraction"] < 1
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])


@pytest.mark.mabel
def test_contact_wrapper_never_invents_permission_for_a_product():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(
        ShelfConfig(
            product_count=1, center_y=-1.5, table_center_y=-1.6, carton_size=(0.01, 0.01, 0.01)
        )
    )
    try:
        hand = CartesianHand(env, ScriptConfig())
        hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
        hand.command = np.clip(hand.command, hand.lower, hand.upper)
        q = hand.ik.measured_q()
        hand.ik.forward(q)
        before = hand.ik.scratch.geom_xpos.copy()
        future = env.data.qpos.copy()
        for name, value in hand.finger_targets(0).items():
            future[int(env.model.joint(name).qposadr[0])] = value
        hand.ik.forward(q, future)
        geoms = [
            g
            for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] in hand.ik.hand and env.model.geom_contype[g]
        ]
        geom = max(geoms, key=lambda g: np.linalg.norm(hand.ik.scratch.geom_xpos[g] - before[g]))
        assert np.linalg.norm(hand.ik.scratch.geom_xpos[geom] - before[geom]) > 0.01
        adr = int(env.model.joint("carton_0_joint").qposadr[0])
        # Scratch/reset rejection fixture at the FUTURE hand posture. It is
        # absent from the current-contact baseline and earns no task credit.
        env.data.qpos[adr : adr + 3] = hand.ik.scratch.geom_xpos[geom]
        mujoco.mj_forward(env.model, env.data)
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        target = hand.ik.forward(hand.command)[0]
        rates = np.r_[np.ones(8), 0.08]
        denied = hand.predict(target, alternatives=False, check_path=False)
        rejected = bounded_contact_prediction(hand, denied, target, env.dt, rates)
        assert not rejected.feasible
        assert rejected.allowed_contact_object is None
        assert any("carton_0_geom" in error["geoms"] for error in rejected.collisions)
        allowed = hand.predict(
            target, allowed_object="carton_0", alternatives=False, check_path=False
        )
        accepted = bounded_contact_prediction(hand, allowed, target, env.dt, rates)
        assert accepted.feasible
        assert accepted.allowed_contact_object == "carton_0"
        # A subsequent carrying forecast cannot silently change the old
        # proposal's contact mode while the physical qpos remains unchanged.
        hand.predict(
            target, allowed_object="carton_0", alternatives=False, carrying=True, check_path=False
        )
        with pytest.raises(ValueError, match="carry/slide context"):
            bounded_contact_prediction(hand, allowed, target, env.dt, rates)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()
