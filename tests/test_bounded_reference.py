"""Projection preserves hard bounds and exposes infeasible bounded searches."""

import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.bounded_reference import ReferenceEnvelope, project_halfspaces
from abstract_stack.cartesian_feedback import bounded_cartesian_step, bounded_contact_prediction
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import CartesianHand, ScriptConfig
from abstract_stack.scene import ShelfConfig


def test_projection_redirects_motion_along_boundary_instead_of_holding():
    # Closest point to (1, 1) with x+y <= 1 and x <= .2 is (.2, .8).
    result, report = project_halfspaces([1, 1], [0, 0], [0.2, 2], [[1, 1]], [1])
    np.testing.assert_allclose(result, [0.2, 0.8], atol=1e-9)
    assert report["maximum_violation"] < 1e-10


def test_projection_reports_infeasibility_without_relaxing_constraints():
    _, report = project_halfspaces([0], [0], [1], [[1]], [-1])
    assert report["maximum_violation"] >= 1
    assert report["iterations"] == 128


@pytest.mark.parametrize("bounds", [[np.nan], [np.inf]])
def test_projection_rejects_nonfinite_constraints(bounds):
    with pytest.raises(ValueError):
        project_halfspaces([0], [-1], [1], [[1]], bounds)


def test_reference_bounds_are_captured_without_aliases():
    lower, upper = np.zeros(3), np.ones(3)
    envelope = ReferenceEnvelope(lower, upper, lower, upper, [0, 0])
    lower[:] = -1
    np.testing.assert_array_equal(envelope.nominal_lower, np.zeros(3))
    with pytest.raises(ValueError):
        envelope.previous_applied[0] = 1


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
def test_paired_projection_keeps_compensated_fk_inside_its_own_corridor(native_hand):
    env, hand = native_hand
    ik = hand.ik
    previous = np.clip(hand.command.copy(), ik.lower, ik.upper)
    applied = previous.copy()
    applied[-1] += 0.01
    point, rotation = ik.forward(previous)
    applied_point, _ = ik.forward(applied)
    before = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    step = bounded_cartesian_step(ik, previous, point + [0.0003, 0, -0.0003], rotation, dt=0.02)
    rates = np.r_[np.ones(len(previous) - 1), 0.08]
    applied_lower = applied_point - 0.02
    applied_lower[2] = applied_point[2] - 1e-7
    envelope = ReferenceEnvelope(
        point - 0.02, point + 0.02, applied_lower, applied_point + 0.02, applied
    )
    q, report = envelope.project(
        ik,
        previous,
        step.q,
        rates=rates,
        dt=0.02,
        reference=env.data.qpos.copy(),
        postures=step.ramped_postures,
    )
    assert report["feasible"], report
    assert report["correction_joint_norm"] > 1e-6
    assert np.linalg.norm(q - previous) > 1e-6
    assert np.all(np.abs(q - previous) <= rates * 0.02 + 1e-12)
    assert np.all(q >= ik.lower) and np.all(q <= ik.upper)
    projected_applied = applied + q - previous
    assert np.all(projected_applied >= ik.original_lower)
    assert np.all(projected_applied <= ik.original_upper)
    assert ik.forward(projected_applied)[0][2] >= applied_lower[2]
    np.testing.assert_allclose(projected_applied - q, applied - previous, atol=1e-14)
    for actual, original in zip((env.data.qpos, env.data.qvel, env.data.ctrl), before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.mabel
def test_ramped_posture_is_preserved_and_impossible_projection_rejected(native_hand):
    env, hand = native_hand
    ik = hand.ik
    previous = np.clip(hand.command.copy(), ik.lower, ik.upper)
    index = ik.names.index("Left Wrist")
    previous[index] = 0
    hand.command = previous.copy()
    ik.constrain_posture("Left Wrist", 0.6)
    point, rotation = ik.forward(previous)
    step = bounded_cartesian_step(ik, previous, point, rotation, dt=0.02)
    rates = np.r_[np.ones(len(previous) - 1), 0.08]
    envelope = ReferenceEnvelope(point - 0.1, point + 0.1, point - 0.1, point + 0.1, previous)
    q, report = envelope.project(
        ik,
        previous,
        step.q,
        rates=rates,
        dt=0.02,
        reference=env.data.qpos.copy(),
        postures=step.ramped_postures,
    )
    assert report["feasible"]
    assert q[index] == pytest.approx(0.01, abs=1e-12)
    assert ik.posture_constraints[index] == 0.6
    hand.active_rotation = rotation
    proposal = hand.predict(point, alternatives=False, check_path=False)
    impossible = ReferenceEnvelope(point + 1, point + 2, point - 0.1, point + 0.1, previous)
    result = bounded_contact_prediction(
        hand, proposal, point, 0.02, rates, reference_envelope=impossible
    )
    assert not result.feasible
    assert not result.bounded_feedback["paired_reference_projection"]["feasible"]
