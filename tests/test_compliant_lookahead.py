"""Admittance uses force ON the robot, with bounded unloading motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.compliant_lookahead import PushAdmittance, generate_push_candidates


def test_contact_feedback_signs_unload_excess_normal_shear_and_downward_load():
    policy = PushAdmittance()
    forward = policy.velocity([1, 0, 0, 0, 0, 0])
    assert forward[0] < 0
    assert policy.velocity([4, 0, 0, 0, 0, 0])[0] > 0
    overloaded = policy.velocity([1, 1, 1, 0, 0, 0])
    assert overloaded[0] == 0  # Raise hand before pushing a downward-loaded carton.
    assert overloaded[1] > 0 and overloaded[2] > 0
    np.testing.assert_array_equal(policy.velocity([1, 0, 0, 0, 0, 0]), forward)


def test_all_feedback_corrections_are_bounded_after_large_inputs():
    policy = PushAdmittance()
    for sign in (-1, 1):
        velocity = policy.velocity(np.full(6, sign * 1e6), lateral_bias=sign * 1e6)
        assert -policy.forward_speed_m_s <= velocity[0] <= policy.reverse_speed_m_s
        assert abs(velocity[1]) <= policy.lateral_speed_m_s
        assert abs(velocity[2]) <= policy.vertical_speed_m_s


def test_invalid_sensor_samples_or_force_targets_do_not_create_commands():
    policy = PushAdmittance()
    for sample in ([0] * 3, [np.nan] * 6):
        with pytest.raises(ValueError, match="finite world wrench"):
            policy.velocity(sample)
    with pytest.raises(ValueError, match="force guard"):
        PushAdmittance(normal_target_N=25)


def test_forecast_rejects_horizon_that_rounds_to_a_different_duration():
    hand = SimpleNamespace(
        command=np.zeros(1),
        ik=SimpleNamespace(lower=np.array([-1]), upper=np.array([1])),
        env=SimpleNamespace(dt=0.02),
        model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
        data=None,
    )
    with pytest.raises(ValueError, match="complete control intervals"):
        generate_push_candidates(
            hand,
            "carton",
            nominal=np.zeros(1),
            wrench_on_hand_world=np.zeros(6),
            lower_xyz=np.full(3, -1),
            upper_xyz=np.ones(3),
            horizon_s=0.401,
        )
