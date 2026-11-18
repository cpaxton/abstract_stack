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


@pytest.mark.mabel
def test_finger_forecasts_validate_ownership_native_limits_and_rate_without_mutation():
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import CartesianHand, ScriptConfig

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        hand = CartesianHand(env, ScriptConfig())
        nominal = np.clip(hand.command, hand.ik.lower, hand.ik.upper)
        hand.command = nominal.copy()  # Planning-controller fixture; native state stays unchanged.
        point = hand.ik.forward(nominal)[0]
        before = [env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()]
        name = "Left Index MCP"
        aid = env.model.actuator(name).id
        value = float(env.data.ctrl[aid])
        options = dict(
            nominal=nominal,
            wrench_on_hand_world=np.zeros(6),
            lower_xyz=point - 0.1,
            upper_xyz=point + 0.1,
            horizon_s=0.2,
            max_candidates=1,
        )
        _, report, rates = generate_push_candidates(
            hand, env.products[0].name, finger_control_targets={name: value + 0.001}, **options
        )
        assert rates[aid] == 0.5
        assert report["selected_finger_control_targets"][name] == value + 0.001
        for invalid in (
            {"Left Arm 1": 0.0},
            {"Right Index MCP": 0.0},
            {name: value + 0.02},
            {name: 99.0},
            {name: True},
        ):
            with pytest.raises(ValueError, match="Finger"):
                generate_push_candidates(
                    hand, env.products[0].name, finger_control_targets=invalid, **options
                )
        for actual, original in zip(
            (env.data.qpos, env.data.qvel, env.data.ctrl), before, strict=True
        ):
            np.testing.assert_array_equal(actual, original)
    finally:
        env.close()
