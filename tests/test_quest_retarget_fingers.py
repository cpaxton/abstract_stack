"""Named-hand geometry, coupled proposals and native finger-response checks."""

import copy
import math

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.retarget_fingers import (
    FINGER_ACTUATORS,
    FingerMappingConfig,
    FingerRetargeter,
    hand_features,
)
from abstract_stack.teleop.retarget_fingers_fixtures import synthetic_hand
from abstract_stack.teleop.sim import MujocoTeleopAdapter


@pytest.fixture(scope="module")
def adapter():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("pinned MABEL assets unavailable")
    with MujocoTeleopAdapter(ShelfConfig(product_count=0, front_x=-1.4)) as sim:
        yield sim


@pytest.fixture
def mapper(adapter):
    state = adapter.reset()
    retargeter = FingerRetargeter(adapter._env.model)
    retargeter.reset(dict(zip(state.actuator_names, state.applied_targets, strict=True)))
    return retargeter


def test_anatomical_features_are_mirrored_and_rigid_motion_invariant():
    left = synthetic_hand("Left", 0.5, abduction_rad=0.1)
    right = synthetic_hand("Right", 0.5, abduction_rad=0.1)
    a = hand_features(left, "Left")
    b = hand_features(right, "Right")
    for name in a:
        assert a[name] == pytest.approx(b[name], abs=1e-12)
    angle = 0.8
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    moved = copy.deepcopy(left)
    for joint in moved["joints"]:
        joint["position_m"] = (rotation @ joint["position_m"] + [1, 2, 0.5]).tolist()
    transformed = hand_features(moved, "Left")
    for name in a:
        assert a[name] == pytest.approx(transformed[name], abs=1e-12)
    plain = hand_features(synthetic_hand("Left", 0.5), "Left")
    assert plain["Index MCP"] == pytest.approx(0.7)
    assert plain["Index PIP"] == pytest.approx(0.645)


@pytest.mark.parametrize("spread", [0.0, 0.12])
@pytest.mark.mabel
def test_index_flexion_through_right_angle_preserves_bilateral_abduction(mapper, spread):
    def hand(side, flexion):
        source = synthetic_hand(side, abduction_rad=spread)
        points = {joint["name"]: joint for joint in source["joints"]}
        pivot = np.asarray(points["Hand_Index1"]["position_m"])
        c, s = math.cos(flexion), math.sin(flexion)
        bend = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        for suffix in ("2", "3", "Tip"):
            joint = points[f"Hand_Index{suffix}"]
            joint["position_m"] = (pivot + bend @ (joint["position_m"] - pivot)).tolist()
        return source

    abduction = ("Left Index Abd", "Right Index Abd")
    baseline = mapper.propose(hand("Left", 0), hand("Right", 0), 0.02)
    assert baseline.valid, baseline.reason
    for degrees in (89.9, 90.0, 90.1):
        flexion = math.radians(degrees)
        left, right = hand("Left", flexion), hand("Right", flexion)
        for side, source in (("Left", left), ("Right", right)):
            features = hand_features(source, side)
            assert features["Index MCP"] == pytest.approx(flexion)
            assert features["Index Abd"] == pytest.approx(spread, abs=1e-12)
        proposed = mapper.propose(left, right, 0.02)
        assert proposed.valid, proposed.reason
        for channel in abduction:
            assert proposed.nominal[channel] == pytest.approx(baseline.nominal[channel])


@pytest.mark.mabel
def test_complete_ownership_nominal_side_identity_and_native_transmissions(mapper, adapter):
    result = mapper.propose(synthetic_hand("Left", 1), synthetic_hand("Right", 0), 0.02)
    assert result.valid and set(result.targets) == set(FINGER_ACTUATORS)
    assert len(result.targets) == 32
    assert not any("Wrist" in name or "Arm" in name for name in result.targets)
    assert result.nominal["Left Index MCP"] == pytest.approx(1.4)
    assert result.nominal["Right Index MCP"] == 0
    for name in result.targets:
        aid = adapter._env.model.actuator(name).id
        assert adapter._env.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
    assert mapper.describe()["transmission"].startswith("32 direct")


@pytest.mark.mabel
def test_gradual_closure_and_pinch_produce_independent_continuous_nominals(mapper):
    previous = -math.inf
    for amount in np.linspace(0, 1, 31):
        result = mapper.propose(synthetic_hand("Left", amount), synthetic_hand("Right", 0), 0.02)
        assert result.valid
        assert result.nominal["Left Index MCP"] >= previous
        previous = result.nominal["Left Index MCP"]
        assert result.nominal["Right Index MCP"] == 0
    pinch = mapper.propose(
        synthetic_hand("Left", pinch=True), synthetic_hand("Right", pinch=True), 0.02
    )
    assert pinch.valid
    assert pinch.nominal["Left Index MCP"] > 1.0
    assert pinch.nominal["Left Middle MCP"] == 0
    assert pinch.nominal["Left Thumb Abd"] > -0.73304


@pytest.mark.mabel
def test_filter_bounds_velocity_acceleration_and_native_ranges_during_reversals(mapper):
    previous = mapper._previous.copy()
    velocity = np.zeros(32)
    for i in range(450):
        amount = 1 if i < 180 or i >= 350 else 0
        result = mapper.propose(
            synthetic_hand("Left", amount), synthetic_hand("Right", 1 - amount), 0.02
        )
        assert result.valid, result.reason
        applied = np.array([result.targets[n] for n in mapper.names])
        measured_velocity = (applied - previous) / 0.02
        measured_acceleration = (measured_velocity - velocity) / 0.02
        assert np.max(np.abs(measured_velocity)) <= 2 + 1e-10
        assert np.max(np.abs(measured_acceleration)) <= 8 + 1e-8
        assert np.all(applied >= mapper.limits[:, 0]) and np.all(applied <= mapper.limits[:, 1])
        previous, velocity = applied, measured_velocity


@pytest.mark.mabel
def test_native_bound_braking_and_model_state_are_not_modified(mapper, adapter):
    model, data = adapter._env.model, adapter._env.data
    before = {
        key: getattr(model, key).copy()
        for key in (
            "actuator_ctrlrange",
            "actuator_gainprm",
            "actuator_gear",
            "jnt_range",
            "body_mass",
            "body_inertia",
            "geom_contype",
            "geom_conaffinity",
            "eq_data",
        )
    }
    qpos, ctrl = data.qpos.copy(), data.ctrl.copy()
    mapper.reset(dict(zip(mapper.names, mapper.limits[:, 1], strict=True)))
    for _ in range(80):
        result = mapper.propose(synthetic_hand("Left", 0), synthetic_hand("Right", 0), 0.02)
        assert result.valid, result.reason
    np.testing.assert_array_equal(qpos, data.qpos)
    np.testing.assert_array_equal(ctrl, data.ctrl)
    for key, value in before.items():
        np.testing.assert_array_equal(value, getattr(model, key))


@pytest.mark.mabel
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "invalid",
        "nan",
        "bool",
        "profile",
        "root",
        "root_null",
        "degenerate",
    ],
)
def test_bad_hand_rejects_whole_pair_and_requires_explicit_reset(mapper, fault):
    left, right = synthetic_hand("Left", 0.5), synthetic_hand("Right", 0.5)
    if fault == "missing":
        right["joints"].pop()
    elif fault == "duplicate":
        right["joints"][-1] = copy.deepcopy(right["joints"][-2])
    elif fault == "invalid":
        right["joints"][-1]["position_valid"] = False
    elif fault == "nan":
        right["joints"][-1]["position_m"] = [0, np.nan, 0]
    elif fault == "bool":
        right["joints"][-1]["position_m"] = [True, False, True]
    elif fault == "profile":
        right["skeleton_profile"] = "openxr_hand_26"
    elif fault == "root":
        right["pose"]["orientation_valid"] = False
    elif fault == "root_null":
        right["pose"] = None
    else:
        for joint in right["joints"]:
            joint["position_m"] = [0, 0, 0]
    result = mapper.propose(left, right, 0.02)
    assert not result.valid and not result.targets and not result.nominal
    recovered = mapper.propose(synthetic_hand("Left"), synthetic_hand("Right"), 0.02)
    assert not recovered.valid and "reset" in recovered.reason


@pytest.mark.mabel
@pytest.mark.parametrize("dt", [0, -0.01, 0.1, float("nan"), True])
def test_time_gap_or_bad_time_fails_closed(mapper, dt):
    assert not mapper.propose(synthetic_hand("Left"), synthetic_hand("Right"), dt).valid


@pytest.mark.mabel
def test_reset_rejects_partial_nonfinite_or_outside_native_references(mapper):
    with pytest.raises(ValueError):
        mapper.reset({"Left Index MCP": 0})
    assert not mapper.propose(synthetic_hand("Left"), synthetic_hand("Right"), 0.02).valid
    values = dict(zip(mapper.names, mapper.opened, strict=True))
    values["Left Index MCP"] = 99
    with pytest.raises(ValueError):
        mapper.reset(values)
    values["Left Index MCP"] = True
    with pytest.raises(ValueError):
        mapper.reset(values)


@pytest.mark.mabel
def test_native_both_hand_response_is_measured_separately_from_targets(adapter):
    state = adapter.reset()
    mapper = FingerRetargeter(adapter._env.model)
    mapper.reset(dict(zip(state.actuator_names, state.applied_targets, strict=True)))
    ids = [state.actuator_names.index(n) for n in mapper.names]
    initial = state.actuator_lengths[ids].copy()
    saw_tracking_error = False
    for i in range(130):
        proposal = mapper.propose(
            synthetic_hand("Left", min(1, i / 70)), synthetic_hand("Right", min(1, i / 70)), 0.02
        )
        assert proposal.valid
        command = state.applied_targets.copy()
        command[ids] = [proposal.targets[n] for n in mapper.names]
        adapter.apply_actuator_targets(command)
        state = adapter.step()
        saw_tracking_error |= np.max(np.abs(state.actuator_lengths[ids] - command[ids])) > 0.01
    for name in ("Left Index MCP", "Right Index MCP", "Left Middle PIP", "Right Middle PIP"):
        i = mapper.names.index(name)
        assert state.actuator_lengths[ids[i]] - initial[i] > 0.1
    assert saw_tracking_error
    assert np.isfinite(state.actuator_forces).all()
    assert all(
        state.applied_targets[state.actuator_names.index(n)] == 0
        for n in ("FL Drive", "FR Drive", "B Drive")
    )


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        FingerMappingConfig(max_velocity_rad_s=float("nan"))
    with pytest.raises(ValueError):
        FingerMappingConfig(filter_tau_s=0)


@pytest.mark.mabel
def test_default_shelf_contact_blocks_tracking_without_hidden_pose_corrections(adapter):
    # Keep this native counterexample: the default shelf is close enough to
    # obstruct the left index. Bounded commands do not prove feasible closure.
    with MujocoTeleopAdapter(ShelfConfig(product_count=0)) as sim:
        model, data = sim._env.model, sim._env.data
        state = sim.read_state()
        mapper = FingerRetargeter(model)
        mapper.reset(dict(zip(state.actuator_names, state.applied_targets, strict=True)))
        maximum = 0.0
        for i in range(130):
            result = mapper.propose(
                synthetic_hand("Left", min(1, i / 70)),
                synthetic_hand("Right", min(1, i / 70)),
                0.02,
            )
            assert result.valid
            command = state.applied_targets.copy()
            for name, value in result.targets.items():
                command[model.actuator(name).id] = value
            sim.apply_actuator_targets(command)
            state = sim.step()
            for contact_id, contact in enumerate(data.contact):
                names = [
                    model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)
                ]
                if "shelf" in names and any(n.startswith("left_index_") for n in names):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_id, force)
                    maximum = max(maximum, float(force[0]))
        index = model.actuator("Left Index MCP").id
        assert maximum > 0
        assert abs(state.applied_targets[index] - state.actuator_lengths[index]) > 0.1
        assert np.isfinite(state.qpos).all()


def test_gate_feedback_uses_applied_instead_of_accumulating_unexecuted_proposals(mapper):
    from abstract_stack.teleop.retarget_fingers_fixtures import synthetic_hand

    actual = dict(zip(mapper.names, mapper.opened, strict=True))
    mapper.reset(actual)
    closed = synthetic_hand("Left", 1), synthetic_hand("Right", 1)
    prior_actual = np.array([actual[n] for n in mapper.names])
    for _ in range(15):
        proposal = mapper.propose(*closed, 0.02, applied_references=actual)
        assert proposal.valid, proposal.reason
        # A stricter outer gate applies only a small bounded part of the request.
        vector = np.array([proposal.targets[n] for n in mapper.names])
        applied = prior_actual + np.clip(vector - prior_actual, -0.0004, 0.0004)
        actual = dict(zip(mapper.names, applied, strict=True))
        prior_actual = applied
        assert np.max(np.abs(mapper._previous - prior_actual)) < 0.004
    bad = dict(actual)
    bad["Left Index MCP"] += 0.1
    rejected = mapper.propose(*closed, 0.02, applied_references=bad)
    assert not rejected.valid and not rejected.targets
