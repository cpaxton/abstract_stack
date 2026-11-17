"""Private-model pose proposals, native limits/collisions and execution evidence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.calibration import Pose
from abstract_stack.teleop.retarget_arms import (
    ALL_NAMES,
    ARM_NAMES,
    HEAD_NAMES,
    ArmHeadRetargeter,
    ArmHeadSettings,
)
from abstract_stack.teleop.sim import MujocoTeleopAdapter

pytestmark = pytest.mark.mabel
EMPTY_SPACE = ShelfConfig(product_count=0, front_x=-1.4, terminate_on_success=False)


@pytest.fixture(scope="module")
def fixture():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("pinned MABEL assets unavailable")
    with MujocoTeleopAdapter(EMPTY_SPACE) as sim, ArmHeadRetargeter(EMPTY_SPACE) as retargeter:
        yield sim, retargeter


@pytest.fixture
def stack(fixture):
    sim, retargeter = fixture
    state = sim.reset(seed=0)
    retargeter.reset(state)
    return sim, retargeter, state, retargeter.measured_poses(state)


def full_targets(state, partial=None):
    result = dict(zip(state.actuator_names, state.applied_targets, strict=True))
    result.update(partial or {})
    return result


def oriented(pose, axis, radians):
    quaternion, matrix = np.empty(4), np.empty(9)
    mujoco.mju_axisAngle2Quat(quaternion, np.asarray(axis, dtype=float), radians)
    mujoco.mju_quat2Mat(matrix, quaternion)
    mujoco.mju_mat2Quat(quaternion, (matrix.reshape(3, 3) @ pose.rotation).ravel())
    return Pose(pose.position_m, tuple(quaternion))


def test_neutral_proposal_is_complete_and_does_not_touch_live_physics_or_native_model(stack):
    sim, rt, before, poses = stack
    protected = {
        name: getattr(rt.model, name).copy()
        for name in (
            "body_mass",
            "body_inertia",
            "geom_contype",
            "geom_conaffinity",
            "jnt_range",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_gainprm",
            "eq_data",
        )
    }
    proposal = rt.propose(before, {s: poses[s] for s in ARM_NAMES}, poses["head"])
    assert proposal.valid, proposal.reason
    assert set(proposal.targets) == set(ALL_NAMES)
    assert not set(proposal.targets) & {"Lift", "Torso", "Left Thumb MCP", "FL Drive"}
    after = sim.read_state()
    np.testing.assert_array_equal(after.qpos, before.qpos)
    np.testing.assert_array_equal(after.qvel, before.qvel)
    np.testing.assert_array_equal(after.applied_targets, before.applied_targets)
    assert after.sim_time_s == before.sim_time_s
    for name, original in protected.items():
        np.testing.assert_array_equal(getattr(rt.model, name), original)
    for side in ARM_NAMES:
        assert proposal.diagnostics["ik"][side]["position_error_m"] < 0.001


@pytest.mark.parametrize("side", ["left", "right"])
def test_bilateral_mapped_target_and_elbow_fallback_are_explicit(stack, side):
    _, rt, state, poses = stack
    target = Pose(
        tuple(np.asarray(poses[side].position_m) + [0, 0, 0.02]), poses[side].quaternion_wxyz
    )
    goals = {s: target if s == side else poses[s] for s in ARM_NAMES}
    result = rt.propose(state, goals, poses["head"], elbows={side: [float("nan"), 0, 0]})
    assert result.valid, result.reason
    assert result.diagnostics["ik"][side]["feasible"]
    assert result.diagnostics["ik"][side]["elbow_fallback"] == "measured-posture seed"
    assert result.diagnostics["ik"][side]["position_error_m"] < rt.settings.position_tolerance_m
    assert (
        result.diagnostics["ik"][side]["orientation_error_rad"]
        < rt.settings.orientation_tolerance_rad
    )
    assert any(
        abs(result.targets[n] - state.applied_targets[state.actuator_names.index(n)]) > 1e-5
        for n in ARM_NAMES[side]
    )
    other = "right" if side == "left" else "left"
    np.testing.assert_allclose(
        [result.targets[n] for n in ARM_NAMES[other]],
        [state.applied_targets[state.actuator_names.index(n)] for n in ARM_NAMES[other]],
        atol=1e-10,
    )


def test_valid_elbow_guides_nullspace_without_becoming_required(stack):
    _, rt, state, poses = stack
    rt._load(state)
    root_r = rt.data.xmat[rt._root].reshape(3, 3)
    elbow = root_r.T @ (rt.data.xpos[rt._elbows["left"]] - rt.data.xpos[rt._root])
    goal = Pose(
        tuple(np.asarray(poses["left"].position_m) + [0, 0, 0.02]), poses["left"].quaternion_wxyz
    )
    result = rt.propose(
        state, {"left": goal, "right": poses["right"]}, poses["head"], elbows={"left": elbow}
    )
    assert result.valid, result.reason
    assert result.diagnostics["ik"]["left"]["elbow_guidance"] is True


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_native_neck_solver_handles_three_orientation_axes(stack, axis):
    _, rt, state, poses = stack
    result = rt.propose(state, head_target=oriented(poses["head"], axis, 0.12), channels=("head",))
    assert result.valid, result.reason
    assert set(result.targets) == set(HEAD_NAMES)
    assert result.diagnostics["ik"]["head"]["position_error_m"] is None
    assert result.diagnostics["ik"]["head"]["orientation_error_rad"] < 0.02


def test_unreachable_pose_and_unsupported_neck_orientation_fail_without_partial_targets(stack):
    _, rt, state, poses = stack
    far = Pose((100, 0, 1), poses["left"].quaternion_wxyz)
    result = rt.propose(state, {"left": far, "right": poses["right"]}, poses["head"])
    assert not result.valid and not result.targets
    assert "workspace" in result.reason
    flipped = oriented(poses["head"], (0, 0, 1), np.pi)
    result = rt.propose(state, head_target=flipped, channels=("head",))
    assert not result.valid and not result.targets
    assert "neck" in result.reason


@pytest.mark.parametrize("bad", [None, {}, {"left": None, "right": None}])
def test_missing_or_invalid_required_pose_never_becomes_identity(stack, bad):
    _, rt, state, poses = stack
    result = rt.propose(state, bad, poses["head"])
    assert not result.valid and result.targets == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("qpos", np.array([np.nan])),
        ("applied_targets", []),
        ("sim_time_s", float("inf")),
        ("reset_generation", True),
    ],
)
def test_bad_detached_state_fails_closed(stack, field, value):
    _, rt, state, poses = stack
    result = rt.propose(
        replace(state, **{field: value}), {s: poses[s] for s in ARM_NAMES}, poses["head"]
    )
    assert not result.valid and not result.targets


def test_rate_and_acceleration_follow_applied_references(
    stack,
):
    _, rt, state, poses = stack
    head = oriented(poses["head"], (0, 0, 1), 0.35)
    previous_velocity = np.zeros(3)
    current = state
    for _step in range(1, 31):
        proposal = rt.propose(current, head_target=head, channels=("head",))
        assert proposal.valid, proposal.reason
        ids = [state.actuator_names.index(n) for n in HEAD_NAMES]
        applied = current.applied_targets.copy()
        next_values = np.array([proposal.targets[n] for n in HEAD_NAMES])
        velocity = (next_values - applied[ids]) / 0.02
        assert max(abs(velocity)) <= rt.settings.max_joint_speed_rad_s + 1e-10
        assert (
            max(abs(velocity - previous_velocity))
            <= rt.settings.max_joint_acceleration_rad_s2 * 0.02 + 1e-9
        )
        previous_velocity = velocity
        applied[ids] = next_values
        current = replace(
            current,
            step_index=current.step_index + 1,
            sim_time_s=current.sim_time_s + 0.02,
            applied_targets=applied,
        )
    # A rejected/unexecuted proposal must not make the next output drift away
    # from actual references; generation reset and fresh snapshot discard history.
    rt.reset(state)
    first = rt.propose(state, head_target=head, channels=("head",))
    unchanged = replace(state, step_index=state.step_index + 1, sim_time_s=state.sim_time_s + 0.02)
    next_proposal = rt.propose(unchanged, head_target=head, channels=("head",))
    np.testing.assert_allclose(list(first.targets.values()), list(next_proposal.targets.values()))


def test_repeated_snapshot_and_long_gap_require_history_reset(stack):
    _, rt, state, poses = stack
    assert rt.propose(state, head_target=poses["head"], channels=("head",)).valid
    repeated = rt.propose(state, head_target=poses["head"], channels=("head",))
    assert not repeated.valid and "advance" in repeated.reason
    rt.reset(state)
    assert rt.propose(state, head_target=poses["head"], channels=("head",)).valid
    gap = replace(state, step_index=state.step_index + 20, sim_time_s=state.sim_time_s + 1)
    assert not rt.propose(gap, head_target=poses["head"], channels=("head",)).valid


def test_generation_reset_discards_old_reference_velocity(stack):
    _, rt, state, poses = stack
    target = oriented(poses["head"], (0, 0, 1), 0.2)
    first = rt.propose(state, head_target=target, channels=("head",))
    reset = replace(state, reset_generation=state.reset_generation + 1, step_index=0, sim_time_s=0)
    second = rt.propose(reset, head_target=target, channels=("head",))
    assert first.valid and second.valid
    np.testing.assert_allclose(list(first.targets.values()), list(second.targets.values()))


def test_composed_guard_checks_proposed_fingers_and_preserves_live_state(stack):
    sim, rt, state, _ = stack
    neutral = full_targets(state)
    assert rt.check_composed(state, neutral)["valid"]
    changed = dict(neutral)
    changed["Left Index Abd"] = min(
        rt.model.actuator("Left Index Abd").ctrlrange[1], rt.model.joint("Left Index Abd").range[1]
    )
    guard = rt.check_composed(state, changed)
    assert not guard["valid"] and guard["collisions"]
    assert "collision" in guard["reason"]
    np.testing.assert_array_equal(sim.read_state().qpos, state.qpos)
    np.testing.assert_array_equal(sim.read_state().applied_targets, state.applied_targets)


@pytest.mark.parametrize(
    "update", [{"FL Drive": 0.1}, {"Lift": 1.0}, {"Neck Yaw": 100}, {"Neck Pitch": True}]
)
def test_composed_guard_rejects_invalid_or_unqualified_commands(stack, update):
    _, rt, state, _ = stack
    assert not rt.check_composed(state, full_targets(state, update))["valid"]


def test_measured_body_hold_and_unchanged_commands_are_qualified(stack):
    _, rt, state, _ = stack
    targets = full_targets(state)
    for name in ("Lift", "Torso"):
        targets[name] = float(state.actuator_lengths[state.actuator_names.index(name)])
    assert rt.check_composed(state, targets)["valid"]


def test_private_model_mismatch_is_rejected():
    with pytest.raises(ValueError, match="differs"):
        ArmHeadRetargeter(EMPTY_SPACE, expected_model_sha256="wrong")


def test_thread_ownership_is_enforced(stack):
    _, rt, state, poses = stack
    with ThreadPoolExecutor(max_workers=1) as worker:
        for operation in (
            rt.describe,
            rt.reset,
            lambda: rt.measured_poses(state),
            lambda: rt.propose(state, head_target=poses["head"], channels=("head",)),
            lambda: rt.check_composed(state, full_targets(state)),
            rt.close,
        ):
            with pytest.raises(RuntimeError, match="creating thread"):
                worker.submit(operation).result()


@pytest.mark.parametrize("side", ["left", "right"])
def test_native_physics_moves_each_arm_with_position_and_orientation_errors_separate(stack, side):
    sim, rt, state, poses = stack
    target = Pose(
        tuple(np.asarray(poses[side].position_m) + [0, 0, 0.02]), poses[side].quaternion_wxyz
    )
    targets = {s: target if s == side else poses[s] for s in ARM_NAMES}
    initial = np.asarray(poses[side].position_m)
    commands = []
    for _ in range(50):
        state = sim.read_state()
        proposal = rt.propose(state, targets, poses["head"])
        if not proposal.valid:
            pytest.fail(proposal.reason)
        full = full_targets(state, proposal.targets)
        assert rt.check_composed(state, full)["valid"]
        commands.append(proposal.targets[f"{side.title()} Arm 4"])
        sim.apply_actuator_targets(full)
        sim.step()
    measured = rt.measured_poses(sim.read_state())[side]
    # Measured actuator/EE response is independent evidence from IK residuals.
    assert np.linalg.norm(np.asarray(measured.position_m) - initial) > 0.001
    assert np.isfinite(measured.position_m).all()
    assert abs(commands[-1] - commands[0]) > 0.001
    assert proposal.diagnostics["ik"][side]["position_error_m"] <= rt.settings.position_tolerance_m


def test_default_scene_contact_guard_retains_blocked_finger_evidence():
    config = ShelfConfig(product_count=0)
    with MujocoTeleopAdapter(config) as sim, ArmHeadRetargeter(config) as rt:
        state = sim.read_state()
        assert rt.check_composed(state, full_targets(state))["valid"]
        targets = full_targets(
            state, {"Left Index MCP": float(rt.model.actuator("Left Index MCP").ctrlrange[1])}
        )
        blocked = rt.check_composed(state, targets)
        assert not blocked["valid"] and blocked["collisions"]
        assert "collision" in blocked["reason"]
        assert state.sim_time_s == sim.read_state().sim_time_s


def test_native_neck_response_is_measured_separately_from_kinematic_prediction(stack):
    sim, rt, state, poses = stack
    target = oriented(poses["head"], (0, 0, 1), 0.12)
    neck = state.actuator_names.index("Neck Yaw")
    measured_before = state.actuator_lengths[neck]
    for _ in range(50):
        current = sim.read_state()
        proposal = rt.propose(current, head_target=target, channels=("head",))
        assert proposal.valid, proposal.reason
        full = full_targets(current, proposal.targets)
        assert rt.check_composed(current, full)["valid"]
        sim.apply_actuator_targets(full)
        sim.step()
    after = sim.read_state()
    assert abs(after.actuator_lengths[neck] - measured_before) > 0.02
    assert after.sim_time_s > state.sim_time_s
    assert proposal.diagnostics["ik"]["head"]["orientation_error_rad"] < 0.005


def test_inside_workspace_failed_local_search_never_promotes_closest_candidate(stack):
    sim, rt, state, poses = stack
    inside = Pose((0, 0, 1.2), poses["left"].quaternion_wxyz)
    proposal = rt.propose(state, {"left": inside, "right": poses["right"]}, poses["head"])
    assert not proposal.valid and not proposal.targets
    assert proposal.reason == "no acceptable arm IK solution found"
    report = proposal.diagnostics["ik"]["left"]
    assert report["feasible"] is False
    assert report["position_error_m"] > rt.settings.position_tolerance_m
    assert report["orientation_error_rad"] > rt.settings.orientation_tolerance_rad
    assert report["endpoint_collisions"]
    np.testing.assert_array_equal(sim.read_state().qpos, state.qpos)


@pytest.mark.parametrize(
    "settings",
    [
        dict(iterations=True),
        dict(damping=0),
        dict(max_joint_speed_rad_s=float("nan")),
        dict(path_samples=1),
    ],
)
def test_invalid_settings_fail_before_model_use(settings):
    with pytest.raises(ValueError):
        ArmHeadSettings(**settings)


def test_composed_guard_admits_only_bounded_owner_measured_body_hold_transition(stack):
    sim, rt, _, _ = stack
    state = sim.step()
    hold = {
        name: float(state.actuator_lengths[state.actuator_names.index(name)])
        for name in ("Lift", "Torso")
    }
    targets = full_targets(state)
    for name in hold:
        targets[name] = (targets[name] + hold[name]) / 2
    assert rt.check_composed(state, targets, body_hold=hold)["valid"]
    targets["Torso"] = 0.05
    assert not rt.check_composed(state, targets, body_hold=hold)["valid"]
    assert not rt.check_composed(state, targets, body_hold={"Lift": float("nan"), "Torso": 0})[
        "valid"
    ]
