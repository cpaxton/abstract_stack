"""Native physics, ownership and stereo snapshot contracts for the sim host."""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.scene import ShelfConfig
from abstract_stack.teleop.sim import BASE_DRIVE, BASE_STEER, MujocoTeleopAdapter

pytestmark = pytest.mark.mabel


@pytest.fixture(scope="module")
def adapter():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("pinned MABEL assets unavailable")
    with MujocoTeleopAdapter(ShelfConfig(product_count=0)) as sim:
        yield sim


@pytest.fixture
def sim(adapter):
    adapter.reset(seed=0)
    return adapter


def test_state_is_detached_and_native_lift_length_is_not_one_slide(sim):
    state = sim.read_state()
    model, data = sim._env.model, sim._env.data
    lift = state.actuator_names.index("Lift")
    lengths = [data.joint(name).qpos[0] for name in ("Lift Lower", "Lift Upper")]
    assert state.actuator_lengths[lift] == pytest.approx(sum(lengths))
    assert model.actuator_trntype[lift] == mujoco.mjtTrn.mjTRN_TENDON
    for array in (
        state.qpos,
        state.qvel,
        state.applied_targets,
        state.actuator_lengths,
        state.actuator_velocities,
        state.actuator_forces,
        state.root_position_m,
        state.root_quaternion_wxyz,
    ):
        array[:] = 999
    reread = sim.read_state()
    assert reread.qpos[0] != 999
    assert reread.actuator_lengths[lift] == pytest.approx(sum(lengths))
    assert np.all(reread.applied_targets != 999)


def test_complete_commands_are_atomic_and_cannot_drive_or_steer_base(sim):
    before = sim.read_state()
    commands = []
    for name, value in (("FL Drive", 0.2), ("FR Steer", 0.1), ("Lift", 100), ("Torso", np.nan)):
        targets = before.applied_targets.copy()
        targets[before.actuator_names.index(name)] = value
        commands.append(targets)
    commands.extend((np.zeros(2), {"Left Arm 1": -0.1}))
    for command in commands:
        with pytest.raises(ValueError):
            sim.apply_actuator_targets(command)
        after = sim.read_state()
        np.testing.assert_array_equal(after.applied_targets, before.applied_targets)
        np.testing.assert_array_equal(after.qpos, before.qpos)
        assert after.sim_time_s == before.sim_time_s


def test_valid_targets_do_not_teleport_and_measured_physics_is_separate(sim):
    before = sim.read_state()
    targets = dict(zip(before.actuator_names, before.applied_targets, strict=True))
    targets["Neck Yaw"] += 0.05
    sim.apply_actuator_targets(targets)
    commanded = sim.read_state()
    np.testing.assert_array_equal(before.qpos, commanded.qpos)
    neck = before.actuator_names.index("Neck Yaw")
    assert commanded.applied_targets[neck] != commanded.actuator_lengths[neck]
    assert commanded.sim_time_s == before.sim_time_s
    moved = sim.step()
    assert moved.sim_time_s == pytest.approx(before.sim_time_s + sim.control_dt)
    assert moved.step_index == 1
    assert moved.actuator_lengths[neck] != before.actuator_lengths[neck]
    assert np.isfinite(moved.actuator_forces).all()
    for name in BASE_DRIVE:
        assert moved.applied_targets[moved.actuator_names.index(name)] == 0


def test_complex_command_is_rejected_without_changing_targets(sim):
    before = sim.read_state()
    complex_targets = before.applied_targets.astype(complex) + 1j
    named_targets = dict(zip(before.actuator_names, complex_targets, strict=True))
    for command in (complex_targets, complex_targets.astype(object), named_targets):
        with pytest.raises(ValueError, match="targets must be real"):
            sim.apply_actuator_targets(command)
        np.testing.assert_array_equal(sim.read_state().applied_targets, before.applied_targets)


def test_reset_and_repeated_physics_are_deterministic(sim):
    initial = sim.read_state()
    first = None
    for _ in range(8):
        first = sim.step()
    reset = sim.reset(seed=0)
    assert reset.reset_generation == initial.reset_generation + 1
    np.testing.assert_array_equal(reset.qpos, initial.qpos)
    np.testing.assert_array_equal(reset.applied_targets, initial.applied_targets)
    second = None
    for _ in range(8):
        second = sim.step()
    np.testing.assert_array_equal(first.qpos, second.qpos)
    np.testing.assert_array_equal(first.qvel, second.qvel)
    for name in BASE_STEER:
        index = initial.actuator_names.index(name)
        assert second.applied_targets[index] == initial.applied_targets[index]


def test_commands_and_stepping_preserve_native_model_and_free_base(sim):
    model = sim._env.model
    fields = (
        "body_mass",
        "body_inertia",
        "geom_contype",
        "geom_conaffinity",
        "jnt_range",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gear",
        "eq_data",
    )
    original = {name: getattr(model, name).copy() for name in fields}
    assert model.joint("base_root").type[0] == mujoco.mjtJoint.mjJNT_FREE
    targets = sim.stationary_command()
    for _ in range(5):
        sim.apply_actuator_targets(targets)
        sim.step()
    for name in fields:
        np.testing.assert_array_equal(getattr(model, name), original[name])


def test_measured_hold_rejects_out_of_command_range_physical_state_atomically(sim):
    # A measured posture need not lie inside the narrower servo command range.
    # Retain this native-physics counterexample rather than clipping it silently.
    for _ in range(10):
        before = sim.read_state()
        try:
            targets = sim.stationary_command()
        except ValueError as error:
            assert "outside" in str(error)
            np.testing.assert_array_equal(sim.read_state().qpos, before.qpos)
            np.testing.assert_array_equal(sim.read_state().applied_targets, before.applied_targets)
            return
        sim.apply_actuator_targets(targets)
        sim.step()
    pytest.fail("Expected the pinned fixture's measured-hold command-range counterexample")


def test_unknown_camera_fails_before_rendering(sim):
    with pytest.raises(ValueError, match="Unknown camera"):
        sim.render_camera("missing")


def test_product_reset_replays_seed_and_resets_previous_commands(adapter):
    with MujocoTeleopAdapter(ShelfConfig(product_count=2), seed=7) as sim:
        initial = sim.read_state()
        targets = initial.applied_targets.copy()
        targets[initial.actuator_names.index("Neck Yaw")] += 0.03
        sim.apply_actuator_targets(targets)
        sim.step()
        reset = sim.reset(seed=7)
        np.testing.assert_array_equal(reset.qpos, initial.qpos)
        np.testing.assert_array_equal(reset.applied_targets, initial.applied_targets)
        assert not np.array_equal(sim.reset(seed=8).qpos, initial.qpos)


def test_invalid_reset_preserves_the_active_episode_and_seed(adapter):
    with MujocoTeleopAdapter(ShelfConfig(product_count=2), seed=7) as sim:
        initial = sim.read_state()
        targets = initial.applied_targets.copy()
        targets[initial.actuator_names.index("Neck Yaw")] += 0.03
        sim.apply_actuator_targets(targets)
        before = sim.step()

        with pytest.raises(ValueError):
            sim.reset(seed=-1)

        after = sim.read_state()
        assert after.reset_generation == before.reset_generation
        assert after.step_index == before.step_index
        assert after.sim_time_s == before.sim_time_s
        np.testing.assert_array_equal(after.qpos, before.qpos)
        np.testing.assert_array_equal(after.qvel, before.qvel)
        np.testing.assert_array_equal(after.applied_targets, before.applied_targets)
        assert sim.step().step_index == before.step_index + 1
        np.testing.assert_array_equal(sim.reset().qpos, initial.qpos)


def test_all_live_operations_reject_other_threads(sim):
    commands = sim.read_state().applied_targets
    operations = (
        sim.read_state,
        sim.step,
        sim.reset,
        sim.stationary_command,
        sim.describe,
        sim.close,
        lambda: sim.apply_actuator_targets(commands),
        lambda: sim.render_camera("zed_left"),
        sim.render_stereo_pair,
    )
    before = sim.read_state()
    with ThreadPoolExecutor(max_workers=1) as worker:
        for operation in operations:
            with pytest.raises(RuntimeError, match="creating thread"):
                worker.submit(operation).result()
    np.testing.assert_array_equal(sim.read_state().qpos, before.qpos)


def test_finished_episode_rejects_commands_and_requires_reset(adapter):
    with MujocoTeleopAdapter(ShelfConfig(product_count=0, time_limit=0.02)) as sim:
        targets = sim.read_state().applied_targets
        assert sim.step().truncated
        with pytest.raises(RuntimeError, match="Episode finished"):
            sim.apply_actuator_targets(targets)
        with pytest.raises(RuntimeError, match="Episode is finished"):
            sim.step()
        assert sim.reset().sim_time_s == 0
        sim.close()
        sim.close()
        with pytest.raises(RuntimeError, match="closed"):
            sim.read_state()


@pytest.mark.rendering
def test_native_stereo_pair_does_not_advance_or_mutate_physics(sim):
    before = sim.read_state()
    pair = sim.render_stereo_pair(width=160, height=120)
    after = sim.read_state()
    assert pair.sim_time_s == before.sim_time_s == after.sim_time_s
    np.testing.assert_array_equal(before.qpos, after.qpos)
    np.testing.assert_array_equal(before.qvel, after.qvel)
    np.testing.assert_array_equal(before.applied_targets, after.applied_targets)
    assert [pair.left.name, pair.right.name] == ["zed_left", "zed_right"]
    for eye in (pair.left, pair.right):
        assert eye.rgb.shape == (120, 160, 3)
        assert eye.rgb.dtype == np.uint8
        assert np.ptp(eye.rgb) > 0
        assert eye.vertical_fov_deg == pytest.approx(83.3)
        assert np.linalg.norm(eye.quaternion_wxyz) == pytest.approx(1)
        assert eye.intrinsics["fx"] == eye.intrinsics["fy"]
    assert np.linalg.norm(pair.right.position_m - pair.left.position_m) == pytest.approx(0.063)
    pair.left.rgb[:] = 0
    next_pair = sim.render_stereo_pair(width=160, height=120)
    assert next_pair.pair_id == pair.pair_id + 1
    assert np.ptp(next_pair.left.rgb) > 0
    sim.reset()
    restarted = sim.render_stereo_pair(width=160, height=120)
    assert restarted.pair_id == next_pair.pair_id + 1
    assert restarted.reset_generation == next_pair.reset_generation + 1


@pytest.mark.parametrize("width,height", [(0, 480), (640, -1), (True, 480), (1.5, 480)])
def test_bad_render_dimensions_fail_before_allocating_renderer(sim, width, height):
    with pytest.raises(ValueError, match="positive integers"):
        sim.render_stereo_pair(width=width, height=height)
