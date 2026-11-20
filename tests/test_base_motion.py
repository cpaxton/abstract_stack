"""Swerve frame/sign mapping and scratch screening preserve native physics."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.base_motion import (
    BaseContactGuard,
    BaseMotionConfig,
    BaseMotionFailure,
    NativeSwerve,
    closest_steering,
    screen_base_translation,
)
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import InteractionFailure, StockingScript
from abstract_stack.packing import dense_shelf_config
from abstract_stack.scene import ShelfConfig


@pytest.mark.parametrize("target,current", [(np.pi, 0), (-3.1, 3.1), (0.1, 7.0), (20, -10)])
def test_shortest_steering_is_continuous_and_reverses_drive(target, current):
    angle, sign = closest_steering(target, current)
    assert abs(angle - current) <= np.pi / 2 + 1e-12
    np.testing.assert_allclose(
        sign * np.array([np.cos(angle), np.sin(angle)]),
        [np.cos(target), np.sin(target)],
        atol=1e-12,
    )


def test_invalid_base_probe_settings():
    with pytest.raises(ValueError):
        BaseMotionConfig(fixture_force_limit_N=26)
    with pytest.raises(ValueError):
        BaseMotionConfig(max_speed_m_s=float("nan"))
    with pytest.raises(ValueError):
        closest_steering(float("nan"), 0)


@pytest.mark.parametrize(
    ("final_carton", "nominal", "position", "velocity", "expected_ticks"),
    [
        (True, 0.5, 0.501, 0.01, 0),
        (True, 0.5, 0.501, 0.03, 75),
        (True, 0.4, 0.501, 0.01, 75),
        (False, 0.5, 0.501, 0.01, 75),
    ],
)
def test_packed_base_move_skips_stow_only_for_the_settled_reference(
    final_carton, nominal, position, velocity, expected_ticks
):
    events, controls = [], []
    joint = SimpleNamespace(qposadr=[0], dofadr=[0])
    script = SimpleNamespace(
        config=SimpleNamespace(defer_final_alignment=True),
        env=SimpleNamespace(
            model=SimpleNamespace(joint=lambda name: joint),
            data=SimpleNamespace(qpos=[position], qvel=[velocity]),
            products=("first", "last"),
            dt=0.02,
        ),
        observer=SimpleNamespace(completed={"first"} if final_carton else set()),
        tracker=SimpleNamespace(NAMES=("Right Arm 3",), nominal=[nominal]),
        events=events,
        _stage=lambda phase: events.append({"phase": phase}),
        _apply_tick=controls.append,
    )

    StockingScript._stow_right_arm(script)

    assert controls == [{"Right Arm 3": 0.5}] * expected_ticks
    assert events[-1].get("already_stowed", False) == (expected_ticks == 0)


@pytest.mark.parametrize("rejected_before_turn", [True, False])
def test_base_turn_rescreens_clearance_only_before_actuation(rejected_before_turn):
    turns, moves, margins = [], [], []

    def turn(angle):
        turns.append(angle)
        if len(turns) == 1:
            if rejected_before_turn:
                raise InteractionFailure("base_turn: yaw sweep rejected before actuation")
            raise BaseMotionFailure("base_motion: native contact-force guard exceeded")

    def screen(angle, offsets, *, margin):
        margins.append(margin)
        return offsets[0] if len(margins) == 1 else offsets[1]

    script = SimpleNamespace(
        config=SimpleNamespace(record_skill_starts=False),
        observer=SimpleNamespace(incidents=[]),
        events=[{}],
        _screen_base_turn_stances=screen,
        _move_base=lambda offset, *, reason: moves.append((offset, reason)),
        _turn_base=turn,
    )
    if rejected_before_turn:
        StockingScript._turn_base_from_clear_workspace(script, 0.6)
        assert turns == [0.6, 0.6]
        assert margins == [0.016, 0.018]
        assert moves == [([0.015, 0], "base_turn_clearance_recovery")]
    else:
        with pytest.raises(BaseMotionFailure, match="contact-force guard"):
            StockingScript._turn_base_from_clear_workspace(script, 0.6)
        assert turns == [0.6]
        assert margins == [0.016]
        assert not moves


def test_a_screened_turn_cannot_translate_or_carry():
    with pytest.raises(ValueError, match="in place, unloaded"):
        screen_base_translation(None, [0.02, 0], turn_rad=-0.5)
    with pytest.raises(ValueError, match="in place, unloaded"):
        screen_base_translation(None, [0, 0], turn_rad=-0.5, carried_product="carton_0")


@pytest.fixture
def native_env():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=10, initially_stocked=9))
    yield env
    env.close()


@pytest.mark.mabel
@pytest.mark.parametrize("yaw", [0.0, 0.37])
def test_native_quarter_turn_offsets_and_drive_sign_reconstruct_module_velocities(native_env, yaw):
    env = native_env
    scratch = env.clone_for_planning()
    adr = int(env.model.joint("base_root").qposadr[0])
    scratch.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(env.model, scratch.data)
    drive = NativeSwerve(scratch.model, scratch.data)
    original = env.data.qpos.copy()
    requested = np.array([0.01, 0.02, 0.03])
    steering, _, _ = drive.module_requests(requested)
    for module, angle in zip(drive.modules, steering, strict=True):
        scratch.data.qpos[int(env.model.jnt_qposadr[module["steer"]])] = angle
    mujoco.mj_forward(env.model, scratch.data)
    steering, speeds, errors = drive.module_requests(requested)
    assert max(errors) < 1e-12
    for module, angle, speed in zip(drive.modules, steering, speeds, strict=True):
        x, y = module["position_body_m"]
        rolling = np.array(
            [np.cos(module["zero_heading_rad"] + angle), np.sin(module["zero_heading_rad"] + angle)]
        )
        rotation = scratch.data.xmat[drive.base].reshape(3, 3)
        expected = rotation[:2, :2].T @ requested[:2] + requested[2] * np.array([-y, x])
        np.testing.assert_allclose(rolling * speed * module["radius_m"], expected, atol=1e-12)
        assert module["radius_m"] == pytest.approx(0.03812355, abs=1e-7)
    np.testing.assert_array_equal(env.data.qpos, original)


@pytest.mark.mabel
def test_native_swerve_targets_are_rate_limited_without_state_or_gain_writes(native_env):
    env = native_env
    drive = NativeSwerve(env.model, env.data)
    before = [
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.data.ctrl.copy(),
        env.model.actuator_gainprm.copy(),
    ]
    twist = np.array([0.0, 1.0, 0.2])
    requested = twist.copy()
    action = drive.targets(twist, dt=0.02)
    np.testing.assert_array_equal(twist, requested)
    for name, value in action.items():
        actuator = env.model.actuator(name)
        if actuator.ctrllimited:
            assert actuator.ctrlrange[0] <= value <= actuator.ctrlrange[1]
        if "Steer" in name:
            assert abs(value) <= drive.config.steering_rate_rad_s * 0.02 + 1e-12
        else:
            assert abs(value) <= drive.config.drive_accel_rad_s2 * 0.02 + 1e-12
    for actual, original in zip(
        [env.data.qpos, env.data.qvel, env.data.ctrl, env.model.actuator_gainprm],
        before,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.mabel
def test_whole_robot_path_rejects_future_chassis_collision_in_scratch_only():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(ShelfConfig(product_count=1, center_y=-1.5, table_center_y=-1.6))
    try:
        body = env.model.body("mabel_chassis").id
        geom = next(
            g
            for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] == body and env.model.geom_contype[g]
        )
        mesh = env.model.geom_dataid[geom]
        start, count = env.model.mesh_vertadr[mesh], env.model.mesh_vertnum[mesh]
        points = (
            env.model.mesh_vert[start : start + count] @ env.data.geom_xmat[geom].reshape(3, 3).T
            + env.data.geom_xpos[geom]
        )
        adr = int(env.model.joint("carton_0_joint").qposadr[0])
        # Reset-only collision-test fixture outside the current chassis but
        # inside its proposed +Y swept volume. No physical rollout is claimed.
        env.data.qpos[adr : adr + 3] = [
            float(points[:, 0].mean()),
            float(points[:, 1].max()) + 0.05,
            float((points[:, 2].min() + points[:, 2].max()) / 2),
        ]
        mujoco.mj_forward(env.model, env.data)
        qpos, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        report = screen_base_translation(env, [0, 0.1])
        assert not report["accepted"]
        assert report["rejection"]["fixture"] == "carton_0_geom"
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
    finally:
        env.close()


@pytest.mark.mabel
def test_reusable_guard_rejects_yaw_and_product_tip_without_touching_live_episode(native_env):
    env = native_env
    guard = BaseContactGuard(env)
    snapshot = env.clone_for_planning()
    fixture = SimpleNamespace(model=env.model, data=snapshot.data)
    before_qpos, before_ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
    base_adr = int(env.model.joint("base_root").qposadr[0])
    snapshot.data.qpos[base_adr + 3 : base_adr + 7] = [np.cos(0.04 / 2), 0, 0, np.sin(0.04 / 2)]
    mujoco.mj_forward(env.model, snapshot.data)
    with pytest.raises(BaseMotionFailure, match="yaw left"):
        guard.step(fixture)
    snapshot = env.clone_for_planning()
    fixture.data = snapshot.data
    active = next(p for p in env.products if not p.initially_stocked)
    adr = int(env.model.joint(active.name + "_joint").qposadr[0])
    snapshot.data.qpos[adr + 3 : adr + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    mujoco.mj_forward(env.model, snapshot.data)
    with pytest.raises(BaseMotionFailure):
        guard.step(fixture)
    assert any(e["product"] == active.name and e["kind"] == "tipped" for e in guard.incidents)
    np.testing.assert_array_equal(env.data.qpos, before_qpos)
    np.testing.assert_array_equal(env.data.ctrl, before_ctrl)
    assert env.interventions == []


@pytest.mark.mabel
def test_reusable_guard_detects_native_robot_carton_contact(native_env):
    env = native_env
    guard = BaseContactGuard(env)
    snapshot = env.clone_for_planning()
    active = next(p for p in env.products if not p.initially_stocked)
    adr = int(env.model.joint(active.name + "_joint").qposadr[0])
    # Scratch-only rejection fixture, not a physical manipulation episode.
    snapshot.data.qpos[adr : adr + 3] = env.data.body("mabel_chassis").xpos + [0, 0, 0.015]
    mujoco.mj_forward(env.model, snapshot.data)
    with pytest.raises(BaseMotionFailure, match="contact-force"):
        guard.step(SimpleNamespace(model=env.model, data=snapshot.data))
    assert guard.peaks["robot_product_N"] > 25


@pytest.mark.mabel
def test_reconstructing_controller_keeps_existing_drive_reference_continuity(native_env):
    env = native_env
    snapshot = env.clone_for_planning()
    aid = env.model.actuator("FL Drive").id
    snapshot.data.ctrl[aid] = 0.4  # Explicit scratch controller-state fixture.
    drive = NativeSwerve(env.model, snapshot.data)
    action = drive.targets([0, 0, 0], dt=0.02)
    assert action["FL Drive"] == pytest.approx(0.4 - 0.8 * 0.02)
    assert snapshot.data.ctrl[aid] == 0.4
    assert env.data.ctrl[aid] == 0.0
