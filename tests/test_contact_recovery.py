"""Released-contact recovery must retain support, force and motion contracts."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.dense_interactions import FingerPusher
from abstract_stack.interactions import InteractionFailure
from abstract_stack.receding_stocking import RecedingStockingScript


@pytest.fixture
def released_push(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    for _ in range(25):
        env.step()
    script = RecedingStockingScript(env, tmp_path, bounded_contact_ik=True, contact_recovery=True)
    script.product = next(p for p in env.products if not p.initially_stocked)
    script.controller = FingerPusher(env, script.config)
    script._stage("finger_push")
    try:
        yield env, script
    finally:
        env.close()


@pytest.mark.mabel
def test_released_supported_carton_may_lose_hand_contact(released_push):
    env, script = released_push
    script.supported_slide = True
    script._free_push_contact_seen = True
    for _ in range(8):
        env.step(on_physics_step=script._sample)
    assert script.last_forces["table"] + script.last_forces["shelf"] > 0.2
    assert script.last_forces["hand"] == 0
    assert script._free_push_loss_seconds > 0.10
    assert script._free_push_support_loss == 0
    assert not np.any(env.data.qfrc_applied)
    assert not np.any(env.data.xfrc_applied)


@pytest.mark.mabel
@pytest.mark.parametrize("contract", ["grasped", "carried", "other_phase"])
def test_recovery_does_not_replace_grasped_slide_contract(released_push, contract):
    env, script = released_push
    script.supported_slide = True
    if contract == "grasped":
        script.active_grip = 0.5
    elif contract == "carried":
        script.carry_reference = np.zeros(3)
        # Retain the slide guard without triggering the independent carry guard.
        script.phase = "release"
    else:
        script.phase = "push"
    with pytest.raises(InteractionFailure, match="lost shelf support or hand contact"):
        for _ in range(8):
            env.step(on_physics_step=script._sample)


@pytest.mark.mabel
def test_free_push_still_aborts_when_surface_support_is_lost(released_push):
    env, script = released_push
    # Explicit invalid reset fixture; no manipulation credit. The carton falls
    # freely for longer than the support-loss budget without reaching the table.
    adr = int(env.model.joint(f"{script.product.name}_joint").qposadr[0])
    env.data.qpos[adr + 2] += 0.25
    mujoco.mj_forward(env.model, env.data)
    with pytest.raises(InteractionFailure, match="lost physical surface support"):
        for _ in range(8):
            env.step(on_physics_step=script._sample)


@pytest.mark.mabel
def test_free_push_retains_non_grasping_link_force_guard(released_push):
    env, script = released_push
    # Invalid penetration fixture before the tested rollout, never a successful
    # action. The normal 25 N guard must run before recovery can act.
    adr = int(env.model.joint(f"{script.product.name}_joint").qposadr[0])
    tower = env.model.body("left_tower").id
    collision_geom = np.flatnonzero(
        (env.model.geom_bodyid == tower) & (env.model.geom_contype != 0)
    )[0]
    env.data.qpos[adr : adr + 3] = env.data.geom_xpos[collision_geom]
    mujoco.mj_forward(env.model, env.data)
    with pytest.raises(InteractionFailure, match="contact force exceeded 25"):
        env.step(on_physics_step=script._sample)
    assert script.last_forces["robot_product"] > 25
    assert script.last_forces["arm_product"] > 0


@pytest.mark.mabel
def test_reacquisition_rejects_out_of_bounds_reference_before_actuation(released_push):
    env, script = released_push
    reference = script.controller.position()
    reference[1] = script.product.target[1] + 0.021
    before = env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time
    with pytest.raises(InteractionFailure, match="reacquisition reference bound"):
        script._reacquire_push_contact(reference, reference, reference[2], {})
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])
    assert env.data.time == before[2]
    script._free_push_recoveries = 3
    with pytest.raises(InteractionFailure, match="recovery count exhausted"):
        script._reacquire_push_contact(reference, reference, reference[2], {})


@pytest.mark.mabel
@pytest.mark.parametrize("phase", ["finger_push", "reacquire_push_contact"])
def test_free_push_tilt_limit_also_applies_inside_recovery(released_push, phase):
    env, script = released_push
    # Explicit airborne, tilted reset fixture avoids any fixture penetration.
    adr = int(env.model.joint(f"{script.product.name}_joint").qposadr[0])
    env.data.qpos[adr + 2] += 0.25
    env.data.qpos[adr + 3 : adr + 7] = [np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    script.phase = phase
    with pytest.raises(InteractionFailure, match="free_push: carton tilting"):
        env.step(on_physics_step=script._sample)


@pytest.mark.mabel
def test_recovery_unloads_downforce_before_forward_advance(released_push, monkeypatch):
    env, script = released_push
    reference = script.controller.position()
    reference[1] = script.product.target[1]
    captured = []
    # Controller-input unit fixture, not a measured-contact or task-success
    # claim. Stop at the planning boundary before any native command is sent.
    script.last_forces = {"hand": 0.7, "hand_force_world_N": [0, 0, -0.7]}

    def reject(target, **kwargs):
        captured.append(target.copy())
        return SimpleNamespace(feasible=False, report=lambda: {"unit_fixture": True})

    monkeypatch.setattr(script, "_contact_prediction", reject)
    before = env.data.ctrl.copy(), float(env.data.time)
    with pytest.raises(InteractionFailure, match="reacquisition IK/path rejected"):
        script._reacquire_push_contact(reference, reference, reference[2], {})
    assert captured[0][0] == reference[0]
    assert captured[0][2] > reference[2]
    np.testing.assert_array_equal(env.data.ctrl, before[0])
    assert env.data.time == before[1]


@pytest.mark.mabel
def test_recovery_rechecks_projected_nominal_bounds(released_push, monkeypatch):
    env, script = released_push
    reference = script.controller.position()
    reference[1] = script.product.target[1]
    projected = reference.copy()
    projected[1] = script.product.target[1] + 0.021
    # A solver result cannot grant permission outside the original envelope.
    monkeypatch.setattr(
        script,
        "_contact_prediction",
        lambda *args, **kwargs: SimpleNamespace(feasible=True, position=projected),
    )
    before = env.data.ctrl.copy(), float(env.data.time)
    with pytest.raises(InteractionFailure, match="projected reacquisition reference bound"):
        script._reacquire_push_contact(reference, reference, reference[2], {})
    np.testing.assert_array_equal(env.data.ctrl, before[0])
    assert env.data.time == before[1]
