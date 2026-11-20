"""Mode selection preserves legacy scenes and reuses the dense skill bodies."""

from types import SimpleNamespace

import numpy as np
import pytest

from abstract_stack.dense_interactions import DenseStockingScript, FingerPusher
from abstract_stack.finger_pusher import FingerPusher as SharedFingerPusher
from abstract_stack.interactions import ScriptConfig, StockingScript


def script_fixture(gap=0.003, *, completed=True, enabled=True, level=0):
    active = SimpleNamespace(
        name="active", level=0, half_size=(0.035, 0.025, 0.12), target=(-0.72, -0.12, 0.92)
    )
    neighbor = SimpleNamespace(name="neighbor", level=level, half_size=(0.035, 0.025, 0.12))
    body = SimpleNamespace(xpos=np.array([-0.72, -0.12 - 0.05 - gap, 0.92]), xmat=np.eye(3).ravel())
    script = object.__new__(StockingScript)
    script.product = active
    script.env = SimpleNamespace(
        products=[active, neighbor],
        data=SimpleNamespace(body=lambda _: body),
        config=SimpleNamespace(row_free_space=None),
    )
    script.observer = SimpleNamespace(completed={"neighbor"} if completed else set())
    script.config = ScriptConfig(ungrasped_insertion=enabled)
    return script


def test_adjacent_neighbor_is_measured_and_must_be_completed_in_same_row():
    assert script_fixture()._adjacent_placed_neighbors()[0]["product"] == "neighbor"
    assert not script_fixture(gap=0.045)._adjacent_placed_neighbors()
    assert not script_fixture(completed=False)._adjacent_placed_neighbors()
    assert not script_fixture(level=1)._adjacent_placed_neighbors()


def test_neighbor_mode_releases_then_pushes_and_legacy_keeps_wrapping_slide():
    for enabled, expected in ((True, "finger"), (False, "wrapping")):
        script = script_fixture(enabled=enabled)
        calls = []
        script.events = []
        script._stage = lambda phase, script=script: script.events.append(dict(phase=phase))
        script._pick_and_place_front = lambda selected, calls=calls: (
            calls.append("pick_place") or 0.5
        )
        script._release_and_insert_front = lambda grip, calls=calls: calls.append("finger")
        script._push_grasped = lambda grip, calls=calls: calls.append("wrapping")
        script._one_box()
        assert calls == ["pick_place", expected]
        assert bool(script.events) == enabled


def test_dense_controller_inherits_shared_skill_without_duplicating_methods():
    assert FingerPusher is SharedFingerPusher
    for name in ("_insert_from_front", "_precision_release", "_plan_pusher", "_change_hand"):
        assert getattr(DenseStockingScript, name) is getattr(StockingScript, name)


def test_front_reference_is_chosen_inside_the_carry_route_before_seating():
    script = script_fixture()
    canonical = script.product.target[1]
    goal = script._front_reference_y()
    assert canonical < goal <= canonical + 0.02
    assert script.product.target[1] == canonical
    script.config = ScriptConfig(ungrasped_insertion=False)
    assert script._front_reference_y() == canonical


def test_edge_carton_uses_an_explicit_interior_handoff_without_changing_final_slot():
    script = script_fixture(completed=False)
    script.config = ScriptConfig(ungrasped_insertion=True, paired_finger_insertion=True)
    script.env.config = SimpleNamespace(row_free_space=0.01, center_y=-0.12, width=0.51)
    for canonical, staging in ((0.109090909, 0.06), (-0.349090909, -0.30)):
        script.product.target = (-0.722, canonical, 0.92)
        assert script._front_reference_y() == pytest.approx(staging)
        assert script.product.target[1] == canonical
        assert script._pusher_class().__name__ == "WallFingerPusher"
        assert script._pushing_finger() == ("Pinky" if canonical > -0.12 else "Index")


def test_native_continuation_sync_keeps_nominal_separate_from_applied():
    from abstract_stack.contact_execution import sync_tracking_state

    hand = SimpleNamespace(
        names=["Left Arm 1", "Left Wrist"],
        command=np.array([0.2, 0.3, 0.4]),
        ik=SimpleNamespace(forward=lambda q, reference: (np.zeros(3), np.eye(3))),
    )
    tracker = SimpleNamespace(
        NAMES=("Left Arm 1", "Right Arm 1", "Left Wrist", "Lift"),
        nominal=np.array([0.0, 0.8, 0.0, 0.0]),
        ids=np.array([0, 3, 1, 2]),
    )
    data = SimpleNamespace(ctrl=np.array([0.21, 0.42, 0.45, 0.9]), qpos=np.zeros(2))
    script = SimpleNamespace(controller=hand, tracker=tracker, env=SimpleNamespace(data=data))
    sync_tracking_state(script)
    np.testing.assert_array_equal(tracker.nominal, [0.2, 0.8, 0.3, 0.4])
    np.testing.assert_array_equal(tracker.applied, [0.21, 0.9, 0.42, 0.45])


def test_dense_handoff_reserves_whole_row_clearance_and_both_walls():
    script = script_fixture(gap=0.001)
    script.env.config = SimpleNamespace(row_free_space=0.003, center_y=-0.12, width=0.103)
    assert script._front_reference_y() == script.product.target[1]
    # A biased nominal target cannot make the staging corridor leave a wall.
    script.observer.completed.clear()
    script.product.target = (-0.72, -0.15, 0.92)
    goal = script._front_reference_y()
    assert goal >= -0.12 - 0.103 / 2 + 0.025 + 0.0004
    assert goal < -0.14


@pytest.mark.mabel
def test_selected_insertion_reserves_lift_clearance_without_changing_native_model():
    import mujoco

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import CartesianHand

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        initial = env.data.qpos.copy(), env.data.ctrl.copy(), env.model.jnt_range.copy()
        hand = CartesianHand(env, ScriptConfig(ungrasped_insertion=True))
        assert hand.ik.lower[-1] == pytest.approx(0.03)
        geometry = [
            next(
                g
                for g in range(env.model.ngeom)
                if env.model.geom_bodyid[g] == env.model.body(name).id and env.model.geom_contype[g]
            )
            for name in ("lift_base", "body_mount")
        ]
        q = hand.ik.measured_q()
        q[-1] = hand.ik.lower[-1]
        hand.ik.forward(q)
        mujoco.mj_fwdPosition(env.model, hand.ik.scratch)
        assert mujoco.mj_geomDistance(env.model, hand.ik.scratch, *geometry, 0.1, None) > 0.01
        for value, before in zip(
            (env.data.qpos, env.data.ctrl, env.model.jnt_range), initial, strict=True
        ):
            np.testing.assert_array_equal(value, before)
    finally:
        env.close()


@pytest.mark.mabel
def test_paired_native_tool_and_release_forecast_own_their_state(tmp_path):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.contact_rollout import integration_state
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.finger_pusher import PairedFingerPusher
    from abstract_stack.release_planning import ReleaseForecast

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        before = integration_state(env.model, env.data)
        ranges = env.model.jnt_range.copy()
        script = StockingScript(
            env, tmp_path, ScriptConfig(ungrasped_insertion=True, paired_finger_insertion=True)
        )
        script.tracker = script._new_tracker()
        script._change_hand(PairedFingerPusher, finger="Middle", tilt_degrees=20, wrist_posture=0.2)
        hand = script.controller
        targets = hand.finger_targets(0)
        assert targets["Left Index MCP"] == targets["Left Ring MCP"] == -0.2
        assert targets["Left Middle MCP"] == targets["Left Pinky MCP"] == 1.5
        assert targets["Left Index Abd"] == 0.15
        assert targets["Left Ring Abd"] == -0.15
        delta = hand.ROTATION @ (hand.native_tip_offset("Ring") - hand.native_tip_offset("Index"))
        assert abs(delta[0]) < 0.002
        assert abs(delta[1]) > 0.015
        assert abs(delta[1]) < 0.048
        hand.open_posture_overrides = {"Left Thumb PIP": -0.2}
        hand.abduction_targets = {"Left Middle Abd": 0.1}
        forecast = ReleaseForecast(script)
        plane_yaw = hand.contact_plane_yaw()
        joint = env.model.joint("Left Index PIP")
        forecast.env.data.qpos[int(joint.qposadr[0])] += 0.05
        import mujoco

        mujoco.mj_forward(env.model, forecast.env.data)
        assert not np.isclose(forecast.controller.contact_plane_yaw(), plane_yaw, atol=1e-5)
        assert hand.contact_plane_yaw() == plane_yaw
        # Restore the independent fixture before the exact clone-state checks.
        mujoco.mj_copyData(forecast.env.data, env.model, env.data)
        assert forecast.controller.open_posture_overrides == hand.open_posture_overrides
        assert forecast.controller.abduction_targets == hand.abduction_targets
        forecast.controller.open_posture_overrides["Left Thumb PIP"] = -0.1
        forecast.controller.abduction_targets["Left Middle Abd"] = 0.2
        assert hand.open_posture_overrides["Left Thumb PIP"] == -0.2
        assert hand.abduction_targets["Left Middle Abd"] == 0.1
        assert forecast.env.model is env.model
        assert forecast.env.data is not env.data
        np.testing.assert_array_equal(integration_state(env.model, forecast.env.data), before)
        forecast.env.data.ctrl[:] += (
            0.001  # Independent scratch fixture, no native step or success claim.
        )
        forecast.controller.command[:] += 0.001
        forecast.controller.OFFSET[:] += 0.001
        np.testing.assert_array_equal(integration_state(env.model, env.data), before)
        np.testing.assert_array_equal(env.model.jnt_range, ranges)
        assert not np.array_equal(forecast.controller.command, hand.command)
        assert not np.array_equal(forecast.controller.OFFSET, hand.OFFSET)
        script._restore_grasp_hand()
        grasp = script.controller
        assert not isinstance(grasp, SharedFingerPusher)
        for name, value in script.grasp_abduction_targets.items():
            assert grasp.finger_targets(0)[name] == value
            assert grasp.finger_targets(0.5)[name] == value
        assert not grasp.open_posture_overrides
        np.testing.assert_array_equal(integration_state(env.model, env.data), before)
    finally:
        env.close()


@pytest.mark.mabel
@pytest.mark.parametrize("finger", ["Index", "Pinky"])
def test_wall_finger_posture_and_headroom_preserve_native_model(finger):
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.config import load_scene_config
    from abstract_stack.contact_rollout import integration_state
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.finger_pusher import WallFingerPusher

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    try:
        before = integration_state(env.model, env.data)
        ranges = env.model.jnt_range.copy()
        hand = WallFingerPusher(
            env.clone_for_planning(),
            ScriptConfig(ungrasped_insertion=True),
            finger=finger,
            wrist_posture=0.2,
        )
        goals = hand.finger_targets(0)
        assert goals[f"Left {finger} MCP"] == -0.2
        assert goals[f"Left {finger} PIP"] == 0.0
        for other in {"Index", "Middle", "Ring", "Pinky"} - {finger}:
            assert goals[f"Left {other} MCP"] == goals[f"Left {other} PIP"] == 1.5
        assert goals["Left Thumb MCP"] == 0.4
        reference = hand.data.qpos.copy()
        for name, value in goals.items():
            reference[int(env.model.joint(name).qposadr[0])] = value
        contacts = hand.ik._contacts(hand.ik.measured_q(), reference, None)
        descendants = {hand.palm}
        for body in range(hand.palm + 1, env.model.nbody):
            if int(env.model.body_parentid[body]) in descendants:
                descendants.add(body)
        self_collisions = [
            info
            for pair, info in contacts.items()
            if all(int(env.model.geom_bodyid[g]) in descendants for g in pair)
            and info["distance_m"] < -0.002
        ]
        assert not self_collisions
        assert np.isfinite(hand.OFFSET).all()
        assert hand.ik.upper[hand.ik.names.index("Left Wrist")] <= 0.324
        np.testing.assert_array_equal(integration_state(env.model, env.data), before)
        np.testing.assert_array_equal(env.model.jnt_range, ranges)
    finally:
        env.close()


def test_wall_packing_uses_measured_neighbor_and_preserves_remaining_row_capacity():
    from abstract_stack.config import load_scene_config
    from abstract_stack.interactions import InteractionFailure
    from abstract_stack.scene import products_for

    script = object.__new__(StockingScript)
    config = load_scene_config("configs/scenes/tight_ten_cartons.json")
    products = products_for(config)
    script.config = ScriptConfig(
        ungrasped_insertion=True, paired_finger_insertion=True, wall_packing=True
    )
    script.product = products[7]
    canonical = script.product.target
    neighbor = SimpleNamespace(xpos=np.array([-0.725, 0.06, 0.92]), xmat=np.eye(3).ravel())
    script.env = SimpleNamespace(
        config=config, products=products, data=SimpleNamespace(body=lambda _: neighbor)
    )
    script.observer = SimpleNamespace(completed={"carton_8", "carton_9"})
    assert script.lateral_goal_y == pytest.approx(0.01)
    assert script.product.target == canonical
    # A leftward disturbance cannot consume room needed by seven later cartons.
    neighbor.xpos[1] = 0.035
    with pytest.raises(InteractionFailure, match="needs rearrangement"):
        _ = script.lateral_goal_y


@pytest.mark.parametrize("direction", [-1, 1])
def test_wall_registration_replans_brace_from_unloaded_staging_pose(monkeypatch, direction):
    from abstract_stack.lateral_contact import LateralStockingScript

    script = script_fixture()
    body = script.env.data.body("active")
    body.xpos[1] = 0.0
    script.env.config.front_x = -0.56
    script.controller = SimpleNamespace(last_target=np.array([-0.5, 0.0, 0.9]))
    script._cartesian_move = lambda *args: None
    goal = 0.051 * direction
    monkeypatch.setattr(StockingScript, "lateral_goal_y", goal)
    calls = []

    def push(script, *, direction, distance_m, brace_wall, wall_staging_y, from_wall_staging, **kw):
        calls.append((direction, distance_m, brace_wall, wall_staging_y, from_wall_staging))
        body.xpos[1] += direction * (distance_m if brace_wall else 0.011)
        return {"direct_wall_exit_verified": False}

    monkeypatch.setattr(LateralStockingScript, "push_laterally", push)
    script._register_front_laterally()
    assert calls[0] == (direction, pytest.approx(0.009), False, goal, False)
    assert calls[1] == (direction, pytest.approx(0.040), True, None, True)
    assert body.xpos[1] == pytest.approx(goal)


def test_dense_reapproach_is_bounded_and_keeps_the_wall_protocol(monkeypatch):
    from abstract_stack.finger_pusher import FoldedFingerPusher
    from abstract_stack.interactions import InteractionFailure

    script = script_fixture()
    script.config = SimpleNamespace(dense_recovery=True, record_skill_starts=False)
    script.phase = "front_registered"
    script.observer.incidents = []
    script.last_forces = {"shelf": 1.0}
    script.events = []
    script._stage = lambda phase: script.events.append({"phase": phase})
    attempts, unloads = [], []

    def advance(*, precontact_standoff_m, kind, reapproach, **_):
        attempts.append((precontact_standoff_m, kind, reapproach))
        if len(attempts) < 3:
            script._stage("finger_push")
            script.events[-1]["applied_tracking_geometry_rejection"] = [{"distance_m": -0.004}]
            raise InteractionFailure("payload_load: compensated arm/finger geometry rejected")

    monkeypatch.setattr(script, "_advance_from_front", advance)
    monkeypatch.setattr(script, "_unload_insertion_recovery", lambda: unloads.append(True))
    script._insert_from_front()
    assert attempts == [(None, None, False), (0.035, None, True), (0.045, FoldedFingerPusher, True)]
    assert len(unloads) == 2

    def force_stop(**_):
        script._stage("finger_push")
        raise InteractionFailure("finger_push: contact force exceeded 25.0 N")

    monkeypatch.setattr(script, "_advance_from_front", force_stop)
    with pytest.raises(InteractionFailure, match="contact force exceeded"):
        script._insert_from_front()
    assert len(unloads) == 2


def test_short_square_handoff_requires_front_depth_for_lateral_registration():
    script = script_fixture()
    script.observer.incidents = []
    script.env.config = SimpleNamespace(center_y=-0.12, width=0.2)
    script.env.products[1].target = (-0.72, -0.173, 0.92)
    body = SimpleNamespace(
        xpos=np.array([-0.625, -0.131, 0.92]),
        xmat=np.eye(3).ravel(),
    )
    assert not script._front_square_handoff_fits(body)
    assert script._front_square_handoff_fits(body, row_front_x=-0.685)
    body.xpos[0] = -0.645
    assert not script._front_square_handoff_fits(body, row_front_x=-0.685)
    body.xpos[1] = -0.122
    assert script._front_square_handoff_fits(body)


def test_provisional_release_has_no_strict_credit_until_the_measured_finish():
    from abstract_stack.evaluation import strict_final_position

    events = [
        {
            "phase": "release",
            "verified_release": True,
            "placement_conditions": True,
            "strict_release_yaw_verified": False,
        },
        {"phase": "dense_position_decision", "provisional": True},
        {"phase": "finger_push", "back_contact_verified": True},
    ]
    assert not strict_final_position(events)
    events.append(
        {
            "phase": "release",
            "verified_release": True,
            "placement_conditions": True,
            "strict_release_yaw_verified": True,
        }
    )
    assert strict_final_position(events)
    events.append({"phase": "dense_final_pose_recheck", "yaw_deg": 0.16, "verified": False})
    assert not strict_final_position(events)
