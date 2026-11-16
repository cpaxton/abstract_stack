"""Measured forces, observer cadence, and physical controller evidence."""

import json

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.ik import rotation_error
from abstract_stack.interactions import (
    CartesianHand,
    InteractionFailure,
    ScriptConfig,
    StockingScript,
    contact_forces,
)
from abstract_stack.random_scenes import random_shelf_config


def test_contact_forces_measure_weight_and_physics_observer_cadence():
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"), robot=False)
    try:
        times = []
        env.step(on_physics_step=lambda state: times.append(float(state.data.time)))
        assert len(times) == env.config.frame_skip
        np.testing.assert_allclose(np.diff(times), env.config.timestep)
        for _ in range(30):
            env.step()
        force = contact_forces(env, "carton_0", set())
        assert force["table"] == pytest.approx(env.products[0].mass * 9.81, abs=0.02)
        assert force["shelf"] == force["back"] == force["hand"] == 0
        assert not force["opposing_contacts"]
    finally:
        env.close()


@pytest.fixture
def robot():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    yield env
    env.close()


@pytest.mark.mabel
def test_ik_changes_only_scratch_state_and_bounds_targets(robot):
    controller = CartesianHand(robot, ScriptConfig())
    before_qpos, before_ctrl = robot.data.qpos.copy(), robot.data.ctrl.copy()
    action = controller.targets(controller.position() + [0.02, 0, 0.03], 0.2)
    np.testing.assert_array_equal(robot.data.qpos, before_qpos)
    np.testing.assert_array_equal(robot.data.ctrl, before_ctrl)
    for name in controller.names:
        assert abs(action[name] - before_ctrl[robot.model.actuator(name).id]) <= robot.dt + 1e-9
    assert abs(action["Lift"] - before_ctrl[robot.model.actuator("Lift").id]) <= 0.08 * robot.dt
    prediction = controller.predict(controller.position(), check_path=False)
    clearance, normal = controller.tower_clearance(prediction, 0)
    assert np.isfinite(clearance)
    assert np.linalg.norm(normal) == pytest.approx(1)
    np.testing.assert_array_equal(robot.data.qpos, before_qpos)
    np.testing.assert_array_equal(robot.data.ctrl, before_ctrl)


@pytest.mark.mabel
@pytest.mark.parametrize("product_name", ["carton_0", "carton_1"])
def test_force_limit_stops_on_real_hand_contact(robot, tmp_path, product_name):
    script = StockingScript(
        robot,
        tmp_path,
        ScriptConfig(
            max_force=0.5,
            grasp_force_target=0.25,
            push_force_target=0.25,
            approach_force_target=0.25,
            fixture_force_target=0.25,
        ),
    )
    # Initial-condition fixture only: deliberately penetrate the hand to exercise
    # the force guard. No live rollout succeeds from this invalid arrangement.
    joint = robot.model.joint(f"{product_name}_joint")
    qadr = int(joint.qposadr[0])
    robot.data.qpos[qadr : qadr + 3] = robot.data.body("left_palm").xpos
    mujoco.mj_forward(robot.model, robot.data)
    with pytest.raises(InteractionFailure, match="contact force"):
        robot.step(on_physics_step=script._sample)
    assert len(script.rows) > 0
    assert script.last_forces["hand" if product_name == "carton_0" else "other_product"] > 0.5


@pytest.mark.mabel
def test_force_guard_includes_non_grasp_links_on_active_carton(robot, tmp_path):
    script = StockingScript(
        robot,
        tmp_path,
        ScriptConfig(
            max_force=0.5,
            grasp_force_target=0.25,
            push_force_target=0.25,
            approach_force_target=0.25,
            fixture_force_target=0.25,
        ),
    )
    # Invalid initial-contact fixture only, never a successful manipulation.
    adr = int(robot.model.joint("carton_0_joint").qposadr[0])
    robot.data.qpos[adr : adr + 3] = robot.data.body("left_tower").xpos
    mujoco.mj_forward(robot.model, robot.data)
    with pytest.raises(InteractionFailure, match="contact force"):
        robot.step(on_physics_step=script._sample)
    force = script.last_forces
    assert force["arm_product"] > 0.5
    assert force["robot_product"] == pytest.approx(force["hand"] + force["arm_product"])


@pytest.mark.mabel
@pytest.mark.parametrize("excursion", [[0.0302, 0, 0], [0, 0, 0.026]])
def test_final_push_bound_is_checked_when_no_clearance_projection_is_needed(
    robot, tmp_path, excursion
):
    script = StockingScript(robot, tmp_path, ScriptConfig(fixture_avoidance=True))
    point = script.controller.position().copy()
    script.controller.active_rotation = robot.data.xmat[script.controller.palm].reshape(3, 3).copy()
    script.payload_rotation_goal = robot.data.body(script.product.name).xmat.reshape(3, 3).copy()
    script.phase = "push"
    script.events = [{"phase": "push", "push_origin_m": (point - np.array(excursion)).tolist()}]
    qpos, ctrl, time = robot.data.qpos.copy(), robot.data.ctrl.copy(), float(robot.data.time)
    with pytest.raises(InteractionFailure, match="final feedback target outside bounded motion"):
        script._tick(point, 0.1)
    np.testing.assert_array_equal(robot.data.qpos, qpos)
    np.testing.assert_array_equal(robot.data.ctrl, ctrl)
    assert robot.data.time == time


@pytest.mark.mabel
def test_script_writes_honest_physics_evidence(robot, tmp_path):
    script = StockingScript(robot, tmp_path, ScriptConfig(adaptive_grasps=False))
    result = script.run()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["physical_success"] == result["physical_success"]
    assert saved["assistance"] == "none"
    assert saved["events"]
    if not result["physical_success"]:
        assert result["failure"]
    trace = np.load(tmp_path / "trace.npz")
    assert trace["samples"].shape[1] == len(result["trace_columns"])
    assert trace["controls"].shape == (len(trace["samples"]), robot.model.nu)
    assert np.isfinite(trace["samples"]).all()
    assert (np.diff(trace["samples"][:, 0]) > 0).all()
    assert robot.info()["intervention_count"] == 0
    assert not result["physical_success"]
    assert result["events"][0]["phase"] == "grasp_precheck"
    assert result["events"][0]["decision"] == "reject_grasp_candidate"
    assert not any(e["phase"] == "close" for e in result["events"])
    assert not result["metrics"]["tipped"]
    assert not result["metrics"]["dropped"]
    stage = result["events"][-1]
    assert stage["decision"] == "bounded_staging_then_replan"
    assert "replan required" in result["failure"]
    assert stage["prediction_execution_gap_m"] < 0.035
    assert stage["requested_position_error_m"] > 0.1
    assert not stage["staging_prediction"]["allowed_target_hand_contact"]
    assert result["last_ik_prediction"] == stage["staging_prediction"]
    components = stage["execution_diagnostics"]
    displacement = sum(np.array(v) for k, v in components.items() if k.endswith("xyz_m"))
    np.testing.assert_allclose(
        displacement,
        np.array(stage["measured_position"])
        - np.array(stage["staging_prediction"]["predicted_position"]),
        atol=1e-12,
    )


@pytest.mark.mabel
def test_prediction_matches_forward_kinematics_without_changing_live_state(robot):
    controller = CartesianHand(robot, ScriptConfig())
    state = (
        robot.data.qpos.copy(),
        robot.data.qvel.copy(),
        robot.data.ctrl.copy(),
        robot.data.time,
    )
    point, rotation = controller.ik.forward(controller.ik.measured_q())
    prediction = controller.ik.predict(point, rotation, controller.command)
    assert prediction.feasible
    assert prediction.path_checked and prediction.safe
    predicted_point, predicted_rotation = controller.ik.forward(
        prediction.q, prediction.reference_qpos
    )
    assert prediction.position_error_m == pytest.approx(np.linalg.norm(predicted_point - point))
    assert prediction.orientation_error_rad == pytest.approx(
        np.linalg.norm(rotation_error(predicted_rotation, prediction.requested_rotation))
    )
    for actual, previous in zip(
        (robot.data.qpos, robot.data.qvel, robot.data.ctrl, robot.data.time), state, strict=True
    ):
        np.testing.assert_array_equal(actual, previous)


@pytest.mark.mabel
def test_orientation_candidates_can_resolve_a_rejected_fixed_pose(robot):
    controller = CartesianHand(robot, ScriptConfig())
    # Explicit restricted-lift planning fixture. Full native lift travel can
    # make both rotations reachable, so it would not exercise this fallback.
    controller.ik.lower[-1], controller.ik.upper[-1] = 0.18, 0.55
    point, rotation = controller.ik.forward(controller.ik.measured_q())
    preferred = dict(controller.ik.orientations(rotation, 40))["axis1:40deg"]
    fixed = controller.ik.predict(
        point, preferred, controller.command, alternatives=False, check_path=False
    )
    relaxed = controller.ik.predict(point, preferred, controller.command, check_path=False)
    assert not fixed.feasible
    assert relaxed.feasible
    assert relaxed.position_error_m < fixed.position_error_m
    assert relaxed.preferred_orientation_error_rad <= np.radians(52)
    assert not relaxed.path_checked
    assert not relaxed.report()["safe_sampled_path"]


@pytest.mark.mabel
def test_unreachable_target_reports_remaining_error_and_bounds(robot):
    controller = CartesianHand(robot, ScriptConfig())
    command = controller.command.copy()
    prediction = controller.predict([5, -5, 5])
    assert not prediction.feasible
    assert prediction.position_error_m > 5
    assert (prediction.q >= controller.lower).all()
    assert (prediction.q <= controller.upper).all()
    report = prediction.report()
    assert report["status"] == "no_acceptable_solution_found"
    assert "position_tolerance" in report["rejection_reasons"]
    assert np.linalg.norm(report["predicted_position_error_xyz_m"]) == pytest.approx(
        report["predicted_position_error_m"]
    )
    np.testing.assert_array_equal(controller.command, command)
    with pytest.raises(InteractionFailure, match="before actuation"):
        controller.targets([5, -5, 5], 0.8)
    np.testing.assert_array_equal(controller.command, command)


@pytest.mark.mabel
def test_collision_rejection_and_bounded_staging(robot):
    controller = CartesianHand(robot, ScriptConfig())
    target = robot.data.body("carton_0").xpos.copy()
    prediction = controller.predict(target)
    assert not prediction.feasible
    assert prediction.blocked_by or prediction.collisions or prediction.path_collisions
    stage = controller.staging_pose(prediction)
    assert stage is not None and not stage.feasible
    start, _ = controller.ik.forward(controller.ik.measured_q(), stage.reference_qpos)
    assert 0.005 < np.linalg.norm(stage.position - start) <= 0.10
    assert stage.position_error_m < np.linalg.norm(start - target)
    assert stage.position_error_m == pytest.approx(np.linalg.norm(stage.position - target))
    assert not controller.ik.check_path(stage.q, stage.reference_qpos, 1.0)
    assert controller.staging_pose(controller.predict(target, check_path=False)) is None


@pytest.mark.mabel
def test_random_two_carton_scene_completes_all_physical_gates(tmp_path):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(random_shelf_config(0, product_count=2), seed=0)
    try:
        result = StockingScript(env, tmp_path).run()
        assert result["physical_success"], result["failure"]
        assert set(result["completed_products"]) == {p.name for p in env.products}
        assert result["metrics"]["stocked_count"] == 2
        assert not result["metrics"]["tipped"]
        assert not result["metrics"]["dropped"]
        assert result["metrics"]["intervention_count"] == 0
        assert result["stage_evaluation"]["no_knockovers"]
        assert all(
            p[k]
            for p in result["stage_evaluation"]["products"].values()
            for k in ("pick", "place", "final_position", "retreat")
        )
        for product in env.products:
            events = [e for e in result["events"] if e["product"] == product.name]
            assert any(e.get("verified_lift") for e in events)
            assert any(e.get("verified_release") for e in events)
            pushes = [e for e in events if e.get("back_contact_verified")]
            assert pushes and pushes[0]["ik_prediction"]["sampled_path_model"] == "cartesian"
            assert pushes[0]["feedback_checks"] > 0
        assert np.all(env.data.qfrc_applied == 0)
        assert np.all(env.data.xfrc_applied == 0)
    finally:
        env.close()


@pytest.mark.mabel
def test_adaptive_solver_verifies_a_physical_lift(robot, tmp_path):
    result = StockingScript(robot, tmp_path).run()
    lifts = [e for e in result["events"] if e.get("verified_lift")]
    assert lifts
    assert lifts[0]["object_rise_m"] > 0.045
    assert lifts[0]["measured_table_normal_N"] < 0.15
    assert lifts[0]["measured_hand_normal_N"] > 0.2
    carries = [e for e in result["events"] if e["phase"] in ("lift", "transfer", "place")]
    assert all(e["ik_prediction"]["carried_object_swept_volume"] for e in carries)
    if not result["physical_success"]:
        assert result["failure"]
    assert robot.info()["intervention_count"] == 0


@pytest.mark.mabel
def test_grasp_selection_changes_object_instead_of_repeating_bad_pose(robot, tmp_path):
    script = StockingScript(robot, tmp_path)
    before = robot.data.qpos.copy()
    target, prediction = script._select_grasp(list(robot.products))
    assert script.product.name == "carton_1"
    assert prediction.feasible
    assert prediction.position_error_m < 0.015
    box = robot.data.body("carton_1").xpos
    np.testing.assert_allclose(target[:2], box[:2])
    assert 0 <= target[2] - box[2] <= robot.products[1].half_size[2]
    np.testing.assert_array_equal(robot.data.qpos, before)


@pytest.mark.mabel
def test_hold_mode_rejects_before_commanding_motion(robot, tmp_path):
    script = StockingScript(
        robot, tmp_path, ScriptConfig(best_effort_staging=False, adaptive_grasps=False)
    )
    result = script.run()
    assert not result["physical_success"]
    assert result["events"][-1]["decision"] == "hold_for_replan"
    assert len(script.rows) == 0
    assert robot.data.time == pytest.approx(1.0)
    assert "replan required" in result["failure"]


@pytest.mark.mabel
@pytest.mark.parametrize("target,rotation", [([np.nan, 0, 0], np.eye(3)), ([0, 0, 1], -np.eye(3))])
def test_invalid_pose_rejected(robot, target, rotation):
    controller = CartesianHand(robot, ScriptConfig())
    before = robot.data.qpos.copy()
    with pytest.raises(ValueError):
        controller.ik.predict(target, rotation, controller.command)
    np.testing.assert_array_equal(robot.data.qpos, before)


def test_rotation_error_handles_half_turn():
    error = rotation_error(np.eye(3), np.diag([1.0, -1.0, -1.0]))
    assert np.linalg.norm(error) == pytest.approx(np.pi)


@pytest.mark.parametrize("values", [{"grip": 2}, {"max_force": 0}, {"lift_height": -0.1}])
def test_bad_script_settings(values):
    with pytest.raises(ValueError):
        ScriptConfig(**values)
