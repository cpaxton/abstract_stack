"""A stale/disallowed action must never dispatch just because its suffix is cheap."""

from dataclasses import replace

import pytest

from abstract_stack.replanning import (
    ActionCheck,
    CartonEstimate,
    RowObservation,
    RowReplanner,
    SearchConfig,
)


def observation(count=3):
    cartons = tuple(
        CartonEstimate(
            f"p{i}",
            (-0.3, -0.1 * i, 0.8),
            (0.04, 0.03, 0.08),
            (-0.8, -0.1 * i, 0.8),
            (0.04, 0.03, 0.08),
        )
        for i in range(count)
    )
    return RowObservation(cartons, tuple(c.name for c in cartons), (0.0, 0.0, 1.0))


@pytest.mark.mabel
def test_grasp_wrist_posture_is_reinstated_after_pushing_tool_change(tmp_path, monkeypatch):
    import numpy as np

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.dense_interactions import FingerPusher
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import CartesianHand, InteractionFailure
    from abstract_stack.mabel_controller import MabelController
    from abstract_stack.packing import dense_shelf_config
    from abstract_stack.receding_stocking import RecedingStockingScript

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0))
    try:
        script = RecedingStockingScript(env, tmp_path, grasp_wrist_posture=0.0)
        script.tracker = MabelController(env.model, env.data)
        before = env.data.qpos.copy(), env.data.ctrl.copy()
        index = script.controller.ik.names.index("Left Wrist")
        assert script.controller.ik.posture_constraints[index] == 0
        script._change_hand(FingerPusher, wrist_posture=0.4)
        assert script.controller.ik.posture_constraints[index] == 0.4
        script._change_hand(CartesianHand)
        assert script.controller.ik.posture_constraints[index] == 0
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
        # run() recreates the hand after bundle settling. Stop at its first
        # planning boundary: no manipulation can conceal a dropped constraint.
        monkeypatch.setattr("abstract_stack.bundle.build_scene_bundle", lambda *a, **k: None)
        checked = []

        def stop(pending):
            checked.append(script.controller.ik.posture_constraints[index])
            raise InteractionFailure("unit fixture stops before first motion")

        monkeypatch.setattr(script, "_select_grasp", stop)
        result = script.run()
        assert checked == [0.0]
        assert result["failure"] == "unit fixture stops before first motion"
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
    finally:
        env.close()


def test_reject_cheap_root_and_check_other_complete_order():
    state = observation()
    attempted = []

    def check(name):
        attempted.append(name)
        return ActionCheck(name != "p0", "blocked" if name == "p0" else "accepted")

    planner = RowReplanner()
    result = planner.plan(state, check)
    assert result.first_action_checked
    assert set(result.order) == set(state.pending)
    assert result.order[0] != "p0"
    assert "p0" in attempted
    assert len(attempted) == len(set(attempted))


def test_reuse_tail_rechecks_continuous_state_and_omits_completed_carton():
    planner = RowReplanner()
    old = observation()
    first = planner.plan(old, lambda _: ActionCheck(True, "accepted"))
    remaining = tuple(name for name in old.pending if name != first.order[0])
    moved = replace(old.cartons[0], position=(-0.75, 0.01, 0.8))
    new = replace(old, pending=remaining, cartons=(moved, *old.cartons[1:]))
    checked = []

    def check(name):
        checked.append(name)
        return ActionCheck(True, "fresh_result")

    second = planner.plan(new, check)
    assert second.reused_tail
    assert second.order == first.order[1:]
    assert checked == [second.order[0]]
    assert second.observation_fingerprint != first.observation_fingerprint
    assert second.expansions == 0


def test_invalidated_tail_reorders_instead_of_reusing_cached_success():
    planner = RowReplanner()
    state = observation()
    first = planner.plan(state, lambda _: ActionCheck(True, "accepted"))
    failed = first.order[0]
    second = planner.plan(state, lambda name: ActionCheck(name != failed, "new_obstacle"))
    assert not second.reused_tail
    assert second.first_action_checked
    assert second.order[0] != failed
    assert second.reuse_rejection == "new_obstacle"
    # Rejection is local to this observation, never a permanent blacklist.
    planner.previous_order = ()
    third = planner.plan(state, lambda _: ActionCheck(True, "accepted_again"))
    assert third.order[0] == failed


def test_no_candidate_and_budget_exhaustion_do_not_dispatch():
    state = observation()
    result = RowReplanner().plan(state, lambda _: ActionCheck(False, "collision"))
    assert not result.first_action_checked
    assert not result.order
    assert len(result.root_checks) == 3
    assert result.stop_reason == "no_checked_root"
    limited = RowReplanner(SearchConfig(max_expansions=1))
    result = limited.plan(state, lambda _: pytest.fail("Incomplete suffix must not be checked"))
    assert not result.first_action_checked
    assert result.stop_reason == "search_budget"


def test_ten_carton_search_is_bounded_and_deterministic():
    state = observation(10)
    plans = [RowReplanner().plan(state, lambda _: ActionCheck(True, "accepted")) for _ in range(2)]
    assert plans[0].order == plans[1].order
    assert len(plans[0].order) == 10
    assert plans[0].expansions <= SearchConfig().max_expansions
    assert plans[0].first_action_checked


def test_root_check_budget_is_explicit():
    result = RowReplanner(SearchConfig(max_root_checks=1)).plan(
        observation(), lambda _: ActionCheck(False, "blocked")
    )
    assert result.stop_reason == "root_check_budget"
    assert len(result.root_checks) == 1


def test_empty_task_and_invalid_observations():
    state = observation()
    result = RowReplanner().plan(replace(state, pending=()), lambda _: pytest.fail("No pending"))
    assert result.stop_reason == "no_pending_cartons"
    with pytest.raises(ValueError):
        replace(state, pending=("absent",))
    with pytest.raises(ValueError):
        replace(state, pending=("p0", "p0"))
    with pytest.raises(ValueError):
        SearchConfig(wall_seconds=float("nan"))


@pytest.mark.mabel
def test_native_replan_does_not_move_live_world_or_reuse_predictions(tmp_path):
    import numpy as np

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config
    from abstract_stack.receding_stocking import RecedingStockingScript

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0), seed=0)
    try:
        solver = RecedingStockingScript(env, tmp_path)
        before = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env.data.time
        first = solver._select_grasp(list(env.products))
        second = solver._select_grasp(list(env.products))
        assert first[1] is not second[1]
        assert solver.order_plans[-1]["reused_tail"]
        assert len(solver.order_plans[-1]["order"]) == 10
        for current, old in zip(
            (env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time), before, strict=True
        ):
            np.testing.assert_array_equal(current, old)
        assert not np.any(env.data.qfrc_applied)
        assert not np.any(env.data.xfrc_applied)
    finally:
        env.close()


@pytest.mark.mabel
def test_empty_base_action_preserves_task_observer_and_native_force_contract(tmp_path):
    import numpy as np

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.evaluation import StageObserver
    from abstract_stack.mabel_controller import MabelController
    from abstract_stack.packing import dense_shelf_config
    from abstract_stack.receding_stocking import RecedingStockingScript

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, initially_stocked=9), seed=0)
    try:
        for _ in range(50):
            env.step()
        script = RecedingStockingScript(env, tmp_path, loaded_route=True, base_reposition=True)
        script.tracker = MabelController(env.model, env.data)
        script.observer = StageObserver(env)
        observer = script.observer
        before = env.data.body("mabel_chassis").xpos.copy()
        nominal = script.tracker.nominal.copy()
        script._move_base([0, 0.1], reason="reset_fixture_contract_test_no_stocking_credit")
        assert script.observer is observer
        assert not observer.incidents
        assert not env.interventions
        assert script.base_repositions[-1]["verified_base_reposition"]
        assert (
            np.linalg.norm(env.data.body("mabel_chassis").xpos[:2] - before[:2] - [0, 0.1]) < 0.002
        )
        np.testing.assert_array_equal(script.tracker.nominal, nominal)
        assert not np.any(env.data.qfrc_applied)
        assert not np.any(env.data.xfrc_applied)
        assert script._base_guard is None
    finally:
        env.close()


@pytest.mark.mabel
def test_contact_feedback_returns_the_rate_bounded_nominal_fk_without_live_changes(tmp_path):
    import numpy as np

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config
    from abstract_stack.receding_stocking import RecedingStockingScript

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=2))
    try:
        script = RecedingStockingScript(env, tmp_path, bounded_contact_ik=True)
        hand = script.controller
        hand.active_rotation = env.data.xmat[hand.palm].reshape(3, 3).copy()
        script._stage("bounded_feedback_contract")
        before = env.data.qpos.copy(), env.data.ctrl.copy(), hand.command.copy()
        prediction = script._contact_prediction(
            hand.position() + [0.0005, 0, 0],
            alternatives=False,
            carrying=False,
        )
        rates = np.r_[np.ones(len(hand.names)), 0.08]
        assert np.all(np.abs(prediction.q - before[2]) <= rates * env.dt + 1e-10)
        np.testing.assert_allclose(
            hand.ik.forward(prediction.q, prediction.reference_qpos)[0],
            prediction.position,
            atol=1e-12,
        )
        assert hand.last_prediction is prediction
        assert prediction.bounded_feedback["solver"]["projected_gradient_inf"] < 1e-6
        np.testing.assert_array_equal(env.data.qpos, before[0])
        np.testing.assert_array_equal(env.data.ctrl, before[1])
        np.testing.assert_array_equal(hand.command, before[2])
    finally:
        env.close()
