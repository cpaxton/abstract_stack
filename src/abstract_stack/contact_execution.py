"""Native square-contact dwell and load-aware retreat for a live stocking skill."""

from dataclasses import replace

import numpy as np

from abstract_stack.contact_finish import BackstopDwell
from abstract_stack.contact_refinement import refine_supported_contact
from abstract_stack.contact_rollout import ContactMonitor, RolloutScene
from abstract_stack.withdrawal import supported_withdrawal
from abstract_stack.withdrawal_feedback import WithdrawalFeedbackConfig


def shelf_exit_distance(env, hand):
    """Retain the folded posture until its tool is 80 mm outside the shelf."""
    distance = max(0.04, float(env.config.front_x + 0.08 - hand.position()[0]))
    if distance > 0.25:
        raise ValueError("Shelf exit exceeds the bounded native withdrawal corridor")
    return distance


def hand_clears_shelf(env, hand_bodies):
    """Conservative world-X clearance for all physical hand collision geoms."""
    model, data = env.model, env.data
    geoms = np.flatnonzero(
        np.isin(model.geom_bodyid, list(hand_bodies))
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    return bool(
        len(geoms)
        and np.all(data.geom_xpos[geoms, 0] - model.geom_rbound[geoms] >= env.config.front_x + 0.01)
    )


def sync_tracking_state(script):
    """Acknowledge executed references without relabeling compensated controls."""
    hand, tracker = script.controller, script.tracker
    nominal = dict(zip((*hand.names, "Lift"), hand.command, strict=True))
    for index, name in enumerate(tracker.NAMES):
        if name in nominal:
            tracker.nominal[index] = nominal[name]
    tracker.applied = script.env.data.ctrl[tracker.ids].copy()
    finger_tracker = getattr(hand, "finger_load_compensation", None)
    if finger_tracker is not None:
        finger_tracker.acknowledge(script.env.data.ctrl)
    hand.last_target = hand.ik.forward(hand.command, script.env.data.qpos.copy())[0]
    tracker.last_report = dict(
        nominal=dict(zip(tracker.NAMES, tracker.nominal.tolist(), strict=True)),
        servo_target=dict(zip(tracker.NAMES, tracker.applied.tolist(), strict=True)),
        scope="executed nominal/applied state synchronization; force estimates in skill trace",
    )


def make_backstop_monitor(script):
    """Use the same native contact criteria for standalone and composed refinement."""
    env, config = script.env, script.config
    scene = replace(RolloutScene.from_env(env, script.product.name), force_limit_N=config.max_force)
    monitor = ContactMonitor(env.model, env.data, scene)
    dwell = BackstopDwell(
        env.config.timestep,
        config.contact_threshold,
        config.contact_dwell,
        config.payload_upright_degrees,
        0.15,
    )
    return monitor, dwell


def squared_contact_retreat(script, push_event, *, unloaded_entry=False):
    """Verify a fresh 2 ms dwell, then reuse the tested native withdrawal skill."""
    from abstract_stack.interactions import InteractionFailure

    env, hand, config = script.env, script.controller, script.config
    monitor, dwell = make_backstop_monitor(script)
    scene = monitor.scene
    from abstract_stack.contact_rollout import integration_state
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.release_planning import (
        ReleaseForecast,
        choose_release_reference,
        choose_unload_feedback,
    )

    unloaded = False
    error = None
    if unloaded_entry:
        error = InteractionFailure("unloaded_clear: resumed from a measured failure state")
    else:
        try:
            refine_supported_contact(script, monitor, dwell)
        except InteractionFailure as failure:
            if str(failure) not in (
                "contact_refine: no fresh guarded candidate",
                "contact_refine: bounded release-goal search exhausted",
            ):
                raise
            error = failure
    coarse_loaded = False
    if error is not None and config.live_guarded_withdrawal and not unloaded_entry:
        candidate_monitor = ContactMonitor(env.model, env.data, scene)
        try:
            candidate_monitor.begin_coarse_loaded_withdrawal(env.data)
        except ValueError:
            pass
        else:
            if all(
                ok
                for p, ok in zip(env.products, env._placement_conditions(), strict=True)
                if p.name in script.observer.completed | {script.product.name}
            ):
                script.events[-1]["coarse_loaded_exit_after"] = str(error)
                monitor = ContactMonitor(env.model, env.data, scene)
                coarse_loaded = True
                error = None
    if error is not None:
        env = script.env
        required = script.observer.completed | {script.product.name}
        if not all(
            ok
            for product, ok in zip(env.products, env._placement_conditions(), strict=True)
            if product.name in required
        ):
            raise error
        monitor = ContactMonitor(env.model, env.data, scene)
        try:
            monitor.begin_unloaded_withdrawal(env.data)
        except ValueError:
            raise error from None
        script._stage("unloaded_clear_forecast")
        forecast_event = script.events[-1]
        state = integration_state(env.model, env.data)
        attempts = []
        nominal_exit = shelf_exit_distance(env, hand)
        shorter_exit = max(0.04, nominal_exit - 0.035)
        for straight_first, clear_distance in (
            (True, nominal_exit),
            (False, nominal_exit),
            (False, shorter_exit),
        ):
            forecast = ReleaseForecast(script)
            guarded = ContactMonitor(forecast.env.model, forecast.env.data, scene)
            guarded.begin_unloaded_withdrawal(forecast.env.data)
            report, _, _ = supported_withdrawal(
                forecast.controller,
                forecast.controller.command.copy(),
                guarded,
                None,
                resume_withdrawal=True,
                clear_distance_m=clear_distance,
                clear_speed_m_s=0.02,
                feedback=WithdrawalFeedbackConfig(
                    yaw_feedback=True, straight_retreat_first=straight_first
                ),
            )
            terminal_placement = ShelfStockingEnv._placement_conditions(forecast.env)
            hand_clear = hand_clears_shelf(forecast.env, script.hand_bodies)
            admitted = (
                report["release_retreat_verified"]
                and hand_clear
                and all(
                    ok
                    for p, ok in zip(env.products, terminal_placement, strict=True)
                    if p.name in required
                )
            )
            attempts.append(
                dict(
                    straight_retreat_first=straight_first,
                    clear_distance_m=clear_distance,
                    hand_clear=hand_clear,
                    admitted=admitted,
                    forecast=report,
                )
            )
            if not np.array_equal(state, integration_state(env.model, env.data)):
                raise InteractionFailure("unloaded_clear: forecast changed the live state")
            if admitted:
                break
        forecast_event.update(
            refinement_failure=str(error), candidates=attempts, fresh_task_credit=0
        )
        if not admitted:
            raise InteractionFailure("unloaded_clear: no verified withdrawal forecast") from error
        monitor = ContactMonitor(env.model, env.data, scene)
        monitor.begin_unloaded_withdrawal(env.data)
        unloaded = True

    if unloaded:
        release_plan = dict(
            yaw_reference_deg=0.0,
            normal_target_N=None,
            mode="already_unloaded",
            straight_retreat_first=straight_first,
            clear_distance_m=clear_distance,
        )
    elif config.live_guarded_withdrawal:
        release_plan = dict(yaw_reference_deg=0.0, normal_target_N=None, mode="live_guards")
    else:
        release_plan = choose_release_reference(script, monitor, dwell)
    if release_plan["yaw_reference_deg"] or release_plan["normal_target_N"] is not None:
        refine_supported_contact(
            script,
            monitor,
            dwell,
            yaw_target_deg=release_plan["yaw_reference_deg"],
            normal_target_N=release_plan["normal_target_N"],
        )
    if unloaded:
        push_event["unloaded_placement_clear"] = True
    elif not coarse_loaded:
        push_event["back_contact_verified"] = True
    script.supported_slide = False
    neighbors = (
        tuple(item["product"] for item in script._adjacent_placed_neighbors())
        if config.wall_packing and not unloaded
        else ()
    )
    release_yaw_limit = 1.0 if config.live_guarded_withdrawal and not unloaded else 0.15
    exit_distance = release_plan.get("clear_distance_m", shelf_exit_distance(env, hand))
    feedback = WithdrawalFeedbackConfig(
        yaw_feedback=True,
        yaw_reference_deg=release_plan["yaw_reference_deg"],
        straight_retreat_first=release_plan.get("straight_retreat_first", True),
        paired_plane_feedback=bool(getattr(hand, "fingers", ())),
    )
    if config.screen_supported_unload and neighbors and feedback.paired_plane_feedback:
        feedback = choose_unload_feedback(
            script,
            monitor,
            dwell,
            feedback,
            clear_distance_m=exit_distance,
            contact_neighbors=neighbors,
            release_yaw_limit_deg=release_yaw_limit,
            coarse_loaded_entry=coarse_loaded,
        )
    script.observer.begin_retreat(script.product.name)
    script._stage("release")
    release_event = script.events[-1]
    release_event["forecast_selected_reference"] = release_plan

    def withdrawal_sample(current):
        current.update_task_state(current.config.timestep)
        sync_tracking_state(script)
        script._sample(current)
        if current.data.time >= current.config.time_limit:
            raise InteractionFailure("release: episode time limit")

    report, _, _ = supported_withdrawal(
        hand,
        hand.command.copy(),
        monitor,
        dwell,
        resume_withdrawal=unloaded,
        coarse_loaded_entry=coarse_loaded,
        release_yaw_limit_deg=release_yaw_limit,
        clear_distance_m=exit_distance,
        clear_speed_m_s=0.02,
        feedback=feedback,
        on_physics_step=withdrawal_sample,
        contact_neighbors=neighbors,
    )
    sync_tracking_state(script)
    release_event["supported_withdrawal"] = report
    if not report["release_retreat_verified"]:
        raise InteractionFailure(f"release: {report['failure']}")
    script.wrist_load = 0.0
    index = next(i for i, p in enumerate(env.products) if p.name == script.product.name)
    release_event.update(
        verified_release=True,
        strict_release_yaw_verified=report["strict_release_yaw_verified"],
        placement_conditions=bool(env._placement_conditions()[index]),
        support_surface="shelf",
        fresh_task_credit=0,
    )
    if unloaded and not hand_clears_shelf(env, script.hand_bodies):
        raise InteractionFailure("unloaded_clear: measured hand did not clear shelf")
    script.observer.end_retreat(script.product.name)
    if (unloaded or config.live_guarded_withdrawal) and (
        not script.observer.retreat_results[script.product.name]["verified"]
    ):
        raise InteractionFailure("release: live retreat did not preserve stable placement")
