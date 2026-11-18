"""Native square-contact dwell and load-aware retreat for a live stocking skill."""

from dataclasses import replace

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


def sync_tracking_state(script):
    """Acknowledge executed references without relabeling compensated controls."""
    hand, tracker = script.controller, script.tracker
    nominal = dict(zip((*hand.names, "Lift"), hand.command, strict=True))
    for index, name in enumerate(tracker.NAMES):
        if name in nominal:
            tracker.nominal[index] = nominal[name]
    tracker.applied = script.env.data.ctrl[tracker.ids].copy()
    hand.last_target = hand.ik.forward(hand.command, script.env.data.qpos.copy())[0]
    tracker.last_report = dict(
        nominal=dict(zip(tracker.NAMES, tracker.nominal.tolist(), strict=True)),
        servo_target=dict(zip(tracker.NAMES, tracker.applied.tolist(), strict=True)),
        scope="executed nominal/applied state synchronization; force estimates in skill trace",
    )


def squared_contact_retreat(script, push_event):
    """Verify a fresh 2 ms dwell, then reuse the tested native withdrawal skill."""
    from abstract_stack.interactions import InteractionFailure

    env, hand, config = script.env, script.controller, script.config
    scene = replace(RolloutScene.from_env(env, script.product.name), force_limit_N=config.max_force)
    monitor = ContactMonitor(env.model, env.data, scene)
    dwell = BackstopDwell(
        env.config.timestep,
        config.contact_threshold,
        config.contact_dwell,
        config.payload_upright_degrees,
        0.15,
    )
    refine_supported_contact(script, monitor, dwell)
    from abstract_stack.release_planning import choose_release_reference

    release_plan = choose_release_reference(script, monitor, dwell)
    if release_plan["yaw_reference_deg"] or release_plan["normal_target_N"] is not None:
        refine_supported_contact(
            script,
            monitor,
            dwell,
            yaw_target_deg=release_plan["yaw_reference_deg"],
            normal_target_N=release_plan["normal_target_N"],
        )
    push_event["back_contact_verified"] = True
    script.supported_slide = False
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
        clear_distance_m=shelf_exit_distance(env, hand),
        clear_speed_m_s=0.02,
        feedback=WithdrawalFeedbackConfig(
            yaw_feedback=True,
            yaw_reference_deg=release_plan["yaw_reference_deg"],
            paired_plane_feedback=bool(getattr(hand, "fingers", ())),
        ),
        on_physics_step=withdrawal_sample,
    )
    sync_tracking_state(script)
    release_event["supported_withdrawal"] = report
    if not report["release_retreat_verified"]:
        raise InteractionFailure(f"release: {report['failure']}")
    script.wrist_load = 0.0
    index = next(i for i, p in enumerate(env.products) if p.name == script.product.name)
    release_event.update(
        verified_release=True,
        placement_conditions=bool(env._placement_conditions()[index]),
        support_surface="shelf",
        fresh_task_credit=0,
    )
    script.observer.end_retreat(script.product.name)
