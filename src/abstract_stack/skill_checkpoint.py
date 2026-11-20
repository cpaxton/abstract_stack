"""Restore the measured handoff state needed by an isolated native skill."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from abstract_stack.finger_pusher import (
    FingerPusher,
    FoldedFingerPusher,
    PairedFingerPusher,
    WallFingerPusher,
)
from abstract_stack.interactions import ScriptConfig, contact_forces
from abstract_stack.lateral_contact import SideFingerPusher
from abstract_stack.mabel_controller import MabelController, TrackingConfig

PUSHER_TYPES = {
    kind.__name__: kind
    for kind in (
        FingerPusher,
        FoldedFingerPusher,
        PairedFingerPusher,
        WallFingerPusher,
        SideFingerPusher,
    )
}


def restore_skill_controller(script, checkpoint, *, replan_released_tool=False):
    """Restore recorded references on a private native state before replanning."""
    state, context = checkpoint.controller, checkpoint.context
    env = script.env
    if state.get("kind") != "stocking_skill_entry" or not context.get("observer_history"):
        raise ValueError("A captured reset requires a stocking skill entry with observer history")
    if not np.array_equal(state["applied_controls"], env.data.ctrl):
        raise ValueError("Recorded applied controls differ from native controls")
    script.product = next(
        p for p in env.products if p.name == context["skill_entry"]["active_product"]
    )
    if context["skill_entry"].get("phase") == "front_squaring_complete":
        # Route admission depends on whether the pusher is starting after a square.
        script.phase = "front_squaring_complete"
    legacy = "tool_settings" not in state or "grasp_abduction_targets" not in state
    script.grasp_abduction_targets = dict(
        state.get("grasp_abduction_targets", script.grasp_abduction_targets)
    )
    tool_config = replace(
        ScriptConfig(**state.get("tool_settings", state["settings"])), record_skill_starts=False
    )
    tool = state["tool_class"]
    parameters = state.get("tool_parameters")
    if tool in PUSHER_TYPES and not parameters:
        parameters = next(
            (
                event["selected"]
                for event in reversed(context.get("prior_events", ()))
                if event["phase"] == "pusher_plan" and event.get("selected")
            ),
            None,
        )
        legacy = True
    if replan_released_tool:
        if state["active_grip"] is not None or state["carry_reference"] is not None:
            raise ValueError("Tool replanning requires a released carton")
        hand = script._new_cartesian_hand()
    elif tool == "CartesianHand":
        hand = script._new_cartesian_hand()
        hand.config = tool_config
    elif tool in PUSHER_TYPES and parameters:
        hand = PUSHER_TYPES[tool](env, tool_config, **parameters)
        hand.tool_parameters = dict(parameters)
    else:
        raise ValueError(f"Missing parameters for recorded tool {tool!r}")
    script.controller = hand
    hand.command = np.array([state["hand_nominal_subset"][n] for n in (*hand.names, "Lift")])
    if replan_released_tool:
        hand.last_target = hand.ik.forward(hand.command)[0]
    else:
        _restore_tool_references(hand, state, env.data.ctrl)

    script.tracker = MabelController(
        env.model,
        env.data,
        TrackingConfig(**state["tracking_config"]),
        nominal_references=state["nominal_subset"],
    )
    script.active_grip = state["active_grip"]
    script.wrist_load = float(state.get("wrist_load_state_Nm", 0))
    if not np.isfinite(script.wrist_load) or abs(script.wrist_load) > 0.3:
        raise ValueError("Recorded wrist load exceeds the controller effort budget")
    for name, key in (
        ("_base_translation_origin", "base_translation_origin_xy_m"),
        ("carry_reference", "carry_reference"),
        ("payload_rotation_goal", "payload_rotation_goal"),
    ):
        value = state.get(key)
        setattr(script, name, None if value is None else np.asarray(value))
    script.observer = script._new_observer()
    for name, value in deepcopy(context["observer_history"]).items():
        if name == "seen":
            value = {tuple(item) for item in value}
        if name == "completed":
            value = set(value)
        setattr(script.observer, name, value)
    script.last_forces = contact_forces(env, script.product.name, script.hand_bodies)
    if script.config.head_tracking and state.get("head_gaze") is not None:
        from abstract_stack.recording import HeadGaze

        script.gaze = HeadGaze.restore(env.model, env.data, state["head_gaze"])
    if replan_released_tool:
        return "new_checked_tool_reference_from_recorded_nominal"
    return "recorded_nominal_with_legacy_tool_defaults" if legacy else "recorded_tool_and_nominal"


def _restore_tool_references(hand, state, controls):
    offset = np.asarray(state["tool_offset_m"], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid recorded tool reference")
    hand.OFFSET = hand.ik.offset = offset.copy()
    hand.ROTATION = np.asarray(state.get("tool_preferred_rotation_world", hand.ROTATION))
    hand.active_rotation = np.asarray(state["active_rotation_world"])
    hand.open_posture_overrides = dict(
        state.get("open_posture_overrides", hand.open_posture_overrides)
    )
    hand.abduction_targets = dict(state.get("abduction_targets", hand.abduction_targets))
    for name, value in state["tool_posture_constraints"].items():
        hand.ik.constrain_posture(name, value)
    if state.get("tip_body_references"):
        hand.tip_body_references = {
            k: np.asarray(v) for k, v in state["tip_body_references"].items()
        }
    hand.last_target = np.asarray(state["nominal_cartesian_target_m"])
    finger_tracker = getattr(hand, "finger_load_compensation", None)
    for name, value in state.get("finger_compensation_applied", {}).items():
        joint = finger_tracker.joints[name]
        if value != float(controls[joint["actuator"]]):
            raise ValueError("Recorded finger compensation differs from applied control")
        joint["previous"] = value
