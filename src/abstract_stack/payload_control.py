"""Shared measured wrist-load compensation and final native geometry admission."""

from types import SimpleNamespace

import mujoco
import numpy as np


def reserve_payload_references(hand, wrist_reserve, lift_reserve):
    """Reserve compensation headroom in planning without editing native limits."""
    for name, lower_reserve, upper_reserve in (
        ("Left Wrist", wrist_reserve, wrist_reserve),
        ("Lift", 0, lift_reserve),
    ):
        index = hand.ik.names.index(name)
        hand.ik.constrain_reference_range(
            name,
            hand.ik.original_lower[index] + lower_reserve,
            hand.ik.original_upper[index] - upper_reserve,
        )


def measured_payload_load(script):
    result = script._wrist_contact_load()
    if script.events:
        script.events[-1].setdefault("payload_wrist_load_samples", []).append(
            [float(script.env.data.time), script.wrist_load_last_requested, script.wrist_load]
        )
    return result


def guard_payload_targets(script, targets):
    """Check final compensated arm and proposed fingers together before stepping."""
    from abstract_stack.interactions import InteractionFailure

    model, hand = script.env.model, script.controller
    applied_q = np.array([targets[model.actuator(name).id] for name in [*hand.names, "Lift"]])
    fingers = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[aid, 0])
        if int(model.jnt_bodyid[jid]) in script.hand_bodies:
            fingers[model.actuator(aid).name] = float(targets[aid])
    intended_contact = script.active_grip is not None or script.phase in (
        "approach",
        "close",
        "place",
        "stage_align",
        "release_unload",
        "release_front",
    )
    errors = hand.check_hand_step(
        SimpleNamespace(q=applied_q),
        0,
        script.product.name if intended_contact else None,
        targets=fingers,
    )
    if errors:
        script.events[-1]["applied_tracking_geometry_rejection"] = errors
        raise InteractionFailure("payload_load: compensated arm/finger geometry rejected")
