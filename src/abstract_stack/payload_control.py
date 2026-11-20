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


def screen_unloaded_servo_path(hand, prediction, tracking_config, *, product):
    """Screen a joint-space approach using gravity-compensated servo references.

    The projection assumes a stationary robot at each sample. Live rate,
    damping, load and geometry checks still govern every issued command.
    """
    from abstract_stack.mabel_controller import MabelController

    model, ik = hand.model, hand.ik
    static = mujoco.MjData(model)
    static.ctrl[:] = hand.data.ctrl
    tracker = MabelController(model, static, tracking_config)
    settle_dt = float(np.max(tracker.offset_limits / tracker.rates))
    names = (*hand.names, "Lift")
    ids = [model.actuator(name).id for name in names]
    start = ik.measured_q()
    baseline = ik._contacts(start, hand.data.qpos, None)
    product_geom = model.geom(f"{product}_geom").id
    robot_geoms = [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] in ik.robot
        and (
            model.geom_contype[geom] & model.geom_conaffinity[product_geom]
            or model.geom_contype[product_geom] & model.geom_conaffinity[geom]
        )
    ]
    minimum = 0.05
    clearance_reserve = 0.002
    rates = np.r_[np.full(len(hand.names), tracking_config.joint_speed), tracking_config.lift_speed]
    waypoints = (
        prediction.joint_path if prediction.joint_path is not None else prediction.q[None, :]
    )
    nominal_samples = [start]
    for first, last in zip(np.vstack((start, waypoints[:-1])), waypoints, strict=True):
        steps = max(1, int(np.ceil(np.max(np.abs(last - first) / rates) / 0.04)))
        if len(waypoints) == 1:
            steps = max(7, steps)
        nominal_samples.extend(np.linspace(first, last, steps + 1)[1:])
    for index, nominal in enumerate(nominal_samples, start=1):
        ik.forward(nominal, prediction.reference_qpos)
        static.qpos[:] = ik.scratch.qpos
        mujoco.mj_forward(model, static)
        applied = np.clip(static.actuator_length[tracker.ids], tracker.lower, tracker.upper)
        tracker.nominal = applied.copy()
        tracker.applied = applied.copy()
        references = tracker.targets({}, dt=settle_dt, load_feedforward=None)
        fingers = hand.finger_load_compensation.gravity_preshape(static)
        reference = static.qpos.copy()
        for name, value in fingers.items():
            reference[int(model.joint(name).qposadr[0])] = value
        errors = ik.collision_errors(references[ids], reference, baseline)
        clearance = min(
            (
                mujoco.mj_geomDistance(model, ik.scratch, g, product_geom, 0.05, None)
                for g in robot_geoms
            ),
            default=minimum,
        )
        minimum = min(minimum, float(clearance))
        if errors:
            return dict(
                admitted=False, samples=index, active_product_clearance_m=minimum, errors=errors
            )
    return dict(
        admitted=minimum >= clearance_reserve,
        samples=len(nominal_samples),
        active_product_clearance_m=minimum,
        active_product_clearance_reserve_m=clearance_reserve,
    )


def guard_payload_targets(script, targets, *, intended_contact=None, admit_contact=None):
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
    if intended_contact is None:
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
        **({"admit_contact": admit_contact} if admit_contact is not None else {}),
    )
    if errors:
        script.events[-1]["applied_tracking_geometry_rejection"] = errors
        raise InteractionFailure("payload_load: compensated arm/finger geometry rejected")
