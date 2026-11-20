"""Measured capacity and finishing decisions for the ordered dense-row script."""

import time

import mujoco
import numpy as np

from abstract_stack.contact_execution import hand_clears_shelf, squared_contact_retreat
from abstract_stack.evaluation import strict_final_position
from abstract_stack.finger_pusher import FoldedFingerPusher
from abstract_stack.interactions import InteractionFailure, RowCapacityFailure, contact_forces
from abstract_stack.lateral_contact import LateralContact, LateralPushConfig
from abstract_stack.packing import exposed_row_capacity, open_slot_interval, row_geometry


def yaw_degrees(body):
    return float(np.degrees(np.arctan2(body.xmat[3], body.xmat[0])))


def require_open_slot(script, product):
    """Try one guarded press when the measured row cannot admit the next carton."""
    env, completed = script.env, script.observer.completed

    def interval():
        return open_slot_interval(env, product, completed)

    lower, upper = interval()
    script._stage("dense_slot_capacity")
    event = script.events[-1]
    event["interval_before_m"] = [lower, upper]
    if lower <= upper:
        return
    if not completed or script.observer.incidents or not hand_clears_shelf(env, script.hand_bodies):
        raise RowCapacityFailure(dict(admissible_center_interval_m=[lower, upper]))
    choices = []
    for direction in (1, -1):
        group = {
            other.name
            for other in env.products
            if other.name in completed and direction * (other.target[1] - product.target[1]) > 0
        }
        if group:
            evidence = exposed_row_capacity(env, group, direction)
            slack = float(evidence["closable_slack_m"])
            if slack > lower - upper:
                choices.append((slack, direction, evidence["exposed_product"]))
    event["press_candidates"] = choices
    if not choices:
        raise RowCapacityFailure(dict(admissible_center_interval_m=[lower, upper]))
    slack, direction, active = max(choices, key=lambda item: item[0])
    script.product = next(item for item in env.products if item.name == active)
    travel = min(0.015, slack, lower - upper + 0.003)
    try:
        report = LateralContact.push_laterally(
            script,
            direction=direction,
            distance_m=travel,
            near_front_corner=True,
            front_contact_fraction=0.0,
            row_ready=lambda: interval()[0] + 0.0005 <= interval()[1],
            settings=LateralPushConfig(
                speed_m_s=0.006,
                force_target_N=6.0,
                distance_limit_m=0.015,
                timeout_s=26.0,
            ),
        )
        after = script.observer.snapshot()
        lower, upper = interval()
        event.update(interval_after_m=[lower, upper], press_retreat=report["retreat_verified"])
        if not (
            lower <= upper
            and all(after[name]["placement_conditions"] for name in completed)
            and hand_clears_shelf(env, script.hand_bodies)
        ):
            raise InteractionFailure("dense_slot_capacity: press did not leave a supported opening")
    finally:
        script.product = product


def _assess_approach(script, *, planning=None):
    """Screen a folded tool from the measured or prospective shelf landing."""
    env, product = script.env, script.product
    body = (env.data if planning is None else planning.data).body(product.name)
    extent_x = float((np.abs(body.xmat.reshape(3, 3)) @ product.half_size)[0])
    height = env.config.levels[product.level] + 0.028
    target = np.array(
        [env.config.front_x - env.config.depth + extent_x * 2 - 0.0005, product.target[1], height]
    )
    approach = np.array([body.xpos[0] + extent_x + 0.035, body.xpos[1], height])
    report = dict(yaw_deg=yaw_degrees(body), admitted=False)
    try:
        script._plan_pusher_at_base(
            approach,
            target,
            kind=FoldedFingerPusher,
            event=report,
            deadline=time.monotonic() + 5.0,
            planning=planning,
            screen_servo=True,
        )
    except InteractionFailure as error:
        report["failure"] = str(error)
    else:
        report["admitted"] = True
    return report


def _assess_next_approach(script, product, interval):
    """Build the prospective free front landing on scratch data for the approach screen."""
    env = script.env
    scratch = env.clone_for_planning()
    hx, _, hz = product.half_size
    y = float(np.clip(product.target[1], *interval))
    qadr = int(scratch.model.joint(f"{product.name}_joint").qposadr[0])
    scratch.data.qpos[qadr : qadr + 7] = [
        env.config.front_x - hx - 0.006,
        y,
        env.config.levels[product.level] + hz + 0.001,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    mujoco.mj_forward(scratch.model, scratch.data)
    previous = script.product
    script.product = product
    try:
        return _assess_approach(script, planning=scratch)
    finally:
        script.product = previous


def _finish(script, report):
    script._stage("dense_finish")
    script.events[-1]["reason"] = report
    script._insert_from_front(retreat=False, precontact_standoff_m=0.035, recovering=True)
    squared_contact_retreat(script, script._active_push_event, strict_finish=True)
    if not hand_clears_shelf(script.env, script.hand_bodies):
        raise InteractionFailure("dense_finish: measured hand did not clear")
    script._stage("dense_finish_verified")


def register_placement(script, remaining):
    """Admit a provisional placement only with capacity and checked recovery access."""
    product = script.product
    own = [event for event in script.events if event["product"] == product.name]
    if strict_final_position(own):
        return
    env, done = script.env, script.observer.completed
    body = env.data.body(product.name)
    support = contact_forces(env, product.name, script.arm_bodies)
    reserve = {p.name: open_slot_interval(env, p, done) for p in remaining}
    supported = (
        product.name in done
        and not script.observer.incidents
        and support["shelf"] > 0.2
        and support["hand"] <= 0.2
        and hand_clears_shelf(env, script.hand_bodies)
    )
    if not supported:
        raise InteractionFailure(
            "dense_position: provisional placement lacks a cleared support state"
        )
    if script.config.defer_final_alignment:
        script._stage("dense_position_decision")
        script.events[-1].update(
            yaw_deg=yaw_degrees(body),
            reserved_centers_m=reserve,
            provisional=True,
            decision="retain_supported_placement_for_later_full_row_alignment",
        )
        return
    access = _assess_approach(script)
    script._stage("dense_position_decision")
    event = script.events[-1]
    event.update(yaw_deg=yaw_degrees(body), reserved_centers_m=reserve, finish_access=access)
    may_defer = (
        remaining
        and access["admitted"]
        and abs(yaw_degrees(body)) <= 1.0
        and all(lower <= upper for lower, upper in reserve.values())
    )
    if may_defer:
        event["next_approach"] = _assess_next_approach(
            script, remaining[0], reserve[remaining[0].name]
        )
    event["provisional"] = bool(may_defer and event["next_approach"]["admitted"])
    if not event["provisional"]:
        _finish(script, "strict position required before advancing")


def strict_positioned_products(script, scheduled):
    """Use verified releases and the current pose, including retained checkpoint evidence."""
    env = script.env
    prior = getattr(script, "_prior_strict_positions", ())
    verified = set()
    for index, product in enumerate(env.products):
        name = product.name
        if name not in scheduled or not env._stocked[index]:
            continue
        if abs(yaw_degrees(env.data.body(name))) > 0.15:
            continue
        own = [event for event in script.events if event["product"] == name]
        if strict_final_position(own) or (
            name in prior and not any(event.get("verified_release") for event in own)
        ):
            verified.add(name)
    return verified


def finish_row(script, scheduled, *, targets=None):
    """Finish the smallest open intervals first and recheck all live poses after each move."""
    env = script.env
    positions = {p.name: p for p in env.products if p.name in scheduled}
    targets = set(positions) if targets is None else set(targets)
    if not targets <= positions.keys():
        raise ValueError("Finish targets must be part of the stocked row")

    def access_width(name):
        lower, upper = open_slot_interval(env, positions[name], scheduled - {name})
        return upper - lower

    verified = strict_positioned_products(script, scheduled)
    unfinished = [name for name in positions if name in targets and name not in verified]
    for name in sorted(unfinished, key=access_width):
        script.product = positions[name]
        _finish(script, "deferred finish")
        row = row_geometry(env, product_names=scheduled)
        script.events[-1]["measured_row"] = row
        if script.observer.incidents:
            raise InteractionFailure("dense_finish: product incident during finishing pass")


def recheck_terminal_positions(script, scheduled):
    """Gate final credits on the measured released row after the native dwell."""
    final = script.observer.snapshot()
    original = script.product
    try:
        for product in script.env.products:
            if product.name in scheduled:
                script.product = product
                yaw = yaw_degrees(script.env.data.body(product.name))
                script._stage("dense_final_pose_recheck")
                script.events[-1].update(
                    yaw_deg=yaw,
                    verified=final[product.name]["stocked"] and abs(yaw) <= 0.15,
                )
    finally:
        script.product = original


def record_packed_ready(script):
    """Retain a post-dwell state for independently scheduled alignment."""
    scheduled = {p.name for p in script.env.products}
    if (
        script.config.record_skill_starts
        and script.observer.completed == scheduled
        and not script.observer.incidents
        and script.active_grip is None
        and script.carry_reference is None
        and hand_clears_shelf(script.env, script.hand_bodies)
    ):
        script._stage("packed_row_ready")
