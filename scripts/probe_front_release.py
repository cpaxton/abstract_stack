#!/usr/bin/env python3
"""Bounded physical release from a retained supported-front entry; zero fresh credit."""

import argparse
import copy
import hashlib
import json
import math
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_verified_milestone(script):
    if (
        script.config.record_skill_starts
        and script.product.name in script.observer.completed
        and not script.observer.incidents
        and script.active_grip is None
        and script.carry_reference is None
    ):
        script._stage("stocking_milestone")
        script.events[-1]["completed_count"] = len(script.observer.completed)


def clear_both_palms(script):
    """Measured unloaded bimanual clearance through one shared-lift controller."""
    import numpy as np

    from abstract_stack.ik import rotation_error
    from abstract_stack.interactions import InteractionFailure, contact_forces
    from abstract_stack.whole_body_control import PalmGoal, StationaryWholeBodyController

    env = script.env
    loads = contact_forces(
        env, script.product.name, script.arm_bodies, all_products=script.product_bodies
    )
    if (
        script.active_grip is not None
        or script.carry_reference is not None
        or script.observer.incidents
        or max(loads["hand"], loads["other_product"]) > 0.2
    ):
        raise InteractionFailure("bimanual_clear: requires an unloaded incident-free state")
    from abstract_stack.mabel_controller import MabelController

    products = {product.name: env.data.body(product.name).xpos.copy() for product in env.products}
    headroom = MabelController(env.model, env.data)
    lower = headroom.lower + headroom.offset_limits
    upper = headroom.upper - headroom.offset_limits
    outside = (script.tracker.nominal < lower) | (script.tracker.nominal > upper)
    if np.any(outside):
        # Stage the unsupported left reference before taking over with both
        # arms. The checkout's existing arm/path and force checks still apply.
        hand = script.controller
        for index in np.flatnonzero(outside):
            if headroom.NAMES[index] not in hand.ik.names:
                raise InteractionFailure(
                    f"bimanual_clear: {headroom.NAMES[index]} needs separate staging"
                )
        for joint, name in enumerate(hand.ik.names):
            index = headroom.NAMES.index(name)
            hand.ik.constrain_reference_range(
                name,
                max(hand.ik.original_lower[joint], lower[index] + 0.02),
                min(hand.ik.original_upper[joint], upper[index] - 0.02),
            )
        goal = hand.position() + [0.14, 0, 0.06]
        script._move("bimanual_headroom_stage", goal, carrying=False)
        if script.observer.incidents or any(
            np.linalg.norm(env.data.body(name).xpos - before) > 0.001
            for name, before in products.items()
        ):
            raise InteractionFailure("bimanual_clear: headroom staging disturbed a product")
    script._stage("bimanual_clear")
    event = script.events[-1]
    event["nominal_headroom_error_rad"] = {
        name: float(max(lower[i] - value, value - upper[i], 0.0))
        for i, (name, value) in enumerate(zip(headroom.NAMES, script.tracker.nominal, strict=True))
    }
    control = StationaryWholeBodyController(
        env.model,
        env.data,
        nominal=dict(zip(script.tracker.NAMES, script.tracker.nominal, strict=True)),
    )
    start = {side: env.data.xpos[body].copy() for side, body in control.palms.items()}
    goals = {
        side: PalmGoal(start[side] + [0.012, 0, 0.012], env.data.xmat[body].reshape(3, 3))
        for side, body in control.palms.items()
    }
    event["goals_world_m"] = {side: goal.position_m.tolist() for side, goal in goals.items()}

    def guard(world):
        script._sample(world)
        if (
            script.observer.incidents
            or max(script.last_forces["robot_product"], script.last_forces["other_product"]) > 0.2
            or any(
                np.linalg.norm(world.data.body(name).xpos - before) > 0.001
                for name, before in products.items()
            )
        ):
            raise InteractionFailure("bimanual_clear: product contact or displacement")

    held = 0.0
    for _ in range(int(2.0 / env.dt)):
        step = control.predict_step(goals, dt=env.dt)
        if not step.report["accepted"]:
            event["rejection"] = step.report
            raise InteractionFailure("bimanual_clear: coupled reference rejected")
        _, _, terminated, truncated, _ = env.step(control.accept(step), on_physics_step=guard)
        if terminated or truncated:
            raise InteractionFailure("bimanual_clear: episode ended during motion")
        errors = {
            side: float(np.linalg.norm(env.data.xpos[control.palms[side]] - goal.position_m))
            for side, goal in goals.items()
        }
        orientation = {
            side: float(
                np.degrees(
                    np.linalg.norm(
                        rotation_error(
                            env.data.xmat[control.palms[side]].reshape(3, 3), goal.rotation_world
                        )
                    )
                )
            )
            for side, goal in goals.items()
        }
        event["measured_error_m"] = errors
        event["measured_orientation_error_deg"] = orientation
        within_goal = (
            max(errors.values()) <= 0.004
            and max(orientation.values()) <= control.config.orientation_tolerance_deg
        )
        held = held + env.dt if within_goal else 0.0
        if held >= 0.1:
            event["palm_displacement_world_m"] = {
                side: (env.data.xpos[body] - start[side]).tolist()
                for side, body in control.palms.items()
            }
            return {"skill": "bimanual-clear", "verified": True}
    raise InteractionFailure("bimanual_clear: measured palm dwell not reached")


def worker(args):
    import numpy as np

    from abstract_stack.checkpoints import load_checkpoint, native_state
    from abstract_stack.continuation import CheckpointContinuationEnv
    from abstract_stack.evaluation import StageObserver
    from abstract_stack.interactions import ScriptConfig, StockingScript, contact_forces
    from abstract_stack.mabel_controller import MabelController, TrackingConfig

    reference = json.loads((args.output / "input_ref.json").read_text())
    path = Path(reference["checkpoint"])
    if hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Front-entry checkpoint changed")
    checkpoint = load_checkpoint(path)
    state = checkpoint.controller
    skill = checkpoint.context["skill_entry"]["skill"]
    expected = (
        {"stocking_milestone"}
        if args.bimanual_clear
        else {"motion_failure"}
        if args.unloaded_clear
        else {"stocking_milestone", "grasp_lift"}
        if args.next_cartons
        else {"side_slide_unloaded"}
        if args.resume_side_square
        else {"free_landing"}
        if args.stop_after == "registration" and not args.side_offset_m
        else {"stocking_milestone", "front_insertion", "front_squaring_complete"}
        if args.stop_after == "registration" and args.side_offset_m
        else {"front_insertion", "contact_finish", "front_squaring_complete"}
        if args.insert_only
        else {"release_retreat", "contact_refinement", "withdrawal_planning", "motion_failure"}
        if args.withdrawal
        else {"motion_failure"}
        if args.lead_landing and args.transfer
        else {"transfer"}
        if args.transfer
        else {"front_release"}
    )
    if state["kind"] != "stocking_skill_entry" or skill not in expected:
        raise ValueError(f"Checkpoint skill {skill!r} does not match {sorted(expected)}")
    if args.withdrawal and skill == "motion_failure":
        case = checkpoint.context["skill_entry"]["case"]
        if (
            case.get("failed_phase") != "contact_refine"
            or case.get("failure") != "contact_refine: bounded release-goal search exhausted"
            or state["active_grip"] is not None
            or state["carry_reference"] is not None
            or checkpoint.context["observer_history"]["incidents"]
        ):
            raise ValueError(
                "Withdrawal retry requires incident-free contact refinement exhaustion"
            )
    if args.lead_landing and args.transfer:
        entry = checkpoint.context["skill_entry"]["case"]
        if (
            entry.get("failed_phase") != "base_reposition_search"
            or entry.get("failure") != "loaded_stage: no checked shelf approach candidate"
            or checkpoint.context["observer_history"]["incidents"]
        ):
            raise ValueError("Lead landing requires an incident-free loaded shelf approach stop")
    env = CheckpointContinuationEnv(checkpoint)
    env.restore_task_memory(checkpoint.context["task_memory"])
    settings = dict(state["settings"])
    if reference.get("controller"):
        from abstract_stack.tuning import load_control_config

        controller = reference["controller"]
        source = Path(controller["snapshot"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != controller["sha256"]:
            raise ValueError("Frozen probe controller input changed")
        settings.update(load_control_config(source))
    settings.update(
        record_skill_starts=args.record_skill_starts,
        continuous_finger_opening=not args.legacy_opening,
    )
    if args.paired_fingers:
        settings["paired_finger_insertion"] = True
    if args.live_guarded_withdrawal:
        settings["live_guarded_withdrawal"] = True
    if args.plan_driven:
        settings["plan_driven"] = True
    if args.guided_wall_entry:
        settings["guided_wall_entry"] = True
    if args.back_force_target is not None:
        settings["insertion_back_force_target"] = args.back_force_target
    if args.alignment_seconds is not None:
        settings["alignment_timeout_seconds"] = args.alignment_seconds
    script_class = StockingScript
    if args.front_row_press:
        from abstract_stack.front_tangential import FrontTangentialPress

        if skill != "stocking_milestone":
            raise ValueError("Front row press requires a verified stocking milestone")

        class CheckpointRowPress(FrontTangentialPress, StockingScript):
            pass

        script_class = CheckpointRowPress
    script = script_class(env, args.output / "episode", ScriptConfig(**settings))
    if args.refinement_pad_angles:
        script.refinement_pad_angles = True
    if args.refinement_horizon:
        script.refinement_horizon_s = args.refinement_horizon
    if args.pusher_tilt is not None:
        script.PUSHER_TILTS = (args.pusher_tilt,)
    if args.guided_entry_deg:
        script.guided_entry_deg = args.guided_entry_deg
    if args.diagonal_entry:
        script.diagonal_entry = True
    if args.clearance_before_square:
        script.clearance_before_square = True
    if args.side_base_dx:
        script.side_base_dx = args.side_base_dx
    if args.lead_landing:
        script.front_lead_y_m = args.lead_landing
        script.front_landing_x_offset_m = 0.004
    if args.front_landing_x_offset_m:
        script.front_landing_x_offset_m = args.front_landing_x_offset_m
    if args.front_landing_y_offset_m:
        script.front_landing_y_offset_m = args.front_landing_y_offset_m
    if args.single_center_finger:
        from abstract_stack.finger_pusher import WallFingerPusher

        script.PAIRED_PUSHER_CLASS = WallFingerPusher
    if "grasp_abduction_targets" in state:
        script.grasp_abduction_targets = dict(state["grasp_abduction_targets"])
        if args.next_cartons:
            script._configure_payload_hand(script.controller)
    script.product = next(
        p for p in env.products if p.name == checkpoint.context["skill_entry"]["active_product"]
    )
    if args.record_skill_starts:
        scene = copy.deepcopy(checkpoint.scene)
        scene["continuation"] = dict(checkpoint=str(path), fresh_task_credit=0)
        initial = script.directory / "initial-scene"
        initial.mkdir()
        write(initial / "manifest.json", scene)
    replan_legacy_tool = (
        (args.next_cartons or args.stop_after == "registration")
        and skill == "stocking_milestone"
        or (args.insert_only and skill == "contact_finish")
    )
    if (
        args.withdrawal
        or args.unloaded_clear
        or (
            (args.insert_only or args.next_cartons)
            and not replan_legacy_tool
            and state.get("tool_class") != "CartesianHand"
        )
    ):
        from abstract_stack.finger_pusher import (
            FingerPusher,
            FoldedFingerPusher,
            PairedFingerPusher,
            WallFingerPusher,
        )
        from abstract_stack.lateral_contact import SideFingerPusher

        kinds = {
            kind.__name__: kind
            for kind in (
                FingerPusher,
                FoldedFingerPusher,
                PairedFingerPusher,
                WallFingerPusher,
                SideFingerPusher,
            )
        }
        if state.get("tool_class") not in kinds:
            raise ValueError("Withdrawal continuation requires an explicitly recorded FingerPusher")
        selected = state.get("tool_parameters") or next(
            (
                e["selected"]
                for e in reversed(checkpoint.context["prior_events"])
                if e["phase"] == "pusher_plan" and e.get("selected")
            ),
            None,
        )
        if selected is None:
            raise ValueError("Checkpoint is missing the recorded pusher parameters")
        kind = kinds[state["tool_class"]]
        tool_settings = dict(state.get("tool_settings", state["settings"]))
        tool_settings.update({k: v for k, v in settings.items() if state["settings"].get(k) != v})
        script.controller = kind(env, ScriptConfig(**tool_settings), **selected)
        script.controller.tool_parameters = dict(selected)
        # Passive distal joints can settle after the original calibration.
        # Restore the recorded tool reference, not a newly calibrated frame.
        offset = np.asarray(state["tool_offset_m"], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Invalid recorded tool reference")
        script.controller.OFFSET = offset.copy()
        script.controller.ik.offset = offset.copy()
        if "tool_preferred_rotation_world" in state:
            script.controller.ROTATION = np.asarray(state["tool_preferred_rotation_world"])
        for name, value in state["tool_posture_constraints"].items():
            script.controller.ik.constrain_posture(name, value)
    hand = script.controller
    finger_tracker = getattr(hand, "finger_load_compensation", None)
    if finger_tracker is not None:
        for name, value in state.get("finger_compensation_applied", {}).items():
            aid = finger_tracker.joints[name]["actuator"]
            if value != float(env.data.ctrl[aid]):
                raise ValueError("Saved finger compensation differs from native applied control")
            finger_tracker.joints[name]["previous"] = value
    if "open_posture_overrides" in state and not args.next_cartons:
        hand.open_posture_overrides = dict(state["open_posture_overrides"])
    if "abduction_targets" in state and not args.next_cartons:
        hand.abduction_targets = dict(state["abduction_targets"])
    if state.get("tip_body_references") and not replan_legacy_tool:
        hand.tip_body_references = {
            k: np.asarray(v) for k, v in state["tip_body_references"].items()
        }
    hand.command = np.array([state["hand_nominal_subset"][n] for n in (*hand.names, "Lift")])
    hand.active_rotation = np.asarray(state["active_rotation_world"])
    if replan_legacy_tool:
        hand.active_rotation = hand.ROTATION.copy()
        hand.last_target = hand.ik.forward(hand.command)[0]
        initialization = "new tool reference for checked replanning from retained push entry"
    elif "nominal_cartesian_target_m" in state:
        hand.last_target = np.asarray(state["nominal_cartesian_target_m"])
        initialization = "recorded nominal Cartesian reference"
    else:
        hand.last_target = hand.ik.forward(hand.command)[0]
        initialization = "new local reference from saved nominal FK at restored base"
    tracking = (
        script._tracking_config()
        if args.next_cartons and skill == "stocking_milestone"
        else TrackingConfig(**state["tracking_config"])
    )
    if args.wrist_offset is not None:
        tracking = replace(tracking, wrist_offset_limit=args.wrist_offset)
    script.tracker = MabelController(
        env.model, env.data, tracking, nominal_references=state["nominal_subset"]
    )
    script.active_grip = state["active_grip"]
    if state.get("base_translation_origin_xy_m") is not None:
        script._base_translation_origin = np.asarray(state["base_translation_origin_xy_m"])
    if "wrist_load_state_Nm" in state:
        if not np.isfinite(state["wrist_load_state_Nm"]) or abs(state["wrist_load_state_Nm"]) > 0.3:
            raise ValueError(
                "Saved wrist holding state exceeds the native-controller effort budget"
            )
        script.wrist_load = float(state["wrist_load_state_Nm"])
    script.carry_reference = (
        np.asarray(state["carry_reference"])
        if args.transfer and state["carry_reference"] is not None
        else None
    )
    script.payload_rotation_goal = (
        None
        if state["payload_rotation_goal"] is None
        else np.asarray(state["payload_rotation_goal"])
    )
    script.observer = StageObserver(env)
    for name, value in copy.deepcopy(checkpoint.context["observer_history"]).items():
        if name == "seen":
            value = {tuple(item) for item in value}
        if name == "completed":
            value = set(value)
        setattr(script.observer, name, value)
    script.last_forces = contact_forces(env, script.product.name, script.hand_bodies)
    initial = native_state(env.model, env.data)
    failure = None
    progress = None
    skill_boundary = None
    try:
        if args.bimanual_clear:
            skill_boundary = clear_both_palms(script)
        elif args.next_cartons:
            if state["active_grip"] is not None or state["carry_reference"] is not None:
                raise ValueError("Stocking milestone must have an unloaded hand")
            before = sorted(script.observer.completed)
            progress = dict(completed_products=before.copy(), starting_completed_products=before)
            script._execute_task(progress, next_cartons=args.next_cartons)
            failure = progress.get("failure")
        elif args.insert_only:
            if state["active_grip"] is not None or state["carry_reference"] is not None:
                raise ValueError("Insertion-only continuation requires a released carton")
            if args.front_row_press:
                from abstract_stack.packing import exposed_row_capacity

                stocked = script.observer.completed
                direction = 1 if args.side_offset_m > 0 else -1
                before = exposed_row_capacity(env, stocked, direction)
                if before["exposed_product"] != script.product.name:
                    raise ValueError("Row press must start from the exposed end carton")
                script.push_front_tangentially(
                    direction=direction,
                    distance_m=abs(args.side_offset_m),
                    force_guided=True,
                    row_products=stocked,
                )
                after = exposed_row_capacity(env, stocked, direction)
                skill_boundary = {
                    "skill": "front-row-press",
                    "verified": after["available_m"] >= after["required_m"],
                    "previous_open_span_m": before["available_m"],
                    "open_span_after_m": after["available_m"],
                    "required_remaining_width_m": after["required_m"],
                }
                if skill_boundary["verified"]:
                    record_verified_milestone(script)
            elif args.side_offset_m:
                from abstract_stack.lateral_contact import LateralPushConfig, LateralStockingScript

                LateralStockingScript.push_laterally(
                    script,
                    direction=1 if args.side_offset_m > 0 else -1,
                    distance_m=abs(args.side_offset_m),
                    near_front_corner=args.side_front_corner,
                    stage_for_squaring=args.interleave_side_square,
                    settings=LateralPushConfig(
                        speed_m_s=0.003 if args.side_front_corner else 0.01,
                        force_target_N=1.2 if args.side_front_corner else 2.5,
                        timeout_s=30 if args.side_front_corner else 20,
                        distance_limit_m=0.025 if args.interleave_side_square else 0.01,
                        position_tolerance_m=0.0005,
                        height_m=0.025 if args.side_front_corner else 0.028,
                    ),
                )
            if args.stop_after == "registration":
                if not args.side_offset_m:
                    script._register_front_laterally()
                if skill_boundary is None:
                    skill_boundary = {"skill": "side-registration", "verified": True}
            elif args.stop_after == "backstop-staging":
                from abstract_stack.skill_library import build_stocking_plan

                push = build_stocking_plan().by_name("push_to_contact")
                push.execute(script, {})
                event = script._active_push_event
                skill_boundary = {
                    "skill": "push-to-contact",
                    "verified": push.verify(script, {}),
                    "backstop_contact_verified": bool(
                        event.get("first_back_contact_verified")
                        or event.get("back_contact_verified")
                    ),
                    "bounded_staging": bool(
                        event.get("bounded_backstop_staging_for_native_refinement")
                    ),
                }
                if script.config.record_skill_starts and skill_boundary["verified"]:
                    script._stage("contact_refine")
            else:
                if args.interleave_side_square or args.resume_side_square:
                    if not args.resume_side_square:
                        script._stage("side_slide_unloaded")
                    # Park the side pusher before opening the paired contacts.
                    # Middle and Ring otherwise sweep through one another.
                    script.controller.park_all_fingers = True
                    script._settle_pusher_preshape()
                    current = env.data.body(script.product.name).xpos[0]
                    script._insert_from_front(
                        partial_stop_x=float(current) - 0.004,
                        front_squaring=True,
                        square_in_place=True,
                    )
                script._insert_from_front()
                record_verified_milestone(script)
            if args.continue_after_insert:
                if (
                    script.product.name not in script.observer.completed
                    or script.observer.incidents
                ):
                    raise ValueError(
                        "Follow-on stocking requires a verified, incident-free retreat"
                    )
                before = sorted(script.observer.completed)
                progress = dict(
                    completed_products=before.copy(), starting_completed_products=before
                )
                script._execute_task(progress, next_cartons=args.continue_after_insert)
                failure = progress.get("failure")
        elif args.transfer:
            if script.carry_reference is None or script.active_grip is None:
                raise ValueError("Transfer continuation requires a recorded retained grasp")
            if args.base_dx or args.base_dy:
                script._move_base(
                    [args.base_dx, args.base_dy],
                    reason="development_loaded_front_pose",
                    carrying=True,
                )
            grip = script._transfer_and_place_front(
                script.active_grip, already_clear=bool(args.lead_landing)
            )
            if args.stop_after == "seat":
                skill_boundary = {"skill": "transfer-seat", "verified": True}
            else:
                script._release_and_insert_front(grip, push=args.stop_after != "release")
                if args.stop_after == "release":
                    skill_boundary = {"skill": "supported-release", "verified": True}
                else:
                    record_verified_milestone(script)
        elif args.stop_after == "squared-contact":
            from abstract_stack.contact_execution import make_backstop_monitor
            from abstract_stack.contact_refinement import refine_supported_contact

            monitor, dwell = make_backstop_monitor(script)
            refine_supported_contact(script, monitor, dwell)
            skill_boundary = {
                "skill": "squared-contact",
                "verified": bool(dwell.verified),
                "native_dwell": dwell.report(),
            }
        elif args.withdrawal or args.unloaded_clear:
            from abstract_stack.contact_execution import squared_contact_retreat

            squared_contact_retreat(script, {}, unloaded_entry=args.unloaded_clear)
            record_verified_milestone(script)
            if args.base_goal is not None:
                script._restore_grasp_hand()
                base = env.data.body("mabel_chassis").xpos[:2].copy()
                script._move("empty_navigation_clear", [base[0] - 0.25, base[1] - 0.20, 1.35])
                script._move_base(
                    np.asarray(args.base_goal) - base, reason="inventory_grasp_workspace"
                )
        elif args.insert:
            script._release_and_insert_front(
                state["active_grip"], push=args.stop_after != "release"
            )
            if args.stop_after == "release":
                skill_boundary = {"skill": "supported-release", "verified": True}
            else:
                record_verified_milestone(script)
        else:
            script._precision_release(state["active_grip"])
            goal = hand.last_target.copy()
            for _ in range(int(0.5 / env.dt)):
                script._tick(goal, 0)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failure = str(error)
    checkpoint_error = None
    if failure:
        from abstract_stack.skill_entries import record_motion_failure

        checkpoint_error = record_motion_failure(script, failure)
    body = env.data.body(script.product.name)
    velocity = env.data.qvel[int(env.model.joint(f"{script.product.name}_joint").dofadr[0]) :][:6]
    write(
        args.output / "summary.json",
        dict(
            kind="supported_withdrawal_skill_continuation"
            if args.withdrawal
            else "supported_loaded_transfer_continuation"
            if args.transfer
            else "supported_front_release_continuation",
            fresh_task_credit=0,
            base_navigation_goal_m=args.base_goal,
            task_success=False,
            failure=failure,
            failure_checkpoint_error=checkpoint_error,
            milestone_progress=progress,
            skill_boundary=skill_boundary,
            reference_initialization=initialization,
            config=vars(script.config),
            single_center_finger=args.single_center_finger,
            guided_entry_deg=args.guided_entry_deg,
            pusher_tilts_degrees=list(getattr(script, "PUSHER_TILTS", (20, 30, 10, 40))),
            tracking_config=vars(tracking),
            active=script.product.name,
            simulated_seconds=float(env.data.time) - float(initial[0]),
            final_position_world_m=body.xpos.tolist(),
            final_yaw_deg=float(
                np.degrees(np.arctan2(body.xmat.reshape(3, 3)[1, 0], body.xmat[0]))
            ),
            final_velocity=velocity.tolist(),
            last_forces=script.last_forces,
            events=script.events,
            incidents=script.observer.incidents,
        ),
    )
    np.savez_compressed(
        args.output / "continuation.npz",
        controls=np.asarray(script.controls),
        initial_state=initial,
        terminal_state=native_state(env.model, env.data),
        samples=np.asarray(script.rows),
    )
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-opening", action="store_true")
    parser.add_argument("--insert", action="store_true")
    parser.add_argument("--insert-only", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("seat", "release", "registration", "backstop-staging", "squared-contact"),
    )
    parser.add_argument("--front-landing-x-offset-m", type=float, choices=(0.003, 0.004))
    parser.add_argument(
        "--front-landing-y-offset-m", type=float, choices=(-0.008, -0.006, 0.006, 0.008)
    )
    parser.add_argument("--front-row-press", action="store_true")
    parser.add_argument("--side-front-corner", action="store_true")
    parser.add_argument("--interleave-side-square", action="store_true")
    parser.add_argument("--resume-side-square", action="store_true")
    parser.add_argument("--side-base-dx", type=float, choices=(-0.14, -0.08, -0.04))
    parser.add_argument("--diagonal-entry", action="store_true")
    parser.add_argument("--clearance-before-square", action="store_true")
    parser.add_argument("--guided-entry-deg", type=float, choices=(-3, -2, 2, 3))
    parser.add_argument(
        "--side-offset-m",
        type=float,
        choices=(-0.024, -0.0079, -0.006, -0.004, -0.003, 0.003, 0.004, 0.006, 0.0079),
    )
    parser.add_argument("--next-cartons", type=int, choices=range(1, 11))
    parser.add_argument("--continue-after-insert", type=int, choices=range(1, 11))
    parser.add_argument("--record-skill-starts", action="store_true")
    parser.add_argument("--withdrawal", action="store_true")
    parser.add_argument("--live-guarded-withdrawal", action="store_true")
    parser.add_argument("--plan-driven", action="store_true")
    parser.add_argument("--guided-wall-entry", action="store_true")
    parser.add_argument("--unloaded-clear", action="store_true")
    parser.add_argument("--bimanual-clear", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--base-dy", type=float, default=0)
    parser.add_argument("--base-dx", type=float, default=0)
    parser.add_argument("--lead-landing", type=float, choices=(0.006, 0.012, 0.024))
    parser.add_argument("--base-goal", type=float, nargs=2)
    parser.add_argument("--wall-seconds", type=float, default=90)
    parser.add_argument("--paired-fingers", action="store_true")
    parser.add_argument("--back-force-target", type=float)
    parser.add_argument("--refinement-pad-angles", action="store_true")
    parser.add_argument("--refinement-horizon", type=float, choices=(0.4, 0.6))
    parser.add_argument("--alignment-seconds", type=float)
    parser.add_argument("--wrist-offset", type=float)
    parser.add_argument("--single-center-finger", action="store_true")
    parser.add_argument("--pusher-tilt", type=float, choices=(0, 10, 20, 30, 40))
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    stop_modes = {
        "seat": args.transfer,
        "squared-contact": args.withdrawal and args.base_goal is None,
        "release": args.insert or args.transfer,
        "registration": args.insert_only,
        "backstop-staging": args.insert_only,
    }
    if args.stop_after and (not stop_modes[args.stop_after] or args.continue_after_insert):
        parser.error("--stop-after requires its matching continuation without follow-on cartons")
    if args.clearance_before_square and not args.diagonal_entry:
        parser.error("--clearance-before-square requires --diagonal-entry")
    if args.diagonal_entry and not (args.insert_only and args.guided_entry_deg):
        parser.error("--diagonal-entry requires --insert-only and --guided-entry-deg")
    if args.guided_entry_deg and not (args.insert_only or args.insert):
        parser.error("--guided-entry-deg requires --insert-only or --insert")
    if args.front_landing_x_offset_m and not args.insert:
        parser.error("--front-landing-x-offset-m requires --insert")
    if args.front_landing_y_offset_m and not args.insert:
        parser.error("--front-landing-y-offset-m requires --insert")
    if args.refinement_pad_angles and not args.withdrawal:
        parser.error("--refinement-pad-angles requires --withdrawal")
    if args.refinement_horizon and not args.withdrawal:
        parser.error("--refinement-horizon requires --withdrawal")
    if args.front_row_press and (
        args.stop_after != "registration" or not args.side_offset_m or args.side_base_dx
    ):
        parser.error("--front-row-press requires an explicit offset and registration stop")
    if args.resume_side_square and (
        not args.insert_only
        or not args.guided_wall_entry
        or args.stop_after
        or args.interleave_side_square
        or args.side_offset_m
    ):
        parser.error("--resume-side-square requires just the guided insertion continuation")
    if args.interleave_side_square and (
        not args.side_front_corner or not args.guided_wall_entry or args.stop_after
    ):
        parser.error("--interleave-side-square needs a front-corner push and guided insertion")
    if args.side_offset_m and abs(args.side_offset_m) > 0.01 and not args.interleave_side_square:
        parser.error("Larger lateral shifts require the separate square and clearance gates")
    if args.side_front_corner and (not args.side_offset_m or args.front_row_press):
        parser.error("--side-front-corner requires an exposed-side offset")
    if args.side_base_dx and not args.side_offset_m:
        parser.error("--side-base-dx requires --side-offset-m")
    if args.side_offset_m and not args.insert_only:
        parser.error("--side-offset-m requires --insert-only")
    if args.continue_after_insert and not args.insert_only:
        parser.error("--continue-after-insert requires --insert-only")
    wall_limit = 900 if args.next_cartons else 300 if args.interleave_side_square else 180
    if not 0 < args.wall_seconds <= wall_limit:
        parser.error(f"Continuation wall budget must be positive and at most {wall_limit} seconds")
    if (
        sum(
            [
                args.withdrawal,
                args.unloaded_clear,
                args.bimanual_clear,
                args.insert,
                args.transfer,
                args.insert_only,
                bool(args.next_cartons),
            ]
        )
        > 1
    ):
        parser.error("Choose front insertion, transfer or withdrawal continuation")
    if args.lead_landing and (not (args.transfer or args.insert) or args.stop_after != "release"):
        parser.error("--lead-landing requires a loaded transfer ending after supported release")
    if (
        not math.isfinite(args.base_dx)
        or abs(args.base_dx) > 0.08
        or (args.base_dx and not args.lead_landing)
    ):
        parser.error("--base-dx is a bounded lead-landing workspace adjustment")
    if not -0.15 <= args.base_dy <= 0.15 or (args.base_dy and not args.transfer):
        parser.error("Loaded base probes need --transfer and at most 15 cm translation")
    if args.base_goal is not None and (
        not args.withdrawal
        or not -1.2 <= args.base_goal[0] <= 0.6
        or not -1 <= args.base_goal[1] <= 0.6
    ):
        parser.error("Empty base goals require --withdrawal and a finite workspace XY pair")
    if args.wrist_offset is not None and not 0.1 <= args.wrist_offset <= 0.5:
        parser.error("Wrist offset search is bounded to .1–.5 rad")
    if args.back_force_target is not None and not 0 < args.back_force_target <= 3:
        parser.error("Development back force targets are bounded to 3 N")
    if args.alignment_seconds is not None and not 0 < args.alignment_seconds <= 12:
        parser.error("Alignment search budget must be at most twelve simulated seconds")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    args.checkpoint = args.checkpoint.resolve()
    reference = dict(
        checkpoint=str(args.checkpoint),
        sha256=hashlib.sha256((args.checkpoint / "manifest.json").read_bytes()).hexdigest(),
    )
    if args.control_config:
        from abstract_stack.tuning import load_control_config

        settings = load_control_config(args.control_config)
        snapshot = args.output / "controller.json"
        write(snapshot, settings)
        snapshot.chmod(0o444)
        reference["controller"] = dict(
            snapshot=str(snapshot), sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        )
    write(args.output / "input_ref.json", reference)
    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/probe_front_release.py",
            "scripts/run_frozen_replanning.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    command = [
        sys.executable,
        str(frozen.runtime_root / "scripts/probe_front_release.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--worker",
        "--wall-seconds",
        str(args.wall_seconds),
    ]
    if args.legacy_opening:
        command.append("--legacy-opening")
    if args.paired_fingers:
        command.append("--paired-fingers")
    if args.single_center_finger:
        command.append("--single-center-finger")
    if args.stop_after:
        command += ["--stop-after", args.stop_after]
    if args.insert:
        command.append("--insert")
    if args.insert_only:
        command.append("--insert-only")
    if args.front_row_press:
        command.append("--front-row-press")
    if args.side_offset_m:
        command += ["--side-offset-m", str(args.side_offset_m)]
    if args.side_front_corner:
        command.append("--side-front-corner")
    if args.interleave_side_square:
        command.append("--interleave-side-square")
    if args.resume_side_square:
        command.append("--resume-side-square")
    if args.side_base_dx:
        command += ["--side-base-dx", str(args.side_base_dx)]
    if args.front_landing_x_offset_m:
        command += ["--front-landing-x-offset-m", str(args.front_landing_x_offset_m)]
    if args.front_landing_y_offset_m:
        command += ["--front-landing-y-offset-m", str(args.front_landing_y_offset_m)]
    if args.diagonal_entry:
        command.append("--diagonal-entry")
    if args.clearance_before_square:
        command.append("--clearance-before-square")
    if args.guided_entry_deg:
        command += ["--guided-entry-deg", str(args.guided_entry_deg)]
    if args.next_cartons:
        command += ["--next-cartons", str(args.next_cartons)]
    if args.continue_after_insert:
        command += ["--continue-after-insert", str(args.continue_after_insert)]
    if args.record_skill_starts:
        command.append("--record-skill-starts")
    if args.plan_driven:
        command.append("--plan-driven")
    if args.guided_wall_entry:
        command.append("--guided-wall-entry")
    if args.withdrawal:
        command.append("--withdrawal")
    if args.live_guarded_withdrawal:
        command.append("--live-guarded-withdrawal")
    if args.unloaded_clear:
        command.append("--unloaded-clear")
    if args.bimanual_clear:
        command.append("--bimanual-clear")
    if args.transfer:
        command += ["--transfer", "--base-dy", str(args.base_dy)]
    if args.lead_landing:
        command += ["--lead-landing", str(args.lead_landing), "--base-dx", str(args.base_dx)]
    if args.base_goal is not None:
        command += ["--base-goal", *(str(value) for value in args.base_goal)]
    if args.refinement_pad_angles:
        command.append("--refinement-pad-angles")
    if args.refinement_horizon:
        command += ["--refinement-horizon", str(args.refinement_horizon)]
    if args.back_force_target is not None:
        command += ["--back-force-target", str(args.back_force_target)]
    if args.alignment_seconds is not None:
        command += ["--alignment-seconds", str(args.alignment_seconds)]
    if args.wrist_offset is not None:
        command += ["--wrist-offset", str(args.wrist_offset)]
    if args.pusher_tilt is not None:
        command += ["--pusher-tilt", str(args.pusher_tilt)]
    supervise = runpy.run_path(str(frozen.runtime_root / "scripts/run_frozen_replanning.py"))[
        "supervise"
    ]
    result = supervise(
        command,
        cwd=root,
        environment=dict(
            os.environ,
            PYTHONPATH=str(frozen.runtime_root / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONFAULTHANDLER="1",
            ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
        ),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["returncode"] == 0 and not result["timeout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
