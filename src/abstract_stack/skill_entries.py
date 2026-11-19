"""Measured skill-entry distributions and restorable native states, not BC records."""

import json
from dataclasses import asdict

import mujoco
import numpy as np

from abstract_stack.checkpoints import save_checkpoint

SKILL_PHASES = {
    "motion_failure": "motion_failure",
    "stocking_milestone": "stocking_milestone",
    "grasp_precheck": "grasp_lift",
    "carry_clear": "transfer",
    "place": "shelf_place",
    "push": "contact_finish",
    "finger_push": "contact_finish",
    "release": "release_retreat",
    "stage_align": "front_release",
    "front_registered": "front_insertion",
    "free_landing_verified": "free_landing",
    "lateral_push": "side_registration",
    "interleaved_row_press": "row_press_entry",
    "interleaved_row_press_complete": "row_press_exit",
    "side_slide_unloaded": "side_slide_unloaded",
    "front_square_plan": "front_squaring",
    "front_squaring_complete": "front_squaring_complete",
    "contact_refine": "contact_refinement",
    "release_forecast": "withdrawal_planning",
    "loaded_push_stall": "loaded_push_recovery",
    "push_recovery_exhausted": "push_recovery_exhausted",
}

SKILL_CONTRACTS = {
    "row_press_entry": {
        "entry": "Two stocked cartons have verified retreats and the hand is unloaded",
        "success": "Close the seam through measured pair contact and retain it after unloading",
    },
    "row_press_exit": {
        "entry": "Verified compaction with both cartons still placed and the hand clear",
        "success": "Replan the next carton from the measured row geometry",
    },
    "side_slide_unloaded": {
        "entry": "Measured sideways displacement retained after checked unloading and exit",
        "success": "Square and verify clearance before advancing; no stocking credit",
    },
    "front_squaring_complete": {
        "entry": "Verified supported intermediate push with the pusher unloaded",
        "success": "Replan final forward insertion from the measured intermediate pose",
    },
    "motion_failure": {
        "entry": "Terminal measured state at a controller stop",
        "success": "Diagnostic fixture for a checked recovery; no task credit",
    },
    "stocking_milestone": {
        "entry": "Verified retreat after a completed carton with no retained load",
        "success": "Resume with the next pending carton using the measured scene",
    },
    "free_landing": {
        "entry": "Measured supported landing with the grasp hand clear",
        "success": "A separate checked push plan starts from the observed landing pose",
    },
    "withdrawal_planning": {
        "entry": "Aligned supported object with the loaded tool still at final depth",
        "success": "A guarded full-exit reference passes pose, motion and neighbor checks",
    },
    "side_registration": {
        "entry": "Shelf-supported object approached on an exposed side",
        "success": "Measured lateral displacement, support, unloaded hand and safe exit",
    },
    "front_squaring": {
        "entry": "Released, skewed carton at shelf front before advancing beside a neighbor",
        "success": "Measured yaw and support at the intermediate pose after unloading",
    },
    "contact_refinement": {
        "entry": "Supported carton near final depth with measured contact and pose error",
        "success": "Verified contact and pose dwell before release",
    },
    "loaded_push_recovery": {
        "entry": "Supported carton with measured pushing force and insufficient advance",
        "success": "Renewed measured advance without a force or support violation",
    },
    "push_recovery_exhausted": {
        "entry": "Terminal lack of advance after the allowed recovery window",
        "success": "Diagnostic failure fixture for future recovery evaluation",
    },
    "front_insertion": {
        "entry": "Released shelf-front carton after optional lateral registration; empty hand",
        "success": "Native insertion, squared contact, unloaded release and complete shelf exit",
    },
    "front_release": {
        "entry": "Shelf-front supported carton; retained wrapping grip before pressure unload",
        "success": "Measured unloaded support with pose/velocity recorded before pusher approach",
    },
    "grasp_lift": {
        "entry": "Upright table-supported carton; empty hand; measured inventory poses",
        "success": "Opposing contacts, measured rise, retained hand contact and table support lost",
    },
    "transfer": {
        "entry": "Verified lift; measured grasp transform and remaining inventory",
        "success": "Load retained through checked route to the shelf approach",
    },
    "shelf_place": {
        "entry": "Carried carton near shelf front; measured rotated extents and grasp transform",
        "success": "Measured shelf support within the original bounded seating range",
    },
    "contact_finish": {
        "entry": "Shelf-supported carton with measured lateral/yaw/depth error and contact forces",
        "success": "Required final placement/contact gates under unchanged task criteria",
    },
    "release_retreat": {
        "entry": "Supported completed contact skill; hand still loaded or touching",
        "success": "Unloaded hand and verified retreat without disturbing any required product",
    },
}


def entry_distribution(entries):
    """Describe observed states only; min/max are not a proposed sampling prior."""
    result = {}
    for skill in SKILL_CONTRACTS:
        own = [entry for entry in entries if entry["skill"] == skill]
        samples = [entry["objects"][entry["active_product"]] for entry in own]
        values = np.asarray([sample["position_world_m"] for sample in samples])
        result[skill] = dict(
            samples=len(samples),
            active_position_min_m=values.min(axis=0).tolist() if samples else None,
            active_position_max_m=values.max(axis=0).tolist() if samples else None,
            active_yaw_degrees=[sample["yaw_deg"] for sample in samples],
            entry_files=[entry["entry_file"] for entry in own],
            sampling_scope="empirical reached states; includes previous-skill disturbances",
            robustness_or_training_claim=False,
        )
    return result


def record_skill_entry(script, phase, *, case=None):
    skill = SKILL_PHASES.get(phase)
    if skill is None or any(
        entry["skill"] == skill and entry["active_product"] == script.product.name
        for entry in script.skill_entries
    ):
        return
    env = script.env
    objects = {}
    from abstract_stack.interactions import contact_forces

    for product in env.products:
        body = env.data.body(product.name)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
        )
        rotation = body.xmat.reshape(3, 3)
        force = contact_forces(env, product.name, script.hand_bodies)
        objects[product.name] = dict(
            position_world_m=body.xpos.tolist(),
            quaternion_wxyz=body.xquat.tolist(),
            yaw_deg=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
            tilt_deg=float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))),
            angular_then_linear_velocity_world=velocity.tolist(),
            half_size_m=list(product.half_size),
            mass_kg=float(env.model.body_mass[body.id]),
            target_world_m=list(product.target),
            table_normal_N=force["table"],
            shelf_normal_N=force["shelf"],
            hand_normal_N=force["hand"],
            initially_stocked=product.initially_stocked,
        )
    directory = (
        script.directory
        / "skill-entries"
        / (f"{len(script.skill_entries):03}-{script.product.name}-{skill}")
    )
    directory.mkdir(parents=True, exist_ok=False)
    entry = dict(
        kind="measured_skill_entry",
        skill=skill,
        contract=SKILL_CONTRACTS[skill],
        phase=phase,
        active_product=script.product.name,
        sim_time_s=float(env.data.time),
        reset_seed=env._seed,
        objects=objects,
        completed_products=sorted(script.observer.completed) if script.observer else [],
        case=dict(case or {}),
        robot_base_position_world_m=env.data.body("mabel_chassis").xpos.tolist(),
        tool_position_world_m=script.controller.position().tolist(),
        checkpoint="checkpoint",
        entry_file=str((directory / "entry.json").relative_to(script.directory)),
        training_eligible=False,
        scope="Privileged entry distribution/checkpoint; not a causal demonstration",
    )
    scene = json.loads((script.directory / "initial-scene/manifest.json").read_text())
    nominal = dict(zip(script.tracker.NAMES, script.tracker.nominal.tolist(), strict=True))
    observer = script.observer
    history = (
        dict(
            initial=observer.initial,
            incidents=observer.incidents,
            seen=[list(value) for value in sorted(observer.seen)],
            completed=sorted(observer.completed),
            retreat=observer.retreat,
            retreat_results=observer.retreat_results,
            maximum_displacements=observer.maximum_displacements,
        )
        if observer is not None
        else None
    )
    save_checkpoint(
        directory / "checkpoint",
        env.model,
        env.data,
        scene=scene,
        controller=dict(
            kind="stocking_skill_entry",
            tool_class=type(script.controller).__name__,
            tool_parameters=dict(getattr(script.controller, "tool_parameters", {})),
            finger_compensation_applied={
                k: v["previous"]
                for k, v in getattr(
                    getattr(script.controller, "finger_load_compensation", None), "joints", {}
                ).items()
            },
            tool_offset_m=script.controller.OFFSET.tolist(),
            tool_preferred_rotation_world=script.controller.ROTATION.tolist(),
            grasp_abduction_targets=dict(script.grasp_abduction_targets),
            open_posture_overrides=dict(script.controller.open_posture_overrides),
            abduction_targets=dict(script.controller.abduction_targets),
            tip_body_references={
                k: v.tolist()
                for k, v in getattr(script.controller, "tip_body_references", {}).items()
            },
            tool_posture_constraints={
                script.controller.ik.names[i]: float(value)
                for i, value in script.controller.ik.posture_constraints.items()
            },
            settings=asdict(script.config),
            tool_settings=asdict(script.controller.config),
            nominal_subset=nominal,
            hand_nominal_subset=dict(
                zip(
                    [*script.controller.names, "Lift"],
                    script.controller.command.tolist(),
                    strict=True,
                )
            ),
            applied_controls=env.data.ctrl.tolist(),
            active_rotation_world=script.controller.active_rotation.tolist(),
            nominal_cartesian_target_m=script.controller.last_target.tolist(),
            active_grip=script.active_grip,
            base_translation_origin_xy_m=None
            if script._base_translation_origin is None
            else script._base_translation_origin.tolist(),
            wrist_load_state_Nm=float(script.wrist_load),
            carry_reference=None
            if script.carry_reference is None
            else script.carry_reference.tolist(),
            payload_rotation_goal=None
            if script.payload_rotation_goal is None
            else script.payload_rotation_goal.tolist(),
            tracking_config=asdict(script.tracker.config),
        ),
        context=dict(
            skill_entry=entry,
            observer_history=history,
            prior_events=script.events if phase == "motion_failure" else script.events[:-1],
            task_memory=dict(stable_time=env._stable_time.tolist(), stocked=env._stocked.tolist()),
            fresh_task_credit=0,
            resume_scope="Exact native state; arbitrary skill-controller resume is not automatic",
        ),
        validation_controls=np.tile(env.data.ctrl, (10, 1)),
        portable_source=False,
        shared_model_checkpoint=(script.directory / script.skill_entries[0]["entry_file"]).parent
        / "checkpoint"
        if script.skill_entries
        else None,
    )
    (directory / "checkpoint/model.mjb").chmod(0o444)
    (directory / "entry.json").write_text(json.dumps(entry, indent=2, allow_nan=False) + "\n")
    script.skill_entries.append(entry)
    event = script.events[-1]
    if phase == script.phase:
        event["skill_entry"] = entry["entry_file"]
    else:
        event.setdefault("recovery_skill_entries", {})[skill] = entry["entry_file"]
    (script.directory / "skill-entries/index.json").write_text(
        json.dumps(
            dict(
                contracts=SKILL_CONTRACTS,
                entries=[entry["entry_file"] for entry in script.skill_entries],
                distribution=entry_distribution(script.skill_entries),
                unreached_skills_count_as_unreached=True,
                task_stage_denominators="summary.json includes all scheduled objects and failures",
                training_eligible=False,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def record_motion_failure(script, failure):
    """Record the stopped state without replacing the original failure."""
    if not script.config.record_skill_starts:
        return None
    try:
        record_skill_entry(
            script,
            "motion_failure",
            case={"failed_phase": script.phase, "failure": failure},
        )
    except (OSError, RuntimeError, ValueError, FloatingPointError) as error:
        return str(error)
    return None
