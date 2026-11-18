#!/usr/bin/env python3
"""Bounded physical release from a retained supported-front entry; zero fresh credit."""

import argparse
import copy
import hashlib
import json
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
    if state["kind"] != "stocking_skill_entry" or checkpoint.context["skill_entry"]["skill"] != (
        "front_insertion"
        if args.insert_only
        else "release_retreat"
        if args.withdrawal
        else "transfer"
        if args.transfer
        else "front_release"
    ):
        raise ValueError("Require the matching recorded front-release or withdrawal skill entry")
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
    if args.back_force_target is not None:
        settings["insertion_back_force_target"] = args.back_force_target
    if args.alignment_seconds is not None:
        settings["alignment_timeout_seconds"] = args.alignment_seconds
    script = StockingScript(env, args.output / "episode", ScriptConfig(**settings))
    if "grasp_abduction_targets" in state:
        script.grasp_abduction_targets = dict(state["grasp_abduction_targets"])
    script.product = next(
        p for p in env.products if p.name == checkpoint.context["skill_entry"]["active_product"]
    )
    if args.record_skill_starts:
        scene = copy.deepcopy(checkpoint.scene)
        scene["continuation"] = dict(checkpoint=str(path), fresh_task_credit=0)
        initial = script.directory / "initial-scene"
        initial.mkdir()
        write(initial / "manifest.json", scene)
    if args.withdrawal or args.insert_only and state.get("tool_class") != "CartesianHand":
        from abstract_stack.finger_pusher import FingerPusher, PairedFingerPusher, WallFingerPusher
        from abstract_stack.lateral_contact import SideFingerPusher

        kinds = {
            kind.__name__: kind
            for kind in (FingerPusher, PairedFingerPusher, WallFingerPusher, SideFingerPusher)
        }
        if state.get("tool_class") not in kinds:
            raise ValueError("Withdrawal continuation requires an explicitly recorded FingerPusher")
        selected = state.get("tool_parameters") or next(
            e["selected"]
            for e in reversed(checkpoint.context["prior_events"])
            if e["phase"] == "pusher_plan" and e.get("selected")
        )
        kind = kinds[state["tool_class"]]
        tool_settings = dict(state.get("tool_settings", state["settings"]))
        tool_settings.update({k: v for k, v in settings.items() if state["settings"].get(k) != v})
        script.controller = kind(env, ScriptConfig(**tool_settings), **selected)
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
    if "open_posture_overrides" in state:
        hand.open_posture_overrides = dict(state["open_posture_overrides"])
    if "abduction_targets" in state:
        hand.abduction_targets = dict(state["abduction_targets"])
    if state.get("tip_body_references"):
        hand.tip_body_references = {
            k: np.asarray(v) for k, v in state["tip_body_references"].items()
        }
    hand.command = np.array([state["hand_nominal_subset"][n] for n in (*hand.names, "Lift")])
    hand.active_rotation = np.asarray(state["active_rotation_world"])
    if "nominal_cartesian_target_m" in state:
        hand.last_target = np.asarray(state["nominal_cartesian_target_m"])
        initialization = "recorded nominal Cartesian reference"
    else:
        hand.last_target = hand.ik.forward(hand.command)[0]
        initialization = "new local reference from saved nominal FK at restored base"
    tracking = TrackingConfig(**state["tracking_config"])
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
    try:
        if args.insert_only:
            if state["active_grip"] is not None or state["carry_reference"] is not None:
                raise ValueError("Insertion-only continuation requires a released carton")
            script._insert_from_front()
        elif args.transfer:
            if script.carry_reference is None or script.active_grip is None:
                raise ValueError("Transfer continuation requires a recorded retained grasp")
            if args.base_dy:
                script._move_base(
                    [0, args.base_dy], reason="development_loaded_front_pose", carrying=True
                )
            grip = script._transfer_and_place_front(script.active_grip)
            script._release_and_insert_front(grip)
        elif args.withdrawal:
            from abstract_stack.contact_execution import squared_contact_retreat

            squared_contact_retreat(script, {})
            if args.base_goal is not None:
                script._restore_grasp_hand()
                base = env.data.body("mabel_chassis").xpos[:2].copy()
                script._move("empty_navigation_clear", [base[0] - 0.25, base[1] - 0.20, 1.35])
                script._move_base(
                    np.asarray(args.base_goal) - base, reason="inventory_grasp_workspace"
                )
        elif args.insert:
            script._release_and_insert_front(state["active_grip"])
        else:
            script._precision_release(state["active_grip"])
            goal = hand.last_target.copy()
            for _ in range(int(0.5 / env.dt)):
                script._tick(goal, 0)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failure = str(error)
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
            reference_initialization=initialization,
            config=vars(script.config),
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
    parser.add_argument("--record-skill-starts", action="store_true")
    parser.add_argument("--withdrawal", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--base-dy", type=float, default=0)
    parser.add_argument("--base-goal", type=float, nargs=2)
    parser.add_argument("--wall-seconds", type=float, default=90)
    parser.add_argument("--paired-fingers", action="store_true")
    parser.add_argument("--back-force-target", type=float)
    parser.add_argument("--alignment-seconds", type=float)
    parser.add_argument("--wrist-offset", type=float)
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 180:
        parser.error("Keep skill continuations within 180 wall seconds")
    if sum([args.withdrawal, args.insert, args.transfer, args.insert_only]) > 1:
        parser.error("Choose front insertion, transfer or withdrawal continuation")
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
    if args.insert:
        command.append("--insert")
    if args.insert_only:
        command.append("--insert-only")
    if args.record_skill_starts:
        command.append("--record-skill-starts")
    if args.withdrawal:
        command.append("--withdrawal")
    if args.transfer:
        command += ["--transfer", "--base-dy", str(args.base_dy)]
    if args.base_goal is not None:
        command += ["--base-goal", *(str(value) for value in args.base_goal)]
    if args.back_force_target is not None:
        command += ["--back-force-target", str(args.back_force_target)]
    if args.alignment_seconds is not None:
        command += ["--alignment-seconds", str(args.alignment_seconds)]
    if args.wrist_offset is not None:
        command += ["--wrist-offset", str(args.wrist_offset)]
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
