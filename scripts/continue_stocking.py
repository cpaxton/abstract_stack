#!/usr/bin/env python3
"""Continue a cleared shelf state into another physical table pickup and stock."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    from dataclasses import asdict, replace

    import mujoco
    import numpy as np

    from abstract_stack.checkpoints import load_checkpoint, native_state, save_checkpoint
    from abstract_stack.contact_rollout import ContactMonitor, RolloutScene
    from abstract_stack.continuation import CheckpointContinuationEnv, reassign_empty_slot
    from abstract_stack.evaluation import StageObserver
    from abstract_stack.interactions import InteractionFailure, ScriptConfig
    from abstract_stack.load_aware_pusher import LoadAwareFingerPusher
    from abstract_stack.mabel_controller import MabelController
    from abstract_stack.receding_stocking import RecedingStockingScript
    from abstract_stack.replanning import SearchConfig
    from abstract_stack.withdrawal import supported_withdrawal

    reference = json.loads((args.output / "checkpoint_ref.json").read_text())
    checkpoint_path = Path(reference["path"])
    if (
        hashlib.sha256((checkpoint_path / "manifest.json").read_bytes()).hexdigest()
        != reference["sha256"]
    ):
        raise ValueError("Source checkpoint reference changed")
    restored = load_checkpoint(checkpoint_path)
    if args.export_push_state:
        source = json.loads(args.source_summary.read_text())
        with np.load(args.source_summary.parent / "continuation.npz") as archive:
            if not np.array_equal(
                archive["terminal_state"], native_state(restored.model, restored.data)
            ):
                raise ValueError("Controller evidence and checkpoint native terminal state differ")
        selected = next(
            e["selected"] for e in reversed(source["events"]) if e["phase"] == "pusher_plan"
        )
        environment = CheckpointContinuationEnv(restored)
        configuration = ScriptConfig(**restored.controller["settings"])
        push_hand = LoadAwareFingerPusher(environment, configuration, **selected)
        active = source["events"][-1]["product"]
        push_scene = RolloutScene.from_env(environment, active)
        push_monitor = ContactMonitor(restored.model, restored.data, push_scene)
        failure = push_monitor.sample(restored.data, elapsed=0)
        if failure or source["stage_evaluation"]["incidents"]:
            raise ValueError("Cannot adopt a guarded push from an inadmissible source state")
        summary = dict(source, controller=asdict(configuration))
        save_checkpoint(
            args.output / "push-checkpoint",
            restored.model,
            restored.data,
            scene=restored.scene,
            controller=dict(
                kind="retained_finger_push",
                settings=asdict(configuration),
                pusher_settings=selected,
                nominal_subset=restored.controller["nominal_subset"],
                tool_offset_m=push_hand.OFFSET.tolist(),
                active_rotation_world=restored.controller["active_rotation_world"],
                applied_controls=restored.data.ctrl.tolist(),
            ),
            context=dict(
                source_summary=summary,
                original_outcome=source["failure"],
                original_episode_outcome=restored.context.get("original_outcome"),
                contact_monitor=push_monitor.state_dict(),
                prior_stage_evaluation=source["stage_evaluation"],
                controller_adoption="Explicit pusher plan and matching full state; no physics step",
                force_peak_scope="New push mode; previous-stage evidence retained separately",
            ),
        )
        write(
            args.output / "summary.json",
            dict(
                kind="guarded_push_controller_adoption",
                task_success=False,
                fresh_task_credit=0,
                native_state_unchanged=True,
                pusher_settings=selected,
                continuation=dict(simulated_seconds=0, failure=None, fresh_task_credit=0),
            ),
        )
        return
    carrying = restored.controller["kind"] == "continuation_grasp"
    if not carrying and restored.controller["kind"] != "retained_finger_withdrawal":
        raise ValueError("Require a completed withdrawal or an explicit carried grasp checkpoint")
    initial = native_state(restored.model, restored.data)
    env = CheckpointContinuationEnv(restored)
    np.testing.assert_array_equal(native_state(env.model, env.data), initial)
    source_summary = restored.context["source_summary"]
    effective_scene = restored.scene
    if args.swap_slot_with:
        env.products, effective_scene = reassign_empty_slot(
            env, restored.scene, args.product, args.swap_slot_with
        )
    config = replace(
        ScriptConfig(**restored.controller.get("settings", source_summary["controller"])),
        pregrasp_height=args.pregrasp_height,
    )
    prior_active = source_summary["events"][-1]["product"]
    nominal_names = [*(f"Left Arm {i}" for i in range(1, 8)), "Left Wrist", "Lift"]
    nominal = np.array([restored.controller["nominal_subset"][n] for n in nominal_names])
    hand, monitor = None, None
    if not carrying:
        hand = LoadAwareFingerPusher(env, config, **restored.controller["pusher_settings"])
        hand.command = nominal.copy()
        wrist = hand.ik.names.index("Left Wrist")
        wrist_range = restored.controller.get("planning_wrist_range")
        if wrist_range:
            hand.ik.posture_constraints.pop(wrist)
            hand.ik.constrain_reference_range("Left Wrist", *wrist_range)
        hand.active_rotation = np.asarray(restored.controller["active_rotation_world"])
        scene = RolloutScene.from_env(env, prior_active)
        monitor = ContactMonitor(env.model, env.data, scene)
        monitor.restore_state(restored.context["contact_monitor"], env.data)
    observer = StageObserver(env)
    observer.incidents.extend((restored.context.get("contact_monitor") or {}).get("incidents", []))
    memory = restored.context.get("handoff", {})
    if carrying:
        if memory["active_product"] != args.product:
            raise ValueError("Carried checkpoint product differs from the scheduled manipulation")
        env.restore_task_memory(memory)
        observer.initial = memory["observer_initial"]
        observer.incidents = memory["observer_incidents"].copy()
        observer.completed = set(memory["observer_completed"])
        observer.maximum_displacements = memory["observer_maximum_displacements"].copy()
    controls, traces, names = [], [], []
    result = dict(
        kind="continuous_checkpoint_next_stock",
        fresh_task_credit=0,
        task_success=False,
        original_outcome=restored.context.get("original_outcome"),
        inherited_contact_history=restored.context.get("contact_monitor"),
        initially_stocked_products=[p.name for p in env.products if p.initially_stocked],
        earlier_continuation_product=prior_active,
        scheduled_pick_products=[args.product],
        failure=None,
        stages=dict.fromkeys(("pick", "place", "final_position", "retreat"), False),
        placement_assignment=effective_scene.get("placement_assignment"),
    )
    solver = None
    source_time = float(env.data.time)

    class RecordedScript(RecedingStockingScript):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            self.native_trace, self.native_names = [], []

        def _apply_tick(self, action):
            before = float(self.env.data.time)
            previous_nominal = self.tracker.nominal.copy()
            previous_applied = self.tracker.applied.copy()
            previous_report = self.tracker.last_report
            previous_grip = self.active_grip
            try:
                return super()._apply_tick(action)
            except (RuntimeError, ValueError, FloatingPointError):
                if self.env.data.time == before:
                    if self.events:
                        self.events[-1]["unexecuted_tracking_proposal"] = self.tracker.last_report
                    self.tracker.nominal = previous_nominal
                    self.tracker.applied = previous_applied
                    self.tracker.last_report = previous_report
                    self.active_grip = previous_grip
                    by_name = dict(zip(self.tracker.NAMES, previous_nominal, strict=True))
                    self.controller.command = np.array(
                        [by_name[n] for n in [*self.controller.names, "Lift"]]
                    )
                raise

        def _sample(self, e):
            before = len(self.controls)
            try:
                super()._sample(e)
            finally:
                if len(self.controls) > before:
                    pairs = {}
                    wrench = np.empty(6)
                    for index, contact in enumerate(e.data.contact):
                        b1, b2 = (
                            int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)
                        )
                        if (
                            b1 in self.product_bodies
                            and b2 in self.product_bodies
                            and contact.efc_address >= 0
                        ):
                            mujoco.mj_contactForce(e.model, e.data, index, wrench)
                            pair = tuple(sorted((b1, b2)))
                            pairs[pair] = pairs.get(pair, 0) + max(0.0, float(wrench[0]))
                    body = e.data.body(self.product.name)
                    rotation = body.xmat.reshape(3, 3)
                    self.native_trace.append(
                        [
                            float(e.data.time),
                            *body.xpos.tolist(),
                            float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                            self.last_forces["hand"],
                            self.last_forces["back"],
                            max(pairs.values(), default=0.0),
                            self.last_forces["support"],
                        ]
                    )
                    self.native_names.append(self.product.name)

    def observe_clear(e):
        e.update_task_state(e.config.timestep)
        observer.sample("extended_clearance", prior_active)

    try:
        if not carrying and hand.position()[0] < env.config.front_x + 0.025:
            distance = env.config.front_x + 0.04 - hand.position()[0]
            report, commands, samples = supported_withdrawal(
                hand,
                nominal,
                monitor,
                None,
                resume_withdrawal=True,
                clear_distance_m=max(0.04, float(distance)),
                clear_height_m=0.025,
                clear_speed_m_s=0.02,
                on_physics_step=observe_clear,
            )
            controls.extend(commands)
            traces.extend(samples)
            names.extend([prior_active] * len(commands))
            result["extended_clearance"] = report
            nominal = np.asarray(report["terminal_nominal"])
            if report["failure"]:
                raise InteractionFailure(report["failure"])
        elif not carrying:
            result["extended_clearance"] = dict(
                already_outside=True, measured_tool_m=hand.position().tolist()
            )
        if not carrying:
            controller = dict(
                restored.controller,
                nominal_subset=dict(zip(nominal_names, nominal.tolist(), strict=True)),
                applied_controls=env.data.ctrl.tolist(),
            )
            save_checkpoint(
                args.output / "outside-checkpoint",
                env.model,
                env.data,
                scene=effective_scene,
                controller=controller,
                context=dict(
                    restored.context,
                    contact_monitor=monitor.state_dict(),
                    parent_checkpoint=restored.manifest["hashes"]["state.npz"],
                ),
            )
        if args.stop_after == "clear":
            return
        solver = RecordedScript(
            env,
            args.output,
            config,
            search_config=SearchConfig(wall_seconds=20, root_wall_seconds=20),
            loaded_route=True,
            route_admission=True,
            table_front=True,
            base_reposition=True,
            finger_load_compensation=True,
            bounded_contact_ik=True,
            square_backstop=True,
            contact_recovery=True,
            payload_load_compensation=True,
            payload_wrist_reserve=args.payload_wrist_reserve,
            split_insertion=not args.no_split_insertion,
            loaded_transit_yaws=(0, -20, 20, -40, 40),
        )
        solver.product = next(p for p in env.products if p.name == args.product)
        if solver.product.initially_stocked or args.product == prior_active:
            raise ValueError("Schedule a remaining table carton, not an established fixture")
        solver.observer = observer
        solver.observer.completed.update(
            p.name
            for i, p in enumerate(env.products)
            if env._stocked[i] and p.name in (prior_active, *result["initially_stocked_products"])
        )
        solver.controller.command = nominal.copy()
        solver.tracker = MabelController(
            env.model,
            env.data,
            solver._tracking_config(),
            nominal_references=dict(zip(nominal_names, nominal.tolist(), strict=True)),
        )
        solver.base_start = env.data.body("mabel_chassis").xpos.copy()
        if carrying:
            solver.active_grip = float(memory["active_grip"])
            solver.carry_reference = np.asarray(memory["carry_reference"])
            solver.payload_rotation_goal = np.asarray(memory["payload_rotation_goal"])
            solver.controller.active_rotation = np.asarray(
                restored.controller["active_rotation_world"]
            )
            solver.last_forces = memory["last_forces"]
            solver.wrist_load = memory["wrist_load"]
            solver.events.extend(memory.get("stage_events", []))
            if (
                not solver.last_forces.get("opposing_contacts")
                or solver.last_forces["table"] > 0.15
            ):
                raise InteractionFailure(
                    "Loaded resume lacks a verified free opposing-contact grasp"
                )
            solver._stage("resume_loaded_transfer")
            solver._move("align_shelf", solver.controller.position(), solver.active_grip)
            grip = solver._seat_front(solver.active_grip)
            if args.stop_after != "place-front":
                solver._release_and_insert_front(grip)
            return
        solver._stage("checked_grasp_preshape")
        desired = solver.controller.finger_targets(0)
        for _ in range(400):
            action = dict(zip(nominal_names, nominal.tolist(), strict=True))
            error = 0.0
            for name, target in desired.items():
                current = env.data.ctrl[env.model.actuator(name).id]
                action[name] = float(
                    current + np.clip(target - current, -0.5 * env.dt, 0.5 * env.dt)
                )
                error = max(error, abs(target - current))
            solver._apply_tick(action)
            if error < 1e-6:
                break
        else:
            raise InteractionFailure("grasp_preshape: native target budget exhausted")
        if args.stop_after == "preshape":
            return
        selected = solver._select_grasp([solver.product])
        if args.stop_after == "plan":
            return
        if args.stop_after == "place-front":
            solver._pick_and_place_front(selected)
        else:
            solver._one_box(selected)
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["failure"] = str(error)
    finally:
        if solver is not None:
            controls.extend(solver.controls)
            traces.extend(solver.native_trace)
            names.extend(solver.native_names)
            evaluation = observer.report(solver.events)
            result.update(
                controller=asdict(config),
                events=solver.events,
                stage_evaluation=evaluation,
                controller_evidence=solver.episode_evidence(),
                tracking_controller=solver.tracker.last_report,
                effective_tracking_config=asdict(solver.tracker.config),
            )
            np.savez_compressed(
                args.output / "physics_trace.npz", samples=np.asarray(solver.rows).reshape(-1, 23)
            )
            result["stages"] = {
                k: evaluation["products"][args.product][k]
                for k in ("pick", "place", "final_position", "retreat")
            }
            memory = dict(
                active_product=solver.product.name,
                active_grip=solver.active_grip,
                carry_reference=None
                if solver.carry_reference is None
                else solver.carry_reference.tolist(),
                payload_rotation_goal=None
                if solver.payload_rotation_goal is None
                else solver.payload_rotation_goal.tolist(),
                wrist_load=solver.wrist_load,
                last_forces=solver.last_forces,
                observer_initial=observer.initial,
                observer_incidents=observer.incidents,
                observer_completed=sorted(observer.completed),
                observer_maximum_displacements=observer.maximum_displacements,
                stable_time=env._stable_time.tolist(),
                stocked=env._stocked.tolist(),
                stage_events=[
                    dict(e, inherited=True)
                    for e in solver.events
                    if any(
                        e.get(k)
                        for k in (
                            "verified_lift",
                            "verified_table_support",
                            "verified_shelf_support",
                            "back_contact_verified",
                        )
                    )
                    or e["phase"] == "grasp_precheck"
                ],
            )
            try:
                known = dict(
                    zip(solver.tracker.NAMES, solver.tracker.nominal.tolist(), strict=True)
                )
                save_checkpoint(
                    args.output / "terminal-checkpoint",
                    env.model,
                    env.data,
                    scene=effective_scene,
                    controller=dict(
                        kind="continuation_grasp"
                        if solver.active_grip is not None
                        else "continuation_empty_hand",
                        settings=asdict(config),
                        nominal_subset=known,
                        applied_controls=env.data.ctrl.tolist(),
                        active_rotation_world=solver.controller.active_rotation.tolist(),
                        tracking_config=asdict(solver.tracker.config),
                    ),
                    context=dict(
                        source_summary=source_summary,
                        original_outcome=restored.context.get("original_outcome"),
                        handoff=memory,
                        previous_continuation_result=result["failure"],
                        parent_checkpoint=restored.manifest["hashes"]["state.npz"],
                    ),
                )
            except (ValueError, OSError, RuntimeError) as error:
                result["checkpoint_error"] = str(error)
        result["continuation"] = dict(
            simulated_seconds=float(env.data.time) - source_time,
            failure=result["failure"],
            withdrawal=None,
            full_stocking_success=False,
            fresh_task_credit=0,
        )
        np.savez_compressed(
            args.output / "continuation.npz",
            controls=np.asarray(controls).reshape(-1, env.model.nu),
            contact_trace=np.asarray(traces).reshape(-1, 9),
            active_names=np.asarray(names),
            terminal_state=native_state(env.model, env.data),
        )
        write(args.output / "summary.json", result)
        print(
            json.dumps(
                dict(
                    failure=result["failure"],
                    stages=result["stages"],
                    simulated_seconds=result["continuation"]["simulated_seconds"],
                )
            )
        )
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--product", default="carton_9")
    parser.add_argument("--pregrasp-height", type=float, default=0.25)
    parser.add_argument("--payload-wrist-reserve", type=float, default=0.4)
    parser.add_argument("--swap-slot-with")
    parser.add_argument("--no-split-insertion", action="store_true")
    parser.add_argument("--export-push-state", action="store_true")
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument(
        "--stop-after",
        choices=("clear", "preshape", "plan", "place-front", "stock"),
        default="stock",
    )
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.source_summary:
        args.source_summary = args.source_summary.resolve()
    if args.export_push_state and args.source_summary is None:
        parser.error("Push adoption requires explicit source summary matching the checkpoint state")
    if (
        not 0 < args.wall_seconds <= 600
        or not 0.15 <= args.pregrasp_height <= 0.35
        or not 0.15 <= args.payload_wrist_reserve <= 0.5
    ):
        parser.error("Bound the native continuation wall time")
    if args.worker:
        worker(args)
        return 0
    from abstract_stack.checkpoint_manifest import inspect_checkpoint
    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    checkpoint, _ = inspect_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "checkpoint_ref.json",
        dict(
            path=str(checkpoint.resolve()),
            sha256=hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest(),
        ),
    )
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/continue_stocking.py",
            "scripts/run_frozen_replanning.py",
            "uv.lock",
            "pyproject.toml",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    write(
        args.output / "manifest.json",
        dict(
            source_content_id=frozen.content_id,
            git=frozen.git,
            settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            sha256={
                name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                for name in ("checkpoint_ref.json", "source_ref.json")
            },
            fresh_task_credit=0,
            task_success=False,
        ),
    )
    runtime = frozen.runtime_root
    supervise = runpy.run_path(str(runtime / "scripts/run_frozen_replanning.py"))["supervise"]
    write(args.output / "process.json", dict(outcome="unknown", fresh_task_credit=0))
    command = [
        sys.executable,
        str(runtime / "scripts/continue_stocking.py"),
        "--worker",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(args.output),
        "--product",
        args.product,
        "--stop-after",
        args.stop_after,
        "--pregrasp-height",
        str(args.pregrasp_height),
        "--payload-wrist-reserve",
        str(args.payload_wrist_reserve),
    ]
    if args.swap_slot_with:
        command += ["--swap-slot-with", args.swap_slot_with]
    if args.no_split_insertion:
        command.append("--no-split-insertion")
    if args.export_push_state:
        command += ["--export-push-state", "--source-summary", str(args.source_summary)]
    result = supervise(
        command,
        cwd=root,
        environment=dict(
            os.environ,
            PYTHONPATH=str(runtime / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONFAULTHANDLER="1",
            ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
        ),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    result.update(
        outcome="completed_continuation"
        if result["returncode"] == 0 and (args.output / "summary.json").exists()
        else "unknown",
        fresh_task_credit=0,
        task_success=False,
    )
    write(args.output / "process.json", result)
    print(json.dumps(result))
    return 0 if result["outcome"] == "completed_continuation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
