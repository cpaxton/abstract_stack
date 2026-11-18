#!/usr/bin/env python3
"""Retry a retained failed-plan grasp after an isolated held-out skill check."""

import argparse
import copy
import hashlib
import json
import os
import runpy
import sys
from dataclasses import asdict
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_passed_heldout(split, heldout, content_id):
    """Require every scheduled isolated skill case, with the identical policy."""
    if (
        split.get("split") != "heldout"
        or split.get("source_content_id") != content_id
        or heldout.get("source_content_id") != content_id
        or heldout.get("evaluation_complete") is not True
        or type(heldout.get("scheduled_grasps")) is not int
        or type(heldout.get("verified_grasps")) is not int
        or heldout["scheduled_grasps"] < 1
        or heldout["verified_grasps"] != heldout["scheduled_grasps"]
        or len(heldout.get("cases", [])) != heldout["scheduled_grasps"]
        or not all(case.get("grasp_verified") is True for case in heldout["cases"])
    ):
        raise ValueError("Isolated held-out grasp check has not passed for this frozen policy")


def worker(args):
    import numpy as np

    from abstract_stack.checkpoints import load_checkpoint, native_state, save_checkpoint
    from abstract_stack.continuation import CheckpointContinuationEnv
    from abstract_stack.evaluation import StageObserver
    from abstract_stack.grasp_skill import GraspSkillProbe
    from abstract_stack.interactions import CartesianHand, ScriptConfig
    from abstract_stack.mabel_controller import TrackingConfig

    reference = json.loads((args.output / "input_ref.json").read_text())
    source = Path(reference["checkpoint"])
    if (
        hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        != reference["manifest_sha256"]
    ):
        raise ValueError("Failed-plan checkpoint changed")
    checkpoint = load_checkpoint(source)
    before = native_state(checkpoint.model, checkpoint.data).copy()
    initialization = "recorded nominal/applied references"
    if checkpoint.controller["kind"] == "native_physics_only":
        if not args.initialize_measured_reference:
            raise ValueError(
                "Historical nominal labels unknown; explicitly initialize new references"
            )
        if (
            checkpoint.context.get("phase") != "pregrasp"
            or checkpoint.context.get("active") != args.product
        ):
            raise ValueError(
                "Measured-reference adapter requires the declared pregrasp focus state"
            )
        env = CheckpointContinuationEnv(checkpoint)
        settings = ScriptConfig(grasp_skill=True)
        hand = CartesianHand(env, settings)
        nominal = dict(zip([*hand.names, "Lift"], hand.ik.measured_q().tolist(), strict=True))
        observer = StageObserver(env)
        context = copy.deepcopy(checkpoint.context)
        initialization = (
            "new measured-pose nominal references; historical nominal labels remain unknown"
        )
        context.update(
            reference_initialization=initialization,
            original_checkpoint=str(source),
            original_checkpoint_manifest_sha256=reference["manifest_sha256"],
            original_controller=checkpoint.controller,
            skill_entry=dict(skill="grasp_lift", active_product=args.product),
            observer_history=dict(
                initial=observer.initial,
                incidents=[],
                seen=[],
                completed=[],
                retreat=None,
                retreat_results={},
                maximum_displacements=observer.maximum_displacements,
            ),
            task_memory=dict(stable_time=env._stable_time.tolist(), stocked=env._stocked.tolist()),
            history_scope="Original outcome retained; observer monitors this prefix only",
            fresh_stocking_credit=0,
        )
        controller = dict(
            kind="stocking_skill_entry",
            settings=asdict(settings),
            nominal_subset=nominal,
            hand_nominal_subset=nominal,
            active_rotation_world=checkpoint.data.xmat[hand.palm].reshape(3, 3).tolist(),
            applied_controls=checkpoint.data.ctrl.tolist(),
            active_grip=None,
            carry_reference=None,
            tracking_config=asdict(TrackingConfig()),
            reference_initialization=initialization,
        )
        saved = save_checkpoint(
            args.output / "adopted-entry",
            checkpoint.model,
            checkpoint.data,
            scene=checkpoint.scene,
            controller=controller,
            context=context,
            validation_controls=np.tile(checkpoint.data.ctrl, (10, 1)),
            portable_source=False,
            shared_model_checkpoint=checkpoint.directory,
        )
        if not np.array_equal(before, native_state(checkpoint.model, checkpoint.data)):
            raise ValueError("Reference initialization changed the native world")
        env.close()
        checkpoint = load_checkpoint(saved)
    if not np.array_equal(before, native_state(checkpoint.model, checkpoint.data)):
        raise ValueError("Recovery did not start at the exact failed-plan native state")
    env = CheckpointContinuationEnv(checkpoint)
    env.restore_task_memory(checkpoint.context["task_memory"])
    try:
        script = GraspSkillProbe(
            env,
            args.output / "episode",
            ScriptConfig(grasp_skill=True),
            product=args.product,
            entry_checkpoint=checkpoint,
            video=args.video,
            video_fps=15,
        )
        result = script.run()
        verified = script.grasp_skill_verified and result["failure"] is None
        write(
            args.output / "result.json",
            dict(
                kind="failed_plan_grasp_recovery",
                exact_starting_integration_state=True,
                reference_initialization=initialization,
                grasp_verified=verified,
                failure=result["failure"],
                peak_forces_N=result.get("peak_forces_N"),
                new_prefix_incidents=result["stage_evaluation"]["incidents"],
                original_outcome=checkpoint.context.get("original_outcome"),
                full_stocking_task_success=False,
                fresh_stocking_credit=0,
                recording_error=result.get("video_error"),
            ),
        )
        return 0 if verified and not result.get("video_error") else 2
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-source-ref", type=Path)
    parser.add_argument("--heldout-root", type=Path)
    parser.add_argument("--product", default="carton_0")
    parser.add_argument("--initialize-measured-reference", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--wall-seconds", type=float, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    if (
        args.checkpoint is None
        or args.policy_source_ref is None
        or args.heldout_root is None
        or not 0 < args.wall_seconds <= 180
    ):
        parser.error(
            "Require frozen policy, passed isolated held-out suite, checkpoint and bounded time"
        )
    from abstract_stack.checkpoint_manifest import inspect_checkpoint
    from abstract_stack.source_store import freeze_source, resolve_source_reference

    root = Path(__file__).resolve().parents[1]
    policy = resolve_source_reference(
        args.policy_source_ref, cache=root / "outputs/cache/source_runtime"
    )
    split = json.loads((args.heldout_root / "split.json").read_text())
    heldout = json.loads((args.heldout_root / "suite/report.json").read_text())
    require_passed_heldout(split, heldout, policy.content_id)
    source, _ = inspect_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    policy.write_reference(args.output / "source_ref.json")
    harness = freeze_source(
        root,
        paths=[
            "scripts/recover_grasp_skill.py",
            "scripts/run_frozen_replanning.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    harness.write_reference(args.output / "harness_source_ref.json")
    write(
        args.output / "input_ref.json",
        dict(
            checkpoint=str(source.resolve()),
            manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
            isolated_heldout_report_sha256=hashlib.sha256(
                (args.heldout_root / "suite/report.json").read_bytes()
            ).hexdigest(),
        ),
    )
    write(
        args.output / "manifest.json",
        dict(
            command=sys.argv,
            policy_source_content_id=policy.content_id,
            harness_content_id=harness.content_id,
            heldout_scope=split["skill"],
            initialize_measured_reference=args.initialize_measured_reference,
            wall_seconds=args.wall_seconds,
            fresh_stocking_credit=0,
        ),
    )
    command = [
        sys.executable,
        str(harness.runtime_root / "scripts/recover_grasp_skill.py"),
        "--worker",
        "--output",
        str(args.output),
        "--product",
        args.product,
    ]
    if args.initialize_measured_reference:
        command.append("--initialize-measured-reference")
    if args.video:
        command.append("--video")
    supervise = runpy.run_path(str(harness.runtime_root / "scripts/run_frozen_replanning.py"))[
        "supervise"
    ]
    result = supervise(
        command,
        cwd=root,
        environment=dict(
            os.environ,
            PYTHONPATH=str(policy.runtime_root / "src"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONFAULTHANDLER="1",
            ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
            ABSTRACT_STACK_SOURCE_STORE=str(policy.archive.parent),
        ),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
    )
    write(args.output / "process.json", result)
    print(json.dumps(dict(output=str(args.output), **result)))
    return result["returncode"] if not result["timeout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
