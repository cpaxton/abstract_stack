#!/usr/bin/env python3
"""Bounded fixed-geometry grasp/lift/hold development suite; no stocking credit."""

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
    from dataclasses import replace

    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.grasp_skill import GraspSkillProbe
    from abstract_stack.interactions import ScriptConfig

    checkpoint = None
    if args.checkpoint is None:
        config = replace(load_scene_config(scene="pick_place"), terminate_on_success=False)
        env = ShelfStockingEnv(config, seed=args.seeds[0])
    else:
        from abstract_stack.checkpoints import load_checkpoint
        from abstract_stack.continuation import CheckpointContinuationEnv

        if (
            hashlib.sha256((args.checkpoint / "manifest.json").read_bytes()).hexdigest()
            != args.checkpoint_sha
        ):
            raise ValueError("Grasp entry checkpoint manifest changed")
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint.scene["seed"] != args.seeds[0]:
            raise ValueError("Grasp entry seed does not match scheduled seed")
        env = CheckpointContinuationEnv(checkpoint)
        env.restore_task_memory(checkpoint.context["task_memory"])
    try:
        script = GraspSkillProbe(
            env,
            args.output / "episode",
            ScriptConfig(grasp_skill=True, record_skill_starts=args.record_skill_starts),
            product=args.products[0],
            entry_checkpoint=checkpoint,
            video=args.video,
            video_fps=15,
        )
        result = script.run()
        verified = script.grasp_skill_verified and result["failure"] is None
        write(
            args.output / "grasp_result.json",
            dict(
                kind="isolated_grasp_skill_result",
                seed=args.seeds[0],
                product=args.products[0],
                grasp_verified=verified,
                failure=result["failure"],
                peak_forces_N=result.get("peak_forces_N"),
                incidents=result["stage_evaluation"]["incidents"],
                initial_distribution=script.skill_entries,
                whole_shelf_success=False,
                training_eligible=False,
                recording_error=result.get("video_error"),
            ),
        )
        return 0 if verified and not result.get("video_error") else 2
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--products", nargs="+", choices=["carton_0", "carton_1"], default=["carton_0", "carton_1"]
    )
    parser.add_argument("--case-wall-seconds", type=float, default=60)
    parser.add_argument("--video", action="store_true", help="Record the first scheduled case")
    parser.add_argument(
        "--record-skill-starts", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint", type=Path, help="One previously recorded grasp-entry checkpoint"
    )
    parser.add_argument("--checkpoint-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not 1 <= len(args.seeds) <= 8
        or len(set(args.seeds)) != len(args.seeds)
        or any(s < 0 for s in args.seeds)
        or len(set(args.products)) != len(args.products)
        or not 0 < args.case_wall_seconds <= 180
    ):
        parser.error("Bound unique development cases and wall time")
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    if args.checkpoint:
        from abstract_stack.checkpoint_manifest import inspect_checkpoint

        args.checkpoint, metadata = inspect_checkpoint(args.checkpoint)
        args.checkpoint = args.checkpoint.resolve()
        if len(args.seeds) != 1 or len(args.products) != 1:
            parser.error("A retained entry requires one seed and one product")
        args.checkpoint_sha = hashlib.sha256(
            (args.checkpoint / "manifest.json").read_bytes()
        ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/evaluate_grasp_skill.py",
            "scripts/run_frozen_replanning.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    rows = [
        dict(seed=s, product=p, process=None, outcome="not_run", grasp_verified=False)
        for s in args.seeds
        for p in args.products
    ]
    write(
        args.output / "manifest.json",
        dict(
            command=sys.argv,
            source_content_id=frozen.content_id,
            git=frozen.git,
            scene="pick_place",
            geometry_randomization=False,
            whole_shelf_success=False,
            force_abort_N=25,
            hold_seconds=0.5,
            case_wall_seconds=args.case_wall_seconds,
            checkpoint=None if args.checkpoint is None else str(args.checkpoint),
            checkpoint_manifest_sha256=args.checkpoint_sha,
            record_skill_starts=args.record_skill_starts,
        ),
    )

    def report():
        write(
            args.output / "report.json",
            dict(
                kind="fixed_geometry_grasp_skill_development",
                source_content_id=frozen.content_id,
                scheduled_grasps=len(rows),
                verified_grasps=sum(r["grasp_verified"] for r in rows),
                evaluation_complete=all(r["outcome"] != "not_run" for r in rows),
                cases=rows,
                statistical_generalization_claim=False,
                whole_shelf_success=False,
                training_eligible=False,
            ),
        )

    report()
    supervise = runpy.run_path(str(frozen.runtime_root / "scripts/run_frozen_replanning.py"))[
        "supervise"
    ]
    for index, row in enumerate(rows):
        directory = args.output / f"{index:03}-seed{row['seed']}-{row['product']}"
        directory.mkdir()
        command = [
            sys.executable,
            str(frozen.runtime_root / "scripts/evaluate_grasp_skill.py"),
            "--worker",
            "--output",
            str(directory),
            "--seeds",
            str(row["seed"]),
            "--products",
            row["product"],
        ]
        if args.video and index == 0:
            command.append("--video")
        if not args.record_skill_starts:
            command.append("--no-record-skill-starts")
        if args.checkpoint:
            command += [
                "--checkpoint",
                str(args.checkpoint),
                "--checkpoint-sha",
                args.checkpoint_sha,
            ]
        process = supervise(
            command,
            cwd=root,
            environment=dict(
                os.environ,
                PYTHONPATH=str(frozen.runtime_root / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONFAULTHANDLER="1",
                ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
                ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ),
            log=directory / "run.log",
            wall_seconds=args.case_wall_seconds,
        )
        row.update(process=process, outcome="process_failure_or_unknown")
        write(directory / "process.json", process)
        report()
        result = directory / "grasp_result.json"
        if process["returncode"] in (0, 2) and not process["timeout"] and result.exists():
            try:
                value = json.loads(result.read_text())
                if value["seed"] != row["seed"] or value["product"] != row["product"]:
                    raise ValueError("Grasp result belongs to another scheduled case")
                row.update(
                    outcome="completed_grasp_probe",
                    failure=value["failure"],
                    grasp_verified=value["grasp_verified"] is True and process["returncode"] == 0,
                    incidents=value["incidents"],
                    peak_forces_N=value["peak_forces_N"],
                    recording_error=value["recording_error"],
                )
            except (ValueError, KeyError, TypeError, OSError):
                row["outcome"] = "invalid_grasp_result"
        report()
    print(
        json.dumps(
            dict(
                output=str(args.output),
                scheduled=len(rows),
                verified=sum(r["grasp_verified"] for r in rows),
            )
        )
    )
    return 0 if all(r["grasp_verified"] for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
