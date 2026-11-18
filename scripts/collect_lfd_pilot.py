#!/usr/bin/env python3
"""Collect at most two explicit DR development sensing fixtures, never task demos."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="cases", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--wall-seconds", type=float, default=60)
    args = parser.parse_args()
    if (
        not 1 <= len(args.cases) <= 2
        or not 1 <= args.steps <= 100
        or not 0 < args.wall_seconds <= 60
    ):
        parser.error("Keep this pilot to at most two cases, 100 transitions and 60 s per process")
    root = Path(__file__).resolve().parents[1]
    cases = []
    for directory in args.cases:
        directory = directory.resolve()
        config_path, context_path = directory / "scene.json", directory / "context.json"
        config_bytes, context_bytes = config_path.read_bytes(), context_path.read_bytes()
        config, context = json.loads(config_bytes), json.loads(context_bytes)
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        if (
            not isinstance(config, dict)
            or not isinstance(context, dict)
            or context.get("split") != "development"
            or type(context.get("scene_seed")) is not int
            or context["scene_seed"] < 0
            or type(context.get("reset_seed")) is not int
            or context["reset_seed"] < 0
            or context.get("scene_config_sha256") != config_hash
        ):
            parser.error("Require an explicit development case with matching config hash and seeds")
        cases.append(
            dict(
                path=str(directory),
                config_path=str(config_path),
                context_path=str(context_path),
                context=context,
                config_sha256=config_hash,
                context_sha256=hashlib.sha256(context_bytes).hexdigest(),
                product_count=config["product_count"],
                shelf_levels=len(config["levels"]),
            )
        )
    if len({c["path"] for c in cases}) != len(cases):
        parser.error("Schedule each development case once")
    from abstract_stack.source_store import freeze_source

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/export_lfd_fixture.py",
            "scripts/collect_lfd_pilot.py",
            "scripts/run_frozen_replanning.py",
            "uv.lock",
            "pyproject.toml",
        ],
    )
    frozen.write_reference(output / "source_ref.json")
    write(
        output / "manifest.json",
        dict(
            kind="bounded_dr_sensing_pilot",
            cases=cases,
            source_content_id=frozen.content_id,
            git=frozen.git,
            source_ref_sha256=hashlib.sha256((output / "source_ref.json").read_bytes()).hexdigest(),
            steps=args.steps,
            case_wall_seconds=args.wall_seconds,
            camera_shape=[120, 160, 3],
            purpose="pipeline_fixture",
            training_eligible=False,
            fresh_task_credit=0,
            task_success=False,
            uploads=False,
        ),
    )
    runtime = frozen.runtime_root
    supervise = runpy.run_path(str(runtime / "scripts/run_frozen_replanning.py"))["supervise"]
    rows = []
    for index, case in enumerate(cases):
        directory = output / f"case-{index:02}"
        directory.mkdir()
        write(directory / "inputs.json", case)
        row = dict(
            case=case,
            outcome="unknown",
            fresh_task_credit=0,
            task_success=False,
            training_eligible=False,
            scheduled_transitions=args.steps,
        )
        write(directory / "process.json", row)
        command = [
            sys.executable,
            str(runtime / "scripts/export_lfd_fixture.py"),
            "--scene-config",
            case["config_path"],
            "--domain-context",
            case["context_path"],
            "--seed",
            str(case["context"]["scene_seed"]),
            "--reset-seed",
            str(case["context"]["reset_seed"]),
            "--steps",
            str(args.steps),
            "--cameras",
            "--output",
            str(directory / "episode"),
        ]
        process = supervise(
            command,
            cwd=root,
            environment=dict(
                os.environ,
                PYTHONPATH=str(runtime / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONFAULTHANDLER="1",
                ABSTRACT_STACK_SOURCE_REFERENCE=str(output / "source_ref.json"),
                ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ),
            log=directory / "run.log",
            wall_seconds=args.wall_seconds,
        )
        row["process"] = process
        episode_manifest = directory / "episode/episode.json"
        if process["returncode"] == 0 and episode_manifest.exists():
            from abstract_stack.lfd import load_episode

            try:
                episode = load_episode(directory / "episode")
                metadata = episode.metadata
                if metadata["purpose"] != "pipeline_fixture" or metadata["training_eligible"]:
                    raise ValueError("The sensing pilot cannot admit task/training demonstrations")
                cameras = [f["name"] for f in metadata["fields"] if f["dtype"] == "uint8"]
                available = {
                    name: int(episode.arrays[name + ".available"].sum()) for name in cameras
                }
                execution = json.loads((directory / "episode/fixture_execution.json").read_text())
                if (
                    not execution["collection_complete"]
                    or len(episode.arrays["action.nominal"]) != args.steps
                ):
                    raise ValueError("Scheduled sensing transitions were not completed")
                row.update(
                    outcome=(
                        "completed_pipeline_fixture"
                        if len(cameras) == 2
                        and all(count == args.steps + 1 for count in available.values())
                        else "completed_fixture_with_missing_images"
                    ),
                    completed_transitions=len(episode.arrays["action.nominal"]),
                    snapshots=len(episode.arrays["observation_time_s"]),
                    available_raw_images=available,
                    scheduled_raw_images_per_camera=args.steps + 1,
                    collection_complete=execution["collection_complete"],
                    guarded_physics_steps=execution["guarded_physics_steps"],
                    guard=execution["guard"],
                    native_outcome=metadata["outcome"],
                    episode_manifest_sha256=hashlib.sha256(
                        episode_manifest.read_bytes()
                    ).hexdigest(),
                )
            except (ValueError, OSError, KeyError) as error:
                row.update(outcome="invalid_collection_artifact", validation_error=str(error))
        write(directory / "process.json", row)
        rows.append(row)
        write(
            output / "report.json",
            dict(
                kind="bounded_dr_sensing_pilot",
                scheduled_cases=len(cases),
                results=rows,
                fresh_task_credit=0,
                task_success=False,
                training_eligible=False,
                robustness_claim=False,
                uploads=False,
            ),
        )
    print(
        json.dumps(
            dict(
                scheduled_cases=len(cases),
                outcomes=[r["outcome"] for r in rows],
                fresh_task_credit=0,
                task_success=False,
                training_eligible=False,
            )
        )
    )
    return 0 if all(r["outcome"] == "completed_pipeline_fixture" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
