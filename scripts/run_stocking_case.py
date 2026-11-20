#!/usr/bin/env python3
"""Supervise one fresh scene with immutable shared source and complete failure evidence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-source-ref", type=Path, help="Reuse an authenticated frozen controller source"
    )
    parser.add_argument("--wall-seconds", type=float, default=300)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        help="Stop if neither the phase nor the physics heartbeat updates for this many seconds",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("Bound each native episode to at most 1800 wall seconds")
    if args.idle_seconds is not None and not 0 < args.idle_seconds <= args.wall_seconds:
        parser.error("The idle budget must be positive and fit inside the wall budget")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    content = args.config.read_bytes()
    scene_sha256 = hashlib.sha256(content).hexdigest()
    configuration = json.loads(content)
    write(args.output / "scene.json", configuration)
    root = Path(__file__).resolve().parents[1]
    from abstract_stack.source_store import freeze_source, resolve_source_reference

    frozen = (
        resolve_source_reference(
            args.policy_source_ref, cache=root / "outputs/cache/source_runtime"
        )
        if args.policy_source_ref
        else freeze_source(
            root,
            paths=[
                "src/abstract_stack",
                "scripts/run_stocking_case.py",
                "scripts/run_frozen_replanning.py",
                "pyproject.toml",
                "uv.lock",
            ],
        )
    )
    frozen.write_reference(args.output / "source_ref.json")
    extra = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    extra = list(extra)
    controller_inputs = []
    for index, argument in enumerate(extra):
        if argument != "--control-config" and not argument.startswith("--control-config="):
            continue
        if controller_inputs:
            parser.error("Use a single explicit controller configuration")
        inline = argument.startswith("--control-config=")
        if not inline and index + 1 == len(extra):
            parser.error("--control-config requires a path")
        original = Path(argument.split("=", 1)[1] if inline else extra[index + 1]).resolve()
        content = original.read_bytes()
        snapshot = args.output / "controller.json"
        snapshot.write_bytes(content)
        snapshot.chmod(0o444)
        controller_inputs.append(
            dict(
                original_path=str(original),
                sha256=hashlib.sha256(content).hexdigest(),
                snapshot=str(snapshot),
            )
        )
        if inline:
            extra[index] = "--control-config=" + str(snapshot)
        else:
            extra[index + 1] = str(snapshot)
    command = [
        sys.executable,
        "-m",
        "abstract_stack.cli",
        "script",
        "--config",
        str(args.output / "scene.json"),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output / "episode"),
        *extra,
    ]
    write(
        args.output / "manifest.json",
        dict(
            kind="fresh_frozen_stocking_case",
            source_content_id=frozen.content_id,
            original_config_sha256=scene_sha256,
            config=configuration,
            seed=args.seed,
            command=command,
            controller_inputs=controller_inputs,
            wall_budget_seconds=args.wall_seconds,
            idle_budget_seconds=args.idle_seconds,
            runtime=dict(
                python=sys.version,
                mujoco=importlib.metadata.version("mujoco"),
                numpy=importlib.metadata.version("numpy"),
                python_allocator=os.environ.get("PYTHONMALLOC", "default"),
            ),
        ),
    )
    write(args.output / "process.json", dict(outcome="scheduled", task_success=False))
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
            ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            MABEL_ROOT=os.environ.get("MABEL_ROOT", str(root / "third_party/open_mabel")),
        ),
        log=args.output / "run.log",
        wall_seconds=args.wall_seconds,
        progress_path=args.output / "episode/physics-progress.json"
        if args.idle_seconds is not None
        else None,
        idle_seconds=args.idle_seconds,
    )
    summary_path = args.output / "episode/summary.json"
    result.update(outcome="unknown_process_failure", task_success=False)
    if not result["timeout"] and result["returncode"] in (0, 2) and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        result.update(
            outcome="completed_episode",
            task_success=summary["task_success"],
            success_criterion=summary.get("success_criterion", "clean_success"),
            failure=summary["failure"],
            stages=summary["stage_evaluation"],
        )
    write(args.output / "process.json", result)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                returncode=result["returncode"],
                timeout=result["timeout"],
                timeout_reason=result["timeout_reason"],
                task_success=result["task_success"],
                failure=result.get("failure"),
                wall_seconds=result["wall_seconds"],
            )
        )
    )
    return 0 if result["task_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
