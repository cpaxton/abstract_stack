#!/usr/bin/env python3
"""Generate a DR curriculum, then validate development resets without task credit."""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    from dataclasses import asdict

    from abstract_stack.assets import mabel_model_path
    from abstract_stack.base_motion import BaseContactGuard
    from abstract_stack.checkpoints import save_checkpoint
    from abstract_stack.config import load_scene_config
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.mabel_controller import MabelController

    context = json.loads((args.output / "inputs/context.json").read_text())
    scene_file = args.output / "inputs/scene.json"
    if context["scene_config_sha256"] != hashlib.sha256(scene_file.read_bytes()).hexdigest():
        raise ValueError("DR scene/context identity mismatch")
    config = load_scene_config(scene_file)
    result = dict(
        case_id=context["case_id"],
        family=context["factors"]["family"],
        outcome="unknown",
        task_success=False,
        fresh_task_credit=0,
        training_eligible=False,
        purpose="native_geometry_and_hold_fixture",
        scheduled_products=context["scheduled_product_names"],
        failure=None,
    )
    env, guard = None, None
    try:
        env = ShelfStockingEnv(config, seed=context["reset_seed"])
        env.validate_robot_fixture_clearance()
        tracker = MabelController(env.model, env.data)
        guard = BaseContactGuard(env, max_product_displacement_m=0.03)
        nominal = dict(zip(tracker.NAMES, tracker.nominal.tolist(), strict=True))
        for _ in range(10):
            env.step(tracker.targets(nominal, dt=env.dt), on_physics_step=guard.step)
        result.update(
            outcome="native_geometry_and_hold_verified",
            guard=guard.report(),
            simulated_seconds=float(env.data.time),
            model_xml_sha256=hashlib.sha256(env.xml.encode()).hexdigest(),
            tracking_config=asdict(tracker.config),
            timing=dict(physics_dt_s=env.config.timestep, control_dt_s=env.dt),
            model_revision=subprocess.run(
                ["git", "-C", str(mabel_model_path().parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None,
        )
        if args.preview:
            from PIL import Image

            Image.fromarray(env.render("overview", width=480, height=360)).save(
                args.output / "preview.png"
            )
        scene = dict(
            config=json.loads(scene_file.read_text()),
            products=[asdict(p) for p in env.products],
            seed=context["reset_seed"],
            robot=True,
            robot_source=dict(revision=result["model_revision"]),
            domain_context=context,
        )
        save_checkpoint(
            args.output / "checkpoint",
            env.model,
            env.data,
            scene=scene,
            controller=dict(
                kind="dr_hold_fixture",
                nominal=nominal,
                applied_controls=env.data.ctrl.tolist(),
                tracking_config=asdict(tracker.config),
            ),
            context=dict(purpose="geometry_validation_only", domain_context=context),
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result.update(outcome="native_reset_or_hold_rejected", failure=str(error))
        if guard:
            result["guard"] = guard.report()
    finally:
        if env:
            env.close()
        write(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--development", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=12)
    parser.add_argument("--development-seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.action == "generate":
        from abstract_stack.domain_randomization import generate_suite

        generate_suite(
            args.output,
            development_count=args.development,
            heldout_count=args.heldout,
            development_seed=args.development_seed,
        )
        print(
            json.dumps(
                dict(
                    output=str(args.output),
                    development_cases=args.development,
                    heldout_cases=args.heldout,
                    task_success=False,
                )
            )
        )
        return 0
    if args.suite is None or not 1 <= args.limit <= 8:
        parser.error("Validation needs a suite and one through eight development cases")
    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    args.suite = args.suite.resolve()
    index = json.loads((args.suite / "development/index.json").read_text())
    if index["split"] != "development":
        raise ValueError("Native development validation must not inspect heldout cases")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = freeze_source(
        root,
        paths=[
            "src/abstract_stack",
            "scripts/domain_randomization.py",
            "scripts/run_frozen_replanning.py",
            "uv.lock",
            "pyproject.toml",
        ],
    )
    frozen.write_reference(args.output / "source_ref.json")
    runtime = frozen.runtime_root
    supervise = runpy.run_path(str(runtime / "scripts/run_frozen_replanning.py"))["supervise"]
    rows = []
    selected = index["cases"][: args.limit]
    for case in selected:
        directory = args.output / case["case_id"]
        (directory / "inputs").mkdir(parents=True)
        for kind in ("scene", "context"):
            source = (args.suite / case[kind]).resolve()
            if (
                not source.is_relative_to(args.suite)
                or hashlib.sha256(source.read_bytes()).hexdigest() != case[f"{kind}_sha256"]
            ):
                raise ValueError("DR case input changed or escaped its suite")
            shutil.copy2(source, directory / f"inputs/{kind}.json")
        command = [
            sys.executable,
            str(runtime / "scripts/domain_randomization.py"),
            "validate",
            "--worker",
            "--output",
            str(directory),
        ]
        if args.preview:
            command.append("--preview")
        process = supervise(
            command,
            cwd=root,
            environment=dict(
                os.environ,
                PYTHONPATH=str(runtime / "src"),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONFAULTHANDLER="1",
                ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
                ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ),
            log=directory / "run.log",
            wall_seconds=60,
        )
        write(directory / "process.json", process)
        result = (
            json.loads((directory / "result.json").read_text())
            if process["returncode"] == 0 and (directory / "result.json").exists()
            else dict(outcome="unknown")
        )
        rows.append(dict(case_id=case["case_id"], process=process, result=result))
        write(
            args.output / "report.json",
            dict(
                kind="dr_development_geometry_validation",
                scheduled_cases=len(selected),
                results=rows,
                task_success=False,
                fresh_task_credit=0,
                robustness_proven=False,
            ),
        )
    print(
        json.dumps(
            dict(
                scheduled_cases=len(selected),
                outcomes=[r["result"]["outcome"] for r in rows],
                task_success=False,
            )
        )
    )
    return (
        0 if all(r["result"]["outcome"] == "native_geometry_and_hold_verified" for r in rows) else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
