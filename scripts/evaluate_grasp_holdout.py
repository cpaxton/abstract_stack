#!/usr/bin/env python3
"""Fresh private fixed-scene sequences for a previously frozen grasp teacher."""

import argparse
import json
import os
import runpy
import secrets
import sys
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_episode(episode, output):
    import mujoco
    import numpy as np

    from abstract_stack.checkpoints import native_state
    from abstract_stack.interactions import _descendants, contact_forces
    from abstract_stack.native_views import retain_data_views, retain_model_views

    summary = json.loads((episode / "summary.json").read_text())
    names = ["carton_0", "carton_1"]
    if sorted(summary["scheduled_pick_products"]) != names:
        raise ValueError("Held-out sequence omitted scheduled products")
    model = mujoco.MjModel.from_xml_string((episode / "initial-scene/scene.xml").read_text())
    views = retain_model_views(model)
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    with np.load(episode / "rollout_start.npz") as saved:
        mujoco.mj_setState(model, data, saved["state"], int(saved["state_spec"]))
    mujoco.mj_forward(model, data)
    from types import SimpleNamespace

    env = SimpleNamespace(model=model, data=data)
    hand = _descendants(model, "left_palm")
    arms = _descendants(model, "l_shoulder_1") | _descendants(model, "r_shoulder_1")
    windows = {}
    for name in names:
        lift = next(
            (
                e
                for e in summary["events"]
                if e["product"] == name and e.get("verified_lift") is True
            ),
            None,
        )
        windows[name] = dict(start=None if lift is None else lift["end_time"], steps=0, valid=True)
    incidents = []
    with np.load(episode / "trace.npz") as archive:
        for control in archive["controls"]:
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if (
                np.any(data.qfrc_applied)
                or np.any(data.xfrc_applied)
                or np.any(data.warning.number)
            ):
                raise ValueError("External force input or native warning in replay")
            for name in names:
                body = data.body(name)
                if body.xmat.reshape(3, 3)[2, 2] < np.cos(np.radians(45)) or body.xpos[2] < 0.25:
                    incidents.append(dict(product=name, sim_time=float(data.time)))
                window = windows[name]
                if (
                    window["start"] is None
                    or not window["start"] + 1e-8 < data.time <= window["start"] + 0.5 + 1e-8
                ):
                    continue
                force = contact_forces(env, name, hand)
                robot_force = contact_forces(env, name, arms)["hand"]
                window["steps"] += 1
                window["valid"] &= bool(
                    force["opposing_contacts"]
                    and force["hand"] >= 0.2
                    and force["table"] <= 0.15
                    and robot_force <= 25
                )
    with np.load(episode / "rollout_terminal.npz") as saved:
        matched = np.array_equal(native_state(model, data), saved["state"])
    required = round(0.5 / model.opt.timestep)
    grasps = {
        name: bool(matched and not incidents and window["valid"] and window["steps"] == required)
        for name, window in windows.items()
    }
    output.mkdir(exist_ok=False)
    report = dict(
        kind="strict_native_grasp_sequence_replay",
        full_terminal_bit_equal=bool(matched),
        required_hold_seconds=0.5,
        required_hold_steps=required,
        hold_windows=windows,
        grasps=grasps,
        incidents=incidents,
        task_success=summary["task_success"],
        original_task_failure=summary["failure"],
        fresh_stocking_credit=0,
    )
    write(output / "result.json", report)
    assert views and data_views
    return 0 if matched else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-source-ref", type=Path)
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--case-wall-seconds", type=float, default=90)
    parser.add_argument("--verify-episode", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.verify_episode:
        return verify_episode(args.verify_episode, args.output)
    if (
        args.policy_source_ref is None
        or not 1 <= args.cases <= 8
        or not 0 < args.case_wall_seconds <= 180
    ):
        parser.error("Require a frozen policy and bounded fresh held-out cases")
    from abstract_stack.source_store import freeze_source, resolve_source_reference

    root = Path(__file__).resolve().parents[1]
    frozen = resolve_source_reference(
        args.policy_source_ref, cache=root / "outputs/cache/source_runtime"
    )
    harness = freeze_source(
        root,
        paths=[
            "scripts/evaluate_grasp_holdout.py",
            "scripts/run_frozen_replanning.py",
            "pyproject.toml",
            "uv.lock",
        ],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    frozen.write_reference(args.output / "source_ref.json")
    harness.write_reference(args.output / "harness_source_ref.json")
    seeds = set()
    while len(seeds) < args.cases:
        seed = secrets.randbits(31)
        if seed not in {*range(8), 112}:
            seeds.add(seed)
    rows = [
        dict(
            seed=seed,
            process=None,
            outcome="not_run",
            grasps=dict.fromkeys(["carton_0", "carton_1"], False),
        )
        for seed in sorted(seeds)
    ]
    write(
        args.output / "manifest.json",
        dict(
            kind="private_fixed_geometry_grasp_holdout",
            policy_source_content_id=frozen.content_id,
            harness_content_id=harness.content_id,
            case_wall_seconds=args.case_wall_seconds,
            verification_wall_seconds=30,
            scheduled_sequences=len(rows),
            scheduled_grasps=2 * len(rows),
            seed_policy="Independent fresh entropy after policy freeze; private outputs only",
            effective_flags=dict(grasp_skill=True, record_skill_starts=False),
            geometry="fixed pick_place",
            reset_jitter_m=0.002,
            force_abort_N=25,
            hold_seconds=0.5,
        ),
    )

    def report():
        write(
            args.output / "report.json",
            dict(
                kind="private_fixed_geometry_grasp_holdout",
                policy_source_content_id=frozen.content_id,
                harness_content_id=harness.content_id,
                scheduled_sequences=len(rows),
                scheduled_grasps=2 * len(rows),
                verified_grasps=sum(sum(r["grasps"].values()) for r in rows),
                evaluation_complete=all(r["outcome"] != "not_run" for r in rows),
                cases=rows,
                whole_shelf_success=False,
                trained_student=False,
            ),
        )

    report()
    supervise = runpy.run_path(str(harness.runtime_root / "scripts/run_frozen_replanning.py"))[
        "supervise"
    ]
    environment = dict(
        os.environ,
        PYTHONPATH=str(frozen.runtime_root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONFAULTHANDLER="1",
        ABSTRACT_STACK_SOURCE_REFERENCE=str(args.output / "source_ref.json"),
        ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
    )
    for index, row in enumerate(rows):
        directory = args.output / f"case-{index:03}"
        directory.mkdir()
        process = supervise(
            [
                sys.executable,
                "-m",
                "abstract_stack.cli",
                "script",
                "--scene",
                "pick_place",
                "--seed",
                str(row["seed"]),
                "--grasp-skill",
                "--output-dir",
                str(directory / "episode"),
            ],
            cwd=root,
            environment=environment,
            log=directory / "run.log",
            wall_seconds=args.case_wall_seconds,
        )
        row.update(process=process, outcome="process_failure_or_unknown")
        write(directory / "process.json", process)
        report()
        if process["returncode"] not in (0, 2) or process["timeout"]:
            continue
        verification = supervise(
            [
                sys.executable,
                str(harness.runtime_root / "scripts/evaluate_grasp_holdout.py"),
                "--verify-episode",
                str(directory / "episode"),
                "--output",
                str(directory / "verification"),
            ],
            cwd=root,
            environment=environment,
            log=directory / "verify.log",
            wall_seconds=30,
        )
        row["verification_process"] = verification
        write(directory / "verification_process.json", verification)
        path = directory / "verification/result.json"
        if verification["returncode"] == 0 and not verification["timeout"] and path.exists():
            value = json.loads(path.read_text())
            row.update(
                outcome="verified_native_sequence",
                grasps=value["grasps"],
                task_success=value["task_success"],
                task_failure=value["original_task_failure"],
            )
        report()
    verified = sum(sum(r["grasps"].values()) for r in rows)
    print(
        json.dumps(
            dict(output=str(args.output), scheduled_grasps=2 * len(rows), verified_grasps=verified)
        )
    )
    return 0 if verified == 2 * len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
