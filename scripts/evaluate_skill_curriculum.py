#!/usr/bin/env python3
"""Run seeded isolated skill episodes with a frozen teacher and retained failures."""

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

from abstract_stack.skill_env import ACTIONS


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    from abstract_stack.skill_env import StockingSkillEnv

    env = StockingSkillEnv(args.skill, args.output)
    try:
        try:
            env.reset(seed=args.seed[0])
        except (RuntimeError, ValueError, FloatingPointError):
            if (args.output / "episode-000/result.json").is_file():
                return 2
            raise
        *_, info = env.step(args.action)
        return 0 if info["success"] else 2
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=(*ACTIONS, "all"), default="all")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--action")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-wall-seconds", type=float, default=150)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not 1 <= len(args.seed) <= 40
        or len(set(args.seed)) != len(args.seed)
        or any(not 0 <= seed < 2**63 for seed in args.seed)
        or not 0 < args.case_wall_seconds <= 600
    ):
        parser.error("Choose distinct nonnegative seeds and a bounded case timeout")
    if args.action is not None and args.action not in ACTIONS.get(args.skill, ()):
        parser.error("Specify a compatible action for one skill")
    if args.worker:
        return worker(args)

    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [
        "src/abstract_stack",
        "scripts/evaluate_skill_curriculum.py",
        "scripts/run_frozen_replanning.py",
        "pyproject.toml",
        "uv.lock",
    ]
    frozen = freeze_source(root, paths=files)
    frozen.write_reference(output / "source_ref.json")
    runtime = frozen.runtime_root
    script = runtime / "scripts/evaluate_skill_curriculum.py"
    supervise = runpy.run_path(str(runtime / "scripts/run_frozen_replanning.py"))["supervise"]
    skills = ACTIONS if args.skill == "all" else (args.skill,)
    cases = [
        dict(skill=skill, seed=seed, action=args.action or ACTIONS[skill][0], outcome="not_run")
        for skill in skills
        for seed in args.seed
    ]

    def report():
        write(
            output / "report.json",
            dict(
                kind="public_development_skill_curriculum",
                cases=cases,
                scheduled=len(cases),
                verified=sum(case["outcome"] == "verified" for case in cases),
                fresh_task_credit=0,
                source_id=frozen.content_id,
                case_wall_budget_s=args.case_wall_seconds,
            ),
        )

    report()
    for case in cases:
        directory = output / f"{case['skill']}-{case['seed']}"
        directory.mkdir()
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--skill",
            case["skill"],
            "--seed",
            str(case["seed"]),
            "--action",
            case["action"],
            "--output",
            str(directory),
        ]
        process = supervise(
            command,
            cwd=root,
            log=directory / "run.log",
            wall_seconds=args.case_wall_seconds,
            environment=dict(
                os.environ,
                PYTHONPATH=str(runtime / "src"),
                PYTHONFAULTHANDLER="1",
                PYTHONDONTWRITEBYTECODE="1",
                ABSTRACT_STACK_SOURCE_REFERENCE=str(output / "source_ref.json"),
                ABSTRACT_STACK_SOURCE_STORE=str(frozen.archive.parent),
            ),
        )
        write(directory / "process.json", process)
        case.update(process=process, outcome="process_failure_or_unknown")
        result = directory / "episode-000/result.json"
        case["result"] = str(result.relative_to(output))
        if process["returncode"] in (0, 2) and not process["timeout"] and result.is_file():
            terminal = json.loads(result.read_text())
            if (terminal["outcome"] == "verified") == (process["returncode"] == 0):
                case.update(outcome=terminal["outcome"], failure=terminal.get("failure"))
        report()
    print(json.dumps({"report": str(output / "report.json"), "scheduled": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
