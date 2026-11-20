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
    if args.gym:
        from abstract_stack.skill_curriculum import (
            StockingCurriculumEnv,
            attempt_case,
            load_curriculum,
        )

        cases = (
            load_curriculum(args.manifest, artifact_root=args.artifact_root)
            if args.manifest
            else ()
        )
        selected = (
            next((case for case in cases if case.id == args.case[0]), None) if args.case else None
        )
        env = StockingCurriculumEnv(
            args.skill,
            args.output,
            cases=cases,
            control_config=args.control_config,
            verify_retreat=args.verify_retreat,
        )
        options = {"source": selected.role, "case_id": selected.id} if selected else None
        try:
            if args.agent_attempts:
                info = attempt_case(
                    env,
                    seed=args.seed[0],
                    case=selected,
                    preferred=args.action,
                    budget=args.agent_attempts,
                    rewind=args.rewind,
                    rewind_preferred=args.rewind_action,
                )
                write(args.output / "skill-result.json", info)
                return 0 if info["outcome"] == "verified" else 2
            try:
                env.reset(seed=args.seed[0], options=options)
            except (OSError, RuntimeError, ValueError, FloatingPointError) as error:
                write(
                    args.output / "skill-result.json",
                    dict(outcome="reset_rejected", failure=str(error)),
                )
                return 2
            *_, truncated, info = env.step(env.actions.index(args.action))
            info["action_mask"] = info["action_mask"].tolist()
            info["outcome"] = (
                "verified" if info["success"] else "timeout" if truncated else "failed"
            )
            info["artifacts"] = str(env.directory.relative_to(args.output))
            write(args.output / "skill-result.json", info)
            return 0 if info["success"] else 2
        finally:
            env.close()

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
    parser.add_argument("--skill", default="all")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--action")
    parser.add_argument("--gym", action="store_true", help="Use the Gymnasium skill options")
    parser.add_argument(
        "--agent-attempts",
        type=int,
        choices=range(1, 5),
        help="Try distinct admissible actions from the same native reset",
    )
    parser.add_argument("--manifest", type=Path, help="Use the captured-case corpus with --gym")
    parser.add_argument(
        "--control-config",
        type=Path,
        help="Bounded controller overrides for captured Gymnasium cases",
    )
    parser.add_argument("--rewind", action="store_true", help="Try a declared earlier checkpoint")
    parser.add_argument("--rewind-action", help="Try this admissible action first after rewinding")
    parser.add_argument(
        "--verify-retreat",
        action="store_true",
        help="Include native retreat after a verified captured push",
    )
    parser.add_argument(
        "--case", action="append", default=[], help="Select a case ID from the corpus"
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-wall-seconds", type=float, default=150)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    actions = ACTIONS
    if args.gym:
        from abstract_stack.skill_curriculum import ACTIONS as GYM_ACTIONS
        from abstract_stack.skill_curriculum import PROCEDURAL, load_curriculum

        actions = GYM_ACTIONS if args.manifest else {key: GYM_ACTIONS[key] for key in PROCEDURAL}
    elif args.manifest or args.case:
        parser.error("Captured cases require --gym")
    if args.agent_attempts and not args.gym:
        parser.error("Agent attempts require --gym")
    if args.control_config and not (args.gym and args.manifest):
        parser.error("Controller overrides require a captured Gymnasium case manifest")
    if args.verify_retreat and not (args.gym and args.manifest and args.skill == "push"):
        parser.error("Retreat verification requires a captured Gymnasium push case")
    if args.rewind and not (args.agent_attempts and args.manifest):
        parser.error("Rewinding requires agent attempts and a captured case manifest")
    if args.rewind_action is not None and (
        not args.rewind or args.rewind_action not in actions.get(args.skill, ())
    ):
        parser.error("Specify a compatible action for one skill with --rewind")
    if args.case and not args.manifest:
        parser.error("Select cases from an explicit manifest")
    if args.skill != "all" and args.skill not in actions:
        parser.error(f"Choose all or one of {tuple(actions)}")
    if args.seed is None:
        args.seed = [0] if args.gym else []
    if args.manifest and len(args.seed) != 1:
        parser.error("Captured cases use their recorded scene seeds and one selector seed")
    if (
        not 1 <= len(args.seed) <= 40
        or len(set(args.seed)) != len(args.seed)
        or any(not 0 <= seed < 2**63 for seed in args.seed)
        or not 0 < args.case_wall_seconds <= 600
    ):
        parser.error("Choose distinct nonnegative seeds and a bounded case timeout")
    if args.action is not None and args.action not in actions.get(args.skill, ()):
        parser.error("Specify a compatible action for one skill")
    if args.worker:
        return worker(args)

    skills = actions if args.skill == "all" else (args.skill,)
    if args.manifest:
        args.artifact_root = args.artifact_root.resolve()
        source_cases = load_curriculum(args.manifest, artifact_root=args.artifact_root)
        selected = [
            c for c in source_cases if c.skill in skills and (not args.case or c.id in args.case)
        ]
        if not selected or set(args.case) - {c.id for c in selected}:
            parser.error("No matching cases or an unknown selected case")
        cases = [
            dict(
                skill=c.skill,
                seed=args.seed[0],
                action=args.action or actions[c.skill][0],
                case_id=c.id,
                source=c.role,
                outcome="not_run",
            )
            for c in selected
        ]
    else:
        cases = [
            dict(skill=skill, seed=seed, action=args.action or actions[skill][0], outcome="not_run")
            for skill in skills
            for seed in args.seed
        ]

    from abstract_stack.source_store import freeze_source

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.manifest:
        corpus = output / "curriculum.json"
        corpus.write_bytes(args.manifest.read_bytes())
    if args.control_config:
        from abstract_stack.tuning import load_control_config

        controls = load_control_config(args.control_config)
        control_config = output / "controls.json"
        write(control_config, controls)
        control_config.chmod(0o444)
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
                verify_retreat=args.verify_retreat,
                rewind_action=args.rewind_action,
                case_wall_budget_s=args.case_wall_seconds,
            ),
        )

    report()
    for case in cases:
        directory = output / case.get("case_id", f"{case['skill']}-{case['seed']}")
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
        if args.gym:
            command.append("--gym")
        if args.agent_attempts:
            command += ["--agent-attempts", str(args.agent_attempts)]
        if args.control_config:
            command += ["--control-config", str(control_config)]
        if args.rewind:
            command.append("--rewind")
        if args.rewind_action:
            command += ["--rewind-action", args.rewind_action]
        if args.verify_retreat:
            command.append("--verify-retreat")
        if args.manifest:
            command += [
                "--manifest",
                str(corpus),
                "--artifact-root",
                str(args.artifact_root),
                "--case",
                case["case_id"],
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
        result = directory / ("skill-result.json" if args.gym else "episode-000/result.json")
        case["result"] = str(result.relative_to(output))
        if process["returncode"] in (0, 2) and not process["timeout"] and result.is_file():
            terminal = json.loads(result.read_text())
            if (terminal["outcome"] == "verified") == (process["returncode"] == 0):
                case.update(outcome=terminal["outcome"], failure=terminal.get("failure"))
                if "attempts" in terminal:
                    case["attempts"] = terminal["attempts"]
                if "rewind" in terminal:
                    case["initial_attempt_outcome"] = terminal["initial_attempt_outcome"]
                    case["rewind"] = terminal["rewind"]
        report()
    print(json.dumps({"report": str(output / "report.json"), "scheduled": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
