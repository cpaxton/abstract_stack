#!/usr/bin/env python3
"""Freeze sources and supervise one native development episode or planning probe."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def freeze_sources(root, snapshot, arguments):
    shutil.copytree(
        root / "src/abstract_stack",
        snapshot / "src/abstract_stack",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(root / "scripts/run_replanning.py", snapshot / "run_replanning.py")
    shutil.copy2(Path(__file__), snapshot / "supervisor.py")
    frozen = list(arguments)
    config_count = 0
    for index, argument in enumerate(arguments):
        if argument == "--control-config":
            if index + 1 >= len(arguments):
                raise ValueError("--control-config requires a path")
            position, original = index + 1, arguments[index + 1]
        elif argument.startswith("--control-config="):
            position, original = index, argument.split("=", 1)[1]
        else:
            continue
        destination = snapshot / f"control_config_{config_count}.json"
        config_count += 1
        content = Path(original).read_bytes()
        json.loads(content)
        destination.write_bytes(content)
        frozen[position] = (
            f"--control-config={destination}" if position == index else str(destination)
        )
    hashes = {
        str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return frozen, hashes


def supervise(command, *, cwd, environment, log, wall_seconds):
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=wall_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            timed_out = True
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return dict(
        returncode=process.returncode,
        timeout=timed_out,
        wall_seconds=time.monotonic() - started,
        wall_budget_seconds=wall_seconds,
    )


def classify(result, episode, *, planning_only):
    name = "plan_probe.json" if planning_only else "summary.json"
    terminal = episode / name
    outcome = dict(
        verification_status="unknown",
        task_success=False,
        credited_stages=dict.fromkeys(("pick", "place", "final_position", "retreat"), 0),
    )
    if result.get("timeout") or result.get("returncode") not in (0, 2) or not terminal.is_file():
        return outcome
    try:
        summary = json.loads(terminal.read_text())
    except (ValueError, OSError) as error:
        return dict(outcome, terminal_read_error=str(error))
    if planning_only:
        success = not summary.get("failure") and summary.get("live_state_unchanged") is True
        if success != (result["returncode"] == 0):
            return outcome
        return dict(
            outcome,
            verification_status="planning_only_zero_task_credit",
            planning_succeeded=success,
            failure=summary.get("failure"),
        )
    success = summary.get("task_success", summary.get("physical_success"))
    if not isinstance(success, bool) or success != (result["returncode"] == 0):
        return outcome
    stages = summary.get("manipulation_evaluation", {}).get("stages")
    scheduled = summary.get("manipulation_evaluation", {}).get("scheduled_manipulations")
    if not isinstance(stages, dict) or any(
        type(stages.get(name)) is not int or stages[name] < 0 for name in outcome["credited_stages"]
    ):
        return outcome
    stages = {name: stages[name] for name in outcome["credited_stages"]}
    if type(scheduled) is not int or scheduled < 1 or any(v > scheduled for v in stages.values()):
        return outcome
    if success and any(value != scheduled for value in stages.values()):
        return outcome
    return dict(
        outcome,
        verification_status="verified_terminal_outcome",
        task_success=success,
        credited_stages=stages,
        failure=summary.get("failure"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not 0 < args.wall_seconds <= 3600:
        parser.error("Wall budget must be positive and at most 3600 seconds")
    if any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in arguments):
        parser.error("Use --output; the supervisor assigns the worker output directory")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(root / "src"):
        parser.error("Keep episode outputs outside the source package")
    output.mkdir(parents=True, exist_ok=False)
    snapshot, episode = output / "source_snapshot", output / "episode"
    result = dict(returncode=None, timeout=False, verification_status="unknown")
    (output / "process.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        frozen, hashes = freeze_sources(root, snapshot, arguments)
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        manifest = dict(
            kind="frozen_development_source_not_population_validation",
            base_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            sha256=hashes,
            requested_arguments=arguments,
            python=sys.version,
            packages={name: importlib.metadata.version(name) for name in ("mujoco", "numpy")},
        )
        (output / "source_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
        command = [
            sys.executable,
            str(snapshot / "run_replanning.py"),
            "--output-dir",
            str(episode),
            *frozen,
        ]
        environment = dict(
            os.environ,
            PYTHONPATH=str(snapshot / "src"),
            PYTHONFAULTHANDLER="1",
            MABEL_ROOT=os.environ.get("MABEL_ROOT", str(root / "third_party/open_mabel")),
        )
        result = supervise(
            command,
            cwd=root,
            environment=environment,
            log=output / "run.log",
            wall_seconds=args.wall_seconds,
        )
        result.update(command=command, kind="frozen_development_attempt")
        result.update(classify(result, episode, planning_only="--plan-only" in arguments))
    except (OSError, ValueError) as error:
        result["supervisor_error"] = str(error)
    (output / "process.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result.get("task_success") or result.get("planning_succeeded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
