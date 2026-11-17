"""Supervised capture/inspection/replay of native simulation checkpoints."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path


def add_commands(parent):
    commands = parent.add_subparsers(dest="checkpoint_action", required=True)
    capture = commands.add_parser("capture", help="Import a retained episode state once")
    capture.add_argument("--episode", type=Path, required=True)
    capture.add_argument("--state", choices=("start", "terminal"), default="terminal")
    capture.add_argument("--witness-steps", type=int, default=50)
    verify = commands.add_parser("verify", help="Strict fresh-process state/force replay")
    verify.add_argument("checkpoint", type=Path)
    inspect = commands.add_parser("inspect", help="Inspect and hash-check without loading a model")
    inspect.add_argument("checkpoint", type=Path)
    for command in (capture, verify):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--wall-seconds", type=float, default=30)
        command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _capture(args):
    import mujoco
    import numpy as np

    from abstract_stack.checkpoints import STATE, save_checkpoint
    from abstract_stack.native_views import retain_data_views, retain_model_views

    scene = json.loads((args.episode / "initial-scene/manifest.json").read_text())
    summary = json.loads((args.episode / "summary.json").read_text())
    xml = (args.episode / "initial-scene/scene.xml").read_text()
    model = mujoco.MjModel.from_xml_string(xml)
    views = retain_model_views(model)
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    source_state = args.episode / f"rollout_{args.state}.npz"
    with np.load(source_state, allow_pickle=False) as saved:
        state, spec = saved["state"], int(saved["state_spec"])
    if (
        spec != int(STATE)
        or state.shape != (mujoco.mj_stateSize(model, STATE),)
        or not np.isfinite(state).all()
    ):
        raise ValueError("Retained episode lacks a matching full integration state")
    mujoco.mj_setState(model, data, state, STATE)
    mujoco.mj_forward(model, data)
    selected = next(
        (
            e["selected"]
            for e in reversed(summary.get("events", []))
            if e.get("phase") == "pusher_plan"
        ),
        None,
    )
    nominal = summary.get("last_nominal_targets") if args.state == "terminal" else None
    controller = dict(
        kind="retained_finger_push" if selected is not None and nominal else "native_physics_only",
        settings=summary.get("controller", {}),
        pusher_settings=selected,
        nominal_subset=nominal,
        nominal_scope="only explicitly recorded named references; omitted channels are unknown",
        applied_controls=data.ctrl.tolist(),
    )
    save_checkpoint(
        args.output / "checkpoint",
        model,
        data,
        scene=scene,
        controller=controller,
        context=dict(
            origin="legacy_episode_import",
            source_episode=str(args.episode),
            source_state_sha256=hashlib.sha256(source_state.read_bytes()).hexdigest(),
            source_summary=summary,
            original_outcome=summary.get("failure"),
            original_live_model_identity_verified=False,
            reconstruction="exported MJCF plus saved integration state; no original MJB existed",
            inherited_history="retained in source_summary; do not reinterpret as a new episode",
        ),
        scene_xml=xml,
        validation_controls=np.tile(data.ctrl, (args.witness_steps, 1)),
    )
    assert views and data_views
    return dict(status="captured", checkpoint=str(args.output / "checkpoint"), fresh_task_credit=0)


def run(args):
    from abstract_stack.checkpoint_manifest import inspect_checkpoint
    from abstract_stack.provenance import policy_provenance

    if args.checkpoint_action == "inspect":
        path, manifest = inspect_checkpoint(args.checkpoint)
        print(
            json.dumps(
                dict(
                    checkpoint=str(path),
                    **{
                        k: manifest[k]
                        for k in (
                            "kind",
                            "runtime",
                            "simulation_time_s",
                            "physics_timestep_s",
                            "dimensions",
                            "witness_steps",
                            "fresh_task_credit",
                        )
                    },
                ),
                indent=2,
            )
        )
        return 0
    if not 0 < args.wall_seconds <= 180 or not 1 <= getattr(args, "witness_steps", 1) <= 500:
        raise ValueError("Keep checkpoint witness/compute budgets bounded")
    args.output = args.output.resolve()
    if args.worker:
        from abstract_stack.checkpoints import verify_checkpoint

        try:
            report = (
                _capture(args)
                if args.checkpoint_action == "capture"
                else verify_checkpoint(args.checkpoint)
            )
            _write(args.output / "result.json", report)
            return 0 if report.get("verified", True) else 2
        except (OSError, ValueError, RuntimeError) as error:
            traceback.print_exc()
            _write(
                args.output / "error.json",
                dict(error=str(error), task_success=False, fresh_task_credit=0),
            )
            return 2
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source_snapshot"
    shutil.copytree(
        Path(__file__).parent,
        snapshot / "abstract_stack",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    command = [sys.executable, "-m", "abstract_stack.checkpoint_cli", args.checkpoint_action]
    if args.checkpoint_action == "capture":
        command += [
            "--episode",
            str(args.episode.resolve()),
            "--state",
            args.state,
            "--witness-steps",
            str(args.witness_steps),
        ]
    else:
        command.append(str(args.checkpoint.resolve()))
    command += ["--output", str(args.output), "--worker"]
    _write(
        args.output / "capture_manifest.json",
        dict(
            policy_source=policy_provenance(),
            command=command,
            wall_budget_seconds=args.wall_seconds,
            fresh_task_credit=0,
        ),
    )
    _write(
        args.output / "process.json", dict(status="running", returncode=None, fresh_task_credit=0)
    )
    started = time.monotonic()
    with (args.output / "run.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            env=dict(os.environ, PYTHONPATH=str(snapshot), PYTHONFAULTHANDLER="1"),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timeout = False
        try:
            process.wait(timeout=args.wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            timeout = True
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report = dict(
        returncode=process.returncode,
        timeout=timeout,
        wall_seconds=time.monotonic() - started,
        wall_budget_seconds=args.wall_seconds,
        outcome="completed" if process.returncode == 0 else "failed_or_unknown",
        fresh_task_credit=0,
        task_success=False,
    )
    _write(args.output / "process.json", report)
    print(json.dumps(report))
    return 0 if process.returncode == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_commands(parser)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
