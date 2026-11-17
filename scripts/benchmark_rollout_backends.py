#!/usr/bin/env python3
"""Frozen, supervised native/Warp throughput probe; zero fresh-task credit."""

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

import mujoco
import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def versions():
    result = {}
    for name in ("mujoco", "numpy", "mujoco-warp", "warp-lang"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_contract(model):
    fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_range",
        "jnt_limited",
        "dof_armature",
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_dynprm",
        "actuator_gear",
        "actuator_trnid",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
        "actuator_forcelimited",
        "actuator_forcerange",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_margin",
        "geom_gap",
        "mesh_vert",
        "mesh_face",
        "eq_data",
        "eq_active0",
    )
    return dict(
        arrays={
            name: hashlib.sha256(getattr(model, name).tobytes()).hexdigest() for name in fields
        },
        options={
            name: float(getattr(model.opt, name))
            for name in (
                "timestep",
                "integrator",
                "solver",
                "jacobian",
                "iterations",
                "ls_iterations",
                "noslip_iterations",
                "tolerance",
                "ls_tolerance",
                "noslip_tolerance",
                "disableflags",
                "enableflags",
                "cone",
            )
        },
    )


def load_case(directory):
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(directory / "rollout_terminal.npz") as saved:
        if int(saved["state_spec"]) != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("Expected full integration-state input")
        mujoco.mj_setState(model, data, saved["state"], int(saved["state_spec"]))
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("External-force inputs are not admitted")
    mujoco.mj_forward(model, data)
    return model, data


def state(model, data, spec):
    result = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, result, spec)
    return result


def native(model, data, args, output):
    import mujoco.rollout

    from abstract_stack.contact_rollout import RolloutScene, evaluate_batch, evaluate_screened_batch

    integration = state(model, data, mujoco.mjtState.mjSTATE_INTEGRATION)
    full = state(model, data, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    user = state(model, data, mujoco.mjtState.mjSTATE_USER)
    controls = np.tile(user, (args.worlds, args.steps, 1))
    started = time.monotonic()
    with mujoco.rollout.Rollout(nthread=args.threads) as pool:
        predictions, _ = pool.rollout(
            model,
            [mujoco.MjData(model) for _ in range(args.threads)],
            full,
            controls,
            control_spec=int(mujoco.mjtState.mjSTATE_USER),
            initial_warmstart=data.qacc_warmstart.copy(),
        )
    fast_seconds = time.monotonic() - started
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite C++ batch trajectory")
    forward_terminal = []
    started = time.monotonic()
    for _ in range(args.worlds):
        scratch = mujoco.MjData(model)
        mujoco.mj_setState(model, scratch, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, scratch)
        for _ in range(args.steps):
            mujoco.mj_step(model, scratch)
            mujoco.mj_forward(model, scratch)
            if np.any(scratch.warning.number):
                raise RuntimeError("Native warning in step+forward reference")
        forward_terminal.append(state(model, scratch, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    forward_seconds = time.monotonic() - started
    context = json.loads((args.input / "rollout_scene.json").read_text())
    scene = RolloutScene(**context)
    applied = np.tile(data.ctrl, (args.steps, 1))
    candidates = [
        dict(name=f"held_native_targets_{i}", controls=applied) for i in range(args.worlds)
    ]
    guarded, guard_seconds = evaluate_batch(
        model, integration, candidates, scene, rates=np.zeros(model.nu), workers=args.threads
    )
    np.savez_compressed(
        output / "terminal_states.npz",
        cpp=predictions[:, -1],
        step_forward=np.asarray(forward_terminal),
        guarded_integration=np.asarray([r[2] for r in guarded]),
    )
    reports = [r[0] for r in guarded]
    save(output / "guarded_candidates.json", reports)
    screened = evaluate_screened_batch(
        model,
        integration,
        candidates,
        scene,
        rates=np.zeros(model.nu),
        workers=args.threads,
        shortlist=min(4, args.worlds),
    )
    save(output / "screened_candidates.json", screened)
    return dict(
        cpp_step_only_seconds=fast_seconds,
        step_forward_seconds=forward_seconds,
        guarded_step_forward_seconds=guard_seconds,
        cpp_physics_steps=args.steps * args.worlds,
        guarded_completed_physics_steps=sum(r["completed_steps"] for r in reports),
        guarded_eligible=sum(r["eligible"] for r in reports),
        screen_then_guard_seconds=screened["total_seconds"],
        screened_guarded_count=screened["guarded_count"],
        screened_candidate_count=screened["screened_count"],
        cpp_vs_step_forward_max_state_difference=float(
            np.max(np.abs(predictions[:, -1] - forward_terminal))
        ),
        threads=args.threads,
        scope="held-native-target forecasts; C++ path has no per-step force guard; no task credit",
    )


def warp_backend(model, data, args, output):
    import mujoco_warp as mjw
    import warp as wp

    wp.config.kernel_cache_dir = os.environ.get(
        "ABSTRACT_STACK_WARP_CACHE", str(output.parents[1] / "warp-cache")
    )
    wp.init()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("GPU benchmark requires CUDA; no CPU fallback")
    started = time.monotonic()
    with wp.ScopedDevice(device):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=args.worlds, nconmax=args.nconmax, njmax=args.njmax)
        # Compile on a disposable forecast, then fork fresh complete data.
        # set_state segfaulted during compilation in the retained first probe.
        mjw.step(m, d)
        mjw.forward(m, d)
        wp.synchronize()
        d = mjw.put_data(model, data, nworld=args.worlds, nconmax=args.nconmax, njmax=args.njmax)
        mjw.forward(m, d)
        wp.synchronize()
        initial_time = d.time.numpy().copy()
        np.testing.assert_array_equal(
            d.qpos.numpy(), np.tile(data.qpos.astype(np.float32), (args.worlds, 1))
        )
        np.testing.assert_array_equal(
            d.ctrl.numpy(), np.tile(data.ctrl.astype(np.float32), (args.worlds, 1))
        )
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mjw.step(m, d)
                mjw.forward(m, d)
        compile_seconds = time.monotonic() - started
        np.testing.assert_array_equal(d.time.numpy(), initial_time)
        wp.synchronize()
        started = time.monotonic()
        for _ in range(args.steps // 10):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        execution_seconds = time.monotonic() - started
        overflow = d.overflow.numpy()
        solver_iterations = d.solver_niter.numpy()
        constraints = d.nefc.numpy()
        contacts = d.nacon.numpy()
        if model.npluginstate or model.nhistory:
            raise ValueError("Plugin/history state unsupported by this explicit forecast exporter")
        final = np.concatenate(
            (d.time.numpy().reshape(-1, 1), d.qpos.numpy(), d.qvel.numpy(), d.act.numpy()), axis=1
        )
    np.savez_compressed(output / "terminal_states.npz", step_forward=final, overflow=overflow)
    if np.any(overflow):
        raise RuntimeError(f"GPU capacity overflow: {overflow.tolist()}")
    if not np.isfinite(final).all():
        raise FloatingPointError("Nonfinite GPU terminal state")
    return dict(
        compile_and_capture_seconds=compile_seconds,
        step_forward_seconds=execution_seconds,
        physics_steps=args.steps * args.worlds,
        nconmax=args.nconmax,
        njmax=args.njmax,
        final_solver_iterations=solver_iterations.tolist(),
        final_constraints=constraints.tolist(),
        final_contact_count=int(contacts[0]),
        device=str(device),
        gpu_name=device.name,
        overflow=overflow.tolist(),
        scope="unchecked held-target throughput only; no force admission or task credit",
        state_initialization="fresh put_data after disposable compilation forecast",
        cuda_capture_preserved_initial_time=True,
        kernel_cache=wp.config.kernel_cache_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nconmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=2048)
    parser.add_argument("--wall-seconds", type=float, default=240)
    parser.add_argument("--worker", choices=("native", "warp"))
    parser.add_argument(
        "--backends", nargs="+", choices=("native", "warp"), default=["native", "warp"]
    )
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if (
        not 1 <= args.worlds <= 32
        or not 10 <= args.steps <= 1000
        or args.steps % 10
        or not 1 <= args.threads <= 8
        or not 0 < args.wall_seconds <= 600
        or not 1 <= args.nconmax <= 1024
        or not 1 <= args.njmax <= 8192
    ):
        parser.error(
            "Bounded probe: worlds1..32, steps10..1000 divisible by10, threads1..8, wall<=600s"
        )
    if args.worker:
        args.output.mkdir(exist_ok=False)
        model, data = load_case(args.input)
        original_contract = model_contract(model)
        if args.steps * model.opt.timestep > 2:
            raise ValueError("Maximum two-second forecast horizon")
        report = (native if args.worker == "native" else warp_backend)(
            model, data, args, args.output
        )
        if original_contract != model_contract(model):
            raise ValueError("Backend changed the native model contract")
        report.update(
            backend=args.worker,
            worlds=args.worlds,
            steps=args.steps,
            dt_seconds=float(model.opt.timestep),
            fresh_task_credit=0,
            versions=versions(),
            model_contract=original_contract,
            requested_settings=dict(
                worlds=args.worlds,
                steps=args.steps,
                threads=args.threads,
                nconmax=args.nconmax,
                njmax=args.njmax,
                worker_wall_seconds=args.wall_seconds,
            ),
        )
        save(args.output / "summary.json", report)
        print(json.dumps(report))
        return 0
    if args.episode is None:
        parser.error("--episode is required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, snapshot = output / "inputs", output / "source_snapshot"
    inputs.mkdir()
    snapshot.mkdir()
    for src, name in (
        (args.episode / "initial-scene/scene.xml", "scene.xml"),
        (args.episode / "initial-scene/manifest.json", "manifest.json"),
        (args.episode / "rollout_terminal.npz", "rollout_terminal.npz"),
        (args.episode / "summary.json", "summary.json"),
    ):
        shutil.copy2(src, inputs / name)
    manifest = json.loads((inputs / "manifest.json").read_text())
    summary = json.loads((inputs / "summary.json").read_text())
    active = summary["events"][-1]["product"]
    product = next(p for p in manifest["products"] if p["name"] == active)
    save(
        inputs / "rollout_scene.json",
        dict(
            products=[p["name"] for p in manifest["products"]],
            active=active,
            robot_root="mabel_chassis",
            hand_root="left_palm",
            support_geoms=["supply_table", f"shelf_board_{product['level']}"],
            active_half_size=product["half_size"],
            support_height=manifest["config"]["levels"][product["level"]],
            base_root="mabel_chassis",
            back_geom="shelf_back",
        ),
    )
    shutil.copy2(__file__, snapshot / "benchmark.py")
    package = snapshot / "src/abstract_stack"
    package.mkdir(parents=True)
    for name in ("__init__.py", "contact_rollout.py"):
        shutil.copy2(root / "src/abstract_stack" / name, package / name)
    shutil.copy2(root / "uv.lock", snapshot / "uv.lock")
    save(
        output / "manifest.json",
        dict(
            kind="frozen_backend_throughput_zero_task_credit",
            input_episode=str(args.episode.resolve()),
            model_revision=manifest["robot_source"]["revision"],
            versions=versions(),
            requested_settings={
                **vars(args),
                "episode": str(args.episode.resolve()),
                "output": str(output),
                "input": None,
            },
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            sha256={
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for directory in (inputs, snapshot)
                for p in directory.rglob("*")
                if p.is_file()
            },
        ),
    )
    outcomes = {
        name: dict(returncode=None, verification_status="unknown", fresh_task_credit=0)
        for name in dict.fromkeys(args.backends)
    }
    save(output / "process.json", outcomes)
    for backend in dict.fromkeys(args.backends):
        started = time.monotonic()
        command = [
            sys.executable,
            str(snapshot / "benchmark.py"),
            "--worker",
            backend,
            "--input",
            str(inputs),
            "--output",
            str(output / backend),
            "--worlds",
            str(args.worlds),
            "--steps",
            str(args.steps),
            "--threads",
            str(args.threads),
            "--wall-seconds",
            str(args.wall_seconds),
            "--nconmax",
            str(args.nconmax),
            "--njmax",
            str(args.njmax),
        ]
        save(
            output / f"{backend}-process.json",
            dict(returncode=None, verification_status="unknown", fresh_task_credit=0),
        )
        with (output / f"{backend}.log").open("w") as stream:
            process = subprocess.Popen(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(os.environ, PYTHONPATH=str(snapshot / "src"), PYTHONFAULTHANDLER="1"),
            )
            try:
                process.wait(timeout=args.wall_seconds)
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
        outcomes[backend] = dict(
            returncode=process.returncode,
            timeout=timed_out,
            wall_seconds=time.monotonic() - started,
            wall_budget_seconds=args.wall_seconds,
            verification_status="completed"
            if process.returncode == 0 and not timed_out
            else "unknown",
            fresh_task_credit=0,
            command=command,
        )
        save(output / f"{backend}-process.json", outcomes[backend])
        save(output / "process.json", outcomes)
        print(json.dumps({backend: outcomes[backend]}), flush=True)
    save(output / "process.json", outcomes)
    return 0 if all(r["verification_status"] == "completed" for r in outcomes.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
