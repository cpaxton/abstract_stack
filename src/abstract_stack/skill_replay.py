"""Seek a retained skill trace at any native physics-step boundary."""

import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.checkpoint_manifest import sha256_file
from abstract_stack.checkpoints import STATE, load_checkpoint, native_state, save_checkpoint
from abstract_stack.scene import ShelfConfig
from abstract_stack.skill_env import _FixtureEnv
from abstract_stack.skill_fixtures import SkillFixture


def _read(path):
    return json.loads(path.read_text())


def _replay_step(model, data, control, time):
    data.ctrl[:] = control
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if abs(float(data.time) - float(time)) > 1e-8:
        raise ValueError("Replay clock differs from the trace")


def _recorded_start(attempt, metadata):
    path = Path(metadata["checkpoint"])
    expected = metadata["manifest_sha256"]
    if sha256_file(path / "manifest.json") != expected:
        raise ValueError("The original skill checkpoint changed")
    checkpoint = load_checkpoint(path)
    return checkpoint, checkpoint.scene, attempt / "continuation.npz"


def _procedural_start(attempt):
    import hashlib

    metadata = _read(attempt / "fixture.json")
    fixture = metadata["fixture"]
    fixture = SkillFixture(**{**fixture, "config": ShelfConfig(**fixture["config"])})
    env = _FixtureEnv(fixture, seed=metadata["seed"], mabel_root=None)
    expected = _read(attempt / "reset.json")["model_xml_sha256"]
    if hashlib.sha256(env.xml.encode()).hexdigest() != expected:
        env.close()
        raise ValueError("Procedural replay compiled a different model")
    scene = dict(
        config=asdict(env.config),
        products=[asdict(p) for p in env.products],
        robot=True,
        seed=env._seed,
    )
    return env, scene, attempt / "trace.npz"


def capture_skill_cut(attempt, physics_step, output, *, model_cache=None):
    """Verify the native suffix and save a private, physics-only branch point."""
    attempt = Path(attempt).resolve()
    metadata = _read(attempt / "reset.json")
    recorded = "checkpoint" in metadata
    if recorded:
        native, scene, trace = _recorded_start(attempt, metadata)
    else:
        native, scene, trace = _procedural_start(attempt)
    try:
        return _capture_cut(native, scene, attempt, trace, physics_step, output, model_cache)
    finally:
        if not recorded:
            native.close()


def _capture_cut(native, scene, attempt, trace, physics_step, output, model_cache):
    model, data = native.model, native.data
    with np.load(trace, allow_pickle=False) as archive:
        controls, samples = archive["controls"], archive["samples"]
        times = samples[:, 0] if samples.size else np.empty(0)
        if "initial_state" in archive.files:
            initial, terminal = archive["initial_state"], archive["terminal_state"]
            if not np.array_equal(initial, native_state(model, data)):
                raise ValueError("Recorded checkpoint is not the start of this trace")
        else:
            with np.load(attempt / "initial.npz", allow_pickle=False) as capture:
                initial = capture["integration"]
            terminal = archive["terminal"]
            mujoco.mj_setState(model, data, initial, STATE)
            mujoco.mj_forward(model, data)
            if not np.array_equal(initial, native_state(model, data)):
                raise ValueError("Procedural initial state does not round-trip")
    if (
        type(physics_step) is not int
        or not 0 <= physics_step <= len(controls)
        or controls.shape != (len(times), model.nu)
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Choose a physics step inside a valid recorded native trace")
    cut = initial.copy() if physics_step == 0 else None
    for index, (control, time) in enumerate(zip(controls, times, strict=True), start=1):
        _replay_step(model, data, control, time)
        if index == physics_step:
            cut = native_state(model, data)
    if not np.array_equal(terminal, native_state(model, data)):
        raise ValueError("Recorded native terminal state did not replay exactly")
    mujoco.mj_setState(model, data, cut, STATE)
    mujoco.mj_forward(model, data)
    if not np.array_equal(cut, native_state(model, data)):
        raise ValueError("Cut state does not round-trip after forward reconstruction")
    result = _read(attempt / "result.json")
    phase = next(
        (e["phase"] for e in reversed(result["events"]) if e["sim_time"] <= data.time + 1e-8),
        "reset",
    )
    save_checkpoint(
        output,
        model,
        data,
        scene=scene,
        controller=dict(kind="native_physics_only", applied_controls=data.ctrl.tolist()),
        context=dict(
            origin="skill_trace_cut",
            attempt=str(attempt),
            trace_sha256=sha256_file(trace),
            result_sha256=sha256_file(attempt / "result.json"),
            physics_step=physics_step,
            total_physics_steps=len(controls),
            phase=phase,
            original_outcome=result.get("failure"),
            suffix_bit_equal=True,
            fresh_task_credit=0,
            controller_initialization="Requires a new reference before closed-loop replanning",
        ),
        portable_source=False,
        shared_model_cache=model_cache,
        validation_controls=np.tile(data.ctrl, (10, 1)),
    )
    return dict(
        checkpoint=str(output),
        phase=phase,
        simulation_time_s=float(data.time),
        physics_step=physics_step,
        suffix_bit_equal=True,
        fresh_task_credit=0,
    )


def resume_skill_cut(directory):
    """Run the recorded suffix from a freshly loaded cut and verify its terminal state."""
    checkpoint = load_checkpoint(directory)
    context = checkpoint.context
    if context.get("origin") != "skill_trace_cut":
        raise ValueError("Replay requires a skill trace cut")
    attempt = Path(context["attempt"])
    recorded = "checkpoint" in _read(attempt / "reset.json")
    trace = attempt / ("continuation.npz" if recorded else "trace.npz")
    if sha256_file(trace) != context["trace_sha256"]:
        raise ValueError("The skill trace changed after this cut was saved")
    with np.load(trace, allow_pickle=False) as archive:
        controls, samples = archive["controls"], archive["samples"]
        terminal = archive["terminal_state" if recorded else "terminal"]
    step = context["physics_step"]
    model, data = checkpoint.model, checkpoint.data
    for control, row in zip(controls[step:], samples[step:], strict=True):
        _replay_step(model, data, control, row[0])
    if not np.array_equal(terminal, native_state(model, data)):
        raise ValueError("Resumed native suffix differs from the recorded terminal state")
    return dict(
        phase=context["phase"],
        remaining_steps=len(controls) - step,
        terminal_bit_equal=True,
        fresh_task_credit=0,
    )
