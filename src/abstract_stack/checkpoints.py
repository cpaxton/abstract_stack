"""Portable-to-the-same-runtime native checkpoints, never fresh task credit.

Only fresh data objects are restored. Saving and reference replay never step or
modify the caller's live world. MJB embeds compiled meshes/textures and avoids
recompiling MJCF on restore; original MJCF and scene metadata remain auditable.
"""

import errno
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.checkpoint_manifest import SCHEMA, inspect_checkpoint
from abstract_stack.checkpoint_manifest import sha256_file as _sha
from abstract_stack.native_views import retain_data_views, retain_model_views
from abstract_stack.provenance import policy_provenance
from abstract_stack.source_store import freeze_source

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    json.dumps(value, allow_nan=False)
    return value


def _write(path, value):
    path.write_text(json.dumps(_json(value), indent=2, allow_nan=False) + "\n")


def native_state(model, data):
    dimensions = {
        "qpos": (model.nq,),
        "qvel": (model.nv,),
        "act": (model.na,),
        "ctrl": (model.nu,),
        "qacc_warmstart": (model.nv,),
        "mocap_pos": (model.nmocap, 3),
        "mocap_quat": (model.nmocap, 4),
        "userdata": (model.nuserdata,),
        "eq_active": (model.neq,),
        "plugin_state": (model.npluginstate,),
        "xpos": (model.nbody, 3),
    }
    if any(getattr(data, name).shape != shape for name, shape in dimensions.items()):
        raise ValueError("Checkpoint model/data dimensions do not match")
    value = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, value, STATE)
    return value


def model_signature(model):
    """Hash native model arrays/options, including original limits and masks."""
    result = {}
    for group, owner in (("model", model), ("options", model.opt)):
        fields = {}
        for name in dir(owner):
            # MuJoCo's compilation signature is reset to zero by MJB load;
            # it is not an encoded physical parameter. Array/options hashes
            # and the complete MJB bytes identify the actual model instead.
            if name.startswith("_") or group == "model" and name == "signature":
                continue
            value = getattr(owner, name)
            if isinstance(value, np.ndarray):
                fields[name] = dict(
                    shape=list(value.shape),
                    dtype=value.dtype.str,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                )
            elif isinstance(value, (bool, int, float, np.generic)):
                fields[name] = _json(value)
            elif isinstance(value, bytes):
                fields[name] = dict(sha256=hashlib.sha256(value).hexdigest())
        result[group] = fields
    return result


def _runtime():
    return dict(
        mujoco=mujoco.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        byteorder=sys.byteorder,
        native_float_bytes=np.dtype(float).itemsize,
    )


def _contacts(model, data):
    records = []
    for index, contact in enumerate(data.contact):
        force = np.zeros(6)
        if contact.efc_address >= 0:
            mujoco.mj_contactForce(model, data, index, force)
        records.append(
            np.r_[
                contact.geom1,
                contact.geom2,
                contact.dist,
                contact.pos,
                contact.frame,
                contact.dim,
                force,
            ]
        )
    result = np.asarray(records, float).reshape(-1, 22)
    if len(result):
        result = result[np.lexsort(tuple(result[:, i] for i in range(5, -1, -1)))]
    return result


def replay_witness(model, data, controls):
    """Replay on a private native copy; no safety/task admission is implied."""
    controls = np.asarray(controls, float)
    if (
        controls.ndim != 2
        or controls.shape[1] != model.nu
        or not 1 <= len(controls) <= 500
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Witness needs 1–500 finite full native applied-control vectors")
    scratch = mujoco.MjData(model)
    views = retain_data_views(model, scratch)
    before = native_state(model, data)
    mujoco.mj_copyData(scratch, model, data)
    mujoco.mj_forward(model, scratch)
    if not np.array_equal(before, native_state(model, scratch)):
        raise ValueError("Forward reconstruction changed the captured integration state")
    states, positions, rotations, accelerations, constraints, contacts, offsets = (
        [],
        [],
        [],
        [],
        [],
        [],
        [0],
    )
    for index in range(len(controls) + 1):
        if index:
            scratch.ctrl[:] = controls[index - 1]
            mujoco.mj_step(model, scratch)
            mujoco.mj_forward(model, scratch)
        state = native_state(model, scratch)
        if not np.isfinite(state).all() or np.any(scratch.warning.number):
            raise ValueError("Native witness encountered nonfinite state or a warning")
        if abs(scratch.time - data.time - index * model.opt.timestep) > 1e-8:
            raise ValueError("Native witness simulation clock reset")
        states.append(state)
        positions.append(scratch.xpos.copy())
        rotations.append(scratch.xmat.copy())
        accelerations.append(scratch.qacc.copy())
        constraints.append(scratch.qfrc_constraint.copy())
        contact = _contacts(model, scratch)
        if any(
            not np.isfinite(value).all()
            for value in (
                contact,
                scratch.xpos,
                scratch.xmat,
                scratch.qacc,
                scratch.qfrc_constraint,
            )
        ):
            raise ValueError("Native witness has nonfinite derived pose/force data")
        contacts.append(contact)
        offsets.append(offsets[-1] + len(contact))
    if not np.array_equal(before, native_state(model, data)):
        raise ValueError("Witness modified the caller's world")
    assert views
    return dict(
        controls=controls.copy(),
        integration=np.asarray(states),
        body_positions=np.asarray(positions),
        body_rotations=np.asarray(rotations),
        accelerations=np.asarray(accelerations),
        constraint_forces=np.asarray(constraints),
        contact_records=np.concatenate(contacts),
        contact_offsets=np.asarray(offsets, dtype=np.int64),
    )


def _share_compiled_model(path, cache):
    """Share exact MJB bytes while keeping a complete file at the checkpoint path."""
    digest = _sha(path)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    canonical = cache / f"{digest}.mjb"
    path.chmod(0o444)
    pending = path.with_suffix(".pending")
    try:
        try:
            os.link(path, canonical)
        except FileExistsError:
            if (
                canonical.is_symlink()
                or canonical.stat().st_mode & 0o222
                or _sha(canonical) != digest
            ):
                raise ValueError(
                    "Shared compiled model cache entry failed byte verification"
                ) from None
            os.link(canonical, pending)
            pending.replace(path)
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EOPNOTSUPP, errno.EPERM, errno.EMLINK):
            raise
        # A standalone checkpoint remains complete on filesystems without sharing.
    finally:
        pending.unlink(missing_ok=True)


def save_checkpoint(
    directory,
    model,
    data,
    *,
    scene,
    controller,
    context,
    scene_xml=None,
    validation_controls=None,
    generator_states=None,
    portable_source=True,
    shared_model_checkpoint=None,
    shared_model_cache=None,
):
    """Capture a paused control boundary and a short scratch-only replay witness.

    Controller/observer memory must be explicitly supplied, not inferred from
    compensated controls. Arbitrary Python controllers are never pickled or
    automatically resumed. All task outcomes remain inherited continuations.
    """
    directory = Path(directory)
    if type(portable_source) is not bool:
        raise ValueError("Checkpoint source portability must be an explicit boolean")
    # Validate JSON before creating a bundle, and refuse overwrite.
    scene, controller, context = (_json(x) for x in (scene, controller, context))
    if not all(isinstance(x, dict) and x for x in (scene, controller, context)):
        raise ValueError("Checkpoint requires explicit scene, controller and lineage context")
    state = native_state(model, data)
    if not np.isfinite(state).all() or data.time < 0:
        raise ValueError("Cannot checkpoint invalid native state")
    if "applied_controls" in controller:
        applied = np.asarray(controller["applied_controls"], float)
        if applied.shape != data.ctrl.shape or not np.array_equal(applied, data.ctrl):
            raise ValueError("Controller applied references differ from captured native controls")
    for name, value in (controller.get("nominal_subset") or {}).items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0 or np.asarray(value).shape != () or not np.isfinite(value):
            raise ValueError("Unknown/nonfinite named nominal reference in checkpoint")
        if (
            model.actuator_ctrllimited[aid]
            and not model.actuator_ctrlrange[aid, 0] <= value <= model.actuator_ctrlrange[aid, 1]
        ):
            raise ValueError("Checkpoint nominal reference exceeds native actuator bounds")
    signature = model_signature(model)
    shared_model = None
    if shared_model_checkpoint is not None:
        previous, previous_manifest = inspect_checkpoint(shared_model_checkpoint)
        if previous_manifest["model_signature"] != signature:
            raise ValueError("Shared checkpoint has a different compiled model")
        shared_model = previous / "model.mjb"
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "manifest.json", dict(schema_version=SCHEMA, status="incomplete"))
    if shared_model is None:
        mujoco.mj_saveModel(model, str(directory / "model.mjb"), None)
        if shared_model_cache is not None:
            _share_compiled_model(directory / "model.mjb", shared_model_cache)
    else:
        # The complete verified MJB remains present at each checkpoint path.
        # Link only on the same filesystem; never silently substitute a model.
        os.link(shared_model, directory / "model.mjb")
    np.savez_compressed(directory / "state.npz", integration=state, state_spec=int(STATE))
    _write(directory / "scene.json", scene)
    _write(directory / "controller.json", controller)
    _write(directory / "context.json", context)
    _write(
        directory / "random.json",
        dict(
            python=random.getstate(),
            numpy_legacy=np.random.get_state(),
            generators=generator_states or {},
            restore_policy="returned explicitly; loading never changes process-global RNG",
        ),
    )
    if scene_xml is not None:
        (directory / "scene.xml").write_text(scene_xml)
    source = Path(__file__).resolve().parent
    source_root = source.parents[1] if source.parent.name == "src" else source.parent
    source_paths = [source.relative_to(source_root).as_posix()]
    source_paths.extend(
        name for name in ("pyproject.toml", "uv.lock") if (source_root / name).is_file()
    )
    frozen = freeze_source(source_root, paths=source_paths, materialize=False)
    frozen.write_reference(directory / "source_ref.json", portable=portable_source)
    controls = np.tile(data.ctrl, (50, 1)) if validation_controls is None else validation_controls
    witness = replay_witness(model, data, controls)
    np.savez_compressed(directory / "witness.npz", **witness)
    if not np.array_equal(state, native_state(model, data)) or signature != model_signature(model):
        raise ValueError("Checkpoint changed the live model or state")
    manifest = dict(
        schema_version=SCHEMA,
        status="complete",
        kind="native_physics_checkpoint_continuation",
        fresh_task_credit=0,
        task_success=False,
        runtime=_runtime(),
        state_spec=int(STATE),
        state_size=len(state),
        simulation_time_s=float(data.time),
        physics_timestep_s=float(model.opt.timestep),
        dimensions={key: int(getattr(model, key)) for key in ("nq", "nv", "nu", "nbody")},
        actuator_order=[
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        ],
        model_signature=signature,
        policy_source=policy_provenance(),
        source_capture=dict(
            format="source_bundle_zip", portable=portable_source, content_id=frozen.content_id
        ),
        controller_restore=(
            "explicit caller-owned controller/observer state; no arbitrary Python resume"
        ),
        external_force_inputs_present=bool(np.any(data.qfrc_applied) or np.any(data.xfrc_applied)),
        witness_steps=len(witness["controls"]),
        witness_step_order="applied ctrl; mj_step; mj_forward; all native point forces",
        hashes={
            p.relative_to(directory).as_posix(): _sha(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "manifest.json"
        },
    )
    _write(directory / "manifest.pending", manifest)
    (directory / "manifest.pending").replace(directory / "manifest.json")
    return directory


@dataclass(frozen=True)
class RestoredCheckpoint:
    directory: Path
    model: object
    data: object
    manifest: dict
    scene: dict
    controller: dict
    context: dict
    random_state: dict
    _views: tuple = field(repr=False)


def load_checkpoint(directory):
    """Load embedded compiled assets into NEW native model/data, then reconstruct."""
    directory, manifest = inspect_checkpoint(directory)
    current = _runtime()
    for key in ("mujoco", "numpy", "byteorder", "native_float_bytes"):
        if manifest["runtime"][key] != current[key]:
            raise ValueError(f"Checkpoint runtime mismatch: {key}; rebuild explicitly, no fallback")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    views = retain_model_views(model)
    if model_signature(model) != manifest["model_signature"]:
        raise ValueError("Compiled checkpoint model differs from captured native arrays/options")
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    with np.load(directory / "state.npz", allow_pickle=False) as saved:
        state = saved["integration"].copy()
        spec = int(saved["state_spec"])
    if (
        spec != int(STATE)
        or spec != manifest["state_spec"]
        or state.shape != (mujoco.mj_stateSize(model, STATE),)
        or state.dtype != np.dtype(float)
        or not np.isfinite(state).all()
    ):
        raise ValueError("Invalid native checkpoint state contract")
    mujoco.mj_setState(model, data, state, STATE)
    mujoco.mj_forward(model, data)
    if not np.array_equal(state, native_state(model, data)):
        raise ValueError("Restored integration state differs after forward reconstruction")
    return RestoredCheckpoint(
        directory,
        model,
        data,
        manifest,
        *(
            json.loads((directory / name).read_text())
            for name in ("scene.json", "controller.json", "context.json", "random.json")
        ),
        (views, data_views),
    )


def verify_checkpoint(directory):
    """Strict state/pose/force replay against the original private-copy witness."""
    checkpoint = load_checkpoint(directory)
    with np.load(checkpoint.directory / "witness.npz", allow_pickle=False) as saved:
        reference = {name: saved[name] for name in saved.files}
    actual = replay_witness(checkpoint.model, checkpoint.data, reference["controls"])
    if reference.keys() != actual.keys():
        raise ValueError("Checkpoint replay witness fields do not match the schema")
    comparisons = {}
    for name, expected in reference.items():
        observed = actual[name]
        equal_shape = observed.shape == expected.shape
        equal = bool(
            equal_shape
            and observed.dtype == expected.dtype
            and observed.tobytes() == expected.tobytes()
        )
        comparisons[name] = dict(
            bit_equal=equal,
            shape_equal=equal_shape,
            maximum_absolute_difference=(
                float(np.max(np.abs(observed - expected), initial=0)) if equal_shape else None
            ),
        )
    return dict(
        kind="strict_native_checkpoint_replay",
        verified=all(row["bit_equal"] for row in comparisons.values()),
        comparisons=comparisons,
        replay_steps=len(reference["controls"]),
        physics_timestep_s=float(checkpoint.model.opt.timestep),
        source_checkpoint=str(checkpoint.directory),
        checkpoint_model_sha256=checkpoint.manifest["hashes"]["model.mjb"],
        fresh_task_credit=0,
        task_success=False,
        control_admission=False,
        runtime=_runtime(),
    )
