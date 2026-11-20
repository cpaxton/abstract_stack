"""Native snapshots preserve dynamics, controls and lineage without live writes."""

import json
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from abstract_stack.checkpoints import (
    load_checkpoint,
    model_signature,
    native_state,
    save_checkpoint,
    verify_checkpoint,
)

XML = """<mujoco><option timestep=".002" integrator="implicitfast"/>
<size nuserdata="3"/>
<worldbody><geom name="floor" type="plane" size="1 1 .1"/>
 <body name="product" pos="0 0 .049"><joint name="height" type="slide" axis="0 0 1"/>
  <geom name="ball" size=".05" mass=".2"/></body>
 <body name="target" mocap="true" pos=".2 .3 .4"/>
</worldbody><equality><joint name="fixture" joint1="height" active="false"/></equality>
<actuator><general name="drive" joint="height" dyntype="filter" dynprm=".02"
 gainprm="20" biastype="affine" biasprm="0 -20 -1" ctrlrange="-.1 .1"/>
</actuator></mujoco>"""


@pytest.fixture
def world():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.003
    data.userdata[:] = [1.0, 2.0, 3.0]
    data.mocap_pos[:] = [0.3, 0.2, 0.4]
    data.mocap_quat[:] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    for _ in range(20):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    return model, data


def capture(path, model, data):
    return save_checkpoint(
        path,
        model,
        data,
        scene=dict(seed=123, kind="contact replay fixture"),
        controller=dict(
            kind="fixture", nominal={"drive": 0.002}, applied={"drive": 0.003}, integral=0.4
        ),
        context=dict(
            kind="continuation", original_outcome="failure", incidents=["retained earlier event"]
        ),
        scene_xml=XML,
        validation_controls=np.linspace(data.ctrl, [0.004], 20),
        generator_states={"teacher": np.random.default_rng(123).bit_generator.state},
    )


def test_shared_compiled_model_preserves_complete_checkpoint_and_rejects_differences(
    tmp_path, world
):
    model, data = world
    first = capture(tmp_path / "first", model, data)
    before = native_state(model, data)
    second = save_checkpoint(
        tmp_path / "second",
        model,
        data,
        scene=dict(seed=123),
        controller=dict(kind="same-model"),
        context=dict(kind="independent_state_checkpoint"),
        shared_model_checkpoint=first,
    )
    assert (first / "model.mjb").stat().st_ino == (second / "model.mjb").stat().st_ino
    loaded = load_checkpoint(second)
    np.testing.assert_array_equal(native_state(loaded.model, loaded.data), before)
    np.testing.assert_array_equal(native_state(model, data), before)
    model.body_mass[1] *= 2
    with pytest.raises(ValueError, match="different compiled model"):
        save_checkpoint(
            tmp_path / "changed",
            model,
            data,
            scene=dict(seed=123),
            controller=dict(kind="different-model"),
            context=dict(kind="must_reject"),
            shared_model_checkpoint=first,
        )
    assert not (tmp_path / "changed").exists()


def test_contact_snapshot_and_replay_are_bit_equal_without_touching_live_world(tmp_path, world):
    model, data = world
    before, signature = native_state(model, data), model_signature(model)
    path = capture(tmp_path / "checkpoint", model, data)
    loaded = load_checkpoint(path)
    np.testing.assert_array_equal(native_state(loaded.model, loaded.data), before)
    assert loaded.model is not model and loaded.data is not data
    assert loaded.controller["nominal"] != loaded.controller["applied"]
    assert loaded.context["incidents"] == ["retained earlier event"]
    assert loaded.manifest["fresh_task_credit"] == 0
    assert loaded.random_state["generators"]["teacher"]["bit_generator"] == "PCG64"
    report = verify_checkpoint(path)
    assert report["verified"] and not report["control_admission"]
    assert not report["task_success"] and report["fresh_task_credit"] == 0
    assert report["comparisons"]["contact_records"]["bit_equal"]
    assert report["comparisons"]["constraint_forces"]["bit_equal"]
    np.testing.assert_array_equal(native_state(model, data), before)
    assert model_signature(model) == signature


def test_compiled_model_cache_preserves_each_path_without_a_cache_dependency(tmp_path, world):
    model, data = world
    cache = tmp_path / "models"
    checkpoints = [
        save_checkpoint(
            tmp_path / name,
            model,
            data,
            scene={"seed": 3},
            controller={"kind": "test"},
            context={"kind": "continuation"},
            shared_model_cache=cache,
        )
        for name in ("a", "b")
    ]
    (canonical,) = cache.glob("*.mjb")
    assert all((p / "model.mjb").samefile(canonical) for p in checkpoints)
    assert canonical.stat().st_mode & 0o222 == 0
    canonical.unlink()
    for directory in checkpoints:
        loaded = load_checkpoint(directory)
        np.testing.assert_array_equal(
            native_state(loaded.model, loaded.data), native_state(model, data)
        )


def test_skill_trace_cut_replays_the_suffix_from_an_interior_physics_step(tmp_path, world):
    from abstract_stack.checkpoint_manifest import sha256_file
    from abstract_stack.skill_replay import capture_skill_cut, resume_skill_cut

    model, data = world
    checkpoint = capture(tmp_path / "start", model, data)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "reset.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(checkpoint),
                manifest_sha256=sha256_file(checkpoint / "manifest.json"),
            )
        )
    )
    (attempt / "result.json").write_text(
        json.dumps(dict(events=[dict(phase="push", sim_time=data.time)]))
    )
    initial = native_state(model, data)
    controls = np.array([[0.01], [0.02], [0.03]])
    states = []
    for control in controls:
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        states.append(native_state(model, data))
    np.savez_compressed(
        attempt / "continuation.npz",
        controls=controls,
        samples=np.asarray(states)[:, :1],
        initial_state=initial,
        terminal_state=states[-1],
    )
    cut = tmp_path / "cut"
    assert capture_skill_cut(attempt, 2, cut)["phase"] == "push"
    loaded = load_checkpoint(cut)
    np.testing.assert_array_equal(native_state(loaded.model, loaded.data), states[1])
    assert resume_skill_cut(cut) == dict(
        phase="push", remaining_steps=1, terminal_bit_equal=True, fresh_task_credit=0
    )


def test_relocated_binary_loads_without_original_external_texture_or_xml(tmp_path, world):
    from PIL import Image

    texture = tmp_path / "external.png"
    Image.fromarray(np.full((8, 8, 3), 37, np.uint8)).save(texture)
    xml = XML.replace(
        "<worldbody>",
        f'<asset><texture name="external" type="2d" file="{texture}"/></asset><worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    path = capture(tmp_path / "checkpoint", model, data)
    # Original assets are test fixtures. Upstream checkouts are never renamed.
    texture.unlink()
    (path / "scene.xml").unlink()
    # Deliberate test relocation without optional audit XML; keep required hashes.
    manifest = json.loads((path / "manifest.json").read_text())
    del manifest["hashes"]["scene.xml"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    moved = tmp_path / "relocated"
    path.rename(moved)
    restored = load_checkpoint(moved)
    assert restored.model.ntex == 1
    assert model_signature(restored.model) == model_signature(model)


def test_verification_runs_in_an_independent_python_process(tmp_path, world):
    model, data = world
    path = capture(tmp_path / "checkpoint", model, data)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;from abstract_stack.checkpoints import verify_checkpoint;"
            "print(json.dumps(verify_checkpoint(sys.argv[1])))",
            str(path),
        ],
        env=dict(os.environ, PYTHONFAULTHANDLER="1"),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"]


def test_wrong_runtime_is_rejected_before_native_binary_load(tmp_path, world):
    path = capture(tmp_path / "checkpoint", *world)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["runtime"]["mujoco"] = "incompatible"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="runtime mismatch"):
        load_checkpoint(path)


@pytest.mark.parametrize("payload", ["model.mjb", "state.npz", "controller.json", "context.json"])
def test_changed_or_truncated_payload_is_not_silently_loaded(tmp_path, world, payload):
    path = capture(tmp_path / "checkpoint", *world)
    with (path / payload).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)


def test_incomplete_capture_and_overwrite_are_rejected(tmp_path, world):
    path = capture(tmp_path / "checkpoint", *world)
    with pytest.raises(FileExistsError):
        capture(path, *world)
    (path / "manifest.json").write_text(json.dumps(dict(schema_version=1, status="incomplete")))
    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(path)


def test_manifest_inspection_has_no_native_simulator_import(tmp_path, world):
    path = capture(tmp_path / "checkpoint", *world)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;from abstract_stack.checkpoint_manifest import inspect_checkpoint;"
            "inspect_checkpoint(sys.argv[1]);assert 'mujoco' not in sys.modules",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_new_checkpoint_keeps_portable_compressed_sources_without_python_tree(
    tmp_path, world, monkeypatch
):
    from abstract_stack.checkpoint_manifest import inspect_checkpoint

    monkeypatch.setenv("ABSTRACT_STACK_SOURCE_STORE", str(tmp_path / "source_bundles"))
    path = capture(tmp_path / "checkpoint", *world)
    _, manifest = inspect_checkpoint(path)
    assert manifest["source_capture"]["portable"]
    assert (path / "source_bundle.zip").is_file() and (path / "source_ref.json").is_file()
    assert not list(path.rglob("*.py")) and not (path / "source_snapshot").exists()
    for archived in (tmp_path / "source_bundles").glob("*.zip"):
        archived.unlink()
    relocated = tmp_path / "relocated"
    path.rename(relocated)
    assert verify_checkpoint(relocated)["verified"]


def test_reference_only_checkpoint_requires_its_shared_source_archive(tmp_path, world, monkeypatch):
    monkeypatch.setenv("ABSTRACT_STACK_SOURCE_STORE", str(tmp_path / "source_bundles"))
    model, data = world
    path = save_checkpoint(
        tmp_path / "checkpoint",
        model,
        data,
        scene=dict(seed=123),
        controller=dict(kind="fixture"),
        context=dict(kind="continuation"),
        portable_source=False,
    )
    assert not (path / "source_bundle.zip").exists()
    assert load_checkpoint(path).model.nu == model.nu
    reference = json.loads((path / "source_ref.json").read_text())
    from pathlib import Path

    Path(reference["archive"]).unlink()
    with pytest.raises(ValueError, match="Source archive missing"):
        load_checkpoint(path)


def test_legacy_embedded_python_source_checkpoint_remains_loadable(tmp_path, world):
    import hashlib

    path = capture(tmp_path / "checkpoint", *world)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.pop("source_capture")
    for name in ("source_ref.json", "source_bundle.zip"):
        (path / name).unlink()
        manifest["hashes"].pop(name)
    legacy = path / "source_snapshot/abstract_stack/legacy.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("# Legacy source fixture\n")
    manifest["hashes"][legacy.relative_to(path).as_posix()] = hashlib.sha256(
        legacy.read_bytes()
    ).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_checkpoint(path)["verified"]


def test_inconsistent_applied_metadata_cannot_be_captured(tmp_path, world):
    model, data = world
    with pytest.raises(ValueError, match="applied references"):
        save_checkpoint(
            tmp_path / "bad",
            model,
            data,
            scene={"seed": 1},
            controller={"applied_controls": [123]},
            context={"purpose": "fixture"},
        )
    assert not (tmp_path / "bad").exists()


def test_mismatched_model_data_is_rejected_before_native_state_copy(world):
    model, _ = world
    other = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="dimensions"):
        native_state(model, mujoco.MjData(other))
