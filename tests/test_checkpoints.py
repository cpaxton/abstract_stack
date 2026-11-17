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
