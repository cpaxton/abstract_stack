"""Causal native observations, honest action labels and local writer admission."""

import hashlib
import io
import json
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.lfd import (
    EpisodeRecorder,
    ObservationField,
    SensorSample,
    Snapshot,
    load_episode,
)
from abstract_stack.lfd.episode import student_observation, teacher_observation
from abstract_stack.lfd.legacy import audit_legacy_bundle, import_legacy_for_bc
from abstract_stack.lfd.lerobot import _auxiliary_features, export_lerobot, verify_lerobot
from abstract_stack.lfd.native import action_contract, proprioception


@pytest.fixture
def plant():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body><joint name="hinge" range="-1 1"/><geom size=".03"/></body></worldbody>
      <actuator><position name="position_ref" joint="hinge" kp="10" ctrlrange="-1 1"/>
      </actuator></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def fields():
    return (
        ObservationField("observation.state", (3,), "rad,rad/s,Nm", "actuator", "native measured"),
        ObservationField(
            "observation.wrist_wrench", (2, 6), "N,Nm", "wrist", "native parent reaction"
        ),
        ObservationField(
            "observation.images.front",
            (4, 4, 3),
            "uint8 RGB",
            "front",
            "native raw RGB",
            dtype="uint8",
        ),
    )


def recorder(tmp_path, plant, *, purpose="pipeline_fixture"):
    model, _ = plant
    return EpisodeRecorder(
        tmp_path / "episode",
        action_contract(model),
        fields(),
        fps=50,
        physics_dt_s=0.002,
        provenance=dict(
            model_revision="fixture",
            controller_source={"sha256": hashlib.sha256(b"teacher").hexdigest()},
            controller_settings={},
            scene_seed=0,
        ),
        task="native recording fixture",
        purpose=purpose,
    )


def collect(tmp_path, plant, *, status="failure", purpose="pipeline_fixture"):
    model, data = plant
    value = recorder(tmp_path, plant, purpose=purpose)
    for nominal, applied in ((0.1, 0.12), (0.2, 0.22)):
        value.begin(proprioception(data), [nominal], teacher={"object_pose": [1, 2, 3]})
        data.ctrl[:] = applied
        for _ in range(10):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        value.end(proprioception(data), [applied])
    value.finish(status=status, task_success=status == "success", reason="fixture completed")
    return value.directory


def test_native_pairs_are_pre_action_and_do_not_relabel_applied_controls(tmp_path, plant):
    directory = collect(tmp_path, plant)
    episode = load_episode(directory)
    assert episode.arrays["observation.state"][0, 0] == 0
    assert episode.arrays["observation.state"][1, 0] != 0
    np.testing.assert_array_equal(episode.arrays["action.nominal"], [[0.1], [0.2]])
    np.testing.assert_array_equal(episode.arrays["action.applied"], [[0.12], [0.22]])
    assert episode.arrays["observation_time_s"].tolist() == pytest.approx([0, 0.02, 0.04])
    assert not episode.metadata["training_eligible"]
    assert all("object_pose" not in frame for frame in episode.student_frames())
    assert "object_pose" in (directory / "teacher.jsonl").read_text()


def test_missing_sensors_are_masked_not_invented_and_raw_images_keep_uint8(tmp_path, plant):
    episode = load_episode(collect(tmp_path, plant))
    name = "observation.images.front"
    assert episode.arrays[name].dtype == np.uint8
    assert not np.any(episode.arrays[name])
    assert not np.any(episode.arrays[name + ".available"])
    np.testing.assert_array_equal(episode.arrays[name + ".sample_time_s"], [-1, -1, -1])
    assert not any(k.startswith("teacher") for k in next(episode.student_frames()))


@pytest.mark.parametrize("value", [1e300, -1e300])
def test_finite_values_that_overflow_declared_observation_dtype_are_rejected(value):
    with pytest.raises(ValueError, match="finite values representable as float32"):
        student_observation(
            Snapshot(0, {"observation.state": SensorSample(np.array([value, 0, 0]), 0)}),
            fields(),
        )


def replace_payload(directory, key, value):
    path = directory / "frames.npz"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays[key][0, 0] = value
    np.savez_compressed(path, **arrays)
    metadata_path = directory / "episode.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"]["frames.npz"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))


def test_nonfinite_stored_observation_is_refused_even_with_matching_payload_hash(tmp_path, plant):
    directory = collect(tmp_path, plant)
    replace_payload(directory, "observation.state", np.inf)
    with pytest.raises(ValueError, match="finite values"):
        load_episode(directory)


@pytest.mark.parametrize("key", ["action.nominal", "action.applied"])
def test_export_rejects_float32_action_overflow_before_creating_dataset(
    tmp_path, plant, monkeypatch, key
):
    directory = collect(tmp_path, plant)
    metadata_path = directory / "episode.json"
    metadata = json.loads(metadata_path.read_text())
    # Unlimited native actuators can accept finite float64 values that cannot
    # be represented by the SDK's declared float32 action feature.
    metadata["action"]["lower"] = metadata["action"]["upper"] = [None]
    metadata_path.write_text(json.dumps(metadata))
    replace_payload(directory, key, 1e300)
    assert np.isfinite(load_episode(directory).arrays[key]).all()
    monkeypatch.setattr(
        "abstract_stack.lfd.lerobot._dataset_class",
        lambda: pytest.fail("Writer must not initialize before action dtype admission"),
    )
    with pytest.raises(ValueError, match="finite values representable as float32"):
        export_lerobot([directory], tmp_path / "refused", include_failed=True)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize(
    "key,changed",
    [
        ("sensor_contract", {"tactile_order": ["different_body_local_geom"]}),
        ("proprioception_order", ["different_native_actuator"]),
        ("proprioception_units", ["different_units"]),
    ],
)
def test_export_rejects_changed_native_channel_meaning_before_writer(
    tmp_path, plant, monkeypatch, key, changed
):
    first = collect(tmp_path / "first", plant)
    second = collect(tmp_path / "second", plant)
    path = second / "episode.json"
    metadata = json.loads(path.read_text())
    metadata["provenance"][key] = changed
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(
        "abstract_stack.lfd.lerobot._dataset_class",
        lambda: pytest.fail("Incompatible native semantics must fail before writer initialization"),
    )
    with pytest.raises(ValueError, match=f"identical native {key} semantics"):
        export_lerobot([first, second], tmp_path / "refused", include_failed=True)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("stamp", [0.001, -0.001, float("nan")])
def test_future_stale_or_nonfinite_sensor_timestamps_cannot_enter_student_data(plant, stamp):
    _, data = plant
    with pytest.raises(ValueError, match="timestamp"):
        student_observation(
            Snapshot(0, {"observation.state": SensorSample(np.zeros(3), stamp)}), fields()
        )


def test_privileged_teacher_has_separate_namespace_and_unknown_fields_fail(plant):
    _, data = plant
    value = teacher_observation(proprioception(data), fields(), {"object_pose": [1, 2, 3]})
    assert value["privileged"]["object_pose"] == [1, 2, 3]
    assert all("object_pose" not in k for k in value["student"])
    with pytest.raises(ValueError, match="declared"):
        ObservationField("observation.object_pose", (3,), "m", "world", "oracle")


def test_unknown_partial_episode_retains_incomplete_action_and_refuses_export(tmp_path, plant):
    model, data = plant
    value = recorder(tmp_path, plant)
    value.begin(proprioception(data), [0.1])
    data.ctrl[:] = 0.12
    for _ in range(10):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
    value.end(proprioception(data), [0.12])
    value.begin(proprioception(data), [0.2], teacher={"object_pose": [1, 2, 3]})
    value.finish(status="unknown", task_success=None, reason="process interrupted after decision")
    episode = load_episode(value.directory)
    assert not episode.metadata["complete"]
    assert episode.metadata["outcome"]["task_success"] is None
    assert episode.metadata["incomplete_action"]["nominal"] == [0.2]
    with pytest.raises(ValueError, match="Unknown"):
        export_lerobot([value.directory], tmp_path / "export", include_failed=True)
    assert not (tmp_path / "export").exists()


def test_out_of_range_and_irregular_action_intervals_rejected(tmp_path, plant):
    _, data = plant
    value = recorder(tmp_path, plant)
    with pytest.raises(ValueError, match="bounds"):
        value.begin(proprioception(data), [2])
    value.begin(proprioception(data), [0.1])
    with pytest.raises(ValueError, match="control period"):
        value.end(Snapshot(0.03, {"observation.state": SensorSample(np.zeros(3), 0.03)}), [0.1])


def test_legacy_unknown_process_never_becomes_successful_teacher_data(tmp_path):
    directory = tmp_path / "episode"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(dict(task_success=True)))
    (tmp_path / "process.json").write_text(json.dumps(dict(returncode=-11, timeout=False)))
    np.savez_compressed(
        directory / "trace.npz",
        samples=np.array([[0.002, 0], [0.004, 0]]),
        controls=np.ones((2, 1)),
    )
    audit = audit_legacy_bundle(directory)
    assert not audit["training_ready"] and audit["verified_task_success"] is None
    assert audit["action_semantics"] == "applied compensated servo controls"
    with pytest.raises(ValueError, match="diagnostic-only"):
        import_legacy_for_bc(directory)


def test_payload_hash_changes_fail_closed(tmp_path, plant):
    directory = collect(tmp_path, plant)
    with (directory / "frames.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_episode(directory)


def test_real_writer_api_contract_keeps_failures_actions_and_finalize(tmp_path, plant, monkeypatch):
    source = collect(tmp_path, plant)
    captured = {}

    class Writer:
        @classmethod
        def create(cls, **kwargs):
            captured.update(kwargs)
            kwargs["root"].mkdir()
            return cls()

        def add_frame(self, frame):
            captured.setdefault("frames", []).append(frame)

        def save_episode(self):
            captured["saved"] = True

        def finalize(self):
            captured["finalized"] = True

    monkeypatch.setattr("abstract_stack.lfd.lerobot._dataset_class", lambda: Writer)
    with pytest.raises(ValueError, match="include_failed"):
        export_lerobot([source], tmp_path / "refused")
    result = export_lerobot([source], tmp_path / "export", include_failed=True)
    assert captured["saved"] and captured["finalized"] and not result["uploaded"]
    assert not any(
        "image" in key and feature["dtype"] not in {"image", "video"}
        for key, feature in captured["features"].items()
    )
    camera_aux = result["sensor_auxiliary_features"]["observation.images.front"]
    assert captured["frames"][0][camera_aux["available"]][0] == 0
    assert captured["frames"][0][camera_aux["sample_time_s"]][0] == -1
    assert "observation.images.front.available" not in captured["frames"][0]
    assert captured["frames"][0]["action"][0] == pytest.approx(0.1)
    assert captured["frames"][0]["diagnostics.applied_action"][0] == pytest.approx(0.12)
    assert "timestamp" not in captured["frames"][0]
    assert captured["frames"][0]["diagnostics.observation_time_s"].dtype == np.float64
    assert captured["frames"][1]["diagnostics.observation_time_s"][0] == pytest.approx(0.02)
    assert captured["frames"][-1]["next.done"][0] == 1
    assert not any(f["quality.training_eligible"][0] for f in captured["frames"])
    assert all(
        not any("object_pose" in k or k.startswith("teacher") for k in f)
        for f in captured["frames"]
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_verification_keeps_new_scalar_mapping_and_legacy_camera_metadata(
    tmp_path, plant, monkeypatch, legacy
):
    from PIL import Image

    source = load_episode(collect(tmp_path, plant))
    output = tmp_path / "dataset"
    (output / "data").mkdir(parents=True)
    (output / "data/part.parquet").touch()
    auxiliary = _auxiliary_features(source.metadata["fields"])
    stored, decoded = [], []
    for i, frame in enumerate(source.student_frames()):
        raw, row = dict(index=i), {}
        for field in source.metadata["fields"]:
            name = field["name"]
            raw[name] = source.arrays[name][i].tolist()
            row[name] = source.arrays[name][i]
            if field["dtype"] == "uint8":
                encoded = io.BytesIO()
                Image.fromarray(source.arrays[name][i]).save(encoded, format="PNG")
                raw[name] = dict(bytes=encoded.getvalue())
                row[name] = source.arrays[name][i].transpose(2, 0, 1).astype(np.float32) / 255
            for suffix in ("available", "sample_time_s"):
                old = name + "." + suffix
                key = old if legacy else auxiliary[name][suffix]
                row[key] = frame[old]
                raw[key] = frame[old].tolist()
        clock = source.arrays["observation_time_s"][i]
        raw["diagnostics.observation_time_s"] = float(clock)
        row.update(
            {
                "diagnostics.observation_time_s": np.asarray(clock, np.float32),
                "action": source.arrays["action.nominal"][i].astype(np.float32),
                "diagnostics.applied_action": source.arrays["action.applied"][i].astype(np.float32),
                "next.done": np.asarray(i == source.metadata["transitions"] - 1, np.float32),
                "next.success": np.asarray(False, np.float32),
                "quality.training_eligible": np.asarray(False, np.float32),
            }
        )
        stored.append(raw)
        decoded.append(row)

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return len(decoded)

        def get_raw_item(self, index):
            return decoded[index]

    metadata = dict(episodes=[dict(source=str(source.directory), metadata=source.metadata)])
    if not legacy:
        metadata["sensor_auxiliary_features"] = auxiliary
    (output / "collection_provenance.json").write_text(json.dumps(metadata))
    parquet = SimpleNamespace(read_table=lambda path: SimpleNamespace(to_pylist=lambda: stored))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr("abstract_stack.lfd.lerobot._dataset_class", lambda: Reader)
    result = verify_lerobot(output)
    assert result["frames"] == 2 and result["masks_and_outcomes_preserved"]
    assert result["native_clock_float64_storage_exact"]
    assert result["raw_image_storage_uint8_pixels_exact"]


@pytest.mark.mabel
def test_mabel_action_units_distinguish_wheel_velocity_and_lift_metres():
    from abstract_stack.assets import mabel_model_path

    try:
        model = mujoco.MjModel.from_xml_path(str(mabel_model_path()))
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    contract = action_contract(model)
    assert len(contract.names) == 59
    units = dict(zip(contract.names, contract.units, strict=True))
    assert units["Lift"] == "m" and units["FL Drive"] == "rad/s" and units["Left Arm 1"] == "rad"
    assert contract.lower[contract.names.index("FL Steer")] is None
