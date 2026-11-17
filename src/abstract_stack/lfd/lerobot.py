"""Optional real LeRobot writer, never a home-grown imitation or Hub publisher."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from abstract_stack.lfd.episode import _finite_cast, load_episode

SUPPORTED_VERSION = "0.7.0"


def _dataset_class():
    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("LeRobot 0.7.0 exporter dependency is not installed") from error
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"LeRobot {version} is not the admitted exporter version {SUPPORTED_VERSION}"
        )
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset


def export_lerobot(directories, output, *, repo_id="local/mabel-lfd", include_failed=False):
    """Write a local official dataset after full preflight; never upload.

    Unknown/incomplete episodes are refused even when including known failures.
    Episode source, task outcome, sensor frames and action contract remain in a
    local sidecar; privileged teacher payload is deliberately not copied.
    """
    episodes = [load_episode(path) for path in directories]
    if not episodes:
        raise ValueError("No demonstration episodes supplied")
    first = episodes[0].metadata
    for episode in episodes:
        metadata = episode.metadata
        if not metadata["complete"] or metadata["outcome"]["status"] == "unknown":
            raise ValueError("Unknown/incomplete outcomes cannot become teacher training episodes")
        if metadata["outcome"]["status"] == "failure" and not include_failed:
            raise ValueError(
                "Known failed episodes require explicit include_failed; nothing silently dropped"
            )
        if any(metadata[k] != first[k] for k in ("action", "fields", "fps")):
            raise ValueError("LeRobot export requires identical action/sensor/clock contracts")
        for name in ("action.nominal", "action.applied"):
            _finite_cast(episode.arrays[name], "float32", name)
    output = Path(output).resolve()
    if output.exists() or any(
        output.is_relative_to(e.directory.resolve()) or e.directory.resolve().is_relative_to(output)
        for e in episodes
    ):
        raise ValueError("Use a fresh output outside the source episodes")
    dataset_class = _dataset_class()
    features = {}
    for field in first["fields"]:
        name = field["name"]
        image = field["dtype"] == "uint8"
        features[name] = dict(
            dtype="image" if image else field["dtype"],
            shape=tuple(field["shape"]),
            names=["height", "width", "channels"] if image else None,
        )
        features[name + ".available"] = dict(dtype="float32", shape=(1,), names=["available"])
        features[name + ".sample_time_s"] = dict(
            dtype="float64", shape=(1,), names=["native_sample_time_s"]
        )
    action = first["action"]
    width = len(action["names"])
    features["action"] = dict(dtype="float32", shape=(width,), names=action["names"])
    features["diagnostics.applied_action"] = dict(
        dtype="float32", shape=(width,), names=action["names"]
    )
    features["quality.training_eligible"] = dict(
        dtype="float32", shape=(1,), names=["training_eligible"]
    )
    features["diagnostics.observation_time_s"] = dict(
        dtype="float64", shape=(1,), names=["native_observation_time_s"]
    )
    features["next.done"] = dict(dtype="float32", shape=(1,), names=["done"])
    features["next.success"] = dict(dtype="float32", shape=(1,), names=["task_success"])
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset_class.create(
        repo_id=repo_id,
        root=output,
        fps=first["fps"],
        features=features,
        robot_type="mabel_native_actuators",
        use_videos=False,
    )
    saved = []
    try:
        for episode in episodes:
            metadata = episode.metadata
            for index, frame in enumerate(episode.student_frames()):
                frame["action"] = episode.arrays["action.nominal"][index].astype(np.float32)
                frame["diagnostics.applied_action"] = episode.arrays["action.applied"][
                    index
                ].astype(np.float32)
                frame["quality.training_eligible"] = np.array(
                    [metadata["training_eligible"]], np.float32
                )
                frame["next.done"] = np.array([index == metadata["transitions"] - 1], np.float32)
                frame["next.success"] = np.array([metadata["outcome"]["task_success"]], np.float32)
                frame["task"] = metadata["task"]
                # v0.7 owns timestamp/frame_index. Explicit preflight above
                # proves the native clock is uniform; preserve its absolute
                # value separately instead of injecting a reserved feature.
                frame["diagnostics.observation_time_s"] = np.array(
                    [episode.arrays["observation_time_s"][index]], np.float64
                )
                dataset.add_frame(frame)
            dataset.save_episode()
            saved.append(dict(source=str(episode.directory.resolve()), metadata=metadata))
    finally:
        dataset.finalize()
    sidecar = dict(
        kind="local_lerobot_export",
        lerobot_version=SUPPORTED_VERSION,
        repo_id=repo_id,
        episodes=saved,
        student_inputs=[f["name"] for f in first["fields"]],
        teacher_privileged_exported=False,
        uploaded=False,
        fresh_task_credit=0,
        timestamp_semantics=(
            "SDK relative i/fps; native clock retained in diagnostics.observation_time_s"
        ),
    )
    (output / "collection_provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def verify_lerobot(output):
    """Offline local reload plus exact stored-clock/action/sensor checks.

    LeRobot's default HF-to-Torch transform casts float64 sensor timestamps to
    float32. Check DOUBLE Parquet storage exactly, then verify the reader's
    explicitly observed cast instead of weakening native timestamp admission.
    """
    import pyarrow.parquet as pq

    output = Path(output)
    sidecar = json.loads((output / "collection_provenance.json").read_text())
    dataset = _dataset_class()(
        sidecar.get("repo_id", "local/mabel-lfd"),
        root=output,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    stored = []
    for path in sorted((output / "data").rglob("*.parquet")):
        stored.extend(pq.read_table(path).to_pylist())
    stored.sort(key=lambda row: row["index"])
    expected_count = sum(item["metadata"]["transitions"] for item in sidecar["episodes"])
    if len(dataset) != expected_count or len(stored) != expected_count:
        raise ValueError("Exported frame count differs from scheduled transitions")
    global_index = 0
    reader_clock_dtypes = set()
    for item in sidecar["episodes"]:
        episode = load_episode(item["source"])
        if episode.metadata["sha256"] != item["metadata"]["sha256"]:
            raise ValueError("Export source changed since collection")
        for index in range(episode.metadata["transitions"]):
            row = dataset.get_raw_item(global_index)
            raw = stored[global_index]
            clock = episode.arrays["observation_time_s"][index]
            # Python float from Arrow DOUBLE retains the original native value.
            if raw["diagnostics.observation_time_s"] != clock:
                raise ValueError("Stored native observation clock changed")
            for key, source in (
                ("action", "action.nominal"),
                ("diagnostics.applied_action", "action.applied"),
            ):
                np.testing.assert_array_equal(
                    np.asarray(row[key]), episode.arrays[source][index].astype(np.float32)
                )
            clock_view = np.asarray(row["diagnostics.observation_time_s"])
            if clock_view.dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("Unexpected SDK timestamp representation")
            reader_clock_dtypes.add(str(clock_view.dtype))
            np.testing.assert_array_equal(clock_view, np.asarray(clock, dtype=clock_view.dtype))
            for field in episode.metadata["fields"]:
                name = field["name"]
                expected = episode.arrays[name][index]
                observed = np.asarray(row[name])
                if field["dtype"] == "uint8" and observed.shape != expected.shape:
                    observed = observed.transpose(1, 2, 0)
                np.testing.assert_array_equal(observed, expected)
                if bool(np.asarray(row[name + ".available"]).item()) != bool(
                    episode.arrays[name + ".available"][index]
                ):
                    raise ValueError("Sensor availability changed")
                if raw[name + ".sample_time_s"] != [
                    float(episode.arrays[name + ".sample_time_s"][index])
                ]:
                    # Arrow may expose a one-element list or a scalar.
                    if raw[name + ".sample_time_s"] != float(
                        episode.arrays[name + ".sample_time_s"][index]
                    ):
                        raise ValueError("Stored native sensor timestamp changed")
            if bool(np.asarray(row["next.done"]).item()) != (
                index == episode.metadata["transitions"] - 1
            ):
                raise ValueError("Episode terminal label changed")
            if (
                bool(np.asarray(row["next.success"]).item())
                != episode.metadata["outcome"]["task_success"]
            ):
                raise ValueError("Task outcome changed")
            if (
                bool(np.asarray(row["quality.training_eligible"]).item())
                != episode.metadata["training_eligible"]
            ):
                raise ValueError("Training eligibility changed")
            if any(key.startswith("teacher") or "object_pose" in key for key in row):
                raise ValueError("Privileged context leaked into exported frames")
            global_index += 1
    return dict(
        kind="official_lerobot_local_roundtrip",
        episodes=len(sidecar["episodes"]),
        frames=len(dataset),
        native_clock_float64_storage_exact=True,
        reader_clock_dtypes=sorted(reader_clock_dtypes),
        nominal_applied_and_sensor_values_exact_at_declared_dtype=True,
        masks_and_outcomes_preserved=True,
        privileged_fields_exported=False,
        uploaded=False,
        video_decode_tested=False,
        fresh_task_credit=0,
        exporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
