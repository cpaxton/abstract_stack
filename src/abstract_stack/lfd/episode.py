"""Control-rate observation/action pairs; applied servo targets are diagnostics.

This format is internal, not a hand-written imitation of LeRobot. No simulator
import is needed to read it. Privileged teacher state is stored separately and
never enters ``student_frames``. Training and hardware readiness are not implied.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != tuple(shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite values with shape {shape}")
    return result


@dataclass(frozen=True)
class ActionContract:
    names: tuple[str, ...]
    units: tuple[str, ...]
    modes: tuple[str, ...]
    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]

    def __post_init__(self):
        for name in ("names", "units", "modes", "lower", "upper"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.names)
        if (
            not n
            or len(set(self.names)) != n
            or any(not name for name in self.names)
            or any(len(getattr(self, key)) != n for key in ("units", "modes", "lower", "upper"))
        ):
            raise ValueError("Invalid action ordering or native bounds")
        for lower, upper in zip(self.lower, self.upper, strict=True):
            if lower is None and upper is None:
                continue
            if (
                lower is None
                or upper is None
                or not np.isfinite([lower, upper]).all()
                or lower >= upper
            ):
                raise ValueError(
                    "Native bounds must be finite ordered pairs or explicit unlimited pairs"
                )
        for mode, unit in zip(self.modes, self.units, strict=True):
            if (mode, unit) not in {
                ("position", "rad"),
                ("position", "m"),
                ("velocity", "rad/s"),
                ("velocity", "m/s"),
            }:
                raise ValueError("Unsupported native actuator mode/units")

    def validate(self, value):
        result = _array(value, (len(self.names),), "action")
        for value, lower, upper in zip(result, self.lower, self.upper, strict=True):
            if lower is not None and not lower <= value <= upper:
                raise ValueError("Action exceeds declared native bounds")
        return result


@dataclass(frozen=True)
class ObservationField:
    name: str
    shape: tuple[int, ...]
    units: str
    frame: str
    provenance: str
    max_age_s: float = 0.0
    dtype: str = "float32"

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        allowed = {
            "observation.state",
            "observation.wrist_wrench",
            "observation.contact_wrench",
            "observation.tactile",
        }
        image = self.name.startswith("observation.images.")
        if (
            self.name not in allowed
            and not image
            or not self.shape
            or any(type(n) is not int or n <= 0 for n in self.shape)
            or not all((self.units, self.frame, self.provenance))
            or not np.isfinite(self.max_age_s)
            or self.max_age_s < 0
        ):
            raise ValueError("Observation must be a declared sensor/proprioception field")
        if self.dtype not in {"float32", "float64", "uint8"} or (
            image and (self.dtype != "uint8" or len(self.shape) != 3 or self.shape[-1] != 3)
        ):
            raise ValueError("Images require HWC uint8 RGB; other sensors require float data")
        if not image and self.dtype == "uint8":
            raise ValueError("Numeric sensors require floating-point data")


def _finite_cast(value, dtype, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value).astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite values representable as {dtype}")
    return result


def _field_value(value, field):
    if field.dtype == "uint8":
        result = np.asarray(value)
        if result.dtype != np.uint8 or result.shape != field.shape:
            raise ValueError("Raw camera observations must be HWC uint8 RGB")
        return result.copy()
    return _finite_cast(_array(value, field.shape, field.name), field.dtype, field.name)


@dataclass(frozen=True)
class SensorSample:
    value: np.ndarray
    sample_time_s: float
    available: bool = True


@dataclass(frozen=True)
class Snapshot:
    time_s: float
    sensors: dict[str, SensorSample]


def _snapshot(value, fields):
    if not isinstance(value, Snapshot) or not np.isfinite(value.time_s) or value.time_s < 0:
        raise ValueError("Snapshot requires a finite nonnegative physics timestamp")
    if set(value.sensors) - {f.name for f in fields}:
        raise ValueError("Undeclared observation field; privileged state is not a student sensor")
    samples = {}
    for field in fields:
        sample = value.sensors.get(field.name)
        if sample is None or sample.available is False:
            if field.name == "observation.state":
                raise ValueError("Measured pre-action proprioception is required")
            samples[field.name] = SensorSample(
                np.zeros(field.shape, dtype=field.dtype), -1.0, False
            )
            continue
        if type(sample.available) is not bool:
            raise ValueError("Sensor availability must be boolean")
        age = value.time_s - sample.sample_time_s
        if not np.isfinite(sample.sample_time_s) or age < -1e-9 or age > field.max_age_s + 1e-9:
            raise ValueError("Sensor timestamp is future, stale or nonfinite")
        samples[field.name] = SensorSample(
            _field_value(sample.value, field), float(sample.sample_time_s), True
        )
    return Snapshot(float(value.time_s), samples)


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class EpisodeRecorder:
    """Explicit pre-action / nominal / applied / next-observation transaction.

    It observes only; the caller applies actuator targets and steps physics.
    On a process crash, the initial unknown-status manifest is retained. A
    caller-handled exception can finalize partial completed transitions as an
    unknown/failure episode, with an incomplete action retained in metadata.
    """

    def __init__(
        self,
        directory,
        action,
        fields,
        *,
        fps,
        physics_dt_s,
        provenance,
        task,
        purpose="demonstration",
    ):
        self.directory = Path(directory)
        self.action = action
        self.fields = tuple(fields)
        if (
            type(fps) is not int
            or not 1 <= fps <= 500
            or not np.isfinite(physics_dt_s)
            or physics_dt_s <= 0
            or abs(1 / fps / physics_dt_s - round(1 / fps / physics_dt_s)) > 1e-8
            or len({f.name for f in fields}) != len(fields)
            or not any(f.name == "observation.state" for f in fields)
            or purpose not in {"demonstration", "pipeline_fixture", "replay_diagnostic"}
            or not task
        ):
            raise ValueError("Invalid control clock, observation contract or collection purpose")
        required = {"model_revision", "controller_source", "controller_settings", "scene_seed"}
        if not isinstance(provenance, dict) or not required <= provenance.keys():
            raise ValueError("Record model, controller source/settings and scene seed provenance")
        self.provenance = _json(provenance)
        self.fps, self.physics_dt_s = fps, float(physics_dt_s)
        self.task, self.purpose = task, purpose
        self.snapshots, self.nominal, self.applied, self.teachers = [], [], [], []
        self.pending, self.closed = None, False
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(
            schema="abstract_stack_lfd",
            schema_version=1,
            action=asdict(action),
            fields=[asdict(f) for f in fields],
            fps=fps,
            physics_dt_s=physics_dt_s,
            task=task,
            purpose=purpose,
            provenance=self.provenance,
            action_semantics="absolute native references BEFORE compensation at observation time",
            applied_semantics=("native compensated servo targets held during the control interval"),
            observation_semantics=("pre-action; next observation belongs to the next boundary"),
            outcome=dict(status="unknown", task_success=None, reason="not finalized"),
            training_eligible=False,
            fresh_task_credit=0,
            complete=False,
            teacher_semantics="row i is privileged context at pre-action observation_time_s[i]",
        )
        _save_json(self.directory / "episode.json", self.metadata)

    def begin(self, observation, nominal_action, *, teacher=None):
        if self.closed or self.pending is not None:
            raise ValueError("Recorder is closed or has an unfinished action")
        observation = _snapshot(observation, self.fields)
        if self.snapshots:
            previous = self.snapshots[-1]
            if abs(previous.time_s - observation.time_s) > 1e-9 or any(
                a.available != b.available
                or a.sample_time_s != b.sample_time_s
                or not np.array_equal(a.value, b.value)
                for a, b in zip(
                    previous.sensors.values(), observation.sensors.values(), strict=True
                )
            ):
                raise ValueError("Pre-action snapshot differs from the preceding next observation")
        self.pending = (observation, self.action.validate(nominal_action), _json(teacher or {}))

    def end(self, next_observation, applied_action):
        if self.closed or self.pending is None:
            raise ValueError("No pre-action decision has been recorded")
        observation, nominal, teacher = self.pending
        following = _snapshot(next_observation, self.fields)
        if abs(following.time_s - observation.time_s - 1 / self.fps) > 1e-9:
            raise ValueError("Next observation must follow exactly one declared control period")
        applied = self.action.validate(applied_action)
        if not self.snapshots:
            self.snapshots.append(observation)
        self.snapshots.append(following)
        self.nominal.append(nominal)
        self.applied.append(applied)
        self.teachers.append(teacher)
        self.pending = None

    def finish(self, *, status, task_success, reason):
        if self.closed or status not in {"success", "failure", "unknown"}:
            raise ValueError("Invalid or repeated episode finalization")
        if (
            (status == "unknown" and task_success is not None)
            or (status == "success" and task_success is not True)
            or (status == "failure" and task_success is not False)
            or not reason
        ):
            raise ValueError("Outcome status and success must agree explicitly")
        if self.pending is not None and status != "unknown":
            raise ValueError("An unfinished action requires unknown outcome")
        n = len(self.nominal)
        arrays = dict(
            observation_time_s=np.array([s.time_s for s in self.snapshots]),
            **{
                "action.nominal": np.asarray(self.nominal).reshape(n, len(self.action.names)),
                "action.applied": np.asarray(self.applied).reshape(n, len(self.action.names)),
            },
        )
        for field in self.fields:
            values = [s.sensors[field.name] for s in self.snapshots]
            arrays[field.name] = np.asarray([s.value for s in values]).reshape(
                len(values), *field.shape
            )
            arrays[field.name + ".available"] = np.array([s.available for s in values], bool)
            arrays[field.name + ".sample_time_s"] = np.array([s.sample_time_s for s in values])
        np.savez_compressed(self.directory / "frames.npz", **arrays)
        teacher_path = self.directory / "teacher.jsonl"
        teacher_path.write_text(
            "".join(json.dumps(v, allow_nan=False) + "\n" for v in self.teachers)
        )
        self.metadata.update(
            outcome=dict(status=status, task_success=task_success, reason=reason),
            complete=self.pending is None,
            transitions=n,
            fresh_task_credit=0,
            training_eligible=status == "success" and self.purpose == "demonstration",
            sha256={
                name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                for name in ("frames.npz", "teacher.jsonl")
            },
            incomplete_action=None
            if self.pending is None
            else dict(
                time_s=self.pending[0].time_s,
                nominal=self.pending[1].tolist(),
                teacher=self.pending[2],
            ),
        )
        _save_json(self.directory / "episode.json", self.metadata)
        self.closed = True
        return dict(self.metadata)


@dataclass
class LoadedEpisode:
    directory: Path
    metadata: dict
    arrays: dict

    def student_frames(self):
        """Yield only declared observations; teacher/evaluation fields never enter."""
        for index in range(self.metadata["transitions"]):
            frame = {}
            for field in self.metadata["fields"]:
                name = field["name"]
                frame[name] = self.arrays[name][index].copy()
                frame[name + ".available"] = np.array(
                    [self.arrays[name + ".available"][index]], np.float32
                )
                frame[name + ".sample_time_s"] = np.array(
                    [self.arrays[name + ".sample_time_s"][index]], np.float64
                )
            yield frame


def load_episode(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "episode.json").read_text())
    if metadata.get("schema") != "abstract_stack_lfd" or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported demonstration schema")
    for name in ("frames.npz", "teacher.jsonl"):
        expected = metadata.get("sha256", {}).get(name)
        if (
            expected is None
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Incomplete or changed demonstration payload")
    action = ActionContract(**metadata["action"])
    fields = tuple(ObservationField(**f) for f in metadata["fields"])
    outcome = metadata.get("outcome", {})
    status = outcome.get("status")
    success = outcome.get("task_success")
    if (
        status not in {"success", "failure", "unknown"}
        or not outcome.get("reason")
        or status == "success"
        and success is not True
        or status == "failure"
        and success is not False
        or status == "unknown"
        and success is not None
        or type(metadata.get("complete")) is not bool
        or not metadata["complete"]
        and status != "unknown"
        or metadata.get("purpose") not in {"demonstration", "pipeline_fixture", "replay_diagnostic"}
        or type(metadata.get("fps")) is not int
        or metadata["fps"] < 1
        or not any(f.name == "observation.state" for f in fields)
        or len({f.name for f in fields}) != len(fields)
    ):
        raise ValueError("Contradictory episode outcome or observation contract")
    physics_dt = metadata.get("physics_dt_s")
    if (
        not isinstance(physics_dt, (int, float))
        or not np.isfinite(physics_dt)
        or physics_dt <= 0
        or abs(1 / metadata["fps"] / physics_dt - round(1 / metadata["fps"] / physics_dt)) > 1e-8
        or metadata.get("fresh_task_credit") != 0
        or type(metadata.get("training_eligible")) is not bool
        or metadata["training_eligible"]
        != (status == "success" and metadata["purpose"] == "demonstration")
    ):
        raise ValueError("Invalid simulation clock or training eligibility")
    _json(metadata)
    with np.load(directory / "frames.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    n = metadata["transitions"]
    if type(n) is not int or n < 1:
        raise ValueError("An exportable episode needs at least one complete transition")
    teachers = (directory / "teacher.jsonl").read_text().splitlines()
    if len(teachers) != n or any(not isinstance(_json(json.loads(row)), dict) for row in teachers):
        raise ValueError("Privileged context rows do not align with control transitions")
    expected = {"observation_time_s", "action.nominal", "action.applied"}
    for field in fields:
        expected.update((field.name, field.name + ".available", field.name + ".sample_time_s"))
    if set(arrays) != expected:
        raise ValueError("Unexpected or missing demonstration tensor")
    times = _array(arrays["observation_time_s"], (n + 1,), "observation times")
    if times[0] < 0 or not np.allclose(np.diff(times), 1 / metadata["fps"], atol=1e-9, rtol=0):
        raise ValueError("Noncausal or irregular demonstration timestamps")
    for key in ("action.nominal", "action.applied"):
        values = _array(arrays[key], (n, len(action.names)), key)
        for value in values:
            action.validate(value)
    for field in fields:
        if arrays[field.name].shape != (n + 1, *field.shape) or arrays[
            field.name
        ].dtype != np.dtype(field.dtype):
            raise ValueError("Malformed observation tensor")
        mask = arrays[field.name + ".available"]
        stamps = arrays[field.name + ".sample_time_s"]
        if mask.shape != (n + 1,) or mask.dtype != np.dtype(bool) or stamps.shape != (n + 1,):
            raise ValueError("Malformed observation masks/timestamps")
    for i, time_s in enumerate(times):
        samples = {}
        for field in fields:
            value = arrays[field.name]
            mask = arrays[field.name + ".available"]
            stamps = arrays[field.name + ".sample_time_s"]
            samples[field.name] = SensorSample(value[i], float(stamps[i]), bool(mask[i]))
            if not mask[i] and (stamps[i] != -1 or np.any(value[i])):
                raise ValueError("Unavailable channels must use explicit zero/-1 sentinels")
        _snapshot(Snapshot(float(time_s), samples), fields)
    return LoadedEpisode(directory, metadata, arrays)


def student_observation(snapshot, fields):
    """A policy sees sensor values, availability and sampling times only."""
    validated = _snapshot(snapshot, fields)
    result = {}
    for name, sample in validated.sensors.items():
        result[name] = sample.value.copy()
        result[name + ".available"] = np.array([sample.available], np.float32)
        result[name + ".sample_time_s"] = np.array([sample.sample_time_s], np.float64)
    return result


def teacher_observation(snapshot, fields, privileged):
    return dict(student=student_observation(snapshot, fields), privileged=_json(privileged))
