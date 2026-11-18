"""Small native-reference sensing fixtures; no manipulation-success claim."""

import hashlib
import json
from pathlib import Path

import numpy as np

MOTION_PROFILES = {
    "hold": {},
    "lift_pan": {"Lift": 0.004, "Neck Yaw": 0.08},
    "left_wrist_pan": {"Left Wrist": 0.04, "Neck Yaw": -0.08},
    "right_wrist_tilt": {"Right Wrist": 0.04, "Neck Pitch": 0.06},
}


def fixture_scene_context(*, scene_seed, reset_seed, scene_config=None, domain_context=None):
    """Bind explicit DR inputs to fixture provenance, never to student features.

    Input hashes identify the exact user-selected case. This function does not
    randomize the model, admit a task, or make a pipeline fixture training-ready.
    """
    if any(type(seed) is not int or seed < 0 for seed in (scene_seed, reset_seed)):
        raise ValueError("Scene and reset seeds must be nonnegative integers")
    configuration = dict(kind="dense_shelf_config", scene_seed=scene_seed)
    identity = None
    if scene_config is not None:
        path = Path(scene_config).resolve()
        content = path.read_bytes()
        if len(content) > 1024 * 1024 or not isinstance(json.loads(content), dict):
            raise ValueError("Scene configuration must be a bounded JSON object")
        configuration = dict(
            kind="explicit_scene_config", path=str(path), sha256=hashlib.sha256(content).hexdigest()
        )
    if domain_context is not None:
        if scene_config is None:
            raise ValueError("Domain identity requires its explicit scene configuration")
        path = Path(domain_context).resolve()
        content = path.read_bytes()
        if len(content) > 1024 * 1024:
            raise ValueError("Domain context must be a bounded JSON object")
        case = json.loads(content)
        if (
            not isinstance(case, dict)
            or not {"suite_version", "split", "scene_seed"} <= case.keys()
            or not (
                isinstance(case["suite_version"], str)
                and case["suite_version"]
                or type(case["suite_version"]) is int
                and case["suite_version"] > 0
            )
            or not isinstance(case["split"], str)
            or not case["split"]
            or type(case["scene_seed"]) is not int
            or case["scene_seed"] != scene_seed
            or "reset_seed" in case
            and case["reset_seed"] != reset_seed
        ):
            raise ValueError("Domain identity must match the declared scene/reset seeds")
        if "scene_config_sha256" in case and case["scene_config_sha256"] != configuration["sha256"]:
            raise ValueError("Domain identity scene configuration hash differs")
        json.dumps(case, allow_nan=False)
        identity = dict(
            path=str(path),
            sha256=hashlib.sha256(content).hexdigest(),
            identity=case,
            scope="privileged collection provenance, excluded from student features",
        )
    return dict(
        scene_configuration_source=configuration,
        domain_context=identity,
        scene_configuration_seed=scene_seed,
        environment_reset_seed=reset_seed,
    )


def motion_reference(baseline, contract, profile, *, index, steps):
    """A bounded half-sine pulse relative to the clean settled nominal hold.

    The final reference returns to baseline. Wrist/lift references still pass
    through the existing controller's rate/offset limits. This is a controller
    fixture, not collision-screened Cartesian IK or a pick/place teacher.
    """
    if profile not in MOTION_PROFILES or not 0 <= index < steps:
        raise ValueError("Unknown fixture profile or invalid transition index")
    command = contract.validate(baseline)
    phase = float(np.sin(np.pi * (index + 1) / steps)) if index < steps - 1 else 0.0
    for name, amplitude in MOTION_PROFILES[profile].items():
        actuator = contract.names.index(name)
        expected_unit = "m" if name == "Lift" else "rad"
        if contract.modes[actuator] != "position" or contract.units[actuator] != expected_unit:
            raise ValueError("Fixture requires the declared native position-reference units")
        command[actuator] += amplitude * phase
    return contract.validate(command)
