#!/usr/bin/env python3
"""Replay native GLOBAL root poses from a local T02 capture; no robot commands.

Personal raw trajectories and neutral artifacts are never emitted. The aggregate
checks handedness/finite conversion and exact neutral identity, not named-motion
semantics or human/robot morphology. Hand-local bones are deliberately excluded.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from abstract_stack.teleop.calibration import Calibration, Pose, TargetSpec, ovr_to_operator


def decode_sample(record):
    result = {}
    head = record["head"]
    if head["position_valid"] and head["orientation_valid"]:
        pose = head["pose"]
        result["head"] = ovr_to_operator(pose["position_m"], pose["quaternion_xyzw"])
    for side in ("left_hand", "right_hand"):
        source = record[side]
        if source["api_success"] and source["tracked"]:
            if source["skeleton_profile"] != "ovr_hand_24":
                raise ValueError("unsupported raw hand profile")
            pose = source["root_pose"]
            result[side] = ovr_to_operator(pose["position_m"], pose["quaternion_xyzw"])
    return result


def run(path):
    counts = {
        "samples": 0,
        "complete_roots": 0,
        "missing_required": 0,
        "conversions": 0,
        "calibrations": 0,
        "reference_changes": 0,
        "head_valid": 0,
        "left_hand_valid": 0,
        "right_hand_valid": 0,
    }
    times = []
    errors = []
    max_neutral_position_error_m = 0.0
    max_neutral_rotation_matrix_error = 0.0
    manager = None
    recenter = None
    session = None
    source_version = None
    for line_number, line in enumerate(path.open(), 1):
        record = json.loads(line)
        if record["format"] != "quest_tracking_probe_v1":
            raise ValueError(f"line {line_number}: unknown raw profile")
        if record["type"] == "session":
            if record["pose_frame"] != "ovr_tracking_space_right_handed":
                raise ValueError("unexpected native pose frame")
            if record["quaternion_order"] != "xyzw" or record["position_unit"] != "m":
                raise ValueError("unexpected pose conventions")
            session = record["session_id"]
            source_version = record["application_version"]
            manager = Calibration(session)
        if record["type"] != "sample":
            continue
        if manager is None or record["session_id"] != session:
            raise ValueError("sample without matching session")
        counts["samples"] += 1
        times.append(record["capture_monotonic_s"])
        if recenter is not None and recenter != record["recenter_count"]:
            manager.invalidate("recorded reference_space_changed")
            counts["reference_changes"] += 1
        recenter = record["recenter_count"]
        try:
            current = decode_sample(record)
            counts["conversions"] += len(current)
            for source in current:
                counts[f"{source}_valid"] += 1
            if set(current) != {"head", "left_hand", "right_hand"}:
                counts["missing_required"] += 1
                continue
            counts["complete_roots"] += 1
            if manager.artifact is None:
                # Explicit numerical reference, NOT a measured MABEL posture.
                numerical = {
                    name: TargetSpec(Pose((0, 0, 0), (1, 0, 0, 0)), max_translation_m=10)
                    for name in current
                }
                artifact = manager.capture(
                    current, numerical, ready_or_paused=True, inputs_usable=True
                )
                counts["calibrations"] += 1
                for target in artifact.targets(current).values():
                    max_neutral_position_error_m = max(
                        max_neutral_position_error_m, np.linalg.norm(target.position_m)
                    )
                    max_neutral_rotation_matrix_error = max(
                        max_neutral_rotation_matrix_error,
                        np.max(np.abs(target.rotation - np.eye(3))),
                    )
            # Validate all converted motion remains finite under relative mapping.
            manager.artifact.targets(current)
        except (ValueError, KeyError, TypeError) as error:
            errors.append({"line": line_number, "reason": str(error)})
    if not counts["conversions"]:
        raise ValueError("no valid global root observations")
    return {
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_app_version": source_version,
        "counts": counts,
        "duration_s": max(times) - min(times),
        "complete_neutral_verified": counts["calibrations"] > 0,
        "max_neutral_position_error_m": (
            float(max_neutral_position_error_m) if counts["calibrations"] else None
        ),
        "max_neutral_rotation_matrix_error": (
            float(max_neutral_rotation_matrix_error) if counts["calibrations"] else None
        ),
        "errors": errors,
        "robot_commands": 0,
        "limitations": [
            "Global head/wrist roots only; legacy hand bones need SDK FK",
            "No labelled movement directions in raw recording",
            "No observation-age/sensor-age inference from replay",
            "Robot neutral references are numerical fixtures, not measured MABEL",
            "No device-to-robot engagement or command-jump validation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.recording)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
