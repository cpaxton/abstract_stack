#!/usr/bin/env python3
"""Summarize native Quest JSONL observations without asserting tracking accuracy."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

FORMAT = "quest_tracking_probe_v1"
MAX_LINE_BYTES = 2 * 1024 * 1024

# Pinned Meta XR Core 209 OVRPlugin.BoneId/SkeletonType. ParentBoneIndex is
# an array index (short), with BoneId.Invalid == -1 for the sole root.
HAND_BONE_NAMES = tuple(
    "Hand_" + name
    for name in (
        "WristRoot ForearmStub Thumb0 Thumb1 Thumb2 Thumb3 Index1 Index2 Index3 "
        "Middle1 Middle2 Middle3 Ring1 Ring2 Ring3 Pinky0 Pinky1 Pinky2 Pinky3 "
        "ThumbTip IndexTip MiddleTip RingTip PinkyTip"
    ).split()
)
BODY_BONE_NAMES = tuple(
    "Body_" + name
    for name in (
        "Root Hips SpineLower SpineMiddle SpineUpper Chest Neck Head LeftShoulder "
        "LeftScapula LeftArmUpper LeftArmLower LeftHandWristTwist RightShoulder "
        "RightScapula RightArmUpper RightArmLower RightHandWristTwist"
    ).split()
) + tuple(
    f"Body_{side}Hand{name}"
    for side in ("Left", "Right")
    for name in (
        "Palm Wrist ThumbMetacarpal ThumbProximal ThumbDistal ThumbTip "
        "IndexMetacarpal IndexProximal IndexIntermediate IndexDistal IndexTip "
        "MiddleMetacarpal MiddleProximal MiddleIntermediate MiddleDistal MiddleTip "
        "RingMetacarpal RingProximal RingIntermediate RingDistal RingTip "
        "LittleMetacarpal LittleProximal LittleIntermediate LittleDistal LittleTip"
    ).split()
)
SKELETON_PROFILES = {
    "left_hand": ("HandLeft", HAND_BONE_NAMES),
    "right_hand": ("HandRight", HAND_BONE_NAMES),
    "body": ("Body", BODY_BONE_NAMES),
}


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or abs(value) > 1e300
        or not math.isfinite(value)
    ):
        raise ValueError("expected a finite number")
    return value


def integer(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("expected a nonnegative integer")
    return value


def boolean(value):
    if not isinstance(value, bool):
        raise ValueError("expected a boolean")
    return value


def vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"expected an array of length {length}")
    return [number(v) for v in value]


def pose(value):
    return vector(value["position_m"], 3), vector(value["quaternion_xyzw"], 4)


def unit_quaternion(rotation):
    norm = math.sqrt(sum(float(v) * float(v) for v in rotation))
    if abs(norm - 1) > 0.01:
        raise ValueError("usable orientation is not a unit quaternion")
    return norm


def validate_skeleton(record):
    """Validate a complete supported bind profile, not just its reported count.

    This verifies structural consistency, not anatomical accuracy or FK. The raw
    format has no per-bind-component availability flags, so an unusable bind
    transform must fail rather than being silently treated as reconstruction data.
    """
    side = record["side"]
    if side not in SKELETON_PROFILES:
        raise ValueError("unsupported skeleton side")
    profile, expected = SKELETON_PROFILES[side]
    if record["profile"] != profile:
        raise ValueError("skeleton profile does not agree with side/declared OVR24 or Body profile")
    bones = record["bones"]
    if not isinstance(bones, list) or len(bones) != len(expected):
        raise ValueError("incomplete skeleton profile")
    names, parents = {}, []
    for index, (bone, expected_name) in enumerate(zip(bones, expected, strict=True)):
        bid = integer(bone["id"])
        if bid != index or bone["name"] != expected_name:
            raise ValueError("skeleton bone ID/name/order does not match SDK profile")
        names[bid] = bone["name"]
        _, rotation = pose(bone["bind_pose"])
        unit_quaternion(rotation)
        parent = bone["parent_index"]
        if type(parent) is not int:
            raise ValueError("skeleton parent index must be an integer")
        if index == 0:
            if parent != -1:
                raise ValueError("SDK skeleton root must use parent sentinel -1")
        elif not 0 <= parent < len(bones):
            raise ValueError("non-root skeleton parent index is outside the profile")
        parents.append(parent)
    rooted = {0}
    for index in range(1, len(bones)):
        path, current = set(), index
        while current not in rooted:
            if current in path:
                raise ValueError("skeleton parent hierarchy contains a cycle")
            path.add(current)
            current = parents[current]
        rooted.update(path)
    return side, names


class Motion:
    def __init__(self):
        self.low = None
        self.high = None
        self.first_rotation = None
        self.angle = 0.0
        self.count = 0

    def add(self, position, rotation):
        if position is not None:
            self.low = position.copy() if self.low is None else list(map(min, self.low, position))
            self.high = (
                position.copy() if self.high is None else list(map(max, self.high, position))
            )
        if rotation is not None:
            norm = unit_quaternion(rotation)
            rotation = [v / norm for v in rotation]
            if self.first_rotation is None:
                self.first_rotation = rotation
            dot = abs(sum(a * b for a, b in zip(self.first_rotation, rotation, strict=True)))
            self.angle = max(self.angle, 2 * math.acos(min(1.0, dot)))
        self.count += 1

    def summary(self):
        return {
            "observations": self.count,
            "axis_range_m": None
            if self.low is None
            else [hi - lo for hi, lo in zip(self.high, self.low, strict=True)],
            "max_rotation_from_first_rad": self.angle,
        }


def summarize(path: Path):
    counts = Counter()
    events = []
    metadata = {}
    motion = {}
    timestamps = {}
    qualities = {side: Counter() for side in ("left_hand", "right_hand", "body")}
    session = None
    sequence = -1
    first_time = last_time = None
    previous_time = -1.0
    dropped = 0
    ended = False
    terminal_detail = None

    def timestamp(source, value, usable):
        number(value)
        if not usable or value <= 0:
            counts[source + "_time_unavailable"] += 1
            return False
        prior = timestamps.get(source)
        kind = (
            "first" if prior is None else "advanced" if value > prior else "repeated_or_regressed"
        )
        counts[source + "_time_" + kind] += 1
        timestamps[source] = value if prior is None else max(prior, value)
        return kind in ("first", "advanced")

    def measure(name, p, position_valid=True, orientation_valid=True):
        position, rotation = pose(p)
        key = f"origin_{record['recenter_count']}/{name}"
        if position_valid or orientation_valid:
            motion.setdefault(key, Motion()).add(
                position if position_valid else None, rotation if orientation_valid else None
            )

    with path.open("rb") as capture:
        line_no = 0
        while line := capture.readline(MAX_LINE_BYTES + 1):
            line_no += 1
            try:
                if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                    raise ValueError("oversized or truncated JSONL record")
                record = json.loads(
                    line, parse_constant=reject_constant, object_pairs_hook=unique_object
                )
                if record["format"] != FORMAT:
                    raise ValueError("unsupported capture format")
                kind = record["type"]
                if session is None:
                    if kind != "session" or record["schema_version"] != 1:
                        raise ValueError("first record must be a v1 session header")
                    if record["pose_frame"] != "ovr_tracking_space_right_handed":
                        raise ValueError("unsupported raw pose frame")
                    if record["quaternion_order"] != "xyzw" or record["position_unit"] != "m":
                        raise ValueError("unsupported pose units/order")
                    session = record
                    if not isinstance(session["session_id"], str) or not session["session_id"]:
                        raise ValueError("missing session identity")
                    for key in (
                        "unity_version",
                        "meta_core",
                        "openxr",
                        "ovrplugin_version",
                        "runtime_version",
                    ):
                        if not isinstance(session[key], str) or not session[key]:
                            raise ValueError("missing version metadata")
                elif record["session_id"] != session["session_id"] or kind == "session":
                    raise ValueError("mixed sessions or duplicate session header")
                counts[kind] += 1
                if kind == "session":
                    continue
                observed = number(record["capture_monotonic_s"])
                if observed < previous_time:
                    raise ValueError("capture monotonic timestamp regressed")
                previous_time = observed
                if kind == "event":
                    events.append(
                        {key: record[key] for key in ("capture_monotonic_s", "name", "detail")}
                    )
                    ended |= record["name"] in ("recording_stopped", "recorder_destroyed")
                    if record["name"] == "recording_stopped":
                        terminal_detail = record["detail"]
                        if "; dropped_records=" in terminal_detail:
                            dropped = max(dropped, int(terminal_detail.rsplit("=", 1)[1]))
                    continue
                if kind == "skeleton":
                    side, names = validate_skeleton(record)
                    metadata[side] = names
                    counts[side + "_verified_skeleton_records"] += 1
                    continue
                if kind != "sample":
                    raise ValueError("unknown record type")
                if integer(record["sequence"]) <= sequence:
                    raise ValueError("sample sequence did not advance")
                sequence = record["sequence"]
                integer(record["recenter_count"])
                first_time = observed if first_time is None else first_time
                last_time = observed
                dropped = max(dropped, integer(record["dropped_records"]))
                counts["worn_observations"] += boolean(record["user_present"])
                counts["focused_observations"] += boolean(record["focused"])
                head = record["head"]
                hv = boolean(head["position_valid"])
                ho = boolean(head["orientation_valid"])
                available = number(head["pose_target_time_s"]) > 0
                hv, ho = hv and available, ho and available
                counts["head_valid_observations"] += hv and ho
                measure("head", head["pose"], hv, ho)
                for side in ("left_hand", "right_hand"):
                    hand = record[side]
                    if hand["skeleton_profile"] != "ovr_hand_24":
                        raise ValueError("unsupported hand profile; expected declared OVR24")
                    usable = boolean(hand["api_success"]) and boolean(hand["tracked"])
                    qualities[side][hand["confidence"]] += 1
                    counts[side + "_tracked_observations"] += usable
                    # Recompute freshness from timestamps; never trust a logged label alone.
                    new_sample = timestamp(side, hand["sample_time_s"], usable)
                    number(hand["requested_time_s"])
                    ids, names = set(), set()
                    for bone in hand["bones"]:
                        bid = integer(bone["id"])
                        if bid in ids or bone["name"] in names:
                            raise ValueError("duplicate hand bone")
                        ids.add(bid)
                        names.add(bone["name"])
                        if side in metadata and metadata[side].get(bid) != bone["name"]:
                            raise ValueError("hand bone does not match skeleton metadata")
                        rotation = vector(bone["rotation_xyzw"], 4)
                        if new_sample:
                            # Legacy hand data lacks per-bone validity; tips may have
                            # absent orientations even while the hand is tracked.
                            norm = math.sqrt(sum(v * v for v in rotation))
                            if abs(norm - 1) > 0.01:
                                counts[side + "_unusable_local_rotations"] += 1
                                continue
                            key = f"origin_{record['recenter_count']}/{side}/{bone['name']}/local"
                            motion.setdefault(key, Motion()).add(None, rotation)
                    if new_sample:
                        measure(side + "/root", hand["root_pose"])
                body = record["body"]
                bg = boolean(body["api_success"])
                counts["body_api_observations"] += bg
                counts["body_permission_observations"] += boolean(body["permission_granted"])
                timestamp("body_state", body["state_time_s"], bg)
                confidence = number(body["confidence"])
                if not 0 <= confidence <= 1:
                    raise ValueError("body confidence outside [0,1]")
                qualities["body"][body["fidelity"]] += 1
                ids, names = set(), set()
                for joint in body["joints"]:
                    jid = integer(joint["id"])
                    if jid in ids or joint["name"] in names:
                        raise ValueError("duplicate body joint")
                    ids.add(jid)
                    names.add(joint["name"])
                    if "body" in metadata and metadata["body"].get(jid) != joint["name"]:
                        raise ValueError("body joint does not match skeleton metadata")
                    valid = boolean(joint["position_valid"])
                    oriented = boolean(joint["orientation_valid"])
                    pt = boolean(joint["position_tracked"])
                    ot = boolean(joint["orientation_tracked"])
                    counts["body_usable_position_tracked_flags"] += bg and valid and pt
                    counts["body_usable_orientation_tracked_flags"] += bg and oriented and ot
                    if bg:
                        measure("body/" + joint["name"], joint["pose"], valid, oriented)
            except (KeyError, TypeError, ValueError, UnicodeError) as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
    if session is None or not counts["sample"]:
        raise ValueError("capture contains no samples")
    duration = last_time - first_time
    return {
        "session": session,
        "counts": dict(counts),
        "duration_s": duration,
        "sample_poll_hz": (counts["sample"] - 1) / duration if duration > 0 else None,
        "hand_distinct_sample_hz": {
            side: (counts[side + "_time_advanced"] / duration if duration > 0 else None)
            for side in ("left_hand", "right_hand")
        },
        "max_reported_dropped_records": dropped,
        "observed_end_event": ended,
        "terminal_detail": terminal_detail,
        "skeleton_metadata_bone_counts": {side: len(names) for side, names in metadata.items()},
        "skeleton_metadata_validation": {
            side: {
                "status": "verified" if side in metadata else "missing",
                "profile": profile if side in metadata else None,
                "bone_count": len(metadata.get(side, {})),
                "records_verified": counts[side + "_verified_skeleton_records"],
            }
            for side, (profile, _) in SKELETON_PROFILES.items()
        },
        "quality_observation_counts": {key: dict(value) for key, value in qualities.items()},
        "motion_by_origin": {key: value.summary() for key, value in sorted(motion.items())},
        "events": events,
        "limitations": [
            "Motion ranges do not establish spatial accuracy or operator acceptance.",
            "Head pose-target and body state timestamps do not prove sensor freshness.",
            "Hand bone rotations are SDK local rotations, not tracking-space joint poses.",
            "Tracked body flags do not establish direct optical measurement of each joint.",
            "Missing skeleton metadata or end events require inspection before acceptance.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = json.dumps(summarize(args.capture), indent=2, allow_nan=False) + "\n"
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
