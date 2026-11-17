"""Recorded tracking evidence must not confuse polling with fresh sensor samples."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_quest_tracking.py"
SPEC = importlib.util.spec_from_file_location("quest_capture_summary", SCRIPT)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def capture_records():
    header = {
        "format": summary.FORMAT,
        "type": "session",
        "schema_version": 1,
        "session_id": "synthetic",
        "pose_frame": "ovr_tracking_space_right_handed",
        "quaternion_order": "xyzw",
        "position_unit": "m",
        **dict.fromkeys(
            ("unity_version", "meta_core", "openxr", "ovrplugin_version", "runtime_version"),
            "fixture",
        ),
    }
    p = {"position_m": [0.0, 1.0, 0.0], "quaternion_xyzw": [0, 0, 0, 1]}
    hand = {
        "skeleton_profile": "ovr_hand_24",
        "api_success": True,
        "tracked": True,
        "confidence": "High",
        "sample_time_s": 50,
        "requested_time_s": 51,
        "root_pose": p,
        "bones": [{"id": 0, "name": "Hand_WristRoot", "rotation_xyzw": [0, 0, 0, 1]}],
        # Deliberately dishonest label: summarizer must recompute freshness.
        "source_freshness": "advanced",
    }
    record = {
        "format": summary.FORMAT,
        "type": "sample",
        "session_id": "synthetic",
        "sequence": 0,
        "capture_monotonic_s": 1,
        "recenter_count": 0,
        "dropped_records": 0,
        "user_present": True,
        "focused": True,
        "head": {
            "pose": p,
            "pose_target_time_s": 51,
            "position_valid": True,
            "orientation_valid": True,
        },
        "left_hand": copy.deepcopy(hand),
        "right_hand": copy.deepcopy(hand),
        "body": {
            "api_success": True,
            "permission_granted": True,
            "state_time_s": 51,
            "confidence": 1,
            "fidelity": "High",
            "joints": [],
        },
    }
    records = [header]
    for i, timestamp in enumerate((50, 50, 49, 50, 51)):
        sample = copy.deepcopy(record)
        sample["sequence"] = i
        sample["capture_monotonic_s"] = i + 1
        sample["head"]["pose_target_time_s"] += i
        sample["body"]["state_time_s"] += i
        sample["left_hand"]["sample_time_s"] = timestamp
        sample["right_hand"]["sample_time_s"] = timestamp
        records.append(sample)
    return records


def write_capture(tmp_path, records):
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def add_skeleton(records, side="left_hand"):
    """Complete named SDK profile with synthetic identity binds/chain topology."""
    profile, names = summary.SKELETON_PROFILES[side]
    skeleton = {
        "format": summary.FORMAT,
        "type": "skeleton",
        "session_id": "synthetic",
        "capture_monotonic_s": 0.5,
        "side": side,
        "profile": profile,
        "bones": [
            {
                "id": index,
                "name": name,
                "parent_index": index - 1,
                "bind_pose": {
                    "position_m": [0, 0, 0],
                    "quaternion_xyzw": [0, 0, 0, 1],
                },
            }
            for index, name in enumerate(names)
        ],
    }
    records.insert(1, skeleton)
    return skeleton


def test_missing_metadata_is_explicitly_unverified(tmp_path):
    result = summary.summarize(write_capture(tmp_path, capture_records()))
    for side in ("left_hand", "right_hand", "body"):
        assert result["skeleton_metadata_validation"][side] == {
            "status": "missing",
            "profile": None,
            "bone_count": 0,
            "records_verified": 0,
        }


def test_complete_profiles_are_verified_independently(tmp_path):
    records = capture_records()
    for side in ("left_hand", "right_hand", "body"):
        add_skeleton(records, side)
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["skeleton_metadata_bone_counts"] == {
        "left_hand": 24,
        "right_hand": 24,
        "body": 70,
    }
    for side, expected_profile in (
        ("left_hand", "HandLeft"),
        ("right_hand", "HandRight"),
        ("body", "Body"),
    ):
        validation = result["skeleton_metadata_validation"][side]
        assert validation["status"] == "verified"
        assert validation["profile"] == expected_profile
        assert validation["records_verified"] == 1


@pytest.mark.parametrize("side", ["left_hand", "right_hand", "body"])
@pytest.mark.parametrize(
    "corruption",
    [
        "profile",
        "side",
        "incomplete",
        "duplicate_id",
        "name",
        "order",
        "root_out_of_range",
        "root_self_cycle",
        "unsigned_root_sentinel",
        "negative_root_sentinel",
        "extra_root",
        "parent_out_of_range",
        "parent_type",
        "fractional_parent",
        "missing_parent",
        "self_cycle",
        "disconnected_cycle",
        "zero_bind",
        "nonunit_bind",
        "extreme_bind",
        "nonfinite_bind",
        "missing_bind",
    ],
)
def test_invalid_bind_metadata_is_rejected(tmp_path, side, corruption):
    records = capture_records()
    skeleton = add_skeleton(records, side)
    bones = skeleton["bones"]
    if corruption == "profile":
        skeleton["profile"] = "XRHandLeft"
    elif corruption == "side":
        skeleton["side"] = "unsupported"
    elif corruption == "incomplete":
        bones.pop()
    elif corruption == "duplicate_id":
        bones[1]["id"] = bones[0]["id"]
    elif corruption == "name":
        bones[1]["name"] = "wrong_sdk_bone"
    elif corruption == "order":
        bones[1], bones[2] = bones[2], bones[1]
    elif corruption == "root_out_of_range":
        bones[0]["parent_index"] = 999
    elif corruption == "root_self_cycle":
        bones[0]["parent_index"] = 0
    elif corruption == "unsigned_root_sentinel":
        bones[0]["parent_index"] = 65535
    elif corruption == "negative_root_sentinel":
        bones[0]["parent_index"] = -2
    elif corruption == "extra_root":
        bones[1]["parent_index"] = -1
    elif corruption == "parent_out_of_range":
        bones[1]["parent_index"] = len(bones)
    elif corruption == "parent_type":
        bones[1]["parent_index"] = True
    elif corruption == "fractional_parent":
        bones[1]["parent_index"] = 0.5
    elif corruption == "missing_parent":
        del bones[1]["parent_index"]
    elif corruption == "self_cycle":
        bones[1]["parent_index"] = 1
    elif corruption == "disconnected_cycle":
        bones[1]["parent_index"], bones[2]["parent_index"] = 2, 1
    elif corruption == "zero_bind":
        bones[1]["bind_pose"]["quaternion_xyzw"] = [0, 0, 0, 0]
    elif corruption == "nonunit_bind":
        bones[1]["bind_pose"]["quaternion_xyzw"] = [0, 0, 0, 2]
    elif corruption == "extreme_bind":
        bones[1]["bind_pose"]["quaternion_xyzw"] = [10**200, 0, 0, 0]
    elif corruption == "nonfinite_bind":
        bones[1]["bind_pose"]["position_m"] = [0, float("nan"), 0]
    elif corruption == "missing_bind":
        del bones[1]["bind_pose"]
    with pytest.raises(ValueError):
        summary.summarize(write_capture(tmp_path, records))


def test_forward_parent_references_and_quaternion_double_cover_are_valid(tmp_path):
    records = capture_records()
    skeleton = add_skeleton(records)
    skeleton["bones"][1]["parent_index"] = 2
    skeleton["bones"][2]["parent_index"] = 0
    skeleton["bones"][2]["bind_pose"]["quaternion_xyzw"] = [0, 0, 0, -1]
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["skeleton_metadata_validation"]["left_hand"]["status"] == "verified"


def test_later_skeleton_revision_cannot_bypass_validation(tmp_path):
    records = capture_records()
    initial = add_skeleton(records, "body")
    changed = copy.deepcopy(initial)
    changed["capture_monotonic_s"] = 6
    changed["bones"][1]["parent_index"] = 999
    records.append(changed)
    with pytest.raises(ValueError, match="parent index"):
        summary.summarize(write_capture(tmp_path, records))


def test_swapped_hand_bind_profile_is_rejected(tmp_path):
    records = capture_records()
    add_skeleton(records)["profile"] = "HandRight"
    with pytest.raises(ValueError, match="profile"):
        summary.summarize(write_capture(tmp_path, records))


def test_sample_dedup_ignores_repeats_regressions_and_untrusted_fresh_labels(tmp_path):
    result = summary.summarize(write_capture(tmp_path, capture_records()))
    assert result["counts"]["sample"] == 5
    assert result["counts"]["left_hand_time_first"] == 1
    assert result["counts"]["left_hand_time_advanced"] == 1
    assert result["counts"]["left_hand_time_repeated_or_regressed"] == 3
    assert result["hand_distinct_sample_hz"]["left_hand"] == 0.25
    assert result["counts"]["body_state_time_advanced"] == 4
    assert "head_time_advanced" not in result["counts"]
    assert result["observed_end_event"] is False


def test_invalid_hand_does_not_refresh_freshness_or_motion(tmp_path):
    records = capture_records()
    for sample in records[1:]:
        sample["left_hand"]["tracked"] = False
        sample["left_hand"]["root_pose"] = None
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["counts"]["left_hand_time_unavailable"] == 5
    assert result["hand_distinct_sample_hz"]["left_hand"] == 0
    assert not any("left_hand" in key for key in result["motion_by_origin"])


def test_motion_is_separated_at_recenter_not_counted_as_operator_translation(tmp_path):
    records = capture_records()
    records[-1]["recenter_count"] = 1
    records[-1]["head"]["pose"]["position_m"] = [100, 1, 0]
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["motion_by_origin"]["origin_0/head"]["axis_range_m"] == [0, 0, 0]
    assert result["motion_by_origin"]["origin_1/head"]["axis_range_m"] == [0, 0, 0]


def test_missing_hand_bone_orientation_is_reported_without_inventing_joint_validity(tmp_path):
    records = capture_records()
    records[-1]["left_hand"]["bones"][0]["rotation_xyzw"] = [0, 0, 0, 0]
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["counts"]["left_hand_unusable_local_rotations"] == 1
    assert (
        result["motion_by_origin"]["origin_0/left_hand/Hand_WristRoot/local"]["observations"] == 1
    )


def test_cached_hand_pose_motion_is_not_counted_as_new_sample_motion(tmp_path):
    records = capture_records()
    records[2]["left_hand"]["root_pose"]["position_m"] = [100, 1, 0]
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["motion_by_origin"]["origin_0/left_hand/root"]["axis_range_m"] == [0, 0, 0]


def test_failed_head_pose_read_cannot_use_independent_validity_flags(tmp_path):
    records = capture_records()
    for record in records[1:]:
        record["head"]["pose_target_time_s"] = 0
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["counts"]["head_valid_observations"] == 0
    assert not any(key.endswith("/head") for key in result["motion_by_origin"])


def test_failed_body_api_cannot_count_stale_tracked_joint_flags(tmp_path):
    records = capture_records()
    for record in records[1:]:
        record["body"]["api_success"] = False
        record["body"]["joints"] = [
            {
                "id": 0,
                "name": "Body_Chest",
                "pose": record["head"]["pose"],
                "position_valid": True,
                "orientation_valid": True,
                "position_tracked": True,
                "orientation_tracked": True,
            }
        ]
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["counts"]["body_usable_position_tracked_flags"] == 0
    assert result["counts"]["body_usable_orientation_tracked_flags"] == 0


def test_terminal_event_retains_late_drop_count_and_failure_reason(tmp_path):
    records = capture_records()
    records.append(
        {
            "format": summary.FORMAT,
            "type": "event",
            "session_id": "synthetic",
            "capture_monotonic_s": 6,
            "name": "recording_stopped",
            "detail": "writer_failure; dropped_records=7",
        }
    )
    result = summary.summarize(write_capture(tmp_path, records))
    assert result["max_reported_dropped_records"] == 7
    assert result["terminal_detail"] == "writer_failure; dropped_records=7"


def test_runtime_profile_cannot_silently_change_bone_transform_semantics(tmp_path):
    records = capture_records()
    add_skeleton(records)
    records[-1]["left_hand"]["skeleton_profile"] = "openxr_hand_26"
    with pytest.raises(ValueError, match="unsupported hand profile"):
        summary.summarize(write_capture(tmp_path, records))


def test_bone_names_must_match_recorded_bind_hierarchy(tmp_path):
    records = capture_records()
    add_skeleton(records)
    records[-1]["left_hand"]["bones"][0]["name"] = "XRHand_Palm"
    with pytest.raises(ValueError, match="does not match skeleton metadata"):
        summary.summarize(write_capture(tmp_path, records))


@pytest.mark.parametrize("corruption", ["session", "sequence", "missing", "nan", "quaternion"])
def test_incomplete_or_corrupt_capture_is_rejected(tmp_path, corruption):
    records = capture_records()
    if corruption == "session":
        records[-1]["session_id"] = "other"
    elif corruption == "sequence":
        records[-1]["sequence"] = 0
    elif corruption == "missing":
        del records[-1]["left_hand"]["sample_time_s"]
    elif corruption == "nan":
        records[-1]["capture_monotonic_s"] = float("nan")
    else:
        records[-1]["head"]["pose"]["quaternion_xyzw"] = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        summary.summarize(write_capture(tmp_path, records))


def test_truncated_record_is_not_silently_accepted(tmp_path):
    path = write_capture(tmp_path, capture_records())
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated"):
        summary.summarize(path)
