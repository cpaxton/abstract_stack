"""Known-axis and lifecycle checks; no commands or physical-success claims."""

import json

import numpy as np
import pytest

from abstract_stack.teleop.calibration import (
    IDENTITY,
    OPERATOR_TO_ROBOT,
    Calibration,
    Pose,
    TargetSpec,
    change_basis,
    ovr_to_operator,
    pose_from_tracking,
    unity_to_operator,
)


def poses():
    return {
        "head": Pose((0.0, 0.0, 1.6), (1.0, 0.0, 0.0, 0.0)),
        "left_hand": Pose((0.3, 0.2, 1.0), (1.0, 0.0, 0.0, 0.0)),
        "right_hand": Pose((0.3, -0.2, 1.0), (1.0, 0.0, 0.0, 0.0)),
    }


def specs():
    return {
        name: TargetSpec(
            Pose((-0.2, -0.1 if name == "left_hand" else 0.1, 1.0), (1.0, 0.0, 0.0, 0.0))
        )
        for name in poses()
    }


def calibrated():
    manager = Calibration("test-session")
    manager.capture(poses(), specs(), ready_or_paused=True, inputs_usable=True)
    return manager


@pytest.mark.parametrize(
    "unity,expected", [((0, 0, 1), (1, 0, 0)), ((-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1))]
)
def test_unity_axis_metres(unity, expected):
    assert unity_to_operator(unity, (0, 0, 0, 1)).position_m == expected


@pytest.mark.parametrize(
    "ovr,expected", [((0, 0, -1), (1, 0, 0)), ((-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1))]
)
def test_native_ovr_sdk_flip_z(ovr, expected):
    assert ovr_to_operator(ovr, (0, 0, 0, 1)).position_m == expected


def test_rotation_reflection_conjugates_active_rotation_and_roundtrip():
    # Unity positive Y rotation takes +Z to +X; operator +X must become -Y.
    q = np.sqrt(0.5)
    converted = unity_to_operator((0.1, 0.2, 0.3), (0, q, 0, q))
    np.testing.assert_allclose(converted.rotation @ [1, 0, 0], [0, -1, 0], atol=1e-15)
    basis = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    restored = change_basis(converted, basis.T)
    np.testing.assert_allclose(restored.position_m, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(restored.rotation @ [0, 0, 1], [1, 0, 0], atol=1e-15)
    native = ovr_to_operator((0.1, 0.2, -0.3), (0, -q, 0, q))
    np.testing.assert_allclose(native.rotation, converted.rotation)


@pytest.mark.parametrize(
    "position,quaternion",
    [
        ((0, 0, np.nan), (1, 0, 0, 0)),
        ((0, 0), (1, 0, 0, 0)),
        ((0, 0, 0), (0, 0, 0, 0)),
        ((0, 0, 0), (2, 0, 0, 0)),
        ((0, 0, 0), (np.inf, 0, 0, 0)),
    ],
)
def test_invalid_pose_rejected(position, quaternion):
    with pytest.raises(ValueError):
        Pose(position, quaternion)


def test_immutability_and_quaternion_double_cover():
    original = [1, 2, 3]
    pose = Pose(original, (-1, 0, 0, 0))
    original[0] = 99
    assert pose.position_m == (1, 2, 3)
    assert pose.quaternion_wxyz == (1, 0, 0, 0)
    with pytest.raises(ValueError):
        pose_from_tracking({"position_valid": True, "orientation_valid": False})


def test_neutral_is_exact_measured_robot_reference_with_rotated_palm_offset():
    offsets = dict(specs())
    q = np.sqrt(0.5)
    offsets["left_hand"] = TargetSpec(
        Pose((-0.2, -0.3, 1.0), (q, 0, q, 0)), wrist_to_palm=Pose((0.05, 0, 0), (q, q, 0, 0))
    )
    manager = Calibration("s")
    artifact = manager.capture(poses(), offsets, ready_or_paused=True, inputs_usable=True)
    for name, target in artifact.targets(poses()).items():
        np.testing.assert_allclose(target.position_m, offsets[name].robot_neutral.position_m)
        np.testing.assert_allclose(
            target.rotation, offsets[name].robot_neutral.rotation, atol=1e-15
        )
    assert manager.engagement_acknowledged is False
    assert json.loads(json.dumps(artifact.to_dict(), allow_nan=False))["generation"] == 1
    with pytest.raises(TypeError):
        artifact.neutral_operator["head"] = IDENTITY


def test_relative_axes_side_identity_scale_and_rotation():
    manager = calibrated()
    artifact = manager.artifact
    current = poses()
    current["left_hand"] = Pose((0.4, 0.3, 1.1), (np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    result = artifact.targets(current)
    np.testing.assert_allclose(
        np.subtract(result["left_hand"].position_m, specs()["left_hand"].robot_neutral.position_m),
        [-0.1, -0.1, 0.1],
    )
    assert result["right_hand"] == specs()["right_hand"].robot_neutral
    np.testing.assert_allclose(result["left_hand"].rotation @ [-1, 0, 0], [0, -1, 0], atol=1e-15)
    configured = specs()
    configured["left_hand"] = TargetSpec(IDENTITY, translation_scale=2)
    manager.capture(poses(), configured, ready_or_paused=True, inputs_usable=True)
    np.testing.assert_allclose(
        manager.artifact.targets(current)["left_hand"].position_m, [-0.2, -0.2, 0.2]
    )
    np.testing.assert_allclose(OPERATOR_TO_ROBOT @ [1, 0, 0], [-1, 0, 0])


def test_palm_translation_rotates_with_wrist():
    configured = specs()
    configured["left_hand"] = TargetSpec(IDENTITY, wrist_to_palm=Pose((0.1, 0, 0), (1, 0, 0, 0)))
    manager = Calibration("s")
    manager.capture(poses(), configured, ready_or_paused=True, inputs_usable=True)
    current = poses()
    current["left_hand"] = Pose(current["left_hand"].position_m, (np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    np.testing.assert_allclose(
        manager.artifact.targets(current)["left_hand"].position_m, [0.1, -0.1, 0], atol=1e-15
    )


def test_optional_body_relative_references_can_be_omitted_without_stale_target():
    neutral = {**poses(), "Body_Hips": Pose((0, 0, 0.9), (1, 0, 0, 0))}
    configured = {**specs(), "Body_Hips": TargetSpec(IDENTITY, translation_scale=0.5)}
    manager = Calibration("s")
    manager.capture(neutral, configured, ready_or_paused=True, inputs_usable=True)
    current = {**poses(), "Body_Hips": Pose((0, 0, 1.1), (1, 0, 0, 0))}
    assert manager.artifact.targets(current)["Body_Hips"].position_m == pytest.approx((0, 0, 0.1))
    assert "Body_Hips" not in manager.artifact.targets(poses())


def test_deliberate_generation_session_engagement_and_invalidation():
    manager = calibrated()
    with pytest.raises(ValueError, match="deliberate"):
        manager.propose(poses(), session_id="test-session", generation=1, inputs_usable=True)
    for session, generation, deliberate in [
        ("old", 1, True),
        ("test-session", 0, True),
        ("test-session", 1, False),
    ]:
        with pytest.raises(ValueError):
            manager.acknowledge_engagement(
                session_id=session, generation=generation, deliberate=deliberate, inputs_usable=True
            )
    manager.acknowledge_engagement(
        session_id="test-session", generation=1, deliberate=True, inputs_usable=True
    )
    assert manager.propose(poses(), session_id="test-session", generation=1, inputs_usable=True)
    for reason in ("recenter", "reset", "reconnect"):
        generation = manager.invalidate(reason)
        assert manager.artifact is None and not manager.engagement_acknowledged
        with pytest.raises(ValueError):
            manager.propose(
                poses(), session_id="test-session", generation=generation - 1, inputs_usable=True
            )
        artifact = manager.capture(poses(), specs(), ready_or_paused=True, inputs_usable=True)
        assert artifact.generation > generation
        assert not manager.engagement_acknowledged


def test_bad_capture_invalidates_and_bounds_loss_require_reengage():
    manager = calibrated()
    with pytest.raises(ValueError):
        manager.capture(poses(), specs(), ready_or_paused=True, inputs_usable=False)
    assert manager.artifact is None
    manager.capture(poses(), specs(), ready_or_paused=True, inputs_usable=True)
    manager.acknowledge_engagement(
        session_id=manager.session_id,
        generation=manager.generation,
        deliberate=True,
        inputs_usable=True,
    )
    current = poses()
    current["left_hand"] = Pose((3, 0, 0), (1, 0, 0, 0))
    with pytest.raises(ValueError, match="bounds"):
        manager.propose(
            current,
            session_id=manager.session_id,
            generation=manager.generation,
            inputs_usable=True,
        )
    assert not manager.engagement_acknowledged
    with pytest.raises(ValueError):
        manager.propose(
            poses(),
            session_id=manager.session_id,
            generation=manager.generation,
            inputs_usable=True,
        )


def test_unavailable_hand_never_returns_partial_arm_proposal():
    manager = calibrated()
    manager.acknowledge_engagement(
        session_id=manager.session_id, generation=1, deliberate=True, inputs_usable=True
    )
    current = poses()
    del current["right_hand"]
    with pytest.raises(ValueError):
        manager.propose(current, session_id=manager.session_id, generation=1, inputs_usable=True)
    assert not manager.engagement_acknowledged


@pytest.mark.parametrize("value", [True, -1, 2**53, 1.2])
def test_generation_rejects_ambiguous_or_exhausted_values(value):
    with pytest.raises(ValueError):
        Calibration("s", value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"translation_scale": 0},
        {"translation_scale": np.nan},
        {"max_translation_m": -1},
        {"max_rotation_rad": 4},
    ],
)
def test_bad_scale_or_bounds(kwargs):
    with pytest.raises(ValueError):
        TargetSpec(IDENTITY, **kwargs)


def test_neutral_heading_alignment_removes_tracking_yaw_without_world_spawn_yaw():
    neutral = poses()
    q = np.sqrt(0.5)
    neutral["head"] = Pose(neutral["head"].position_m, (q, 0, 0, q))
    manager = Calibration("s")
    artifact = manager.capture(neutral, specs(), ready_or_paused=True, inputs_usable=True)
    current = dict(neutral)
    # Head faces operator +Y, so neutral-forward movement maps to native -X.
    current["left_hand"] = Pose((0.3, 0.3, 1.0), (1, 0, 0, 0))
    delta = np.subtract(
        artifact.targets(current)["left_hand"].position_m,
        specs()["left_hand"].robot_neutral.position_m,
    )
    np.testing.assert_allclose(delta, [-0.1, 0, 0], atol=1e-15)


def test_vertical_neutral_head_has_no_heading_and_invalidates_previous_alignment():
    neutral = poses()
    q = np.sqrt(0.5)
    neutral["head"] = Pose((0, 0, 1.6), (q, 0, q, 0))
    manager = calibrated()
    with pytest.raises(ValueError, match="vertical"):
        manager.capture(neutral, specs(), ready_or_paused=True, inputs_usable=True)
    assert manager.artifact is None


def test_replay_reports_zero_simultaneous_neutrals_without_inventing_missing_hand(tmp_path):
    import runpy
    from pathlib import Path

    probe = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "probe_quest_calibration.py")
    )
    pose = {"position_m": [0, 1, -0.3], "quaternion_xyzw": [0, 0, 0, 1]}
    records = [
        {
            "format": "quest_tracking_probe_v1",
            "type": "session",
            "session_id": "s",
            "pose_frame": "ovr_tracking_space_right_handed",
            "quaternion_order": "xyzw",
            "position_unit": "m",
            "application_version": "synthetic",
        }
    ]
    for index in range(2):
        records.append(
            {
                "format": "quest_tracking_probe_v1",
                "type": "sample",
                "session_id": "s",
                "capture_monotonic_s": index,
                "recenter_count": index,
                "head": {"position_valid": True, "orientation_valid": True, "pose": pose},
                "left_hand": {
                    "api_success": True,
                    "tracked": index == 0,
                    "root_pose": pose,
                    "skeleton_profile": "ovr_hand_24",
                },
                "right_hand": {
                    "api_success": True,
                    "tracked": index == 1,
                    "root_pose": pose,
                    "skeleton_profile": "ovr_hand_24",
                },
            }
        )
    path = tmp_path / "synthetic.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    report = probe["run"](path)
    assert report["counts"]["conversions"] == 4
    assert report["complete_neutral_verified"] is False
    assert report["max_neutral_position_error_m"] is None
    assert report["counts"]["reference_changes"] == 1
    assert report["errors"] == []
    records[-1]["left_hand"]["tracked"] = True
    path.write_text("\n".join(json.dumps(r) for r in records))
    report = probe["run"](path)
    assert report["complete_neutral_verified"] is True
    assert report["max_neutral_position_error_m"] == 0
    assert report["max_neutral_rotation_matrix_error"] < 1e-12


def test_input_loss_does_not_auto_resume_and_rotation_bound_rejects():
    manager = calibrated()
    manager.acknowledge_engagement(
        session_id=manager.session_id, generation=1, deliberate=True, inputs_usable=True
    )
    with pytest.raises(ValueError, match="input unavailable"):
        manager.propose(poses(), session_id=manager.session_id, generation=1, inputs_usable=False)
    with pytest.raises(ValueError, match="deliberate"):
        manager.propose(poses(), session_id=manager.session_id, generation=1, inputs_usable=True)
    configured = specs()
    configured["left_hand"] = TargetSpec(IDENTITY, max_rotation_rad=0.1)
    artifact = manager.capture(poses(), configured, ready_or_paused=True, inputs_usable=True)
    current = poses()
    current["left_hand"] = Pose(current["left_hand"].position_m, (np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    with pytest.raises(ValueError, match="bounds"):
        artifact.targets(current)
