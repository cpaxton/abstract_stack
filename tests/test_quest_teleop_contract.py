"""Check the published interface fixtures against axes and pinned native semantics."""

import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import MABEL_REVISION, mabel_model_path
from abstract_stack.env import ShelfStockingEnv

CONTRACT_DIR = Path(__file__).resolve().parents[1] / "docs" / "quest_teleop"


def load(name):
    return json.loads((CONTRACT_DIR / name).read_text())


def test_published_coordinate_basis_preserves_pose_rotation_and_named_directions():
    frames = load("contract.json")["frames"]
    basis = np.asarray(frames["unity_to_operator_basis"])
    neutral = np.asarray(frames["operator_to_robot_native_neutral_basis"])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3))
    assert np.linalg.det(basis) == -1
    np.testing.assert_array_equal(basis @ [0, 0, 1], [1, 0, 0])  # Unity forward
    np.testing.assert_array_equal(basis @ [-1, 0, 0], [0, 1, 0])  # Unity left
    np.testing.assert_array_equal(basis @ [0, 1, 0], [0, 0, 1])  # Unity up
    np.testing.assert_array_equal(neutral @ [1, 0, 0], [-1, 0, 0])  # MABEL forward
    # A reflected basis still conjugates a proper rotation into a proper rotation.
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    converted = basis @ rotation @ basis.T
    assert np.linalg.det(converted) == 1
    point = np.array([0.2, 0.4, 0.8])
    np.testing.assert_allclose(converted @ (basis @ point), basis @ (rotation @ point))


def test_tracking_fixture_does_not_fabricate_head_freshness_or_skeleton_layout():
    contract = load("contract.json")
    fixture = load("examples/tracking.json")
    assert fixture["schema_version"] == contract["schema_version"]
    assert fixture["frame"] == contract["frames"]["operator"]
    assert fixture["head"]["source_time_s"] is None
    assert fixture["head"]["fresh"] is None
    assert fixture["head"]["source_clock"] == "unavailable"
    for name in ("left_hand", "right_hand", "body"):
        stream = fixture[name]
        declared = contract["skeleton_profiles"][stream["skeleton_profile"]]
        names = [joint["name"] for joint in stream["joints"]]
        assert names == declared
        assert len(set(names)) == len(names)
        for joint in stream["joints"]:
            assert len(joint["position_m"]) == 3
            assert np.isfinite(joint["position_m"]).all()
            assert np.linalg.norm(joint["quaternion_wxyz"]) == pytest.approx(1)
    assert len(contract["skeleton_profiles"]["ovr_hand_24"]) == 24
    assert contract["skeleton_profiles"]["ovr_hand_24"][0] == "Hand_WristRoot"
    assert contract["skeleton_profiles"]["ovr_hand_24"][23] == "Hand_PinkyTip"


def native_env():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("pinned MABEL assets unavailable")
    return ShelfStockingEnv(seed=0)


@pytest.mark.mabel
def test_declared_actuator_ownership_matches_every_native_transmission():
    contract = load("contract.json")
    assert contract["baseline"]["mabel_revision"] == MABEL_REVISION
    env = native_env()
    model = env.model
    ownership = contract["actuators"]
    assert set(ownership) == {model.actuator(i).name for i in range(model.nu)}
    assert Counter(a["owner"] for a in ownership.values()) == {
        "stationary_base": 6,
        "body": 2,
        "head": 3,
        "arms": 16,
        "fingers": 32,
    }
    for aid in range(model.nu):
        declared = ownership[model.actuator(aid).name]
        if declared["command"] == "velocity":
            assert model.actuator_biasprm[aid, 1] == 0
            assert model.actuator_biasprm[aid, 2] < 0
            assert declared["owner"] == "stationary_base"
            assert declared["unit"] == "rad/s"
        else:
            assert declared["command"] == "position"
            assert model.actuator_biasprm[aid, 1] < 0
            assert declared["unit"] == ("m" if model.actuator(aid).name == "Lift" else "rad")
    assert ownership["Left Wrist"]["owner"] == ownership["Right Wrist"]["owner"] == "arms"
    lift = model.actuator("Lift").id
    assert model.actuator_trntype[lift] == mujoco.mjtTrn.mjTRN_TENDON
    assert model.tendon(int(model.actuator_trnid[lift, 0])).name == "Total Lift"
    assert model.equality("Sync Lift").id >= 0


@pytest.mark.mabel
def test_stereo_fixture_is_actual_reset_geometry_and_native_baseline():
    contract = load("contract.json")
    pair = load("examples/stereo_pair.json")
    env = native_env()
    model, data = env.model, env.data
    root = model.body("mabel_chassis").id
    root_rotation = data.xmat[root].reshape(3, 3)
    positions = []
    for eye in pair["eye_order"]:
        camera = pair["cameras"][eye]
        cid = model.camera(camera["name"]).id
        assert camera["name"] == contract["cameras"][eye]
        assert model.cam_bodyid[cid] == model.body("head").id
        assert camera["vertical_fov_deg"] == pytest.approx(model.cam_fovy[cid])
        expected_position = root_rotation.T @ (data.cam_xpos[cid] - data.xpos[root])
        np.testing.assert_allclose(camera["position_m"], expected_position, atol=1e-12)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.array(camera["quaternion_wxyz"]))
        np.testing.assert_allclose(
            matrix.reshape(3, 3),
            root_rotation.T @ data.cam_xmat[cid].reshape(3, 3),
            atol=1e-12,
        )
        positions.append(expected_position)
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(
        contract["cameras"]["baseline_m"]
    )
    assert pair["width"] / pair["height"] == 4 / 3
    assert pair["sim_time_s"] == 0
