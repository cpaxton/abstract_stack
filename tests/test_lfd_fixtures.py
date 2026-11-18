"""Admission and repeatability of short native controller fixtures."""

import numpy as np
import pytest

from abstract_stack.lfd.episode import ActionContract
from abstract_stack.lfd.fixtures import fixture_scene_context, motion_reference


def contract():
    return ActionContract(
        ("Lift", "Neck Yaw", "Neck Pitch", "Left Wrist", "Right Wrist"),
        ("m", "rad", "rad", "rad", "rad"),
        ("position",) * 5,
        (0.0, -1.0, -1.0, -1.0, -1.0),
        (0.5, 1.0, 1.0, 1.0, 1.0),
    )


@pytest.mark.parametrize("profile", ["lift_pan", "left_wrist_pan", "right_wrist_tilt"])
def test_fixture_is_bounded_returns_to_nominal_and_does_not_mutate_baseline(profile):
    reference = np.array([0.1, 0, 0, 0, 0])
    original = reference.copy()
    commands = np.asarray(
        [motion_reference(reference, contract(), profile, index=i, steps=20) for i in range(20)]
    )
    np.testing.assert_array_equal(reference, original)
    np.testing.assert_array_equal(commands[-1], reference)
    assert np.any(np.ptp(commands, axis=0) > 0.01)
    # 20 ms cadence stays below the native controller's configured 1 rad/s
    # and 0.08 m/s bounds, including the first nominal transition.
    rates = np.abs(np.diff(np.vstack([reference, commands]), axis=0)) / 0.02
    assert rates[:, 0].max() < 0.08 and rates[:, 1:].max() < 1.0


def test_fixture_exceeding_native_bounds_is_rejected_instead_of_clipped():
    with pytest.raises(ValueError, match="native bounds"):
        motion_reference([0.499, 0, 0, 0, 0], contract(), "lift_pan", index=9, steps=20)


def test_domain_case_identity_and_effective_reset_are_bound_to_exact_inputs(tmp_path):
    import hashlib
    import json

    config = tmp_path / "scene.json"
    config.write_text('{"product_count": 6, "levels": [0.7, 1.0]}\n')
    identity = tmp_path / "case.json"
    case = dict(
        suite_version=1,
        split="held_out",
        scene_seed=1204,
        reset_seed=3,
        factors=dict(product_count=6, shelf_levels=2),
        scene_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    identity.write_text(json.dumps(case))
    context = fixture_scene_context(
        scene_seed=1204, reset_seed=3, scene_config=config, domain_context=identity
    )
    assert context["environment_reset_seed"] == 3
    assert context["scene_configuration_seed"] == 1204
    assert (
        context["scene_configuration_source"]["sha256"]
        == hashlib.sha256(config.read_bytes()).hexdigest()
    )
    assert context["domain_context"]["identity"] == case
    assert context["domain_context"]["sha256"] == hashlib.sha256(identity.read_bytes()).hexdigest()
    assert "student" in context["domain_context"]["scope"]
    config.write_text('{"product_count": 3}')
    with pytest.raises(ValueError, match="configuration hash"):
        fixture_scene_context(
            scene_seed=1204, reset_seed=3, scene_config=config, domain_context=identity
        )


def test_wrong_domain_seed_or_unbound_scene_is_rejected(tmp_path):
    import json

    config = tmp_path / "scene.json"
    config.write_text("{}")
    identity = tmp_path / "case.json"
    identity.write_text(json.dumps(dict(suite_version=1, split="development", scene_seed=9)))
    with pytest.raises(ValueError, match="scene/reset seeds"):
        fixture_scene_context(
            scene_seed=10, reset_seed=0, scene_config=config, domain_context=identity
        )
    with pytest.raises(ValueError, match="explicit scene"):
        fixture_scene_context(scene_seed=9, reset_seed=0, domain_context=identity)


def test_legacy_fixture_identity_keeps_distinct_generation_and_reset_seeds():
    context = fixture_scene_context(scene_seed=1201, reset_seed=0)
    assert context["scene_configuration_seed"] == 1201
    assert context["environment_reset_seed"] == 0
    assert context["domain_context"] is None
