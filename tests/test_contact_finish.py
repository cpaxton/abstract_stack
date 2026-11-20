"""A contact flash, sparse samples or an unsquared carton cannot pass dwell."""

import numpy as np
import pytest

from abstract_stack.contact_finish import BackstopDwell


def gate():
    return BackstopDwell(0.002, 0.4, 0.12, 5.0, 0.15)


def test_supported_squared_contact_must_persist_for_full_physics_dwell():
    dwell = gate()
    forces = dict(back_N=0.5, hand_N=1.0, support_N=2.0)
    assert not dwell.update(100, forces, np.eye(3))
    for step in range(1, 60):
        assert not dwell.update(100 + step * 0.002, forces, np.eye(3))
    assert dwell.update(100.12, forces, np.eye(3))
    report = dwell.report()
    assert report["verified"] and report["held_seconds"] == pytest.approx(0.12)
    assert not report["task_success"] and not report["release_verified"]


@pytest.mark.parametrize("missing", ["back_N", "hand_N", "support_N"])
def test_each_required_contact_break_resets_dwell(missing):
    dwell = gate()
    forces = dict(back_N=0.5, hand_N=1.0, support_N=2.0)
    for step in range(31):
        dwell.update(step * 0.002, forces, np.eye(3))
    assert dwell.held_seconds == pytest.approx(0.06)
    dwell.update(0.062, {**forces, missing: 0.0}, np.eye(3))
    assert dwell.held_seconds == 0 and not dwell.verified


@pytest.mark.parametrize("angle,axis", [(0.2, "yaw"), (6.0, "roll")])
def test_orientation_is_measured_and_not_replaced_by_the_command(angle, axis):
    dwell = gate()
    a = np.radians(angle)
    rotation = (
        [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        if axis == "yaw"
        else [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    for step in range(100):
        assert not dwell.update(step * 0.002, dict(back_N=0.5, hand_N=1.0, support_N=2.0), rotation)


@pytest.mark.parametrize("next_time", [0.0, -0.002, 0.004])
def test_skipped_repeated_or_reversed_physics_samples_fail_closed(next_time):
    dwell = gate()
    forces = dict(back_N=0.5, hand_N=1.0, support_N=2.0)
    dwell.update(0, forces, np.eye(3))
    with pytest.raises(ValueError, match="every physics step"):
        dwell.update(next_time, forces, np.eye(3))


def test_checkpoint_preserves_partial_dwell_and_rejects_changed_threshold():
    original = gate()
    forces = dict(back_N=0.5, hand_N=1.0, support_N=2.0)
    for step in range(31):
        original.update(step * 0.002, forces, np.eye(3))
    restored = gate()
    restored.restore_state(original.report(), 0.06)
    for step in range(31, 61):
        assert restored.update(step * 0.002, forces, np.eye(3)) == original.update(
            step * 0.002, forces, np.eye(3)
        )
    assert restored.verified and restored.held_seconds == pytest.approx(0.12)
    changed = {**original.report(), "force_threshold_N": 0.3}
    with pytest.raises(ValueError, match="acceptance threshold"):
        gate().restore_state(changed, 0.12)
