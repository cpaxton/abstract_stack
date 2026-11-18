import pytest

from abstract_stack.contact_progress import ContactProgress


def test_side_push_progress_ignores_transverse_motion_and_force_dropout():
    progress = ContactProgress((0, -1, 0))
    assert progress.observe([0, 0, 0], 0, contact=False) is None
    assert progress.observe([0, 0, 0], 1, contact=True) == (0, 0)
    assert progress.observe([0.1, 0, 0], 2, contact=False) == (1, 0)
    assert progress.observe([0.1, 0.0005, 0], 3, contact=True) == pytest.approx((2, -0.0005))
    assert progress.observe([0.1, -0.002, 0], 4, contact=True) == (0, 0)
    progress.reset()
    assert progress.observe([0, 0, 0], 5, contact=False) is None
