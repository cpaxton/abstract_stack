"""Recovery eligibility follows the user's passed-held-out-first requirement."""

import runpy
from copy import deepcopy
from pathlib import Path

import pytest


def test_recovery_requires_complete_same_policy_skill_holdout():
    check = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/recover_grasp_skill.py")
    )["require_passed_heldout"]
    split = dict(split="heldout", source_content_id="frozen")
    report = dict(
        source_content_id="frozen",
        evaluation_complete=True,
        scheduled_grasps=2,
        verified_grasps=2,
        cases=[dict(grasp_verified=True), dict(grasp_verified=True)],
    )
    check(split, report, "frozen")
    for field, value in [
        ("source_content_id", "other"),
        ("evaluation_complete", "true"),
        ("verified_grasps", 1),
        ("scheduled_grasps", True),
        ("cases", []),
    ]:
        invalid = deepcopy(report)
        invalid[field] = value
        with pytest.raises(ValueError):
            check(split, invalid, "frozen")
    with pytest.raises(ValueError):
        check(dict(split="development", source_content_id="frozen"), report, "frozen")
