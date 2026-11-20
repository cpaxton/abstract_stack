import json
from types import SimpleNamespace

import pytest

from abstract_stack.agent_history import AgentHistory, inspect_history, read_history
from abstract_stack.skill_library import Skill, TaskPlan, run_plan


def test_interrupted_execution_retains_selected_carton_and_step(tmp_path):
    episode = tmp_path / "episode"
    journal = AgentHistory(episode, run_kind="stocking", scheduled_products=["first", "second"])
    journal.record("carton_selection_started", carton_step=1, choices=["first", "second"])
    journal.record("carton_selected", carton_step=1, product="first")
    journal.record("carton_plan", product="first", steps=["pick", "place"])
    journal.record("phase_started", carton_step=1, product="first", skill="place", phase="seat")
    journal.record("checkpoint_saved", product="first", checkpoint="skill-entries/first/checkpoint")
    with journal.path.open("ab") as stream:
        stream.write(b'{"sequence":6,')
    (tmp_path / "process.json").write_text(json.dumps(dict(returncode=-11, timeout=False)))
    status = inspect_history(tmp_path)
    assert status["status"] == "process_failure_or_unknown"
    assert status["current"]["skill_step"] == 2
    assert status["current"]["product"] == "first"
    assert status["products"]["second"]["skills_used"] == []
    assert status["latest_checkpoints"]["first"]["sequence"] == 5
    assert status["last_sequence"] == 5


def test_rewind_keeps_both_attempt_outcomes(tmp_path):
    journal = AgentHistory(tmp_path, run_kind="skill_attempts")
    for branch, phase, outcome in (
        ("initial", "push", "failed"),
        ("rewind", "contact_refine", "verified"),
    ):
        journal.record("branch_started", branch=branch)
        journal.record("attempt_started", branch=branch, product="carton", skill="push", attempt=1)
        journal.record("phase_started", branch=branch, product="carton", skill="push", phase=phase)
        journal.record(
            "attempt_finished", branch=branch, product="carton", skill="push", outcome=outcome
        )
    journal.record("run_finished", outcome="verified")
    status = inspect_history(tmp_path)
    assert status["current"]["phase"] == "contact_refine"
    assert [a["outcome"] for a in status["products"]["carton"]["attempts"]] == [
        "failed",
        "verified",
    ]
    assert status["products"]["carton"]["skills_verified"] == ["push"]


def test_plan_history_reports_rejected_entry_and_preserves_original_failure(tmp_path):
    journal = AgentHistory(tmp_path, run_kind="stocking")
    script = SimpleNamespace(
        env=SimpleNamespace(data=SimpleNamespace(time=0.0)),
        _stage=lambda _: None,
        _record_agent_event=journal.record,
    )

    def reject(script, context):
        raise RuntimeError("observed entry failure")

    plan = TaskPlan("stock", (Skill("pick", "Pick", "empty hand", "lift", precondition=reject),))
    with pytest.raises(RuntimeError, match="observed entry failure"):
        run_plan(script, plan)
    rows = list(read_history(journal.path))
    assert [row["event"] for row in rows] == [
        "run_started",
        "plan_started",
        "skill_started",
        "skill_finished",
    ]
    assert rows[-1]["outcome"] == "entry_rejected"
    assert rows[-1]["exit_ok"] is False
