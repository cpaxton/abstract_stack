"""Skill library and task-plan composition are declarative and re-executable."""

import json
from types import SimpleNamespace

import pytest

from abstract_stack.interactions import InteractionFailure
from abstract_stack.skill_library import (
    Skill,
    TaskPlan,
    annotate_events,
    build_stocking_plan,
    describe_stocking_skills,
    run_plan,
)


class FakeScript:
    def __init__(self, *, time=0.0):
        self.events = []
        self.env = SimpleNamespace(data=SimpleNamespace(time=time))
        self.task_plan = None
        self.plan_outcome = None

    def _stage(self, name):
        self.events.append({"phase": name})
        self.env.data.time += 1.0


def _skill(name, *, requires=(), execute=None, verify=None, precondition=None):
    return Skill(
        name,
        name.title(),
        f"entry {name}",
        f"gate {name}",
        requires,
        execute,
        precondition,
        verify,
    )


def test_skill_requires_valid_identifier_and_unique_entries():
    with pytest.raises(ValueError):
        Skill("", "x", "entry", "gate")
    with pytest.raises(ValueError):
        Skill("bad name", "x", "entry", "gate")
    with pytest.raises(ValueError):
        Skill("ok", "x", "", "gate")
    with pytest.raises(ValueError):
        Skill("ok", "x", "entry", "gate", requires=("a", "a"))


def test_task_plan_requires_upstream_skills_in_order():
    plan = TaskPlan("p", (_skill("a"), _skill("b", requires=("a",))))
    assert plan.by_name("a").name == "a"
    assert len(plan) == 2
    with pytest.raises(ValueError):
        TaskPlan("p", (_skill("a"), _skill("b", requires=("c",))))
    with pytest.raises(ValueError):
        TaskPlan("p", (_skill("a"), _skill("a")))
    with pytest.raises(ValueError):
        TaskPlan("p", (_skill("b", requires=("a",)), _skill("a")))


def test_build_stocking_plan_is_a_valid_composable_sequence():
    plan = build_stocking_plan()
    assert [skill.name for skill in plan] == [
        "pick",
        "place",
        "insert",
        "push_to_contact",
        "retreat",
    ]
    assert all(skill.execute is not None for skill in plan)
    description = plan.describe()
    assert description["name"] == "per_carton_stocking"
    assert description["skill_count"] == 5
    assert plan.phase_map()["finger_push"] == "push_to_contact"
    assert plan.phase_map()["front_registered"] == "insert"
    assert plan.phase_map()["release"] == "retreat"


def test_agent_skill_catalog_exports_contracts_and_actual_entry_points():
    catalog = json.loads(json.dumps(describe_stocking_skills()))
    skills = {skill["name"]: skill for skill in catalog["skills"]}
    assert len(skills) == len(catalog["skills"])
    assert catalog["plan_order"] == [skill.name for skill in build_stocking_plan()]
    for skill in skills.values():
        assert skill["description"]
        assert skill["preconditions"] == skill["entry_condition"]
        assert skill["postconditions"] == skill["exit_gate"]
        assert skill["callable"] or skill["entry_point"]
    assert "--side-press" in skills["side_press"]["entry_point"]


def test_run_plan_executes_each_skill_in_order_and_verifies():
    script = FakeScript()
    context = {}

    def execute_a(s, ctx):
        ctx["a"] = True

    def execute_b(s, ctx):
        ctx["b"] = True

    def verify_b(s, ctx):
        return ctx.get("a") is True

    plan = TaskPlan(
        "ab",
        (
            _skill("a", execute=execute_a),
            _skill("b", requires=("a",), execute=execute_b, verify=verify_b),
        ),
    )
    outcome = run_plan(script, plan, context)
    assert outcome.completed == ["a", "b"]
    assert context == {"a": True, "b": True}
    assert [event["phase"] for event in script.events] == ["skill_a", "skill_b"]


def test_run_plan_records_failure_and_reraises_original():
    script = FakeScript()

    def execute_a(s, ctx):
        raise RuntimeError("boom")

    plan = TaskPlan("one", (_skill("a", execute=execute_a),))
    with pytest.raises(RuntimeError):
        run_plan(script, plan, {})
    assert script.plan_outcome.failure == "a: execution failed"
    assert script.plan_outcome.steps[0].error == "execute: boom"
    assert script.plan_outcome.completed == []


def test_run_plan_raises_on_unmet_exit_gate():
    script = FakeScript()

    def execute_a(s, ctx):
        return None

    def verify_a(s, ctx):
        return False

    plan = TaskPlan("one", (_skill("a", execute=execute_a, verify=verify_a),))
    with pytest.raises(InteractionFailure, match="exit gate"):
        run_plan(script, plan, {})
    assert script.plan_outcome.failure == "a: exit gate not satisfied"


def test_run_plan_records_failed_entry_precondition():
    script = FakeScript()

    def precondition_a(s, ctx):
        raise ValueError("empty hand required")

    plan = TaskPlan("one", (_skill("a", precondition=precondition_a),))
    with pytest.raises(ValueError):
        run_plan(script, plan, {})
    assert script.plan_outcome.steps[0].entry_ok is False
    assert script.plan_outcome.steps[0].error == "entry: empty hand required"


def test_annotate_events_marks_known_phases_and_ignores_unknown():
    events = [{"phase": "finger_push"}, {"phase": "not_a_phase"}]
    annotate_events(build_stocking_plan(), events)
    assert events[0]["skill"] == "push_to_contact"
    assert events[1]["skill"] is None


def test_release_front_can_stop_before_the_push_when_composed():
    from abstract_stack.interactions import ScriptConfig, StockingScript

    script = object.__new__(StockingScript)
    calls = []
    script.carry_reference = None
    script.payload_rotation_goal = None
    script.config = ScriptConfig()
    script.product = SimpleNamespace(target=(-0.72, -0.12, 0.92))
    script.controller = SimpleNamespace(last_target=[-0.5, -0.12, 0.9])
    script._precision_release = lambda grip: calls.append("precision_release")
    script._move = lambda *args, **kwargs: calls.append(args[0])
    script._wall_front_staging_y = lambda: script.product.target[1]
    script._stage = lambda name: calls.append(f"stage:{name}")
    script._insert_from_front = lambda **kwargs: calls.append("insert_from_front")
    script._release_and_insert_front(0.5, push=False)
    assert "stage:front_registered" in calls
    assert "insert_from_front" not in calls


def test_plan_driven_one_box_composes_all_five_skills(monkeypatch):
    from abstract_stack import contact_execution
    from abstract_stack.interactions import ScriptConfig, StockingScript

    monkeypatch.setattr(contact_execution, "squared_contact_retreat", lambda script, event: None)
    script = object.__new__(StockingScript)
    script.config = ScriptConfig(
        plan_driven=True, ungrasped_insertion=True, paired_finger_insertion=True
    )
    script.env = SimpleNamespace(
        config=SimpleNamespace(row_free_space=0.01), data=SimpleNamespace(time=0.0)
    )
    script.product = SimpleNamespace(name="carton_0", target=(-0.72, -0.12, 0.92))
    script._adjacent_placed_neighbors = lambda: []
    script._wall_front_staging_y = lambda: script.product.target[1]
    script.observer = SimpleNamespace(retreat_results={"carton_0": {"verified": True}})
    script.active_grip = None
    script.carry_reference = None
    script.events = []
    calls = []

    def stage(name):
        script.events.append({"phase": name, "product": "carton_0"})
        calls.append(f"stage:{name}")

    script._stage = stage

    def grasp_and_lift(selected):
        script.active_grip = 0.8
        script.carry_reference = "held"
        script.events.append({"phase": "lift", "product": "carton_0", "verified_lift": True})
        calls.append("pick")
        return 0.8

    def transfer_and_place(grip):
        script.events.append(
            {"phase": "seat", "product": "carton_0", "verified_shelf_support": True}
        )
        calls.append("place")
        return grip

    def release_and_insert(grip, *, push=True):
        calls.append(f"insert(push={push})")
        script.active_grip = None
        script.carry_reference = None
        script.events.append({"phase": "front_registered", "product": "carton_0"})
        script.events.append(
            {"phase": "release_front", "product": "carton_0", "verified_release": True}
        )

    def insert_from_front(*, partial_stop_x=None, front_squaring=False, retreat=True):
        calls.append(f"push_to_contact(retreat={retreat})")
        script._active_push_event = {"first_back_contact_verified": True}

    script._grasp_and_lift = grasp_and_lift
    script._transfer_and_place_front = transfer_and_place
    script._release_and_insert_front = release_and_insert
    script._insert_from_front = insert_from_front

    script._one_box(None)

    assert "pick" in calls and "place" in calls
    assert "insert(push=False)" in calls
    assert "push_to_contact(retreat=False)" in calls
    assert [call for call in calls if call.startswith("stage:skill_")] == [
        "stage:skill_pick",
        "stage:skill_place",
        "stage:skill_insert",
        "stage:skill_push_to_contact",
        "stage:skill_retreat",
    ]
    assert script.plan_outcome.completed == [
        "pick",
        "place",
        "insert",
        "push_to_contact",
        "retreat",
    ]


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"bounded_backstop_staging_for_native_refinement": True}, True),
        ({"verified_shelf_support": True}, False),
    ],
)
def test_contact_handoff_requires_contact_or_admitted_staging(evidence, expected):
    skill = build_stocking_plan().by_name("push_to_contact")
    script = SimpleNamespace(_active_push_event=evidence)

    assert skill.verify(script, {}) is expected
