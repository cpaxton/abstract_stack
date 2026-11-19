"""Composable shelf-stocking skill library and per-carton task plan.

Each :class:`Skill` is a named, reusable unit with a declared measured entry
precondition, a declared measured exit gate, optional upstream dependencies and
an ``execute`` adapter that drives the existing scripted controller. A
:class:`TaskPlan` composes skills in order and :func:`run_plan` executes them,
recording reach/entry/exit evidence at every boundary.

The library changes bookkeeping and call structure only. Physics, acceptance
criteria, force guards, displacement limits and tolerances are unchanged from
the underlying ``StockingScript`` methods; the adapters here call those methods
verbatim, and the split seams (``push`` on the front release and ``retreat`` on
the finger push) default to the current fused behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from abstract_stack.interactions import StockingScript


@dataclass(frozen=True)
class Skill:
    """A named, reusable unit of shelf-stocking with declared entry/exit gates.

    ``execute`` runs the physical skill on a live ``StockingScript`` and must
    raise ``InteractionFailure`` (a ``RuntimeError``) on any measured failure.
    ``precondition`` and ``verify`` are optional read-only measured checks that
    document, but never replace, the internal acceptance gates.
    """

    name: str
    label: str
    entry: str
    gate: str
    requires: tuple[str, ...] = ()
    execute: Callable[[StockingScript, dict], None] | None = None
    precondition: Callable[[StockingScript, dict], None] | None = None
    verify: Callable[[StockingScript, dict], bool] | None = None
    phases: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Skill name must be a nonempty identifier: {self.name!r}")
        if not self.label or not self.entry or not self.gate:
            raise ValueError("Skill label, entry and gate descriptions must be nonempty")
        for attribute in ("requires", "phases"):
            value = getattr(self, attribute)
            if not isinstance(value, tuple):
                raise ValueError(f"Skill {attribute!r} must be a tuple")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError("Skill requirements must be unique")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError("Skill phase names must be unique")

    def describe(self):
        return {
            "name": self.name,
            "label": self.label,
            "entry_condition": self.entry,
            "exit_gate": self.gate,
            "requires": list(self.requires),
            "phases": list(self.phases),
            "callable": self.execute is not None,
            "verifies_exit": self.verify is not None,
        }


@dataclass(frozen=True)
class SkillStep:
    """Recorded outcome of one plan step, regardless of success."""

    skill: str
    reached: bool
    entry_ok: bool
    exit_ok: bool
    error: str | None
    sim_time_s: float


@dataclass
class PlanOutcome:
    """Mutable execution record; a failure still retains every reached step."""

    plan: TaskPlan | None = None
    steps: list[SkillStep] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    failure: str | None = None

    def describe(self):
        return {
            "plan": self.plan.name if self.plan is not None else None,
            "completed": list(self.completed),
            "failure": self.failure,
            "steps": [
                {
                    "skill": step.skill,
                    "reached": step.reached,
                    "entry_ok": step.entry_ok,
                    "exit_ok": step.exit_ok,
                    "error": step.error,
                    "sim_time_s": step.sim_time_s,
                }
                for step in self.steps
            ],
        }


@dataclass(frozen=True)
class TaskPlan:
    """An ordered, validated composition of skills for one stocking cycle."""

    name: str
    skills: tuple[Skill, ...]

    def __post_init__(self):
        if not self.name:
            raise ValueError("Task plan requires a nonempty name")
        names = [skill.name for skill in self.skills]
        if len(names) != len(set(names)):
            raise ValueError("Task plan skill names must be unique")
        known: set[str] = set()
        for skill in self.skills:
            missing = set(skill.requires) - known
            if missing:
                raise ValueError(
                    f"{self.name}: skill {skill.name!r} requires unknown or out-of-order "
                    f"upstream skills {sorted(missing)}"
                )
            known.add(skill.name)

    def __iter__(self):
        return iter(self.skills)

    def __len__(self):
        return len(self.skills)

    def __getitem__(self, index):
        return self.skills[index]

    def by_name(self, name):
        return next((skill for skill in self.skills if skill.name == name), None)

    def describe(self):
        return {
            "name": self.name,
            "skill_count": len(self.skills),
            "skills": [skill.describe() for skill in self.skills],
        }

    def phase_map(self) -> dict[str, str]:
        """Best-effort phase name to skill name lookup for event annotation."""
        mapping: dict[str, str] = {}
        for skill in self.skills:
            for phase in skill.phases:
                if phase in mapping and mapping[phase] != skill.name:
                    raise ValueError(f"Phase {phase!r} maps to multiple skills")
                mapping[phase] = skill.name
        return mapping


def run_plan(script, plan: TaskPlan, context: dict | None = None) -> PlanOutcome:
    """Execute a validated plan, recording reach/entry/exit evidence per skill.

    The original failure is re-raised (with the failing step retained on
    ``script.plan_outcome``) so the episode summary keeps the same phase-prefixed
    failure message as the monolithic path.
    """
    from abstract_stack.interactions import InteractionFailure

    context = {} if context is None else context
    outcome = PlanOutcome(plan=plan)
    script.task_plan = plan
    script.plan_outcome = outcome
    for skill in plan:
        script._stage(f"skill_{skill.name}")
        try:
            if skill.precondition is not None:
                skill.precondition(script, context)
        except (RuntimeError, ValueError, FloatingPointError) as failure:
            outcome.steps.append(
                SkillStep(skill.name, True, False, False, f"entry: {failure}", _time(script))
            )
            outcome.failure = f"{skill.name}: entry precondition failed"
            raise
        try:
            if skill.execute is not None:
                skill.execute(script, context)
        except (RuntimeError, ValueError, FloatingPointError) as failure:
            outcome.steps.append(
                SkillStep(skill.name, True, True, False, f"execute: {failure}", _time(script))
            )
            outcome.failure = f"{skill.name}: execution failed"
            raise
        exit_ok = skill.verify(script, context) if skill.verify is not None else True
        if not exit_ok:
            outcome.steps.append(SkillStep(skill.name, True, True, False, None, _time(script)))
            outcome.failure = f"{skill.name}: exit gate not satisfied"
            raise InteractionFailure(outcome.failure)
        outcome.steps.append(SkillStep(skill.name, True, True, True, None, _time(script)))
        outcome.completed.append(skill.name)
    return outcome


def _time(script):
    return float(script.env.data.time)


# --- Read-only measured precondition helpers ---------------------------------


def _empty_hand(script, context):
    from abstract_stack.interactions import InteractionFailure

    if script.active_grip is not None or script.carry_reference is not None:
        raise InteractionFailure("pick requires an empty hand")


def _retained_load(script, context):
    from abstract_stack.interactions import InteractionFailure

    if script.active_grip is None:
        raise InteractionFailure("place requires a retained grasped load")


# --- Read-only measured exit-gate helpers ------------------------------------


def _pick_verified(script, context):
    return any(
        event.get("verified_lift")
        for event in script.events
        if event.get("product") == script.product.name
    )


def _place_verified(script, context):
    return any(
        event.get("verified_shelf_support")
        for event in script.events
        if event.get("product") == script.product.name
    )


def _insert_verified(script, context):
    events = [event for event in script.events if event.get("product") == script.product.name]
    return any(event.get("phase") == "front_registered" for event in events) and any(
        event.get("verified_release") for event in events
    )


def _push_to_contact_verified(script, context):
    event = getattr(script, "_active_push_event", None) or {}
    return any(
        event.get(key)
        for key in (
            "first_back_contact_verified",
            "back_contact_verified",
            "bounded_backstop_staging_for_native_refinement",
        )
    )


def _retreat_verified(script, context):
    if script.observer is None:
        return False
    result = script.observer.retreat_results.get(script.product.name, {})
    return bool(result.get("verified"))


# --- Execution adapters ------------------------------------------------------


def _pick(script, context):
    """Grasp/lift/hold one carton; retains the measured grasp in the context."""
    context["grasp_grip"] = script._grasp_and_lift(context.get("selected"))


def _place(script, context):
    """Transfer the carried carton and seat it at the shelf front."""
    context["grasp_grip"] = script._transfer_and_place_front(context["grasp_grip"])


def _insert(script, context):
    """Release, register and clear the hand, stopping before the finger push."""
    script._release_and_insert_front(context["grasp_grip"], push=False)


def _push_to_contact(script, context):
    """Empty-hand push to back contact, stopping before release/withdrawal."""
    script._insert_from_front(retreat=False)


def _retreat(script, context):
    """Verify squared contact and execute the guarded native withdrawal."""
    from abstract_stack.contact_execution import squared_contact_retreat

    squared_contact_retreat(script, script._active_push_event)


# --- Canonical plan ----------------------------------------------------------


def build_stocking_plan() -> TaskPlan:
    """The released selected-finger per-carton sequence.

    The grasped supported-slide protocol is a separate fused path and is not
    represented here; its ``pick`` and ``place`` skills are identical, while its
    contact finishing and retreat remain composed inside ``_push_grasped``.
    """
    return TaskPlan(
        "per_carton_stocking",
        (
            Skill(
                "pick",
                "Grasp/lift/hold",
                "Empty hand; upright table-supported carton with measured inventory poses",
                "Opposing contacts, measured rise, retained hand contact, table support lost",
                (),
                _pick,
                _empty_hand,
                _pick_verified,
                (
                    "grasp_precheck",
                    "grasp_candidate",
                    "grasp_staging",
                    "grasp_transit",
                    "pregrasp",
                    "approach",
                    "close",
                    "lift",
                    "grasp_hold",
                ),
            ),
            Skill(
                "place",
                "Transfer and seat",
                "Verified lift; measured grasp transform and remaining inventory",
                "Measured shelf support within the original bounded seating range",
                ("pick",),
                _place,
                _retained_load,
                _place_verified,
                (
                    "carry_clear",
                    "carry_front_clear",
                    "loaded_workspace_plan",
                    "loaded_stage_plan",
                    "align_shelf",
                    "transfer",
                    "place",
                    "seat",
                ),
            ),
            Skill(
                "insert",
                "Front insertion",
                "Seated shelf-front carton; retained wrapping grip before pressure unload",
                "Released, unloaded, upright and laterally registered front carton",
                ("place",),
                _insert,
                None,
                _insert_verified,
                (
                    "insertion_mode",
                    "stage_align",
                    "release_unload",
                    "release_front",
                    "release_front_separate",
                    "staging_clear",
                    "free_landing_verified",
                    "free_landing_rejected",
                    "front_registered",
                ),
            ),
            Skill(
                "push_to_contact",
                "Contact finishing",
                "Released shelf-front carton; empty preshaped pusher at a checked precontact pose",
                "Measured back contact or bounded supported staging for guarded refinement",
                ("insert",),
                _push_to_contact,
                None,
                _push_to_contact_verified,
                (
                    "pusher_plan",
                    "pusher_base_plan",
                    "pusher_preshape",
                    "pusher_approach",
                    "finger_push",
                    "front_square_plan",
                    "front_square_clear",
                    "front_squaring_complete",
                ),
            ),
            Skill(
                "retreat",
                "Release/withdrawal",
                "Supported back contact or bounded staging; hand still loaded or touching",
                "Unloaded hand and verified retreat without disturbing any required product",
                ("push_to_contact",),
                _retreat,
                None,
                _retreat_verified,
                (
                    "contact_refine",
                    "release",
                    "release_forecast",
                    "unloaded_clear_forecast",
                    "clear",
                    "partial_clear",
                ),
            ),
        ),
    )


def annotate_events(plan: TaskPlan, events: Iterable[dict]) -> None:
    """Best-effort skill annotation for retained phase events (in place)."""
    mapping = plan.phase_map()
    for event in events:
        event.setdefault("skill", mapping.get(event.get("phase")))
