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
    description: str = ""
    entry_point: str = ""

    def __post_init__(self):
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Skill name must be a nonempty identifier: {self.name!r}")
        if not self.label or not self.entry or not self.gate:
            raise ValueError("Skill label, entry and gate descriptions must be nonempty")
        for attribute in ("label", "entry", "gate", "description", "entry_point"):
            if not isinstance(getattr(self, attribute), str):
                raise ValueError(f"Skill {attribute!r} must be a string")
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
            "description": self.description or self.label,
            "preconditions": self.entry,
            "postconditions": self.gate,
            "entry_condition": self.entry,
            "exit_gate": self.gate,
            "entry_point": self.entry_point or None,
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
    journal = getattr(script, "_record_agent_event", None)
    previous_skill = getattr(script, "_agent_skill", None)
    if journal is not None:
        journal("plan_started", plan=plan.describe())
    try:
        for index, skill in enumerate(plan, start=1):
            if journal is not None:
                script._agent_skill = skill.name
                journal("skill_started", skill=skill.name, skill_step=index, plan_name=plan.name)
            script._stage(f"skill_{skill.name}")
            entry_ok, error = False, None
            try:
                if skill.precondition is not None:
                    skill.precondition(script, context)
                entry_ok = True
                if skill.execute is not None:
                    skill.execute(script, context)
                exit_ok = skill.verify(script, context) if skill.verify is not None else True
            except (RuntimeError, ValueError, FloatingPointError) as failure:
                error = f"{'execute' if entry_ok else 'entry'}: {failure}"
                outcome.failure = (
                    f"{skill.name}: execution failed"
                    if entry_ok
                    else f"{skill.name}: entry precondition failed"
                )
                _record_step(script, outcome, skill, index, entry_ok, False, error, journal)
                raise
            _record_step(script, outcome, skill, index, entry_ok, exit_ok, error, journal)
            if not exit_ok:
                outcome.failure = f"{skill.name}: exit gate not satisfied"
                raise InteractionFailure(outcome.failure)
            outcome.completed.append(skill.name)
    finally:
        if journal is not None:
            script._agent_skill = previous_skill
    return outcome


def _record_step(script, outcome, skill, index, entry_ok, exit_ok, error, journal):
    outcome.steps.append(SkillStep(skill.name, True, entry_ok, exit_ok, error, _time(script)))
    if journal is not None:
        status = "failed"
        if not entry_ok:
            status = "entry_rejected"
        elif exit_ok:
            status = "verified" if skill.verify is not None else "executed"
        journal(
            "skill_finished",
            skill=skill.name,
            skill_step=index,
            entry_ok=entry_ok,
            exit_ok=exit_ok,
            outcome=status,
            failure=error or ("exit gate not satisfied" if not exit_ok else None),
        )


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
                description="Acquire the supply carton and hold it for a loaded transfer.",
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
                description="Carry the carton along a checked route and seat it at a free landing.",
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
                    "release_pressure_relief",
                    "release_front",
                    "release_front_separate",
                    "staging_clear",
                    "free_landing_verified",
                    "free_landing_rejected",
                    "front_registered",
                ),
                description="Open and clear the wrapping hand, then register the released carton.",
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
                description="Advance with a checked fingertip posture and measured force feedback.",
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
                    "release_unload_forecast",
                    "unloaded_clear_forecast",
                    "clear",
                    "partial_clear",
                ),
                description="Refine supported placement, unload, and clear the hand from the row.",
            ),
        ),
    )


def describe_stocking_skills(plan: TaskPlan | None = None) -> dict:
    """Return the physical skill contracts and entry points for an agent."""
    plan = build_stocking_plan() if plan is None else plan
    return {
        "schema_version": 1,
        "plan_order": [skill.name for skill in plan],
        "skills": [skill.describe() for skill in (*plan.skills, *_ADDITIONAL_SKILLS)],
    }


_ADDITIONAL_SKILLS = (
    Skill(
        "base_reposition",
        "Base relocation or turn",
        "A new workspace is needed, with a checked route or swept turn and hand clearance.",
        "The measured base pose permits a checked next arm action. The load and row stay stable.",
        description="Move or turn the base to reach inventory, an exposed face, or a withdrawal.",
        entry_point="Native workspace planner with --base-reposition",
    ),
    Skill(
        "side_press",
        "Active-carton side press",
        "Released upright shelf-supported carton, unloaded arms, and a reachable exposed side.",
        "Measured lateral shift survives hand clearance. Support and prior stocks are preserved.",
        description="Press the side center to reduce yaw or the front patch to allow squaring.",
        entry_point="probe_front_release.py --insert-only --side-press {centered,corner}",
    ),
    Skill(
        "front_square",
        "Front square",
        "Released carton with residual yaw and sufficient free depth ahead of the row.",
        "Support, measured lateral clearance and the released yaw gate pass before advancing.",
        description="Use the front fingertips to square a supported carton ahead of its neighbors.",
        entry_point="Guided insertion or probe_front_release.py --insert-only --resume-side-square",
    ),
    Skill(
        "wall_registration",
        "Sidewall registration",
        "Supported released edge carton with a checked side approach and wall reaction.",
        "Wall contact and registration pass, followed by clearance for the next tool.",
        description="Use the sidewall as opposing contact before the forward advance.",
        entry_point="Paired edge insertion or probe_front_release.py --insert-only --register-wall",
    ),
    Skill(
        "grasped_slide",
        "Grasped supported slide",
        "A retained grasp and room for the wrapping hand along a supported advance and exit.",
        "The carton reaches supported placement and the wrapping hand withdraws cleanly.",
        description="Slide into a spacious slot with a retained grasp and measured load.",
        entry_point="StockingScript grasped contact branch",
    ),
    Skill(
        "contact_recovery",
        "Breakaway or recontact",
        "The supported carton is short of the goal and depth stalls or finger contact is lost.",
        "Measured contact and signed progress resume under the configured motion and force guards.",
        description="Attempt bounded breakaway or reacquire from the measured stop.",
        entry_point="Guided insertion stall guard or RecedingStockingScript contact recovery",
    ),
    Skill(
        "contact_refine",
        "Supported contact refinement",
        "Supported carton near final depth, with a checked narrow tool and residual pose error.",
        "Required back contact, support and squared dwell are measured before release.",
        description="Use measured pose and wrench to refine depth, yaw and lateral registration.",
        entry_point="probe_front_release.py --withdrawal --stop-after=squared-contact",
    ),
    Skill(
        "unloaded_clear",
        "Unloaded clear",
        "Carton already satisfies the placement gate with stable support and a clearable hand.",
        "The hand clears the shelf and all required products retain their placement.",
        description="Screen and execute a withdrawal from a supported, unloaded stop.",
        entry_point="probe_front_release.py --unloaded-clear",
    ),
    Skill(
        "row_compaction",
        "Row compaction",
        "The supported partial row has internal gaps and insufficient remaining usable span.",
        "Pair contacts develop and the usable opening survives unloading under support guards.",
        description="Press an exposed end to close gaps across the row.",
        entry_point="StockingCurriculumEnv slide or the opt-in between-carton row press",
    ),
    Skill(
        "row_finish",
        "Strict row finish",
        "A supported compacted row has the needed opening and residual yaw or overlap.",
        "Measured contact, alignment, envelope and unloaded stability gates all pass.",
        description="Refine a compacted row to the strict final geometry.",
        entry_point="StockingSkillEnv row-press",
    ),
    Skill(
        "bimanual_clear",
        "Bimanual clearance",
        "Both palms are unloaded and a native plan can clear them with a shared lift reference.",
        "Both measured palm goals pass and the shelf contents are preserved.",
        description="Move both unloaded palms through a coupled clearance motion.",
        entry_point="probe_front_release.py --bimanual-clear",
    ),
    Skill(
        "stop_and_snapshot",
        "Stop and snapshot",
        "A guard trips, the bounded recovery exhausts its budget, or no motion is admitted.",
        "Record the phase, failure and last valid native state for practice or replanning.",
        description="Retain a handled failure and its actual controller and physics state.",
        entry_point="StockingScript with --record-skill-starts",
    ),
)


def annotate_events(plan: TaskPlan, events: Iterable[dict]) -> None:
    """Best-effort skill annotation for retained phase events (in place)."""
    mapping = plan.phase_map()
    for event in events:
        event.setdefault("skill", mapping.get(event.get("phase")))
