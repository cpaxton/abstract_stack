"""Bounded order search; geometric forecasts are heuristics, never certificates.

Only a freshly checked first action may be dispatched. Discrete suffixes can be
reused across observations; continuous trajectories and IK results cannot.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CartonEstimate:
    name: str
    position: tuple[float, float, float]
    extent: tuple[float, float, float]
    target: tuple[float, float, float]
    half_size: tuple[float, float, float]

    def __post_init__(self):
        vectors = (self.position, self.extent, self.target, self.half_size)
        if not self.name or any(len(v) != 3 for v in vectors):
            raise ValueError("Cartons need a name and four three-vectors")
        if not np.isfinite(vectors).all() or min(*self.extent, *self.half_size) <= 0:
            raise ValueError("Carton coordinates must be finite and extents positive")


@dataclass(frozen=True)
class RowObservation:
    cartons: tuple[CartonEstimate, ...]
    pending: tuple[str, ...]
    tool_position: tuple[float, float, float]

    def __post_init__(self):
        names = [c.name for c in self.cartons]
        if len(names) != len(set(names)) or len(self.pending) != len(set(self.pending)):
            raise ValueError("Carton and pending names must be unique")
        if not set(self.pending) <= set(names):
            raise ValueError("Pending cartons must occur in the observation")
        if len(self.tool_position) != 3 or not np.isfinite(self.tool_position).all():
            raise ValueError("Tool position must be a finite three-vector")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 32
    max_expansions: int = 4000
    max_root_checks: int = 10
    wall_seconds: float = 5.0
    root_wall_seconds: float = 5.0

    def __post_init__(self):
        for value in (self.beam_width, self.max_expansions, self.max_root_checks):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Search counts must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (self.wall_seconds, self.root_wall_seconds)):
            raise ValueError("Search wall budgets must be finite and positive")


@dataclass(frozen=True)
class ActionCheck:
    feasible: bool
    reason: str


@dataclass(frozen=True)
class OrderPlan:
    observation_fingerprint: str
    order: tuple[str, ...]
    first_action_checked: bool
    reused_tail: bool
    expansions: int
    root_checks: tuple[tuple[str, ActionCheck], ...]
    elapsed_seconds: float
    stop_reason: str
    reuse_rejection: str | None = None

    def report(self):
        return {
            **asdict(self),
            "algorithm": "deterministic_beam_order_search_with_tail_reuse",
            "action_scope": "whole_carton_stock_to_fixed_slot",
            "forecast_scope": "nominal_order_cost_only_not_loaded_motion_or_contact_proof",
        }


def _extend_cost(observation, order, name):
    """Travel and local crowding proxy, in heuristic seconds (not measured time).

    Completed hypothetical moves replace only this local geometric forecast.
    They do not change observations or simulate a physical success.
    """
    by_name = {c.name: c for c in observation.cartons}
    carton = by_name[name]
    origin = observation.tool_position if not order else by_name[order[-1]].target
    cost = np.linalg.norm(np.subtract(origin, carton.position)) / 0.1
    cost += np.linalg.norm(np.subtract(carton.position, carton.target)) / 0.05
    for other in observation.cartons:
        if other.name == name:
            continue
        moved = other.name in order
        position = other.target if moved else other.position
        extent = other.half_size if moved else other.extent
        separation = np.abs(np.subtract(carton.position[:2], position[:2]))
        clearance = separation - np.add(carton.extent[:2], extent[:2])
        # Penalize crowded pickup; this is not a swept-arm collision test.
        if max(clearance) < 0.025:
            cost += 2.0
    return float(cost)


class RowReplanner:
    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self.previous_order: tuple[str, ...] = ()

    def plan(
        self,
        observation: RowObservation,
        check_first: Callable[[str], ActionCheck],
    ) -> OrderPlan:
        """Reuse a discrete tail, otherwise search; freshly check the next action.

        Callback must not step or modify the live world. Budget is checked between
        callbacks; a single native IK call is non-preemptible and may overrun it.
        No cached continuous result is retained by this planner.
        """
        started = time.monotonic()
        deadline = started + self.config.wall_seconds
        checks: dict[str, ActionCheck] = {}
        expansions = 0
        reuse_rejection = None

        def finish(order, checked, reused, reason):
            self.previous_order = order if checked else ()
            return OrderPlan(
                observation.fingerprint,
                order,
                checked,
                reused,
                expansions,
                tuple(checks.items()),
                time.monotonic() - started,
                reason,
                reuse_rejection,
            )

        def check(name):
            if name not in checks:
                checks[name] = check_first(name)
            return checks[name].feasible

        if not observation.pending:
            self.previous_order = ()
            return finish((), False, False, "no_pending_cartons")
        tail = tuple(name for name in self.previous_order if name in observation.pending)
        if tail and set(tail) == set(observation.pending):
            if check(tail[0]):
                return finish(tail, True, True, "tail_rechecked")
            reuse_rejection = checks[tail[0]].reason

        # Retain at least one suffix for each root so an infeasible cheap root
        # cannot erase a more costly feasible alternative from the frontier.
        frontier = [(0.0, ())]
        complete = []
        stopped = "tree_exhausted"
        for _ in observation.pending:
            candidates = []
            for cost, order in frontier:
                for name in sorted(set(observation.pending) - set(order)):
                    if expansions >= self.config.max_expansions or time.monotonic() >= deadline:
                        stopped = "search_budget"
                        break
                    candidates.append(
                        (cost + _extend_cost(observation, order, name), (*order, name))
                    )
                    expansions += 1
                if stopped == "search_budget":
                    break
            if not candidates:
                break
            candidates.sort()
            best_per_root = {}
            for candidate in candidates:
                best_per_root.setdefault(candidate[1][0], candidate)
            frontier = list(best_per_root.values())
            selected = {order for _, order in frontier}
            for candidate in candidates:
                if len(frontier) >= max(self.config.beam_width, len(best_per_root)):
                    break
                if candidate[1] not in selected:
                    frontier.append(candidate)
                    selected.add(candidate[1])
            if len(frontier[0][1]) == len(observation.pending):
                complete = frontier
            if stopped == "search_budget":
                break

        # Only a complete symbolic suffix can be returned. Budget exhaustion
        # never upgrades an unchecked or incomplete prefix to an execution plan.
        ranked = sorted(complete)
        for _, order in ranked:
            name = order[0]
            if name in checks:
                if checks[name].feasible:
                    return finish(order, True, False, "new_order_checked")
                continue
            if len(checks) >= self.config.max_root_checks or time.monotonic() >= deadline:
                return finish((), False, False, "root_check_budget")
            if check(name):
                return finish(order, True, False, "new_order_checked")
        return finish((), False, False, stopped if not ranked else "no_checked_root")
