"""Room-level task ownership; robot adapters execute and verify the physical skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from abstract_stack.room_scene import StockingRoom

Skill = Literal["stock", "stage", "stock_staged"]


@dataclass(frozen=True)
class RobotProfile:
    id: str
    skills: frozenset[Skill]
    stations: frozenset[str]
    product_kinds: frozenset[str]


@dataclass(frozen=True)
class RoomAssignment:
    token: int
    robot_id: str
    station_id: str
    product_body: str
    skill: Skill
    source_geom: str
    destination_geom: str
    state_revision: int


@dataclass(frozen=True)
class PlacementReceipt:
    """An adapter's physical verification after release and guarded retreat."""

    state_revision: int
    product_body: str
    support_geom: str
    placement_verified: bool
    retreat_verified: bool


class RoomDispatcher:
    """Serialized dispatcher reserving one robot and one station per active task."""

    def __init__(self, room: StockingRoom) -> None:
        self._stations = {station.id: station for station in room.stations}
        self._active: dict[str, RoomAssignment] = {}
        self._staged: dict[str, str] = {}
        self._stocked: set[str] = set()
        self._failed: dict[str, RoomAssignment] = {}
        self._next_token = 1
        self._station_revision = dict.fromkeys(self._stations, 0)

    @property
    def stocked(self) -> frozenset[str]:
        return frozenset(self._stocked)

    @property
    def blocked(self) -> frozenset[str]:
        return frozenset(self._failed)

    def claim(
        self, robot: RobotProfile, station_id: str, *, state_revision: int
    ) -> RoomAssignment | None:
        station = self._stations[station_id]
        if (
            state_revision < self._station_revision[station_id]
            or station_id not in robot.stations
            or station_id in self._active
            or station_id in self._failed
            or any(
                a.robot_id == robot.id
                for assignments in (self._active, self._failed)
                for a in assignments.values()
            )
        ):
            return None

        staged = self._staged.get(station_id)
        for product in station.products:
            if product.kind not in robot.product_kinds or product.body in self._stocked:
                continue
            if staged is not None:
                if staged != product.body or "stock_staged" not in robot.skills:
                    continue
                skill: Skill = "stock_staged"
                source, destination = station.handoff_geom, product.target_support_geom
            elif "stock" in robot.skills:
                skill = "stock"
                source, destination = station.supply_geom, product.target_support_geom
            elif "stage" in robot.skills:
                skill = "stage"
                source, destination = station.supply_geom, station.handoff_geom
            else:
                continue

            assignment = RoomAssignment(
                self._next_token,
                robot.id,
                station_id,
                product.body,
                skill,
                source,
                destination,
                state_revision,
            )
            self._next_token += 1
            self._active[station_id] = assignment
            self._station_revision[station_id] = state_revision
            return assignment
        return None

    def complete(self, assignment: RoomAssignment, receipt: PlacementReceipt) -> None:
        self._require_active(assignment)
        if (
            receipt.state_revision <= assignment.state_revision
            or receipt.product_body != assignment.product_body
            or receipt.support_geom != assignment.destination_geom
            or not receipt.placement_verified
            or not receipt.retreat_verified
        ):
            raise ValueError("Completion needs fresh verified placement and retreat for the task")
        if assignment.skill == "stage":
            self._staged[assignment.station_id] = assignment.product_body
        else:
            self._stocked.add(assignment.product_body)
            self._staged.pop(assignment.station_id, None)
        self._station_revision[assignment.station_id] = receipt.state_revision
        del self._active[assignment.station_id]

    def report_failure(self, assignment: RoomAssignment, *, state_revision: int) -> None:
        """Block new claims until a supervisor reconciles the physical world state."""
        self._require_active(assignment)
        if state_revision < assignment.state_revision:
            raise ValueError("Failure observation predates this assignment")
        self._station_revision[assignment.station_id] = state_revision
        self._failed[assignment.station_id] = assignment
        del self._active[assignment.station_id]

    def confirm_source_restored(
        self, assignment: RoomAssignment, receipt: PlacementReceipt
    ) -> None:
        """Admit retries only once the product and robot are safe at the previous source."""
        if (
            self._failed.get(assignment.station_id) != assignment
            or receipt.state_revision <= self._station_revision[assignment.station_id]
            or receipt.product_body != assignment.product_body
            or receipt.support_geom != assignment.source_geom
            or not receipt.placement_verified
            or not receipt.retreat_verified
        ):
            raise ValueError("Retry needs a fresh verified restoration of the original source")
        self._station_revision[assignment.station_id] = receipt.state_revision
        del self._failed[assignment.station_id]

    def _require_active(self, assignment: RoomAssignment) -> None:
        if self._active.get(assignment.station_id) != assignment:
            raise ValueError("Task assignment is stale or unknown")
