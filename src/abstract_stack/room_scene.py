"""Deterministic multi-station room assets and world-frame task descriptions."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from abstract_stack.scene import (
    ShelfConfig,
    _box,
    _camera,
    _numbers,
    build_scene_xml,
    initial_positions,
    products_for,
)

Vec3 = tuple[float, float, float]
Bounds = tuple[float, float, float, float]
ROOM_X = (-1.55, 2.2)
STATION_PITCH = 2.6


@dataclass(frozen=True)
class RoomConfig:
    station_count: int = 2
    products_per_station: int = 4
    product_kind: str = "mixed"
    seed: int = 0

    def __post_init__(self) -> None:
        for value, label, lower, upper in (
            (self.station_count, "station_count", 1, 4),
            (self.products_per_station, "products_per_station", 1, 9),
            (self.seed, "seed", 0, 2**32 - 1),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{label} must be an integer in [{lower}, {upper}]")
        if self.product_kind not in ("mixed", "carton"):
            raise ValueError("product_kind must be mixed or carton")


@dataclass(frozen=True)
class RoomProduct:
    body: str
    kind: str
    mass_kg: float
    half_size_m: Vec3
    spawn_xyz: Vec3
    target_xyz: Vec3
    target_site: str
    target_support_geom: str


@dataclass(frozen=True)
class StockingStation:
    id: str
    bounds_xy: Bounds
    approach_xy_yaw: Vec3
    shelf_body: str
    supply_geom: str
    handoff_geom: str
    handoff_xy: tuple[float, float]
    handoff_half_size_xy: tuple[float, float]
    handoff_top_z: float
    products: tuple[RoomProduct, ...]


@dataclass(frozen=True)
class StockingRoom:
    config: RoomConfig
    bounds_xy: Bounds
    stations: tuple[StockingStation, ...]
    robot_spawn: str | None
    model_xml: str

    def manifest(self) -> dict:
        return {
            "schema_version": 1,
            "frame": "world",
            "config": asdict(self.config),
            "bounds_xy": self.bounds_xy,
            "stations": [asdict(station) for station in self.stations],
            "robot_spawn": self.robot_spawn,
        }


def build_stocking_room(
    config: RoomConfig = RoomConfig(), *, mabel: bool = False, mabel_root: Path | None = None
) -> StockingRoom:
    """Build a fixture with optional canonical MABEL at station zero."""
    cfg = ShelfConfig(
        levels=(0.64, 0.99),
        table_center_y=-0.83,
        product_count=config.products_per_station,
        product_kind=config.product_kind,
        table_size=(0.66, 0.54),
        target_back_gap=0.08,
    )
    products = products_for(cfg)
    # Start with the canonical world/robot and replace its one station with named copies.
    root = ET.fromstring(build_scene_xml(cfg, robot=mabel, mabel_root=mabel_root))
    root.set("model", "abstract_stack_stocking_room")
    world = root.find("worldbody")
    task_bodies = {"shelf", "supply", *(p.name for p in products)}
    templates = [body for body in world.findall("body") if body.get("name") in task_bodies]
    templates += [
        site for site in world.findall("site") if site.get("name", "").startswith("target_")
    ]
    for node in templates:
        world.remove(node)
    station_seeds = np.random.SeedSequence(config.seed).spawn(config.station_count)
    stations = tuple(
        _add_station(world, templates, cfg, products, i, seed)
        for i, seed in enumerate(station_seeds)
    )
    bounds = (*ROOM_X, -STATION_PITCH / 2, stations[-1].bounds_xy[3])
    _add_room_shell(world, bounds)
    ymin, ymax = bounds[2:]
    _camera(world, "room_overview", (5, ymin - 2, ymax - ymin + 2), (0, (ymin + ymax) / 2, 0))
    return StockingRoom(
        config,
        bounds,
        stations,
        "mabel_chassis" if mabel else None,
        ET.tostring(root, encoding="unicode"),
    )


def _add_station(world, templates, cfg, products, index, seed) -> StockingStation:
    station = f"station_{index}"
    prefix = f"{station}_"
    y = index * STATION_PITCH
    supply_positions = dict(
        zip((p.name for p in products), initial_positions(cfg, products, seed), strict=True)
    )
    for template in templates:
        node = ET.fromstring(ET.tostring(template))
        name = node.get("name")
        x, local_y, z = supply_positions.get(name, tuple(map(float, node.get("pos").split())))
        node.set("pos", _numbers((x, local_y + y, z)))
        for element in node.iter():
            if "name" in element.attrib:
                element.set("name", prefix + element.get("name"))
        world.append(node)
    handoff_xy = (0.5, y + 0.80)
    handoff_half_size = (0.26, 0.23)
    handoff_top = 0.70
    handoff_geom = f"{prefix}handoff_top"
    _box(
        world,
        handoff_geom,
        (*handoff_xy, handoff_top - 0.02),
        (*handoff_half_size, 0.02),
        (0.22, 0.51, 0.44, 1),
    )
    _box(
        world,
        f"{prefix}handoff_leg",
        (*handoff_xy, (handoff_top - 0.04) / 2),
        (0.075, 0.075, (handoff_top - 0.04) / 2),
        (0.20, 0.28, 0.26, 1),
    )
    ET.SubElement(
        world,
        "light",
        name=f"{prefix}light",
        pos=_numbers((0.75, y, 2.5)),
        dir="-0.2 0 -1",
        diffuse="0.5 0.5 0.5",
        castshadow="false",
    )
    _camera(world, f"{prefix}view", (1.65, y - 1.05, 1.75), (-0.52, y, 0.72))
    return StockingStation(
        id=station,
        bounds_xy=(ROOM_X[0], 1.05, y - STATION_PITCH / 2, y + STATION_PITCH / 2),
        approach_xy_yaw=(0.18, y, 0.0),
        shelf_body=f"{prefix}shelf",
        supply_geom=f"{prefix}supply_table",
        handoff_geom=handoff_geom,
        handoff_xy=handoff_xy,
        handoff_half_size_xy=handoff_half_size,
        handoff_top_z=handoff_top,
        products=tuple(
            RoomProduct(
                body=prefix + p.name,
                kind=p.kind,
                mass_kg=p.mass,
                half_size_m=p.half_size,
                spawn_xyz=tuple(float(v) for v in supply_positions[p.name] + (0.0, y, 0.0)),
                target_xyz=(p.target[0], p.target[1] + y, p.target[2]),
                target_site=f"{prefix}target_{p.name}",
                target_support_geom=f"{prefix}shelf_board_{p.level}",
            )
            for p in products
        ),
    )


def _add_room_shell(world, bounds: Bounds) -> None:
    xmin, xmax, ymin, ymax = bounds
    color = (0.84, 0.83, 0.76, 1)
    # The +X side is open for room ingress; the walkable shared aisle is +X.
    _box(
        world,
        "room_west_wall",
        (xmin, (ymin + ymax) / 2, 1.2),
        (0.05, (ymax - ymin) / 2, 1.2),
        color,
    )
    for label, y in (("south", ymin), ("north", ymax)):
        # Keep the walls physically solid. Only visibility may be cut away.
        _box(
            world,
            f"room_{label}_wall",
            ((xmin + xmax) / 2, y, 1.2),
            ((xmax - xmin) / 2, 0.05, 1.2),
            color,
            group="5",
        )
    _box(
        world,
        "room_floor_finish",
        ((xmin + xmax) / 2, (ymin + ymax) / 2, -0.0075),
        ((xmax - xmin) / 2, (ymax - ymin) / 2, 0.008),
        (0.62, 0.57, 0.48, 1),
    ).attrib.update(contype="0", conaffinity="0")
    _box(
        world,
        "room_aisle_stripe",
        (1.05, (ymin + ymax) / 2, -0.009),
        (0.012, (ymax - ymin) / 2, 0.01),
        (0.95, 0.74, 0.28, 1),
    ).attrib.update(contype="0", conaffinity="0")
