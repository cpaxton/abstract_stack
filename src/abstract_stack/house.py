"""Small furnished house built from MuJoCo primitives; no external scene assets."""

from __future__ import annotations

import xml.etree.ElementTree as ET

HOUSE_BOUNDS = (-3.0, 3.0, -2.8, 2.8)
WALL_HEIGHT = 2.6
CUTAWAY_GROUP = 5
CUTAWAY_CAMERAS = ("overview", "house_overhead")


def house_layout() -> dict:
    """World-frame rectangles describe rooms, not a navigation or reachability map."""
    return {
        "kind": "small_house",
        "bounds_xy": list(HOUSE_BOUNDS),
        "wall_height": WALL_HEIGHT,
        "wall_thickness": 0.12,
        "rooms": [
            {"name": "kitchen", "bounds_xy": [-3.0, 0.55, -2.8, 0.35]},
            {"name": "living_room", "bounds_xy": [-3.0, 0.55, 0.35, 2.8]},
            {"name": "bedroom", "bounds_xy": [0.55, 3.0, -1.0, 2.8]},
            {"name": "bathroom", "bounds_xy": [0.55, 3.0, -2.8, -1.0]},
        ],
        "doorways": [
            {"name": "entrance", "axis": "x", "range": [-0.75, 0.35], "y": -2.8},
            {"name": "bedroom", "axis": "y", "range": [-0.65, 0.45], "x": 0.55},
            {"name": "bathroom", "axis": "x", "range": [1.0, 2.0], "y": -1.0},
        ],
        "doorway_height": 2.15,
        "cutaway_geom_group": CUTAWAY_GROUP,
        "furniture_dynamic": False,
    }


def validate_house_placement(config) -> None:
    """Reject task fixtures or base footprints outside the house's interior."""
    xmin, xmax, ymin, ymax = HOUSE_BOUNDS
    tx, ty = config.table_xy
    footprints = {
        "shelf": (
            config.front_x - config.depth - config.board_thickness,
            config.front_x,
            config.center_y - config.width / 2 - config.board_thickness,
            config.center_y + config.width / 2 + config.board_thickness,
        ),
        "supply table": (
            tx - config.table_size[0] / 2,
            tx + config.table_size[0] / 2,
            ty - config.table_size[1] / 2,
            ty + config.table_size[1] / 2,
        ),
        "robot spawn": (
            config.robot_xy[0] - 0.30,
            config.robot_xy[0] + 0.30,
            config.robot_xy[1] - 0.30,
            config.robot_xy[1] + 0.30,
        ),
    }
    for name, (left, right, bottom, top) in footprints.items():
        if left < xmin + 0.06 or right > xmax - 0.06 or bottom < ymin + 0.06 or top > ymax - 0.06:
            raise ValueError(f"The {name} must fit inside the small house's exterior walls")
    if config.levels[-1] + config.top_clearance >= WALL_HEIGHT:
        raise ValueError("The shelf must fit below the small house's 2.6 m ceiling")


def _numbers(values) -> str:
    return " ".join(f"{value:.8g}" for value in values)


def _body(world, name, pos):
    return ET.SubElement(world, "body", name=f"house_{name}", pos=_numbers(pos))


def _box(parent, name, pos, size, color, *, group=0, collision=True, **attrs):
    return ET.SubElement(
        parent,
        "geom",
        name=f"house_{name}",
        type="box",
        pos=_numbers(pos),
        size=_numbers(size),
        rgba=_numbers(color),
        group=str(group),
        contype="1" if collision else "0",
        conaffinity="1" if collision else "0",
        friction="0.8 0.02 0.002",
        **attrs,
    )


def _cylinder(parent, name, pos, size, color, **attrs):
    return ET.SubElement(
        parent,
        "geom",
        name=f"house_{name}",
        type="cylinder",
        pos=_numbers(pos),
        size=_numbers(size),
        rgba=_numbers(color),
        contype="1",
        conaffinity="1",
        **attrs,
    )


def _wall(parent, name, axis, start, end, fixed, *, cutaway=True):
    """Full-height physical wall; split its upper portion for preview cutaways."""
    half = (end - start) / 2
    center = (start + end) / 2
    for part, z, half_height, group in (
        ("lower", 0.225, 0.225, 0),
        (
            "upper",
            (WALL_HEIGHT + 0.45) / 2,
            (WALL_HEIGHT - 0.45) / 2,
            CUTAWAY_GROUP if cutaway else 0,
        ),
    ):
        pos = (center, fixed, z) if axis == "x" else (fixed, center, z)
        size = (half, 0.06, half_height) if axis == "x" else (0.06, half, half_height)
        _box(parent, f"{name}_{part}", pos, size, (0.88, 0.86, 0.80, 1), group=group)


def _lintel(parent, name, axis, start, end, fixed):
    center, half = (start + end) / 2, (end - start) / 2
    z, hz = (WALL_HEIGHT + 2.15) / 2, (WALL_HEIGHT - 2.15) / 2
    pos = (center, fixed, z) if axis == "x" else (fixed, center, z)
    size = (half, 0.06, hz) if axis == "x" else (0.06, half, hz)
    _box(parent, name, pos, size, (0.88, 0.86, 0.80, 1), group=CUTAWAY_GROUP)


def _table(world, name, xy, half_size, height, color):
    table = _body(world, name, (*xy, 0))
    sx, sy = half_size
    _box(table, f"{name}_top", (0, 0, height - 0.025), (sx, sy, 0.025), color)
    for i, x in enumerate((-sx + 0.05, sx - 0.05)):
        for j, y in enumerate((-sy + 0.05, sy - 0.05)):
            _box(
                table,
                f"{name}_leg_{i}_{j}",
                (x, y, (height - 0.05) / 2),
                (0.025, 0.025, (height - 0.05) / 2),
                (0.25, 0.20, 0.16, 1),
            )
    return table


def add_small_house(root) -> None:
    """Add fixed architecture/furniture. Stocking objects are added by the task builder."""
    world, assets = root.find("worldbody"), root.find("asset")
    ET.SubElement(
        assets,
        "texture",
        name="house_wood",
        type="2d",
        builtin="checker",
        rgb1="0.57 0.43 0.30",
        rgb2="0.63 0.49 0.35",
        width="128",
        height="128",
    )
    ET.SubElement(
        assets,
        "material",
        name="house_floor_material",
        texture="house_wood",
        texrepeat="12 12",
        texuniform="true",
        reflectance="0.05",
    )
    shell = _body(world, "shell", (0, 0, 0))
    # Visible slab/rugs do not duplicate the existing z=0 physics ground plane.
    _box(
        shell,
        "floor_finish",
        (0, 0, -0.039),
        (3.0, 2.8, 0.04),
        (1, 1, 1, 1),
        collision=False,
        material="house_floor_material",
    )
    _wall(shell, "west_wall", "y", -2.8, 2.8, -3.0, cutaway=False)
    _wall(shell, "north_wall", "x", -3.0, 3.0, 2.8, cutaway=False)
    _wall(shell, "east_wall", "y", -2.8, 2.8, 3.0)
    _wall(shell, "south_wall_left", "x", -3.0, -0.75, -2.8)
    _wall(shell, "south_wall_right", "x", 0.35, 3.0, -2.8)
    _lintel(shell, "entrance_lintel", "x", -0.75, 0.35, -2.8)
    _wall(shell, "partition_south", "y", -2.8, -0.65, 0.55)
    _wall(shell, "partition_north", "y", 0.45, 2.8, 0.55)
    _lintel(shell, "bedroom_lintel", "y", -0.65, 0.45, 0.55)
    _wall(shell, "bath_partition_left", "x", 0.55, 1.0, -1.0)
    _wall(shell, "bath_partition_right", "x", 2.0, 3.0, -1.0)
    _lintel(shell, "bathroom_lintel", "x", 1.0, 2.0, -1.0)
    _box(
        shell, "ceiling", (0, 0, 2.64), (3.0, 2.8, 0.04), (0.96, 0.96, 0.94, 1), group=CUTAWAY_GROUP
    )
    # Windows are decorative surfaces on solid walls, not traversable openings.
    for i, x in enumerate((-1.4, 1.9)):
        _box(
            shell, f"window_{i}_frame", (x, 2.726, 1.65), (0.57, 0.015, 0.48), (0.95, 0.93, 0.87, 1)
        )
        _box(
            shell, f"window_{i}_glass", (x, 2.705, 1.65), (0.51, 0.008, 0.42), (0.48, 0.72, 0.86, 1)
        )
        _box(
            shell,
            f"window_{i}_divider",
            (x, 2.69, 1.65),
            (0.018, 0.01, 0.42),
            (0.95, 0.93, 0.87, 1),
        )

    # Kitchen counter, sink and hob. The separate task shelf/table stay movable in configuration.
    kitchen = _body(world, "kitchen_counter", (-2.61, 0.55, 0))
    _box(kitchen, "counter_base", (0, 0, 0.43), (0.29, 0.72, 0.43), (0.30, 0.43, 0.40, 1))
    _box(kitchen, "counter_top", (0, 0, 0.88), (0.32, 0.75, 0.025), (0.84, 0.82, 0.75, 1))
    for i, y in enumerate((-0.48, 0, 0.48)):
        _box(
            kitchen,
            f"cupboard_door_{i}",
            (0.298, y, 0.43),
            (0.012, 0.225, 0.37),
            (0.37, 0.52, 0.48, 1),
        )
        _box(
            kitchen,
            f"cupboard_handle_{i}",
            (0.32, y, 0.58),
            (0.018, 0.07, 0.008),
            (0.72, 0.73, 0.71, 1),
        )
    _box(kitchen, "sink_rim", (0, -0.30, 0.912), (0.20, 0.21, 0.009), (0.63, 0.68, 0.69, 1))
    _box(kitchen, "sink_basin", (0, -0.30, 0.922), (0.16, 0.17, 0.004), (0.20, 0.27, 0.29, 1))
    _cylinder(kitchen, "tap_stem", (-0.20, -0.30, 1.02), (0.018, 0.12), (0.7, 0.73, 0.73, 1))
    _box(kitchen, "tap_spout", (-0.13, -0.30, 1.13), (0.09, 0.018, 0.018), (0.7, 0.73, 0.73, 1))
    _box(kitchen, "hob", (0, 0.37, 0.914), (0.24, 0.24, 0.008), (0.10, 0.13, 0.14, 1))
    for i, y in enumerate((0.25, 0.51)):
        _cylinder(kitchen, f"hob_ring_{i}", (0, y, 0.925), (0.085, 0.003), (0.28, 0.32, 0.34, 1))
    fridge = _body(world, "fridge", (-2.57, 1.84, 0))
    _box(fridge, "fridge_case", (0, 0, 0.90), (0.32, 0.35, 0.90), (0.80, 0.83, 0.81, 1))
    for name, z, height in (("freezer", 1.48, 0.29), ("fridge_door", 0.60, 0.57)):
        _box(fridge, name, (0.331, 0, z), (0.012, 0.32, height), (0.9, 0.91, 0.88, 1))
        _box(fridge, f"{name}_handle", (0.36, -0.23, z), (0.02, 0.012, 0.12), (0.24, 0.28, 0.29, 1))

    # Living area, leaving the central approach to both internal doorways unobstructed.
    sofa = _body(world, "sofa", (-0.78, 2.14, 0))
    _box(sofa, "sofa_plinth", (0, 0, 0.20), (0.94, 0.38, 0.20), (0.19, 0.26, 0.31, 1))
    _box(sofa, "sofa_back", (0, 0.32, 0.59), (0.94, 0.10, 0.31), (0.25, 0.39, 0.44, 1))
    for i, x in enumerate((-0.88, 0.88)):
        _box(sofa, f"sofa_arm_{i}", (x, 0, 0.49), (0.09, 0.38, 0.21), (0.25, 0.39, 0.44, 1))
    for i, x in enumerate((-0.40, 0.40)):
        _box(sofa, f"sofa_cushion_{i}", (x, -0.03, 0.43), (0.37, 0.28, 0.07), (0.39, 0.54, 0.57, 1))
    rug = _body(world, "rug", (-0.73, 1.06, 0))
    _box(rug, "rug", (0, 0, 0.003), (1.03, 0.62, 0.002), (0.72, 0.66, 0.53, 1), collision=False)
    coffee = _table(world, "coffee_table", (-0.75, 0.94), (0.50, 0.26), 0.42, (0.54, 0.35, 0.21, 1))
    for i, (x, color) in enumerate(((-0.12, (0.75, 0.29, 0.16, 1)), (0, (0.82, 0.68, 0.36, 1)))):
        _box(coffee, f"book_{i}", (x, 0, 0.435 + 0.03 * i), (0.13, 0.095, 0.015), color)
    television = _body(world, "television", (0.29, 1.15, 0))
    _box(television, "tv_console", (0, 0, 0.24), (0.17, 0.53, 0.24), (0.40, 0.28, 0.20, 1))
    _box(television, "tv_frame", (0.03, 0, 1.10), (0.035, 0.48, 0.31), (0.08, 0.10, 0.11, 1))
    _box(television, "tv_screen", (-0.009, 0, 1.10), (0.005, 0.44, 0.27), (0.18, 0.31, 0.38, 1))

    bed = _body(world, "bed", (2.13, 1.55, 0))
    _box(bed, "bed_frame", (0, 0, 0.20), (0.72, 0.98, 0.20), (0.47, 0.32, 0.22, 1))
    _box(bed, "mattress", (0, 0, 0.46), (0.69, 0.94, 0.10), (0.94, 0.91, 0.84, 1))
    _box(bed, "duvet", (0, -0.24, 0.57), (0.69, 0.68, 0.035), (0.60, 0.42, 0.30, 1))
    _box(bed, "headboard", (0, 0.96, 0.50), (0.74, 0.045, 0.50), (0.47, 0.32, 0.22, 1))
    for i, x in enumerate((-0.34, 0.34)):
        _box(bed, f"pillow_{i}", (x, 0.67, 0.60), (0.28, 0.20, 0.055), (0.93, 0.88, 0.75, 1))
    nightstand = _body(world, "nightstand", (1.00, 2.18, 0))
    _box(nightstand, "nightstand", (0, 0, 0.25), (0.22, 0.25, 0.25), (0.52, 0.37, 0.25, 1))
    _cylinder(nightstand, "lamp_base", (0, 0, 0.52), (0.08, 0.02), (0.3, 0.29, 0.25, 1))
    _cylinder(nightstand, "lamp_stem", (0, 0, 0.67), (0.015, 0.15), (0.3, 0.29, 0.25, 1))
    _cylinder(nightstand, "lamp_shade", (0, 0, 0.83), (0.12, 0.10), (0.92, 0.80, 0.56, 1))

    bathroom = _body(world, "bathroom", (0, 0, 0))
    _box(
        bathroom,
        "bath_tiles",
        (1.775, -1.93, 0.003),
        (1.15, 0.80, 0.002),
        (0.73, 0.81, 0.80, 1),
        collision=False,
    )
    _box(bathroom, "vanity", (1.05, -2.46, 0.40), (0.30, 0.24, 0.40), (0.54, 0.63, 0.59, 1))
    _box(bathroom, "basin", (1.05, -2.46, 0.84), (0.32, 0.26, 0.04), (0.91, 0.93, 0.89, 1))
    _box(bathroom, "basin_inset", (1.05, -2.46, 0.884), (0.22, 0.16, 0.004), (0.66, 0.76, 0.77, 1))
    _box(bathroom, "mirror", (1.05, -2.72, 1.42), (0.30, 0.012, 0.36), (0.58, 0.73, 0.77, 1))
    _box(bathroom, "shower_tray", (2.30, -2.37, 0.04), (0.49, 0.32, 0.04), (0.88, 0.91, 0.88, 1))
    _cylinder(bathroom, "shower_drain", (2.30, -2.37, 0.082), (0.045, 0.002), (0.3, 0.34, 0.34, 1))
    _cylinder(bathroom, "toilet_pedestal", (2.52, -1.56, 0.20), (0.16, 0.20), (0.91, 0.93, 0.89, 1))
    _cylinder(bathroom, "toilet_seat", (2.52, -1.56, 0.43), (0.22, 0.035), (0.95, 0.96, 0.92, 1))
    _box(bathroom, "toilet_cistern", (2.77, -1.56, 0.51), (0.11, 0.23, 0.29), (0.91, 0.93, 0.89, 1))

    for name, pos in (
        ("kitchen", (-1.7, -1.2, 2.45)),
        ("living", (-0.8, 1.2, 2.45)),
        ("bedroom", (1.8, 1.0, 2.45)),
        ("bathroom", (1.8, -1.9, 2.45)),
    ):
        ET.SubElement(
            world,
            "light",
            name=f"house_{name}_light",
            pos=_numbers(pos),
            dir="0 0 -1",
            diffuse="0.65 0.62 0.56",
            ambient="0.15 0.15 0.15",
            castshadow="false",
        )
