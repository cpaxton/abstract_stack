"""Procedural shelf MJCF, composed with MABEL's canonical full robot model."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from abstract_stack.assets import mabel_model_path


@dataclass(frozen=True)
class ShelfConfig:
    """Dimensions in metres; shelf faces +X and MABEL faces -X."""

    width: float = 0.90
    depth: float = 0.36
    front_x: float = -0.62
    levels: tuple[float, ...] = (0.65, 0.98, 1.31)
    board_thickness: float = 0.02
    table_top: float = 0.70
    top_clearance: float = 0.33
    table_size: tuple[float, float] = (0.60, 0.50)
    table_leg_inset: float = 0.04
    table_center_x: float = -0.30
    table_center_y: float | None = None
    supply_center_y: float | None = None
    center_y: float = 0.0
    product_count: int = 6
    initially_stocked: int = 0
    target_gap: float = 0.045
    spawn_jitter: float = 0.008
    robot_xy: tuple[float, float] = (0.0, 0.0)
    robot_yaw: float = 0.0
    initial_joint_positions: tuple[tuple[str, float], ...] = ()
    timestep: float = 0.002
    frame_skip: int = 10
    time_limit: float = 120.0
    stable_seconds: float = 0.5
    terminate_on_success: bool = True
    target_tolerance: float = 0.055
    enable_sensors: bool = False
    environment: str = "studio"
    product_kind: str = "mixed"
    target_back_gap: float | None = None
    carton_size: tuple[float, float, float] = (0.09, 0.07, 0.17)
    carton_masses_kg: tuple[float, ...] | None = None
    carton_support_friction: float | None = None
    row_free_space: float | None = None
    supply_order: tuple[int, ...] | None = None
    # Optional numerical contact recipe; None preserves the upstream defaults.
    contact_impratio: float | None = None
    rigid_fixture_contacts: bool = False
    scene_contact_time_constant_s: float | None = None

    def __post_init__(self) -> None:
        positive = (
            self.width,
            self.depth,
            self.board_thickness,
            self.table_top,
            self.timestep,
            self.time_limit,
            self.stable_seconds,
            self.target_tolerance,
            self.top_clearance,
            *self.table_size,
        )
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Shelf dimensions, times, and tolerances must be finite and positive")
        if type(self.rigid_fixture_contacts) is not bool:
            raise ValueError("rigid_fixture_contacts must be boolean")
        if self.rigid_fixture_contacts and (
            self.product_kind != "carton"
            or self.contact_impratio is not None
            or self.scene_contact_time_constant_s is not None
        ):
            raise ValueError(
                "Rigid fixture pairs require cartons and unchanged global/geom contacts"
            )
        if self.contact_impratio is not None and (
            isinstance(self.contact_impratio, bool)
            or not math.isfinite(self.contact_impratio)
            or not 1 <= self.contact_impratio <= 100
        ):
            raise ValueError("contact_impratio must be between 1 and 100")
        if self.scene_contact_time_constant_s is not None and (
            isinstance(self.scene_contact_time_constant_s, bool)
            or not math.isfinite(self.scene_contact_time_constant_s)
            or not 2 * self.timestep <= self.scene_contact_time_constant_s <= 0.05
        ):
            raise ValueError(
                "Scene contact time constant must resolve >=2 timesteps and be <=.05 s"
            )
        if not math.isfinite(self.front_x) or self.front_x >= -0.4:
            raise ValueError("Shelf front must be in front of MABEL (front_x < -0.4)")
        if (
            self.frame_skip < 1
            or not isinstance(self.frame_skip, int)
            or isinstance(self.frame_skip, bool)
        ):
            raise ValueError("frame_skip must be a positive integer")
        if not self.levels or any(not math.isfinite(z) or z < 0.25 for z in self.levels):
            raise ValueError("At least one shelf level >= 0.25 m is required")
        if any(
            b - a - self.board_thickness < 0.24 - 1e-9
            for a, b in zip(self.levels, self.levels[1:], strict=False)
        ):
            raise ValueError("Shelf levels must leave at least 0.24 m of clear space")
        if self.row_free_space is not None and (
            isinstance(self.row_free_space, bool)
            or not math.isfinite(self.row_free_space)
            or self.row_free_space <= 0
            or len(self.levels) != 1
            or self.product_kind != "carton"
            or self.product_count < 2
        ):
            raise ValueError("Dense rows need positive total free space, one level and >=2 cartons")
        minimum_width = 0.08 if self.row_free_space is not None else 0.5
        if self.width < minimum_width or self.depth < 0.2:
            raise ValueError(f"Shelf must be at least {minimum_width} m wide and 0.2 m deep")
        if len(self.table_size) != 2 or len(self.robot_xy) != 2:
            raise ValueError("table_size and robot_xy must each contain two numbers")
        if min(self.table_size) < 0.12:
            raise ValueError("Supply table dimensions must each be >= 0.12 m")
        if not math.isfinite(self.table_leg_inset) or not (
            0.02 <= self.table_leg_inset <= min(self.table_size) / 2 - 0.02 + 1e-12
        ):
            raise ValueError("table_leg_inset must leave distinct supported legs inside the table")
        coordinates = (*self.robot_xy, self.robot_yaw, self.center_y, self.table_center_x)
        if self.table_center_y is not None:
            coordinates += (self.table_center_y,)
        if self.supply_center_y is not None:
            coordinates += (self.supply_center_y,)
        if not all(math.isfinite(x) for x in coordinates):
            raise ValueError("Scene positions and angles must be finite")
        try:
            names = [name for name, value in self.initial_joint_positions]
            valid_initial = all(
                isinstance(name, str)
                and name
                and not isinstance(value, bool)
                and math.isfinite(value)
                for name, value in self.initial_joint_positions
            )
        except (ValueError, TypeError):
            valid_initial, names = False, []
        if not valid_initial or len(names) != len(set(names)):
            raise ValueError("Initial joint positions need unique names and finite scalar values")
        if self.top_clearance - self.board_thickness < 0.24 - 1e-9 or self.table_top < 0.10:
            raise ValueError(
                "Top shelf needs >= 0.24 m clear space and table_top must be >= 0.10 m"
            )
        if not isinstance(self.enable_sensors, bool):
            raise ValueError("enable_sensors must be true or false")
        if not isinstance(self.terminate_on_success, bool):
            raise ValueError("terminate_on_success must be true or false")
        if (
            not isinstance(self.product_count, int)
            or isinstance(self.product_count, bool)
            or self.product_count < 0
            or not isinstance(self.initially_stocked, int)
            or isinstance(self.initially_stocked, bool)
            or not 0 <= self.initially_stocked <= self.product_count
        ):
            raise ValueError(
                "Product counts must be integers with 0 <= initially_stocked <= product_count"
            )
        if not math.isfinite(self.target_gap) or self.target_gap < 0:
            raise ValueError("target_gap must be finite and nonnegative")
        if not math.isfinite(self.spawn_jitter) or not 0 <= self.spawn_jitter <= 0.02:
            raise ValueError("spawn_jitter must be between 0 and 0.02 m")
        if self.environment not in ("studio", "small_house"):
            raise ValueError("environment must be 'studio' or 'small_house'")
        if self.product_kind not in ("mixed", "carton"):
            raise ValueError("product_kind must be 'mixed' or 'carton'")
        carton_count = (
            self.product_count if self.product_kind == "carton" else (self.product_count + 1) // 2
        )
        masses = self.carton_masses_kg
        if masses is not None and (
            not isinstance(masses, (tuple, list))
            or len(masses) != carton_count
            or any(
                type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0
                for mass in masses
            )
        ):
            raise ValueError(
                "carton_masses_kg needs one positive finite mass per carton, in ID order"
            )
        friction = self.carton_support_friction
        if friction is not None and (
            type(friction) not in (int, float) or not math.isfinite(friction) or friction <= 0
        ):
            raise ValueError("carton_support_friction must be finite and positive")
        if len(self.carton_size) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.carton_size
        ):
            raise ValueError("carton_size must contain three positive finite full dimensions")
        if self.target_back_gap is not None and (
            not math.isfinite(self.target_back_gap)
            or self.target_back_gap < 0
            or self.target_back_gap + self.carton_size[0] > self.depth
        ):
            raise ValueError("target_back_gap must leave room for a carton inside the shelf")
        if self.environment == "small_house":
            from abstract_stack.house import validate_house_placement

            validate_house_placement(self)

    @property
    def table_xy(self) -> tuple[float, float]:
        y = self.table_center_y
        if y is None:
            # Keep the supply table beside the shelf as its width changes.
            y = (
                self.center_y
                - self.width / 2
                - self.board_thickness
                - self.table_size[1] / 2
                - 0.06
            )
        return self.table_center_x, y

    @property
    def supply_xy(self) -> tuple[float, float]:
        x, y = self.table_xy
        return x, y if self.supply_center_y is None else self.supply_center_y


@dataclass(frozen=True)
class Product:
    name: str
    kind: str
    half_size: tuple[float, float, float]
    mass: float
    color: tuple[float, float, float, float]
    level: int
    target: tuple[float, float, float]
    spawn: tuple[float, float, float]
    initially_stocked: bool = False


def products_for(config: ShelfConfig) -> tuple[Product, ...]:
    products = []
    colors = ((0.89, 0.35, 0.17, 1), (0.15, 0.60, 0.77, 1), (0.91, 0.72, 0.20, 1))
    count = config.product_count
    # Leave every third horizontal slot for insertion. Consecutive gap columns
    # alternate product type instead of selecting only cartons or only bottles.
    fill_order = sorted(
        range(count),
        key=lambda i: (
            (i // len(config.levels)) % 3 == 1,
            i // len(config.levels),
            i % len(config.levels),
        ),
    )
    prefilled = set(fill_order[: config.initially_stocked])
    supply_count = count - config.initially_stocked
    columns = min(3, supply_count)
    rows = math.ceil(supply_count / columns) if columns else 0
    max_x = max(config.carton_size[0], 0.068 if config.product_kind == "mixed" else 0)
    max_y = max(config.carton_size[1], 0.068 if config.product_kind == "mixed" else 0)
    stride_x = max(0.15, max_x + 2 * config.spawn_jitter + 0.02)
    stride_y = max(0.14, max_y + 2 * config.spawn_jitter + 0.02)
    if supply_count and (
        (columns - 1) * stride_x + max_x + 2 * config.spawn_jitter > config.table_size[0]
        or (rows - 1) * stride_y + max_y + 2 * config.spawn_jitter > config.table_size[1]
        or abs(config.supply_xy[1] - config.table_xy[1])
        + ((rows - 1) * stride_y + max_y + 2 * config.spawn_jitter) / 2
        > config.table_size[1] / 2
    ):
        raise ValueError(
            "Supply products do not fit on the table; enlarge table_size or prefill more items"
        )
    supply_order = (
        sorted(set(range(count)) - prefilled)
        if config.supply_order is None
        else config.supply_order
    )
    if (
        any(type(index) is not int for index in supply_order)
        or len(supply_order) != supply_count
        or set(supply_order) != set(range(count)) - prefilled
    ):
        raise ValueError("Supply order must list each non-prefilled product index once")
    supply_index_by_product = {index: rank for rank, index in enumerate(supply_order)}
    for i in range(count):
        bottle = config.product_kind == "mixed" and i % 2 == 1
        half_size = (0.034, 0.034, 0.11) if bottle else tuple(v / 2 for v in config.carton_size)
        level = i % len(config.levels)
        upper_board = (
            config.levels[level + 1]
            if level + 1 < len(config.levels)
            else config.levels[level] + config.top_clearance
        )
        if upper_board - config.levels[level] - config.board_thickness < 2 * half_size[2] + 0.005:
            raise ValueError("Product height does not fit between shelf boards")
        row = list(range(level, count, len(config.levels)))
        widths = [
            0.068 if config.product_kind == "mixed" and j % 2 else config.carton_size[1]
            for j in row
        ]
        gap = config.target_gap
        if config.row_free_space is not None:
            if not math.isclose(config.width - sum(widths), config.row_free_space, abs_tol=1e-9):
                raise ValueError("Shelf width minus all carton widths must equal row_free_space")
            # N+1 gaps include both walls. This is a whole-row budget, not a
            # per-product clearance and not a gap to add repeatedly.
            gap = config.row_free_space / (len(row) + 1)
        used_width = sum(widths) + gap * (len(row) - 1)
        reserve = 0.02 if config.row_free_space is None else 0.0
        if used_width > config.width - reserve:
            raise ValueError("Products do not fit across the configured shelf")
        slot = row.index(i)
        target_y = (
            config.center_y - used_width / 2 + sum(widths[:slot]) + gap * slot + widths[slot] / 2
        )
        target = (
            config.front_x - config.depth / 2
            if config.target_back_gap is None
            else config.front_x - config.depth + half_size[0] + config.target_back_gap,
            target_y,
            config.levels[level] + half_size[2],
        )
        if i in prefilled:
            spawn = (target[0], target[1], target[2] + 0.003)
        else:
            supply_index = supply_index_by_product[i]
            spawn = (
                config.supply_xy[0] + (supply_index % columns - (columns - 1) / 2) * stride_x,
                config.supply_xy[1] + (supply_index // columns - (rows - 1) / 2) * stride_y,
                config.table_top + half_size[2] + 0.003,
            )
        products.append(
            Product(
                name=(
                    f"{'bottle' if bottle else 'carton'}_"
                    f"{i // 2 if config.product_kind == 'mixed' else i}"
                ),
                kind="bottle" if bottle else "carton",
                half_size=half_size,
                mass=(
                    0.25
                    if bottle
                    else 0.18
                    if config.carton_masses_kg is None
                    else config.carton_masses_kg[i // 2 if config.product_kind == "mixed" else i]
                ),
                color=colors[(i // 2 if config.product_kind == "mixed" else i) % len(colors)],
                level=level,
                target=target,
                spawn=spawn,
                initially_stocked=i in prefilled,
            )
        )
    return tuple(products)


def initial_positions(config: ShelfConfig, products: tuple[Product, ...], seed: int) -> np.ndarray:
    """Use the same seeded initial state for MJCF export and simulator reset."""
    positions = np.array([p.spawn for p in products], dtype=float).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    for i, product in enumerate(products):
        if not product.initially_stocked:
            positions[i, :2] += rng.uniform(-config.spawn_jitter, config.spawn_jitter, size=2)
    return positions


def _numbers(values) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _box(parent, name, pos, size, color, **attributes):
    return ET.SubElement(
        parent,
        "geom",
        name=name,
        type="box",
        pos=_numbers(pos),
        size=_numbers(size),
        rgba=_numbers(color),
        contype="1",
        conaffinity="1",
        friction="0.8 0.02 0.002",
        **attributes,
    )


def _carton_contact_pairs(root, shelf, config, products):
    friction = config.carton_support_friction
    if friction is None and not config.rigid_fixture_contacts:
        return
    # Explicit pairs set effective friction even when it is below the default
    # of the two geoms. Only load-bearing boards and the tabletop vary.
    supports = {f"shelf_board_{i}" for i in range(len(config.levels))}
    pairs = {}
    if config.rigid_fixture_contacts:
        for geom in shelf.findall("geom"):
            fixture = geom.get("name")
            sliding = friction if fixture in supports and friction is not None else 0.8
            pairs[fixture] = dict(
                friction=_numbers((sliding, sliding, 0.02, 0.002, 0.002)),
                solref="0.005 1",
                solimp="0.99 0.999 0.001 0.5 2",
            )
    if friction is not None:
        for fixture in sorted(supports | {"supply_table"}):
            pairs.setdefault(
                fixture,
                dict(
                    friction=_numbers((friction, friction, 0.02, 0.002, 0.002)),
                    solref=f"{config.scene_contact_time_constant_s or 0.02} 1",
                    solimp="0.9 0.95 0.001 0.5 2",
                ),
            )
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for product in products:
        if product.kind != "carton":
            continue
        for fixture, attributes in pairs.items():
            ET.SubElement(
                contact,
                "pair",
                geom1=f"{product.name}_geom",
                geom2=fixture,
                condim="3",
                **attributes,
            )


def _camera(world, name, pos, look_at):
    z = np.asarray(pos, dtype=float) - np.asarray(look_at)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(
        world,
        "camera",
        name=name,
        pos=_numbers(pos),
        xyaxes=_numbers((*x, *y)),
        fovy="55",
    )


def build_scene_xml(
    config: ShelfConfig | None = None,
    *,
    robot: bool = True,
    mabel_root: str | Path | None = None,
    seed: int = 0,
) -> str:
    """Return self-contained MJCF with absolute mesh paths and no upstream edits."""
    config = config or ShelfConfig()
    if robot:
        model_path = mabel_model_path(mabel_root)
        root = ET.parse(model_path).getroot()
        # Asset paths in the upstream model are relative to models/mabel_full.xml.
        for element in root.iter():
            if "file" in element.attrib:
                element.set("file", str((model_path.parent / element.get("file")).resolve()))
        chassis = root.find("./worldbody/body[@name='mabel_chassis']")
        if chassis is None:
            raise ValueError("Expected the canonical MABEL model with body 'mabel_chassis'")
        height = float(chassis.get("pos", "0 0 0").split()[2])
        chassis.set("pos", _numbers((*config.robot_xy, height)))
        for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
            chassis.attrib.pop(attribute, None)
        chassis.set(
            "quat", _numbers((math.cos(config.robot_yaw / 2), 0, 0, math.sin(config.robot_yaw / 2)))
        )
    else:
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "option", integrator="implicitfast", cone="elliptic")
    root.set("model", "abstract_stack_shelf")
    option = root.find("option")
    option.set("timestep", str(config.timestep))
    option.set("gravity", "0 0 -9.81")
    if config.contact_impratio is not None:
        option.set("impratio", str(config.contact_impratio))
        option.set("cone", "elliptic")
    if not config.enable_sensors:
        flag = option.find("flag")
        if flag is None:
            flag = ET.SubElement(option, "flag")
        flag.set("sensor", "disable")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="shelf_key",
        pos="-1 -2 4",
        dir="0 0 -1",
        ambient="0.25 0.25 0.25",
        diffuse="0.75 0.75 0.75",
    )
    ET.SubElement(
        world,
        "light",
        name="shelf_fill",
        pos="1 2 3",
        dir="-1 -1 -1",
        diffuse="0.5 0.5 0.5",
        castshadow="false",
    )
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    if not any(t.get("type") == "skybox" for t in assets.findall("texture")):
        ET.SubElement(
            assets,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.72 0.80 0.88",
            rgb2="0.94 0.96 0.98",
            width="512",
            height="512",
        )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="4 4 0.05",
        rgba="0.22 0.25 0.28 1",
        contype="1",
        conaffinity="1",
        friction="1.5 0.05 0.001",
    )
    board = (0.70, 0.63, 0.52, 1)
    frame = (0.25, 0.31, 0.35, 1)
    shelf = ET.SubElement(
        world, "body", name="shelf", pos=_numbers((config.front_x, config.center_y, 0))
    )
    x = -config.depth / 2
    half_t = config.board_thickness / 2
    roof = config.levels[-1] + config.top_clearance
    for i, height in enumerate((*config.levels, roof)):
        _box(
            shelf,
            f"shelf_board_{i}",
            (x, 0, height - half_t),
            (config.depth / 2, config.width / 2, half_t),
            board,
        )
    for side in (-1, 1):
        _box(
            shelf,
            f"shelf_side_{side}",
            (x, side * (config.width / 2 + half_t), roof / 2),
            (config.depth / 2, half_t, roof / 2),
            frame,
        )
    _box(
        shelf,
        "shelf_back",
        (-config.depth - half_t, 0, roof / 2),
        (half_t, config.width / 2 + config.board_thickness, roof / 2),
        frame,
    )
    table = ET.SubElement(world, "body", name="supply", pos=_numbers((*config.table_xy, 0)))
    table_half_x, table_half_y = (s / 2 for s in config.table_size)
    _box(
        table,
        "supply_table",
        (0, 0, config.table_top - 0.02),
        (table_half_x, table_half_y, 0.02),
        board,
    )
    for dx in (-table_half_x + config.table_leg_inset, table_half_x - config.table_leg_inset):
        for dy in (-table_half_y + config.table_leg_inset, table_half_y - config.table_leg_inset):
            _box(
                table,
                f"table_leg_{dx}_{dy}",
                (dx, dy, (config.table_top - 0.04) / 2),
                (0.02, 0.02, (config.table_top - 0.04) / 2),
                frame,
            )
    products = products_for(config)
    for product, pos in zip(products, initial_positions(config, products, seed), strict=True):
        body = ET.SubElement(world, "body", name=product.name, pos=_numbers(pos))
        ET.SubElement(body, "freejoint", name=f"{product.name}_joint")
        if product.kind == "carton":
            _box(
                body,
                f"{product.name}_geom",
                (0, 0, 0),
                product.half_size,
                product.color,
                mass=str(product.mass),
            )
        else:
            # One rigid body with cylindrical body, neck and cap; half_size bounds all three.
            radius = product.half_size[0]
            for suffix, z, r, half_height, mass in (
                ("geom", -0.02, radius, 0.09, product.mass * 0.9),
                ("neck", 0.08, radius * 0.48, 0.01, product.mass * 0.07),
                ("cap", 0.10, radius * 0.55, 0.01, product.mass * 0.03),
            ):
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{product.name}_{suffix}",
                    type="cylinder",
                    pos=f"0 0 {z}",
                    size=_numbers((r, half_height)),
                    mass=str(mass),
                    rgba=_numbers(product.color if suffix != "cap" else (0.9, 0.9, 0.9, 1)),
                    contype="1",
                    conaffinity="1",
                    condim="4",
                    friction="0.8 0.02 0.002",
                )
        ET.SubElement(
            world,
            "site",
            name=f"target_{product.name}",
            type="box",
            pos=_numbers(
                (product.target[0], product.target[1], config.levels[product.level] + 0.002)
            ),
            size=_numbers((product.half_size[0], product.half_size[1], 0.001)),
            rgba=_numbers((*product.color[:3], 0.3)),
            group="4",
        )
    _carton_contact_pairs(root, shelf, config, products)
    if config.scene_contact_time_constant_s is not None:
        # Apply only to our generated rigid objects/fixtures. Native MABEL
        # contact parameters, geometry, inertias, gains and masks stay intact.
        prefixes = ("shelf_", "supply_", *(p.name + "_" for p in products))
        for geom in world.iter("geom"):
            if geom.get("name", "").startswith(prefixes):
                geom.set("solref", f"{config.scene_contact_time_constant_s} 1")
    center = (config.front_x - config.depth / 2, config.center_y, roof / 2)
    _camera(world, "overview", (1.7, config.center_y - 2.5, roof + 0.5), center)
    _camera(
        world, "stocking_third_person", (1.25, config.center_y - 1.8, 1.7), (-0.30, -0.18, 0.82)
    )
    if config.row_free_space is not None:
        _camera(
            world,
            "dense_front",
            (
                config.front_x + 0.28,
                config.center_y,
                config.levels[0] + config.carton_size[2] * 0.65,
            ),
            (
                config.front_x - config.depth + 0.04,
                config.center_y,
                config.levels[0] + config.carton_size[2] / 2,
            ),
        )
    # An oblique front view keeps the shelf visible around MABEL's torso.
    _camera(
        world, "shelf_front", (config.front_x + 1.35, config.center_y + 1.10, roof * 0.8), center
    )
    tx, ty = config.table_xy
    _camera(
        world,
        "supply_overhead",
        (tx, ty - 0.001, config.table_top + 1.6),
        (tx, ty, config.table_top),
    )
    if config.environment == "small_house":
        from abstract_stack.house import add_small_house

        add_small_house(root)
        root.find("./worldbody/camera[@name='overview']").set("name", "task_overview")
        _camera(world, "overview", (5.8, -7.2, 6.6), (0, 0, 0.5))
        _camera(world, "house_overhead", (0, -0.001, 8.2), (0, 0, 0))
        _camera(world, "kitchen", (0.15, -2.35, 1.95), (-2.2, -0.65, 0.85))
        _camera(world, "living_room", (-0.20, -0.70, 1.8), (-0.8, 1.7, 0.65))
        _camera(world, "bedroom", (0.9, -0.6, 1.8), (2.1, 1.5, 0.6))
        _camera(world, "bathroom", (1.1, -1.3, 1.8), (2.2, -2.3, 0.5))
    return ET.tostring(root, encoding="unicode")
