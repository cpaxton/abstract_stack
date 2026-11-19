"""Deterministic reset layouts for isolated native stocking skills."""

from dataclasses import dataclass, replace

import numpy as np

from abstract_stack.config import load_scene_config
from abstract_stack.scene import ShelfConfig, products_for


@dataclass(frozen=True)
class SkillFixture:
    config: ShelfConfig
    active: str
    # Reset-only poses as (x, y, z, yaw radians), in world coordinates.
    poses: dict[str, tuple[float, float, float, float]]
    supported_neighbors: tuple[str, ...] = ()
    direction: int | None = None
    intermediate_x_m: float | None = None


def sample_skill_fixture(skill, seed):
    """Sample the public development distribution without changing a live model."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Choose a nonnegative integer reset seed below 2**63")
    if skill == "row-compaction":
        skill = "row-press"
    rng = np.random.default_rng(seed ^ 0x528A0)
    base = replace(
        load_scene_config(scene="pick_place"),
        table_size=(0.42, 0.42),
        initial_joint_positions=(("Right Arm 3", 0.5),),
        time_limit=100.0,
        terminate_on_success=False,
    )
    if skill == "grasp-lift":
        config = replace(base, product_count=3, spawn_jitter=0.006)
        active = products_for(config)[-1]
        return SkillFixture(
            config,
            active.name,
            {
                active.name: (
                    active.spawn[0] + float(rng.uniform(-0.006, 0.006)),
                    active.spawn[1] + float(rng.uniform(-0.006, 0.006)),
                    active.spawn[2],
                    float(rng.uniform(-0.12, 0.12)),
                )
            },
        )
    if skill not in ("insert", "push-to-contact", "row-press"):
        raise ValueError(f"Unknown skill fixture: {skill}")

    # Keep both side panels in the physical model. Use ten potential slots so
    # the shelf/arm workspace is comparable to the fixed ten-carton benchmark.
    width = float(rng.uniform(0.049, 0.051))
    gap_budget = float(rng.uniform(0.015, 0.019))
    height = 0.24
    if skill == "row-press":
        size_rng = np.random.default_rng(seed ^ 0xA41E)
        height = float(size_rng.choice((0.18, 0.24)))
    config = replace(
        base,
        product_count=10,
        table_size=(0.42, 0.56),
        table_center_x=-0.35 if skill == "row-press" else -0.34,
        # Isolate shelf contact from inventory navigation in the row-press
        # task. The same unstocked cartons still determine the required span.
        table_center_y=-1.10 if skill == "row-press" else -0.62,
        time_limit=180.0 if skill == "row-press" else base.time_limit,
        carton_size=(0.07, width, height),
        width=10 * width + gap_budget,
        row_free_space=gap_budget,
        center_y=-0.125,
        robot_xy=(0.15, 0.0),
    )
    products = products_for(config)
    if skill == "row-press":
        direction = int(rng.choice((-1, 1)))
        # From the occupied wall toward the exposed end. Preserve canonical
        # slot identities so a pose change cannot earn placement by relabeling.
        ordered = products if direction < 0 else products[::-1]
        neighbors = ordered[: int(rng.integers(2, 5))]
        gap = float(rng.uniform(0.0008, 0.0015))
        deficit = float(rng.uniform(0.002, 0.004))
        wall = config.center_y + direction * config.width / 2
        wall_gap = gap_budget + deficit - (len(neighbors) - 1) * gap
        # Shelf depth is a separate task factor. Some rows can react against
        # the backstop. Others leave space behind and need an exposed-side push.
        depth_offset = float(rng.choice((-0.003, 0.020)) + rng.uniform(0.0005, 0.002))
        poses = {
            p.name: (
                p.target[0] + depth_offset,
                wall - direction * (p.half_size[1] + wall_gap + i * (width + gap)),
                p.target[2] + 0.001,
                0.0,
            )
            for i, p in enumerate(neighbors)
        }
        return SkillFixture(
            config, neighbors[-1].name, poses, tuple(p.name for p in neighbors), direction
        )

    index = int(rng.choice((3, 4, 5)))
    active = products[index]
    neighbors = (products[index - 1], products[index + 1])
    poses = {p.name: (*p.target[:2], p.target[2] + 0.001, 0.0) for p in neighbors}
    # Keep the active carton at a measured shelf staging pose ahead of the
    # neighboring cartons. Insert episodes start farther forward than presses.
    offset_x = float(rng.uniform(0.006, 0.012)) if skill == "insert" else 0.045
    yaw = float(rng.uniform(-0.025, 0.025)) if skill == "insert" else 0.0
    poses[active.name] = (
        config.front_x - active.half_size[0] - offset_x,
        active.target[1] + float(rng.uniform(-0.003, 0.003)),
        active.target[2] + 0.001,
        yaw,
    )
    intermediate = config.front_x - 0.076 if skill == "insert" else None
    return SkillFixture(
        config, active.name, poses, tuple(p.name for p in neighbors), None, intermediate
    )
