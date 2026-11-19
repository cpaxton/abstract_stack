"""Dense row geometry and read-only packing measurements, separate from control."""

from dataclasses import replace

import numpy as np

from abstract_stack.random_scenes import random_shelf_config
from abstract_stack.scene import products_for


def dense_shelf_config(seed, *, product_count=10, total_free_space=0.01, initially_stocked=0):
    """Compact carton shelf with a coplanar transfer table in front of its opening.

    All lateral wall/inter-product gaps SHARE total_free_space. The table is an
    explicit staging aid, not evidence of insertion without such a surface.
    """
    if (
        isinstance(total_free_space, bool)
        or not np.isfinite(total_free_space)
        or not (0 < total_free_space <= 0.05)
    ):
        raise ValueError("total_free_space must be within (0, 0.05] metres")
    if (
        isinstance(product_count, bool)
        or not isinstance(product_count, int)
        or not 2 <= product_count <= 12
    ):
        raise ValueError("dense rows support 2 through 12 cartons")
    base = random_shelf_config(seed, product_count=min(product_count, 6))
    width = product_count * base.carton_size[1] + total_free_space
    config = replace(
        base,
        product_count=product_count,
        width=width,
        center_y=-0.24,
        row_free_space=total_free_space,
        target_gap=0.0,
        initially_stocked=initially_stocked,
        table_size=(0.42, max(0.5, width + 0.24)),
        table_leg_inset=0.19,
        table_center_x=base.front_x + 0.21,
        table_center_y=-0.36,
        supply_center_y=-0.48,
        # Explicit reset placement keeps the back-contact pose in the arm's
        # workspace; this controller never commands base motion during a run.
        robot_xy=(-0.04, 0.0),
        # Reset-only shoulder/elbow fold parks the canonical hand above the supply
        # cartons, clear of narrow panels/prefilled objects. Native limits,
        # geometry and servo parameters remain unchanged.
        initial_joint_positions=(("Left Arm 1", -0.5), ("Left Arm 4", -0.7)),
        time_limit=max(240.0, product_count * 90.0),
    )
    products_for(config)
    return config


def outside_in_order(products):
    """Alternate ends of a row, starting at positive Y, so the last slot is central."""
    row = sorted(products, key=lambda product: product.target[1])
    ordered = []
    for index in range(len(row)):
        ordered.append(row[-1 - index // 2] if index % 2 == 0 else row[index // 2])
    return ordered


def is_two_sided_slot(env, product, completed):
    """Whether both immediately neighboring target slots are already occupied."""
    row = sorted(
        (item for item in env.products if item.level == product.level),
        key=lambda item: item.target[1],
    )
    index = row.index(product)
    return (
        0 < index < len(row) - 1
        and row[index - 1].name in completed
        and row[index + 1].name in completed
    )


def open_slot_interval(env, product, completed):
    """Center bounds after reserving room for every unplaced carton on both sides."""
    row = sorted(
        (item for item in env.products if item.level == product.level),
        key=lambda item: item.target[1],
    )
    index = row.index(product)
    lower = env.config.center_y - env.config.width / 2 + product.half_size[1]
    upper = env.config.center_y + env.config.width / 2 - product.half_size[1]
    bounds = [lower, upper]
    for side, neighbors in enumerate((reversed(row[:index]), iter(row[index + 1 :]))):
        direction = 1 if side == 0 else -1
        reserved = 0.0
        for neighbor in neighbors:
            if neighbor.name in completed:
                body = env.data.body(neighbor.name)
                extent = (np.abs(body.xmat.reshape(3, 3)) @ neighbor.half_size)[1]
                bounds[side] = float(
                    body.xpos[1] + direction * (extent + product.half_size[1] + reserved)
                )
                break
            reserved += 2 * neighbor.half_size[1]
        else:
            bounds[side] += direction * reserved
    return tuple(bounds)


def row_geometry(env, *, product_names=None):
    """Projected outer envelopes; numerical overlap is never counted as space.

    This is conservative for yawed cuboids. It is meaningful as a packed-row
    verdict only when every product also satisfies the shelf's support/pose
    conditions. Objects on the table still produce diagnostics, not success.
    """
    config = env.config
    if config.row_free_space is None:
        raise ValueError("row_geometry requires a dense row configuration")
    intervals = []
    for product in env.products:
        if product_names is not None and product.name not in product_names:
            continue
        body = env.data.body(product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ np.asarray(product.half_size)
        intervals.append(
            (float(body.xpos[1] - extent[1]), float(body.xpos[1] + extent[1]), product.name)
        )
    if not intervals:
        raise ValueError("row_geometry requires at least one product")
    intervals.sort()
    left, right = config.center_y - config.width / 2, config.center_y + config.width / 2
    gaps = [intervals[0][0] - left]
    gaps.extend(b[0] - a[1] for a, b in zip(intervals, intervals[1:], strict=False))
    gaps.append(right - intervals[-1][1])
    # A 0.2 mm final envelope-overlap bound is a new, stricter dense-task
    # criterion. It does not change the original spacious benchmark's gates.
    return {
        "nominal_total_free_space_m": config.row_free_space,
        "projected_total_free_space_m": float(sum(gaps)),
        "positive_gap_sum_m": float(sum(max(0, g) for g in gaps)),
        "overlap_sum_m": float(sum(max(0, -g) for g in gaps)),
        "minimum_gap_m": float(min(gaps)),
        "gaps_including_walls_m": list(map(float, gaps)),
        "product_order": [item[2] for item in intervals],
        "envelopes_fit": bool(min(gaps) >= -0.0002 and sum(max(0, -g) for g in gaps) <= 0.0002),
        "maximum_allowed_envelope_overlap_m": 0.0002,
        "maximum_allowed_total_envelope_overlap_m": 0.0002,
    }
