"""Deterministic geometry curricula and private, separately seeded evaluation suites."""

import hashlib
import json
import math
import secrets
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from abstract_stack.config import load_scene_config
from abstract_stack.packing import dense_shelf_config
from abstract_stack.scene import initial_positions, products_for

VERSION = "stocking_geometry_dr_v1"
FAMILIES = ("bench_cartons", "dense_cartons", "multi_shelf_cartons", "mixed_house")


def sample_scene(seed, family):
    """Sample only explicit procedural configuration; never change a live model."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**63
        or family not in FAMILIES
    ):
        raise ValueError("Choose a nonnegative scene seed and a declared DR family")
    rng = np.random.default_rng(seed)
    if family == "dense_cartons":
        config = dense_shelf_config(
            seed,
            product_count=int(rng.integers(2, 11)),
            total_free_space=float(rng.choice([0.005, 0.01])),
        )
    else:
        levels = 1 if family == "bench_cartons" else int(rng.integers(2, 4))
        count = int(rng.integers(max(2, levels), 7 if family == "bench_cartons" else 11))
        kind = "mixed" if family == "mixed_house" else "carton"
        size = (
            float(rng.uniform(0.06, 0.10)),
            float(rng.uniform(0.045, 0.08)),
            float(rng.uniform(0.16, 0.25)),
        )
        base_height = float(rng.uniform(0.68, 0.83))
        spacing = float(rng.uniform(0.31, 0.39))
        gap = float(rng.uniform(0.02, 0.055))
        row_count = math.ceil(count / levels)
        maximum_width = max(size[1], 0.068 if kind == "mixed" else size[1])
        required = row_count * maximum_width + (row_count - 1) * gap + 0.04
        width = float(rng.uniform(max(0.50, required), max(0.80, required + 0.16)))
        right_edge = float(rng.uniform(-0.015, 0.045))
        front = float(rng.uniform(-0.60, -0.54))
        columns = min(3, count)
        rows = math.ceil(count / columns)
        table_y = max(0.45, (rows - 1) * 0.14 + maximum_width + 0.04)
        table_x = max(0.46, (columns - 1) * 0.15 + max(size[0], 0.068) + 0.04)
        config = replace(
            load_scene_config(scene="pick_place"),
            product_count=count,
            product_kind=kind,
            initially_stocked=0,
            carton_size=size,
            front_x=front,
            depth=float(rng.uniform(0.22, 0.34)),
            width=width,
            center_y=right_edge - width / 2,
            levels=tuple(base_height + i * spacing for i in range(levels)),
            top_clearance=float(rng.uniform(0.36, 0.46)),
            table_top=base_height,
            table_size=(table_x, table_y),
            table_leg_inset=0.04,
            table_center_x=front + table_x / 2,
            table_center_y=-0.28 - table_y / 2,
            supply_center_y=None,
            target_gap=gap,
            spawn_jitter=float(rng.uniform(0.001, 0.005)),
            environment="small_house" if family == "mixed_house" else "studio",
            time_limit=900.0,
            terminate_on_success=False,
        )
    products = products_for(config)
    reset_seed = int(np.random.default_rng(seed ^ 0x5A17C3).integers(0, 2**31))
    poses = initial_positions(config, products, reset_seed)
    # Supply layout uses the maximum product envelope and full jitter budget.
    # Shelf generation validates each row/level; neither check proves reachability.
    for p, position in zip(products, poses, strict=True):
        tx, ty = config.table_xy
        if (
            abs(position[0] - tx) + p.half_size[0] > config.table_size[0] / 2
            or abs(position[1] - ty) + p.half_size[1] > config.table_size[1] / 2
        ):
            raise ValueError("Sampled supply layout does not fit its declared table")
    adapter = (
        "carton_bench"
        if family == "bench_cartons"
        else "receding_dense"
        if family == "dense_cartons"
        else None
    )
    factors = dict(
        family=family,
        product_count=len(products),
        shelf_levels=len(config.levels),
        object_kinds=sorted({p.kind for p in products}),
        carton_full_size_m=list(config.carton_size),
        shelf_width_m=config.width,
        shelf_depth_m=config.depth,
        levels_m=list(config.levels),
        environment=config.environment,
        spawn_jitter_m=config.spawn_jitter,
        whole_row_free_space_m=config.row_free_space,
    )
    return config, dict(
        scene_seed=seed,
        reset_seed=reset_seed,
        factors=factors,
        scheduled_product_names=[p.name for p in products],
        teacher_adapter=adapter,
        teacher_capability_status="unvalidated" if adapter else "unsupported_by_current_teacher",
        validation_scope="geometric layout only; no IK, contact or manipulation proof",
        unchanged=[
            "MABEL geometry/inertias/actuators/collision masks",
            "object material parameters",
            "native camera calibration",
            "sensor noise/delay model",
        ],
    )


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate_suite(
    output, *, development_count=8, heldout_count=12, development_seed=2026, heldout_seed=None
):
    """Emit configs/context plus an all-object ledger; no simulation or task credit.

    Heldout entropy is independent and private by default. Do not inspect/tune on
    that split; bind a frozen controller before running the eventual evaluation.
    """
    if (
        any(
            isinstance(n, bool) or not isinstance(n, int)
            for n in (development_count, heldout_count, development_seed)
        )
        or not 1 <= development_count <= 100
        or not 1 <= heldout_count <= 100
        or development_seed < 0
        or heldout_seed is not None
        and (
            isinstance(heldout_seed, bool) or not isinstance(heldout_seed, int) or heldout_seed < 0
        )
    ):
        raise ValueError("Keep suite counts and master seeds bounded and nonnegative")
    if heldout_seed is None:
        heldout_seed = secrets.randbits(63)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    used = set()
    splits = {}
    for split, count, master in (
        ("development", development_count, development_seed),
        ("heldout", heldout_count, heldout_seed),
    ):
        rng = np.random.default_rng(master)
        cases = []
        for index in range(count):
            seed = int(rng.integers(0, 2**31))
            while seed in used:
                seed = int(rng.integers(0, 2**31))
            used.add(seed)
            family = FAMILIES[index % len(FAMILIES)]
            config, context = sample_scene(seed, family)
            case_id = f"{split}-{index:03}"
            directory = output / "cases" / case_id
            _write(directory / "scene.json", asdict(config))
            context.update(
                suite_version=VERSION,
                split=split,
                case_id=case_id,
                scene_config_sha256=hashlib.sha256(
                    (directory / "scene.json").read_bytes()
                ).hexdigest(),
            )
            _write(directory / "context.json", context)
            cases.append(
                dict(
                    case_id=case_id,
                    scene=f"cases/{case_id}/scene.json",
                    context=f"cases/{case_id}/context.json",
                    scene_sha256=context["scene_config_sha256"],
                    context_sha256=hashlib.sha256(
                        (directory / "context.json").read_bytes()
                    ).hexdigest(),
                    scheduled_product_names=context["scheduled_product_names"],
                    teacher_adapter=context["teacher_adapter"],
                )
            )
        _write(
            output / split / "index.json",
            dict(suite_version=VERSION, split=split, master_seed=master, cases=cases),
        )
        _write(
            output / split / "evaluation.json",
            dict(
                suite_version=VERSION,
                split=split,
                controller_source_id=None,
                task_success=False,
                fresh_task_credit=0,
                scheduled_scenes=count,
                cases=[
                    dict(
                        case_id=c["case_id"],
                        outcome="not_run",
                        process=None,
                        products={
                            name: dict.fromkeys(
                                ("pick", "place", "final_position", "retreat"), False
                            )
                            for name in c["scheduled_product_names"]
                        },
                    )
                    for c in cases
                ],
            ),
        )
        splits[split] = dict(
            index=f"{split}/index.json",
            evaluation=f"{split}/evaluation.json",
            cases=count,
            scheduled_objects=sum(len(c["scheduled_product_names"]) for c in cases),
        )
    _write(
        output / "suite.json",
        dict(
            suite_version=VERSION,
            splits=splits,
            source_purpose="geometry curriculum and future robustness evaluation",
            frozen_policy_bound=False,
            task_success=False,
            fresh_task_credit=0,
            evaluation_scope="Heldout stays private until controller freeze; retain every object",
        ),
    )
    return output
