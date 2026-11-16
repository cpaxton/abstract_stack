"""Seeded, validated stocking scene families; generation does not imply solvability."""

from dataclasses import asdict, replace

import numpy as np

from abstract_stack.config import load_scene_config
from abstract_stack.scene import products_for


def random_shelf_config(seed: int, *, product_count: int = 2):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("scene seed must be a nonnegative integer")
    if (
        isinstance(product_count, bool)
        or not isinstance(product_count, int)
        or not 1 <= product_count <= 6
    ):
        raise ValueError("random bench scenes support 1 through 6 products")
    rng = np.random.default_rng(seed)
    height = float(rng.uniform(0.795, 0.805))
    config = replace(
        load_scene_config(scene="pick_place"),
        product_count=product_count,
        carton_size=(
            float(rng.uniform(0.066, 0.074)),
            float(rng.uniform(0.047, 0.053)),
            float(rng.uniform(0.232, 0.248)),
        ),
        front_x=float(rng.uniform(-0.565, -0.555)),
        depth=float(rng.uniform(0.20, 0.21)),
        width=float(rng.uniform(0.50, 0.60)),
        center_y=float(rng.uniform(-0.13, -0.11)),
        levels=(height,),
        table_top=height,
        table_center_x=float(rng.uniform(-0.345, -0.335)),
        table_center_y=float(rng.uniform(-0.49, -0.47)),
        table_size=(0.42 if product_count >= 3 else 0.34, 0.40),
        target_gap=float(rng.uniform(0.035, 0.055)),
        terminate_on_success=False,
        time_limit=240.0,
    )
    # Keep the shelf's right edge clear of MABEL's idle right hand as width
    # varies; the shelf expands leftward rather than through the robot.
    config = replace(
        config,
        width=max(
            config.width,
            product_count * config.carton_size[1] + (product_count - 1) * config.target_gap + 0.025,
        ),
    )
    config = replace(config, center_y=float(rng.uniform(0.09, 0.11)) - config.width / 2)
    products_for(config)
    return config


def scene_manifest(config, seed):
    return {
        "scene_family": "random_carton_bench_v1",
        "seed": seed,
        "config": asdict(config),
        "generation_scope": "geometry and initial poses; no solvability guarantee",
        "state_access": "privileged simulator world state",
    }
