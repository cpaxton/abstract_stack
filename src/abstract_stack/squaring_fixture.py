"""Declared dynamic-neighbor reset fixtures, never manipulation continuations."""

import copy

import mujoco
import numpy as np


def neighbor_reset(model, source, scene, active, side, *, gap_m=0.0005):
    """Return new data with one adjacent carton; leave source/model untouched.

    Only fixture construction writes poses. The active carton, robot, controls,
    clock and all other products retain their captured state. The neighbor stays
    a native free body with unchanged mass, friction and collision properties.
    AABB separation is deliberately conservative for the yawed active carton.
    """
    if side not in ("left", "right") or not np.isfinite(gap_m) or not 0 < gap_m <= 0.005:
        raise ValueError("Choose left/right and a positive gap up to 5 mm")
    products = {p["name"]: p for p in scene["products"]}
    product = products[active]
    if product["kind"] != "carton":
        raise ValueError("Squaring fixture requires rigid cartons")
    sign = -1 if side == "left" else 1
    candidates = [
        p
        for p in products.values()
        if p["name"] != active
        and p["kind"] == "carton"
        and p["level"] == product["level"]
        and sign * (p["target"][1] - product["target"][1]) > 0
    ]
    if not candidates:
        raise ValueError("No declared neighboring carton on the requested side")
    neighbor = min(candidates, key=lambda p: abs(p["target"][1] - product["target"][1]))
    body = model.body(active).id
    extent = np.abs(source.xmat[body].reshape(3, 3)) @ product["half_size"]
    half = np.asarray(neighbor["half_size"])
    config = scene["config"]
    if config["initially_stocked"] or scene.get("reset_fixture"):
        raise ValueError("Use an original zero-prefill checkpoint for this reset fixture")
    level = config["levels"][product["level"]]
    position = np.array(
        [
            config["front_x"] - config["depth"] + half[0] + gap_m,
            source.xpos[body, 1] + sign * (extent[1] + half[1] + gap_m),
            level + half[2] + 0.00005,
        ]
    )
    walls = config["center_y"] + np.array([-1, 1]) * config["width"] / 2
    row_fit = {}
    for label, direction, wall in (("left", -1, walls[0]), ("right", 1, walls[1])):
        names = [
            p
            for p in products.values()
            if p["name"] != active
            and p["level"] == product["level"]
            and direction * (p["target"][1] - product["target"][1]) > 0
        ]
        available = direction * (wall - source.xpos[body, 1]) - extent[1]
        required = sum(2 * p["half_size"][1] for p in names)
        row_fit[label] = dict(
            products=[p["name"] for p in names],
            available_width_m=float(available),
            required_width_m=required,
            unused_width_m=float(available - required),
        )
    if position[1] - half[1] < walls[0] or position[1] + half[1] > walls[1]:
        raise ValueError("Neighbor fixture does not fit inside the physical shelf walls")
    neighbor_body = model.body(neighbor["name"]).id
    joint = int(model.body_jntadr[neighbor_body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Neighbor must remain a native free body")
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, source)
    q = int(model.jnt_qposadr[joint])
    v = int(model.jnt_dofadr[joint])
    data.qpos[q : q + 7] = np.r_[position, [1, 0, 0, 0]]
    data.qvel[v : v + 6] = 0
    data.qacc_warmstart[v : v + 6] = 0
    mujoco.mj_forward(model, data)
    for contact in data.contact:
        bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        if neighbor_body in bodies and contact.dist < -1e-7:
            raise ValueError("Reset neighbor intersects existing geometry")
    manifest = copy.deepcopy(scene)
    for p in manifest["products"]:
        p["initially_stocked"] = p["name"] == neighbor["name"]
        if p["name"] == neighbor["name"]:
            p["spawn"] = position.tolist()
    manifest["config"]["initially_stocked"] = 1
    fixture = dict(
        kind="declared_neighbor_reset_not_continuation",
        side=side,
        direction_world_y=sign,
        active=active,
        neighbor=neighbor["name"],
        initial_aabb_gap_m=gap_m,
        neighbor_position_m=position.tolist(),
        prefilled_products=[neighbor["name"]],
        scheduled_manipulations=[active],
        remaining_table_products=[p for p in products if p not in (active, neighbor["name"])],
        fresh_task_credit=0,
        pick_credit=0,
        task_success=False,
        source_time_s=float(source.time),
        whole_row_fit_at_captured_pose=dict(
            fits=all(v["unused_width_m"] >= 0 for v in row_fit.values()), sides=row_fit
        ),
        limitations="Single dynamic neighbor; does not establish a filled row or wall brace",
    )
    manifest["reset_fixture"] = fixture
    return data, manifest, fixture
