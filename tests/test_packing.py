"""Whole-row budgets, numerical overlap and reproducible collision-free initial states."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.dense_interactions import DenseStockingScript
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import InteractionFailure
from abstract_stack.packing import dense_shelf_config, row_geometry
from abstract_stack.scene import products_for


def test_carton_force_guard_sums_all_contact_points_in_each_pair():
    # Independent material/contact fixture: four contacts each below 25 N,
    # but the combined pair load supports a 3 kg upper box (>25 N).
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="2 2 .1"/>
      <body name="a" pos="0 0 .1"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="1"/></body>
      <body name="b" pos="0 0 .3"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="3"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    script = DenseStockingScript.__new__(DenseStockingScript)
    script.product_bodies = {model.body("a").id, model.body("b").id}
    script.object_pair_peak, script.object_pair_bins = 0.0, {}
    script.config = SimpleNamespace(max_force=25.0)
    script._carton_wall_geoms, script.carton_wall_peak = set(), 0.0
    env = SimpleNamespace(model=model, data=data, dt=0.02)
    before = data.qpos.copy()
    with pytest.raises(InteractionFailure, match="carton/carton force limit"):
        script._sample_carton_pairs(env)
    assert script.object_pair_peak == pytest.approx(3 * 9.81, abs=0.01)
    assert len(script.object_pair_bins) == 1
    np.testing.assert_array_equal(data.qpos, before)


def test_sidewall_guard_sums_all_normals_instead_of_max_contact_point():
    model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 -9.81 0"/>
      <worldbody><geom name="wall" type="box" size=".2 .1 .2"/>
        <body name="carton" pos="0 .2 0"><freejoint/>
          <geom type="box" size=".1 .1 .1" mass="3"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    script = DenseStockingScript.__new__(DenseStockingScript)
    script.product_bodies = {model.body("carton").id}
    script._carton_wall_geoms = {model.geom("wall").id}
    script.carton_wall_peak, script.object_pair_peak, script.object_pair_bins = 0.0, 0.0, {}
    script.config = SimpleNamespace(max_force=25.0)
    with pytest.raises(InteractionFailure, match="summed carton/sidewall force limit"):
        script._sample_carton_pairs(SimpleNamespace(model=model, data=data, dt=0.02))
    assert script.carton_wall_peak == pytest.approx(3 * 9.81, abs=0.01)


@pytest.mark.parametrize("count", [2, 3, 6, 10, 12])
@pytest.mark.parametrize("free", [0.01, 0.005])
def test_total_free_space_includes_both_wall_gaps(count, free):
    config = dense_shelf_config(0, product_count=count, total_free_space=free)
    assert config == dense_shelf_config(0, product_count=count, total_free_space=free)
    products = products_for(config)
    assert config.width - sum(2 * p.half_size[1] for p in products) == pytest.approx(free)
    intervals = [(p.target[1] - p.half_size[1], p.target[1] + p.half_size[1]) for p in products]
    gaps = [intervals[0][0] - (config.center_y - config.width / 2)]
    gaps += [b[0] - a[1] for a, b in zip(intervals, intervals[1:], strict=False)]
    gaps += [config.center_y + config.width / 2 - intervals[-1][1]]
    assert gaps == pytest.approx([free / (count + 1)] * (count + 1))
    with pytest.raises(ValueError, match="minus all carton widths"):
        products_for(replace(config, width=config.width + 0.001))


def test_dense_configuration_round_trips(tmp_path):
    config = dense_shelf_config(7, product_count=10, initially_stocked=9, total_free_space=0.005)
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(asdict(config)))
    assert load_scene_config(path) == config


@pytest.mark.parametrize("free", [0.01, 0.005])
def test_reset_fixture_fits_but_does_not_treat_yaw_overlap_as_free_space(free):
    # Entirely prefilled, robot-free initial fixtures test evaluation only.
    # This is not evidence that a controller stocked a shelf.
    env = ShelfStockingEnv(
        dense_shelf_config(0, product_count=10, initially_stocked=10, total_free_space=free),
        robot=False,
    )
    try:
        for _ in range(50):
            env.step()
        assert env.info()["success"]
        assert env.info()["packing"]["projected_total_free_space_m"] == pytest.approx(free)
        adr = int(env.model.joint("carton_5_joint").qposadr[0])
        angle = np.radians(3)
        env.data.qpos[adr + 3 : adr + 7] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        mujoco.mj_forward(env.model, env.data)
        assert env._placement_conditions().all()
        env.update_task_state(env.config.timestep)
        assert not env.info()["success"]
        assert not row_geometry(env)["envelopes_fit"]
        assert row_geometry(env)["overlap_sum_m"] > 0.0002
    finally:
        env.close()


def test_total_numerical_overlap_has_a_whole_row_bound():
    env = ShelfStockingEnv(
        dense_shelf_config(0, product_count=4, initially_stocked=4, total_free_space=0.005),
        robot=False,
    )
    try:
        # Reset geometry only: two individually small overlaps must not add up
        # to a hidden packing allowance across multiple neighbors.
        for name in ("carton_1", "carton_3"):
            adr = int(env.model.joint(f"{name}_joint").qposadr[0])
            env.data.qpos[adr + 1] -= 0.00115
        mujoco.mj_forward(env.model, env.data)
        before = env.data.qpos.copy()
        report = row_geometry(env)
        np.testing.assert_array_equal(env.data.qpos, before)
        assert report["minimum_gap_m"] > -0.0002
        assert report["overlap_sum_m"] == pytest.approx(0.0003)
        assert not report["envelopes_fit"]
    finally:
        env.close()


@pytest.mark.mabel
@pytest.mark.parametrize("count,prefilled", [(3, 0), (10, 9)])
def test_dense_native_reset_and_export_preserve_explicit_parked_pose(tmp_path, count, prefilled):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=count, initially_stocked=prefilled))
    try:
        env.validate_robot_fixture_clearance()
        for name, value in env.config.initial_joint_positions:
            adr = int(env.model.joint(name).qposadr[0])
            assert env.data.qpos[adr] == value
            assert env.model.qpos0[adr] == 0.0  # Native joint references are untouched.
        path = env.export(tmp_path / "dense.xml")
        exported = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(exported)
        mujoco.mj_resetDataKeyframe(exported, data, 0)
        assert exported.key(0).name.startswith("scene_start")
        np.testing.assert_array_equal(data.qpos, env.data.qpos)
        np.testing.assert_array_equal(data.ctrl, env.data.ctrl)
        origin = env.data.body("mabel_chassis").xpos.copy()
        for _ in range(50):
            env.step()
        env.validate_robot_fixture_clearance()
        assert np.linalg.norm(env.data.body("mabel_chassis").xpos - origin) < 0.01
    finally:
        env.close()


def test_outside_in_order_leaves_central_slot_and_checks_both_boundaries():
    from abstract_stack.packing import is_two_sided_slot, open_slot_interval, outside_in_order

    config = load_scene_config("configs/scenes/tight_ten_cartons_20mm.json")
    products = products_for(config)
    ordered = outside_in_order(products)
    assert [p.name for p in ordered] == [
        "carton_9",
        "carton_0",
        "carton_8",
        "carton_1",
        "carton_7",
        "carton_2",
        "carton_6",
        "carton_3",
        "carton_5",
        "carton_4",
    ]
    bodies = {
        product.name: SimpleNamespace(xpos=np.asarray(product.target), xmat=np.eye(3).ravel())
        for product in products
    }
    env = SimpleNamespace(
        config=config, products=products, data=SimpleNamespace(body=bodies.__getitem__)
    )
    active = ordered[-1]
    assert not is_two_sided_slot(env, active, {"carton_9"})
    lower, upper = open_slot_interval(env, active, {"carton_9"})
    assert lower == pytest.approx(config.center_y - config.width / 2 + 4.5 * 0.05)
    assert upper == pytest.approx(products[9].target[1] - 5 * 0.05)
    complete = {p.name for p in ordered[:-1]}
    assert is_two_sided_slot(env, active, complete)
    lower, upper = open_slot_interval(env, active, complete)
    assert lower < active.target[1] < upper
    bodies["carton_3"].xpos[1] += 0.005
    assert open_slot_interval(env, active, complete)[0] > upper


def test_center_last_supply_order_only_reassigns_inventory_positions():
    original = products_for(load_scene_config("configs/scenes/tight_ten_cartons_20mm.json"))
    config = load_scene_config("configs/scenes/tight_ten_cartons_20mm_center_last.json")
    reordered = products_for(config)

    assert [p.target for p in reordered] == [p.target for p in original]
    assert sorted(p.spawn for p in reordered) == sorted(p.spawn for p in original)
    assert reordered[0].spawn == original[8].spawn
    with pytest.raises(ValueError, match="Supply order"):
        products_for(replace(config, supply_order=(0,) * 10))
