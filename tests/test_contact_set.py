"""Explicit adjacent contact is narrower than an unrestricted row permission."""

import mujoco
import numpy as np
import pytest

from abstract_stack.contact_set import (
    AdjacentContactPlan,
    ContactSetMonitor,
    NamedContactFingerLoadCompensator,
    assess_neighbor_contact,
    check_coupled_contact_set,
    plan_adjacent_contact,
)
from abstract_stack.interactions import InteractionFailure, ScriptConfig
from abstract_stack.native_views import retain_data_views, retain_model_views


def test_corner_contract_rejects_deep_penetration_wrong_surface_and_other_fingers():
    plan = AdjacentContactPlan("carton_1", "carton_2", 1)
    sample = dict(
        body="left_middle_ip",
        distance=-0.0003,
        local_position=[0.034, -0.024, -0.09],
        half_size=[0.034, 0.024, 0.116],
        force_direction=[-0.7, 0.7, 0.0],
    )
    assert assess_neighbor_contact(plan, **sample) is None
    assert "0.5 mm" in assess_neighbor_contact(plan, **(sample | {"distance": -0.002068}))
    assert "distal" in assess_neighbor_contact(plan, **(sample | {"body": "left_palm"}))
    assert "corner" in assess_neighbor_contact(
        plan, **(sample | {"local_position": [0.0, -0.024, -0.09]})
    )
    assert "normal" in assess_neighbor_contact(plan, **(sample | {"force_direction": [-1, 0, 0]}))
    assert "corner" in assess_neighbor_contact(
        plan, **(sample | {"local_position": [0.034, -0.024, 0]})
    )
    opposite = AdjacentContactPlan("carton_1", "carton_0", -1)
    assert (
        assess_neighbor_contact(
            opposite,
            **(
                sample
                | {"local_position": [0.034, 0.024, -0.09], "force_direction": [-0.7, -0.7, 0]}
            ),
        )
        is None
    )


@pytest.fixture
def native_row():
    from abstract_stack.assets import mabel_model_path
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=3))
    try:
        for _ in range(25):
            env.step()
        yield env
    finally:
        env.close()


@pytest.mark.mabel
def test_plan_uses_measured_immediate_neighbor_and_rejects_reordered_identity(native_row):
    env = native_row
    ordered = sorted(env.products, key=lambda p: env.data.body(p.name).xpos[1])
    plan = plan_adjacent_contact(env, ordered[0].name, 1)
    assert plan.neighbor == ordered[1].name
    assert plan.distal_body == "left_middle_ip"
    with pytest.raises(ValueError, match="No adjacent"):
        plan_adjacent_contact(env, ordered[-1].name, 1)
    # Separate reset/scratch fixture: swapping measured positions must invalidate
    # the named plan, even though target slots and product ordering are unchanged.
    first = int(env.model.joint(f"{ordered[1].name}_joint").qposadr[0])
    second = int(env.model.joint(f"{ordered[2].name}_joint").qposadr[0])
    env.data.qpos[first + 1], env.data.qpos[second + 1] = (
        env.data.qpos[second + 1],
        env.data.qpos[first + 1],
    )
    mujoco.mj_forward(env.model, env.data)
    with pytest.raises(InteractionFailure, match="identity changed"):
        ContactSetMonitor(env, plan)


@pytest.mark.mabel
def test_coupled_forecast_checks_complete_postures_without_live_state_writes(native_row):
    from abstract_stack.load_aware_pusher import LoadAwareFingerPusher

    env = native_row
    ordered = sorted(env.products, key=lambda p: env.data.body(p.name).xpos[1])
    plan = plan_adjacent_contact(env, ordered[1].name, 1)
    hand = LoadAwareFingerPusher(env, ScriptConfig())
    nominal = hand.finger_targets(0)
    applied = nominal | {"Left Middle Abd": 0.1}
    before = (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.data.ctrl.copy(),
        hand.command.copy(),
        env.data.time,
    )
    report = check_coupled_contact_set(hand, plan, hand.command, nominal, applied)
    assert report["samples"] == 99
    assert report["maximum_neighbor_penetration_m"] == 0.0005
    assert "accepted" in report
    for observed, expected in zip(
        (env.data.qpos, env.data.qvel, env.data.ctrl, hand.command), before[:4], strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert env.data.time == before[4]
    with pytest.raises(ValueError, match="native bounds"):
        check_coupled_contact_set(
            hand, plan, hand.command, nominal, nominal | {"Left Middle Abd": 20.0}
        )


@pytest.fixture
def two_contact_fixture():
    # Explicit synthetic overlap fixture, not manipulation evidence. A single
    # native revolute finger touches two dynamic cartons at distinct lever arms.
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody>
      <body name="mabel_chassis"><body name="left_middle_ip">
        <joint name="finger" axis="0 0 1" range="-1 1"/>
        <geom type="box" size=".2 .12 .02" pos=".2 .10 0" mass="1"/>
      </body></body>
      <body name="carton_1" pos=".435 .04 0"><freejoint/>
        <geom type="box" size=".04 .025 .025" mass="1"/>
      </body>
      <body name="carton_2" pos=".435 .16 0"><freejoint/>
        <geom type="box" size=".04 .025 .025" mass="1"/>
      </body>
      <body name="carton_3" pos="2 2 0"><freejoint/>
        <geom type="box" size=".04 .025 .025" mass="1"/>
      </body>
    </worldbody><actuator>
      <position name="finger" joint="finger" kp="50" ctrlrange="-1 1"/>
    </actuator>
    </mujoco>""")
    views = retain_model_views(model)
    data = mujoco.MjData(model)
    data_views = retain_data_views(model, data)
    mujoco.mj_forward(model, data)
    yield model, data
    assert views and data_views


def test_named_loads_sum_both_cartons_match_native_joint_constraint_and_stay_bounded(
    two_contact_fixture,
):
    model, data = two_contact_fixture
    plan = AdjacentContactPlan("carton_1", "carton_2", 1)
    controller = NamedContactFingerLoadCompensator(model, data, ["finger"], plan=plan)
    before = data.qpos.copy(), data.ctrl.copy(), data.time
    targets = controller.targets({"finger": 0.0}, dt=0.02)
    report = controller.last_report["finger"]
    efforts = report["contact_effort_by_carton_Nm"]
    assert set(efforts) == {"carton_1", "carton_2"}
    assert all(abs(v) > 1e-4 for v in efforts.values())
    np.testing.assert_allclose(sum(efforts.values()), data.qfrc_constraint[0], atol=1e-8)
    np.testing.assert_allclose(report["measured_contact_effort_Nm"], sum(efforts.values()))
    assert np.sign(targets["finger"]) == -np.sign(data.qfrc_constraint[0])
    assert abs(targets["finger"]) <= 0.02 + 1e-12
    assert -1 <= targets["finger"] <= 1
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.ctrl, before[1])
    assert data.time == before[2]


@pytest.mark.mabel
def test_monitor_requires_every_physics_step_and_keeps_native_guards(native_row):
    env = native_row
    ordered = sorted(env.products, key=lambda p: env.data.body(p.name).xpos[1])
    plan = plan_adjacent_contact(env, ordered[1].name, 1)
    monitor = ContactSetMonitor(env, plan)
    env.step(on_physics_step=monitor.step)
    assert set(monitor.report()["peak_robot_normal_N"]) == {p.name for p in env.products}
    assert set(monitor.support_loss) == set(plan.cartons)
    # Monitor misuse cannot silently omit control-frame interior physics steps.
    env.step()
    with pytest.raises(InteractionFailure, match="missed a physics step"):
        monitor.step(env)


def test_named_loads_exclude_an_undeclared_third_carton(two_contact_fixture):
    from abstract_stack.finger_load_control import FingerLoadCompensator

    model, data = two_contact_fixture
    # Separate synthetic reset fixture puts the third carton under the finger.
    jid = int(model.body("carton_3").jntadr[0])
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr : adr + 3] = [0.435, 0.10, 0]
    mujoco.mj_forward(model, data)
    plan = AdjacentContactPlan("carton_1", "carton_2", 1)
    declared = NamedContactFingerLoadCompensator(model, data, ["finger"], plan=plan)
    third = FingerLoadCompensator(model, data, ["finger"], product_name="carton_3")
    declared.targets({"finger": 0.0}, dt=0.02)
    third.targets({"finger": 0.0}, dt=0.02)
    named_effort = declared.last_report["finger"]["measured_contact_effort_Nm"]
    undeclared_effort = third.last_report["finger"]["measured_contact_effort_Nm"]
    assert abs(undeclared_effort) > 1e-4
    np.testing.assert_allclose(named_effort + undeclared_effort, data.qfrc_constraint[0], atol=1e-8)
    assert not np.isclose(named_effort, data.qfrc_constraint[0])
    assert "carton_3" not in declared.last_report["finger"]["contact_effort_by_carton_Nm"]


@pytest.mark.mabel
def test_same_level_table_carton_between_y_positions_is_not_a_row_neighbor(native_row):
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.packing import dense_shelf_config

    # All labels are initially_stocked=False. Explicit reset positions, not
    # those labels, construct two real shelf occupants and one table carton.
    env = ShelfStockingEnv(dense_shelf_config(0, product_count=3, initially_stocked=0))
    try:
        ordered = sorted(env.products, key=lambda p: p.target[1])
        active, table, neighbor = ordered
        for product in (active, neighbor):
            adr = int(env.model.joint(f"{product.name}_joint").qposadr[0])
            env.data.qpos[adr : adr + 3] = np.asarray(product.target) + [0, 0, 0.001]
        adr = int(env.model.joint(f"{table.name}_joint").qposadr[0])
        env.data.qpos[adr : adr + 3] = [
            env.config.table_center_x,
            (active.target[1] + neighbor.target[1]) / 2,
            env.config.table_top + table.half_size[2] + 0.001,
        ]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(25):
            env.step()
        assert all(not p.initially_stocked for p in env.products)
        assert (
            env.data.body(active.name).xpos[1]
            < env.data.body(table.name).xpos[1]
            < env.data.body(neighbor.name).xpos[1]
        )
        assert active.level == table.level == neighbor.level
        assert plan_adjacent_contact(env, active.name, 1).neighbor == neighbor.name
        assert plan_adjacent_contact(env, neighbor.name, -1).neighbor == active.name
        # An X/Z-compatible carton without board contact must also be excluded;
        # merely being near the shelf height does not establish physical support.
        env.data.qpos[adr : adr + 3] = [
            active.target[0],
            table.target[1],
            env.config.levels[0] + table.half_size[2] + 0.002,
        ]
        mujoco.mj_forward(env.model, env.data)
        assert plan_adjacent_contact(env, active.name, 1).neighbor == neighbor.name
    finally:
        env.close()
