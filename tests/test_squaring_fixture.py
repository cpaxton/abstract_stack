"""Reset-only fixture changes cannot become undocumented live manipulation."""

import copy

import mujoco
import numpy as np
import pytest

from abstract_stack.checkpoints import model_signature, native_state
from abstract_stack.squaring_fixture import neighbor_reset


@pytest.fixture
def world():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="active" pos="-.4 0 .1"><freejoint/>
        <geom type="box" size=".05 .04 .1" mass=".18"/></body>
      <body name="left" pos=".3 -.2 .1"><freejoint/>
        <geom type="box" size=".05 .04 .1" mass=".18"/></body>
      <body name="right" pos=".3 .2 .1"><freejoint/>
        <geom type="box" size=".05 .04 .1" mass=".18"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(0.04), 0, 0, np.sin(0.04)]
    data.time = 2.0
    mujoco.mj_forward(model, data)
    scene = dict(
        config=dict(
            front_x=-0.2, depth=0.3, center_y=0, width=0.5, levels=[0], initially_stocked=0
        ),
        products=[
            dict(
                name=name,
                kind="carton",
                level=0,
                half_size=[0.05, 0.04, 0.1],
                target=[-0.4, y, 0.1],
                initially_stocked=False,
            )
            for name, y in (("active", 0), ("left", -0.1), ("right", 0.1))
        ],
    )
    return model, data, scene


@pytest.mark.parametrize("side", ["left", "right"])
def test_new_dynamic_fixture_preserves_live_source_robot_and_model(world, side):
    model, data, scene = world
    before, signature, manifest = (
        native_state(model, data),
        model_signature(model),
        copy.deepcopy(scene),
    )
    reset, planned, evidence = neighbor_reset(model, data, scene, "active", side)
    np.testing.assert_array_equal(native_state(model, data), before)
    assert model_signature(model) == signature and scene == manifest
    np.testing.assert_array_equal(reset.qpos[:7], data.qpos[:7])
    np.testing.assert_array_equal(reset.ctrl, data.ctrl)
    assert reset.time == data.time and reset is not data
    other = "left" if side == "right" else "right"
    np.testing.assert_array_equal(reset.xpos[model.body(other).id], data.xpos[model.body(other).id])
    active_extent = np.abs(data.xmat[model.body("active").id].reshape(3, 3)) @ [0.05, 0.04, 0.1]
    distance = abs(reset.xpos[model.body(side).id, 1] - data.xpos[model.body("active").id, 1])
    assert distance - active_extent[1] - 0.04 == pytest.approx(0.0005)
    assert not np.any(reset.xfrc_applied) and not np.any(reset.qfrc_applied) and model.neq == 0
    assert evidence["fresh_task_credit"] == evidence["pick_credit"] == 0
    assert evidence["prefilled_products"] == [side]
    assert evidence["scheduled_manipulations"] == ["active"]
    assert planned["config"]["initially_stocked"] == 1


def test_fixture_rejects_nonfitting_neighbor_instead_of_clipping(world):
    model, data, scene = world
    scene["config"]["width"] = 0.12
    with pytest.raises(ValueError, match="does not fit"):
        neighbor_reset(model, data, scene, "active", "right")


def test_fixture_rejects_intersections_and_invalid_gap(world):
    model, data, scene = world
    # Place the other product where the planned neighbor would be constructed.
    data.qpos[7:10] = [-0.4495, 0.084367, 0.1]
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError, match="intersects"):
        neighbor_reset(model, data, scene, "active", "right")
    with pytest.raises(ValueError, match="positive gap"):
        neighbor_reset(model, data, scene, "active", "right", gap_m=0)
