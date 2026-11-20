"""Physical room fixtures and cooperative dispatch contracts."""

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.room_dispatch import PlacementReceipt, RobotProfile, RoomDispatcher
from abstract_stack.room_scene import RoomConfig, build_stocking_room
from abstract_stack.scene import build_scene_xml


def test_seeded_multi_station_room_compiles_and_products_settle_on_supply():
    config = RoomConfig(station_count=4, products_per_station=9, seed=8)
    room = build_stocking_room(config)
    assert room == build_stocking_room(config)
    different = build_stocking_room(RoomConfig(4, 9, seed=9))
    assert room.stations[0].products[0].spawn_xyz != different.stations[0].products[0].spawn_xyz

    model = mujoco.MjModel.from_xml_string(room.model_xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=500)
    mujoco.mj_forward(model, data)
    contacts = list(data.contact)
    for station in room.stations:
        support = model.geom(station.supply_geom).id
        assert model.geom(station.handoff_geom).contype[0]
        for product in station.products:
            body = model.body(product.body)
            assert body.mass[0] == pytest.approx(product.mass_kg)
            assert len(body.jntadr) == 1
            assert np.linalg.norm(data.body(body.id).xpos[:2] - product.spawn_xyz[:2]) < 0.005
            assert any(
                support in (c.geom1, c.geom2) and body.id in model.geom_bodyid[c.geom]
                for c in contacts
            )
            assert model.site(product.target_site).pos[:2] == pytest.approx(product.target_xyz[:2])
            target_support = model.geom(product.target_support_geom)
            assert target_support.pos[2] + target_support.size[2] == pytest.approx(
                product.target_xyz[2] - product.half_size_m[2]
            )


@pytest.mark.mabel
def test_room_uses_canonical_mabel_limits_and_mass():
    try:
        mabel_model_path()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    base = mujoco.MjModel.from_xml_string(build_scene_xml())
    room = build_stocking_room(mabel=True)
    model = mujoco.MjModel.from_xml_string(room.model_xml)
    assert room.robot_spawn == "mabel_chassis"
    assert model.nu == base.nu
    np.testing.assert_array_equal(model.actuator_ctrlrange, base.actuator_ctrlrange)
    np.testing.assert_array_equal(model.actuator_forcerange, base.actuator_forcerange)
    np.testing.assert_array_equal(model.body("mabel_chassis").mass, base.body("mabel_chassis").mass)


def test_heterogeneous_handoff_requires_fresh_placement_and_retreat():
    room = build_stocking_room()
    dispatcher = RoomDispatcher(room)
    stations = frozenset(s.id for s in room.stations)
    products = frozenset(("carton", "bottle"))
    feeder = RobotProfile("feeder", frozenset(("stage",)), stations, products)
    arm = RobotProfile("arm", frozenset(("stock_staged",)), stations, products)
    spare = RobotProfile("spare", frozenset(("stock",)), stations, products)

    stage = dispatcher.claim(feeder, "station_0", state_revision=3)
    assert stage is not None and stage.skill == "stage"
    assert dispatcher.claim(arm, "station_0", state_revision=3) is None
    assert dispatcher.claim(feeder, "station_1", state_revision=3) is None
    assert dispatcher.claim(spare, "station_1", state_revision=3) is not None
    for revision, retreated in ((3, True), (4, False)):
        with pytest.raises(ValueError, match="fresh verified"):
            dispatcher.complete(
                stage,
                PlacementReceipt(
                    revision, stage.product_body, stage.destination_geom, True, retreated
                ),
            )
    dispatcher.complete(
        stage, PlacementReceipt(4, stage.product_body, stage.destination_geom, True, True)
    )
    assert not dispatcher.stocked
    assert dispatcher.claim(feeder, "station_0", state_revision=5) is None
    stock = dispatcher.claim(arm, "station_0", state_revision=5)
    assert stock is not None and stock.skill == "stock_staged"
    assert stock.product_body == stage.product_body
    assert stock.source_geom == stage.destination_geom
    dispatcher.report_failure(stock, state_revision=6)
    assert dispatcher.blocked == {"station_0"}
    assert dispatcher.claim(arm, "station_0", state_revision=6) is None
    dispatcher.confirm_source_restored(
        stock, PlacementReceipt(7, stock.product_body, stock.source_geom, True, True)
    )
    assert dispatcher.claim(arm, "station_0", state_revision=6) is None
    retry = dispatcher.claim(arm, "station_0", state_revision=7)
    assert retry is not None and retry.token != stock.token
    with pytest.raises(ValueError, match="stale"):
        dispatcher.complete(
            stock, PlacementReceipt(8, stock.product_body, stock.destination_geom, True, True)
        )
    dispatcher.complete(
        retry, PlacementReceipt(8, retry.product_body, retry.destination_geom, True, True)
    )
    assert dispatcher.stocked == {stock.product_body}
