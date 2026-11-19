"""Integration checks using the actual upstream robot and actuator semantics."""

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.control import stationary_targets
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.scene import ShelfConfig

pytestmark = pytest.mark.mabel


@pytest.fixture(scope="module")
def model_env():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("Run abstract-stack fetch-mabel to test the real MABEL model")
    env = ShelfStockingEnv()
    yield env
    env.close()


@pytest.fixture
def env(model_env):
    model_env.reset()
    return model_env


def test_full_robot_loads_and_stays_upright(env):
    assert (env.model.nq, env.model.nv, env.model.nu) == (109, 102, 59)
    assert "Left Thumb MCP" in env.actuator_names
    assert "Right Pinky PIP" in env.actuator_names
    assert env.data.actuator("Lift").ctrl[0] == pytest.approx(0.3175)
    initial = env.data.body("mabel_chassis").xpos.copy()
    for _ in range(100):
        env.step()
    assert np.isfinite(env.data.qpos).all()
    assert np.linalg.norm(env.data.body("mabel_chassis").xpos - initial) < 0.025
    assert env.data.body("mabel_chassis").xmat.reshape(3, 3)[2, 2] > 0.99
    assert not env.info()["dropped"]


def test_tendon_lift_and_velocity_wheels_use_correct_units(env):
    for name, value in (("Lift Lower", 0.17), ("Lift Upper", 0.21), ("FL Drive", 0.8)):
        env.data.qpos[env.model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(env.model, env.data)
    targets = stationary_targets(env.model, env.data)
    assert targets[env.model.actuator("Lift").id] == pytest.approx(0.38)
    assert targets[env.model.actuator("FL Drive").id] == 0


def test_partial_targets_persist_and_invalid_actions_are_atomic(env):
    env.step({"Left Arm 1": -0.1})
    env.step()
    assert env.data.actuator("Left Arm 1").ctrl[0] == pytest.approx(-0.1)
    before = env.data.ctrl.copy()
    for action in ({"missing": 1}, {"Lift": 100}, {"Torso": float("nan")}, np.zeros(2)):
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(before, env.data.ctrl)


def test_emet_robot_spec_and_stationary_interface(env):
    pytest.importorskip("emet.robots")
    from emet.robots import get_robot_backend

    from abstract_stack.robots.mabel import MabelBackend, MabelStationaryControl, register_mabel

    spec = MabelBackend().get_spec()
    assert spec.dof == 59
    assert tuple(spec.actuator_names) == env.actuator_names
    assert len(spec.arm_chains["left"].joint_names) == 7
    assert spec.arm_chains["right"].ee_body == "right_palm"
    hold = np.zeros(spec.dof)
    adapter = MabelStationaryControl()
    adapter.sync_ctrl_and_spec_hold(env.model, env.data, spec, hold)
    assert hold[spec.actuator_names.index("Lift")] == pytest.approx(0.3175)
    hold[spec.actuator_names.index("Left Arm 1")] = -0.2
    adapter.write_ctrl_with_spec_hold(env.model, env.data, spec, hold)
    assert env.data.actuator("Left Arm 1").ctrl[0] == pytest.approx(-0.2)
    register_mabel()
    assert get_robot_backend("mabel").get_spec().name == "mabel"


def test_robot_placement_and_source_meshes_are_preserved(model_env):
    env = ShelfStockingEnv(ShelfConfig(product_count=0, robot_xy=(0.15, 0.10), robot_yaw=0.25))
    try:
        np.testing.assert_allclose(env.data.body("mabel_chassis").xpos[:2], [0.15, 0.10])
        rotation = env.data.body("mabel_chassis").xmat.reshape(3, 3)
        assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(0.25)
        assert env.data.actuator("Lift").ctrl[0] == pytest.approx(0.3175)
        assert env.model.nmesh == model_env.model.nmesh
    finally:
        env.close()


@pytest.mark.parametrize("scene,prefilled", [("house", 0), ("house_tight", 12)])
def test_mabel_is_stable_in_furnished_house(model_env, scene, prefilled):
    env = ShelfStockingEnv(load_scene_config(scene=scene))
    try:
        assert env.model.nu == 59
        assert env.model.nmesh == model_env.model.nmesh
        robot_root = env.model.body("mabel_chassis").id
        robot_bodies = set()
        for bid in range(env.model.nbody):
            ancestor = bid
            while ancestor:
                if ancestor == robot_root:
                    robot_bodies.add(bid)
                    break
                ancestor = env.model.body_parentid[ancestor]
        for contact in env.data.contact:
            names = [env.model.geom(g).name or "" for g in (contact.geom1, contact.geom2)]
            bodies = [env.model.geom_bodyid[g] for g in (contact.geom1, contact.geom2)]
            assert not (
                any(name.startswith("house_") for name in names)
                and any(bid in robot_bodies for bid in bodies)
                and contact.dist < -1e-4
            )
        initial = env.data.body("mabel_chassis").xpos.copy()
        for _ in range(100):
            env.step()
        assert np.linalg.norm(env.data.body("mabel_chassis").xpos - initial) < 0.025
        assert env.data.body("mabel_chassis").xmat.reshape(3, 3)[2, 2] > 0.99
        assert env.info()["stocked_count"] == prefilled
        assert not env.info()["tipped"]
        assert not env.info()["dropped"]
    finally:
        env.close()


def _robot_fixture_contacts(env):
    """(depth_m, robot body, other body) for robot contacts with anything but the floor."""
    model, data = env.model, env.data
    robot = model.body_rootid[model.body("mabel_chassis").id]
    contacts = []
    for contact in data.contact[: data.ncon]:
        bodies = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
        mine = [model.body_rootid[b] == robot for b in bodies]
        if mine[0] != mine[1]:
            own, other = bodies if mine[0] else bodies[::-1]
            if other != 0:
                contacts.append((contact.dist, model.body(own).name, model.body(other).name))
    return contacts


# Documented exceptions, measured at reset with upstream MABEL d0181f0. pick_place keeps
# its right-thumb/shelf-side contact (docs/mabel_controller.md). house_tight's narrow,
# offset shelf grazes the right ring finger against its side; it clears on the first step.
RESET_CONTACT_ALLOWANCE_M = {"pick_place": 0.005, "house_tight": 0.002}


@pytest.mark.parametrize(
    "scene", ["stocking", "tight", "empty", "house", "house_tight", "pick_place"]
)
def test_scenes_reset_without_robot_fixture_penetration(scene):
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("Run abstract-stack fetch-mabel to test the real MABEL model")
    env = ShelfStockingEnv(load_scene_config(scene=scene))
    try:
        allowance = RESET_CONTACT_ALLOWANCE_M.get(scene, 0.0)
        contacts = _robot_fixture_contacts(env)
        assert all(-dist <= allowance for dist, _, _ in contacts), contacts
        if not allowance:
            assert not contacts, contacts
            # The lowered lift is a real, held pose: it settles with no new contact.
            lift = env.data.actuator("Lift").ctrl[0]
            for _ in range(50):
                env.step()
                assert not _robot_fixture_contacts(env)
            assert env.data.actuator("Lift").length[0] == pytest.approx(lift, abs=0.005)
    finally:
        env.close()


def test_spawn_lift_offset_lowers_only_the_lift():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("Run abstract-stack fetch-mabel to test the real MABEL model")
    base = ShelfStockingEnv(ShelfConfig(product_count=0))
    lowered = ShelfStockingEnv(ShelfConfig(product_count=0, spawn_lift_offset=0.05))
    try:
        lift = [base.model.joint(n).qposadr[0] for n in ("Lift Lower", "Lift Upper")]
        delta = lowered.data.qpos - base.data.qpos
        assert delta[lift].sum() == pytest.approx(-0.05)
        delta[lift] = 0
        assert np.allclose(delta, 0)
        assert lowered.data.actuator("Lift").ctrl[0] == pytest.approx(0.3175 - 0.05)
    finally:
        base.close()
        lowered.close()
    with pytest.raises(ValueError, match="spawn_lift_offset"):
        ShelfConfig(spawn_lift_offset=0.2)
