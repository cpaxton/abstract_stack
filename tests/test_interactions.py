"""Measured forces, observer cadence, and physical controller evidence."""

import json

import mujoco
import numpy as np
import pytest

from abstract_stack.assets import mabel_model_path
from abstract_stack.config import load_scene_config
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.interactions import (
    CartesianHand,
    InteractionFailure,
    ScriptConfig,
    StockingScript,
    contact_forces,
)


def test_contact_forces_measure_weight_and_physics_observer_cadence():
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"), robot=False)
    try:
        times = []
        env.step(on_physics_step=lambda state: times.append(float(state.data.time)))
        assert len(times) == env.config.frame_skip
        np.testing.assert_allclose(np.diff(times), env.config.timestep)
        for _ in range(30):
            env.step()
        force = contact_forces(env, "carton_0", set())
        assert force["table"] == pytest.approx(env.products[0].mass * 9.81, abs=0.02)
        assert force["shelf"] == force["back"] == force["hand"] == 0
        assert not force["opposing_contacts"]
    finally:
        env.close()


@pytest.fixture
def robot():
    try:
        mabel_model_path()
    except FileNotFoundError:
        pytest.skip("MABEL assets not downloaded")
    env = ShelfStockingEnv(load_scene_config(scene="pick_place"))
    yield env
    env.close()


@pytest.mark.mabel
def test_ik_changes_only_scratch_state_and_bounds_targets(robot):
    controller = CartesianHand(robot, ScriptConfig())
    before_qpos, before_ctrl = robot.data.qpos.copy(), robot.data.ctrl.copy()
    action = controller.targets(controller.position() + [0.02, 0, 0.03], 0.2)
    np.testing.assert_array_equal(robot.data.qpos, before_qpos)
    np.testing.assert_array_equal(robot.data.ctrl, before_ctrl)
    for name in controller.names:
        assert abs(action[name] - before_ctrl[robot.model.actuator(name).id]) <= robot.dt + 1e-9
    assert abs(action["Lift"] - before_ctrl[robot.model.actuator("Lift").id]) <= 0.08 * robot.dt


@pytest.mark.mabel
def test_force_limit_stops_on_real_hand_contact(robot, tmp_path):
    script = StockingScript(robot, tmp_path, ScriptConfig(max_force=0.5))
    # Initial-condition fixture only: deliberately penetrate the hand to exercise
    # the force guard. No live rollout succeeds from this invalid arrangement.
    joint = robot.model.joint("carton_0_joint")
    qadr = int(joint.qposadr[0])
    robot.data.qpos[qadr : qadr + 3] = robot.data.body("left_palm").xpos
    mujoco.mj_forward(robot.model, robot.data)
    with pytest.raises(InteractionFailure, match="contact force"):
        robot.step(on_physics_step=script._sample)
    assert len(script.rows) > 0
    assert script.rows[-1][7] > 0.5


@pytest.mark.mabel
def test_script_writes_honest_physics_evidence(robot, tmp_path):
    script = StockingScript(robot, tmp_path)
    result = script.run()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["physical_success"] == result["physical_success"]
    assert saved["assistance"] == "none"
    assert saved["events"]
    if not result["physical_success"]:
        assert result["failure"]
    trace = np.load(tmp_path / "trace.npz")
    assert trace["samples"].shape[1] == len(result["trace_columns"])
    assert trace["controls"].shape == (len(trace["samples"]), robot.model.nu)
    assert np.isfinite(trace["samples"]).all()
    assert (np.diff(trace["samples"][:, 0]) > 0).all()
    assert robot.info()["intervention_count"] == 0


@pytest.mark.parametrize("values", [{"grip": 2}, {"max_force": 0}, {"lift_height": -0.1}])
def test_bad_script_settings(values):
    with pytest.raises(ValueError):
        ScriptConfig(**values)
