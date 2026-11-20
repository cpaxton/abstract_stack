"""Native lightweight rollout isolation, actuator and continuous force contracts."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from abstract_stack.contact_rollout import (
    ContactMonitor,
    RolloutScene,
    evaluate_batch,
    evaluate_screened_batch,
    integration_state,
)


@pytest.fixture
def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.002" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <geom name="table" type="box" pos="0 0 .3" size="1 1 .3"/>
        <geom name="back" type="box" pos="-.8 0 .65" size=".02 .3 .3"/>
        <body name="robot" pos=".079 0 .65">
          <joint name="drive" type="slide" axis="1 0 0" range="-.1 .1"/>
          <geom type="sphere" size=".01" mass=".1" pos="0 .3 0"/>
          <body name="hand"><geom type="sphere" size=".03" mass=".1"/></body>
        </body>
        <body name="active" pos="0 0 .65"><freejoint name="active_joint"/>
          <geom type="box" size=".05 .05 .05" mass=".18"/>
        </body>
        <body name="other" pos="-.25 0 .65"><freejoint name="other_joint"/>
          <geom type="box" size=".05 .05 .05" mass=".18"/>
        </body>
      </worldbody>
      <actuator><position name="drive" joint="drive" kp="200" ctrlrange="-.1 .1"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    context = RolloutScene(
        products=("active", "other"),
        active="active",
        robot_root="robot",
        hand_root="hand",
        support_geoms=("table",),
        active_half_size=(0.05, 0.05, 0.05),
        support_height=0.6,
    )
    return model, data, context


def test_parallel_forecasts_are_bit_equal_and_do_not_touch_input_model_or_state(fixture):
    model, data, scene = fixture
    initial = integration_state(model, data)
    arrays = [
        a.copy()
        for a in (
            model.body_mass,
            model.dof_damping,
            model.actuator_gainprm,
            model.geom_contype,
            model.geom_conaffinity,
        )
    ]
    controls = np.zeros((10, model.nu))
    candidates = [dict(name="a", controls=controls), dict(name="b", controls=controls)]
    serial, _ = evaluate_batch(
        model, initial, candidates, scene, rates=np.ones(model.nu), workers=1
    )
    parallel, _ = evaluate_batch(
        model, initial, candidates, scene, rates=np.ones(model.nu), workers=2
    )
    for left, right in zip(serial, parallel, strict=True):
        assert left[0]["failure"] == right[0]["failure"]
        assert left[0]["fresh_task_credit"] == right[0]["fresh_task_credit"] == 0
        np.testing.assert_array_equal(left[1], right[1])
        np.testing.assert_array_equal(left[2], right[2])
    np.testing.assert_array_equal(integration_state(model, data), initial)
    np.testing.assert_array_equal(controls, np.zeros_like(controls))
    for original, current in zip(
        arrays,
        (
            model.body_mass,
            model.dof_damping,
            model.actuator_gainprm,
            model.geom_contype,
            model.geom_conaffinity,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(current, original)


@pytest.mark.parametrize(
    "control,reason", [(0.11, "native_control_limit"), (0.01, "native_target_rate")]
)
def test_invalid_applied_controls_rejected_without_a_physics_step(fixture, control, reason):
    model, data, scene = fixture
    result, _ = evaluate_batch(
        model,
        integration_state(model, data),
        [dict(name="invalid", controls=np.full((5, 1), control))],
        scene,
        rates=np.ones(1),
    )
    assert result[0][0]["failure"] == reason
    assert result[0][0]["completed_steps"] == 0
    assert not result[0][0]["eligible"]


def test_non_grasping_robot_contacts_are_in_the_same_force_guard(fixture):
    model, data, scene = fixture
    # Construct a fixture contact with the remote robot link, away from hand.
    address = int(model.joint("active_joint").qposadr[0])
    data.qpos[address : address + 3] = [0.025, 0.3, 0.65]
    mujoco.mj_forward(model, data)
    monitor = ContactMonitor(model, data, scene)
    monitor.sample(data, elapsed=0)
    assert monitor.peaks["robot_product_N"] > 0
    assert monitor.peaks["nonhand_product_N"] == monitor.peaks["robot_product_N"]
    assert monitor.last["hand_N"] == 0


def test_carton_pair_guard_sums_all_native_contact_points(fixture):
    model, data, scene = fixture
    active_address = int(model.joint("active_joint").qposadr[0])
    data.qpos[active_address : active_address + 3] = [-0.4, 0.5, 0.65]
    address = int(model.joint("other_joint").qposadr[0])
    data.qpos[address : address + 3] = [-0.31, 0.5, 0.65]
    mujoco.mj_forward(model, data)
    active, other = model.body("active").id, model.body("other").id
    normals = []
    for index, contact in enumerate(data.contact):
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if contact.efc_address >= 0 and bodies == {active, other}:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            normals.append(max(0, float(force[0])))
    assert len(normals) > 1 and sum(normals) > max(normals)
    monitor = ContactMonitor(model, data, scene)
    monitor.sample(data, elapsed=0)
    assert monitor.peaks["carton_pair_N"] == pytest.approx(sum(normals))
    # A stricter fixture limit lying between the largest point and pair sum
    # must reject; using the maximum single contact would incorrectly pass.
    threshold = (sum(normals) + max(normals)) / 2
    assert threshold < 25
    strict = ContactMonitor(model, data, replace(scene, force_limit_N=threshold))
    assert strict.sample(data, elapsed=0) == "contact_force_guard"


def test_tip_incident_survives_a_later_upright_fixture_state(fixture):
    model, data, scene = fixture
    monitor = ContactMonitor(model, data, scene)
    # Fixture/reset state construction only; no physical recovery is claimed.
    address = int(model.joint("other_joint").qposadr[0])
    data.qpos[address + 3 : address + 7] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0, 0]
    mujoco.mj_forward(model, data)
    assert monitor.sample(data, elapsed=0.002) == "product_tip_or_drop"
    data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    assert monitor.sample(data, elapsed=0.002) == "product_tip_or_drop"
    assert monitor.incidents[0]["product"] == "other"
    restored = ContactMonitor(model, data, scene)
    restored.restore_state(monitor.state_dict(), data)
    assert restored.sample(data, elapsed=0) == "product_tip_or_drop"
    assert restored.incidents[0] == monitor.incidents[0]


def test_monitor_checkpoint_preserves_contact_loss_timer_and_original_positions(fixture):
    model, data, scene = fixture
    # Fixture construction only: put the actuator-controlled hand clear.
    data.qpos[int(model.joint("drive").qposadr[0])] = 0.1
    mujoco.mj_forward(model, data)
    monitor = ContactMonitor(model, data, scene)
    assert monitor.sample(data, elapsed=0.03) is None
    restored = ContactMonitor(model, data, scene)
    restored.restore_state(monitor.state_dict(), data)
    assert restored.sample(data, elapsed=0.012) == "lost_push_contact"
    assert monitor.sample(data, elapsed=0.012) == "lost_push_contact"
    bad = restored.state_dict()
    bad["scene"]["force_limit_N"] = 26
    with pytest.raises(ValueError, match="guard contract"):
        ContactMonitor(model, data, scene).restore_state(bad, data)


def test_initial_external_force_cannot_receive_forecast_admission(fixture):
    model, data, scene = fixture
    data.qfrc_applied[0] = 0.1
    result, _ = evaluate_batch(
        model,
        integration_state(model, data),
        [dict(name="forced", controls=np.zeros((5, 1)))],
        scene,
        rates=np.ones(1),
    )
    assert result[0][0]["failure"] == "external_force_input"
    assert result[0][0]["completed_steps"] == 0


@pytest.mark.parametrize("yaw", [-0.1, 0.1])
@pytest.mark.parametrize(
    "objective", ["square_backstop", "lateral_left", "lateral_right", "align_backstop"]
)
def test_yaw_objective_keeps_force_admission_and_rejects_unknown_objectives(
    fixture, yaw, objective
):
    model, data, scene = fixture
    # Reset-only orientation fixture: compare screened free-joint quaternion
    # yaw against independently guarded native forward kinematics.
    address = int(model.jnt_qposadr[model.body_jntadr[model.body("active").id]])
    data.qpos[address + 3 : address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    candidates = [
        dict(name=name, controls=np.full((10, 1), value))
        for name, value in (("a", -0.001), ("b", 0.001))
    ]
    target_y = float(data.qpos[address + 1] + 0.001)
    scales = 0.0005, np.radians(0.15)
    result = evaluate_screened_batch(
        model,
        integration_state(model, data),
        candidates,
        scene,
        rates=np.ones(1),
        shortlist=1,
        objective=objective,
        alignment_goal=(target_y, *scales) if objective == "align_backstop" else None,
    )
    admitted = [r for r in result["candidates"] if r["eligible"]]
    assert len(admitted) == 1 and result["guarded_count"] == 1
    assert admitted[0]["active_yaw_rad"] == pytest.approx(
        admitted[0]["screen_active_yaw_rad"], abs=1e-10
    )
    assert admitted[0]["displacement_xyz_m"][1] == pytest.approx(
        admitted[0]["screen_displacement_y_m"], abs=1e-10
    )
    assert result["selected_name"] == admitted[0]["name"]
    if objective.startswith("lateral_"):
        direction = -1 if objective == "lateral_left" else 1
        assert (
            abs(
                result["candidates"][0]["screen_displacement_y_m"]
                - result["candidates"][1]["screen_displacement_y_m"]
            )
            > 1e-10
        )
        expected = max(result["candidates"], key=lambda r: direction * r["screen_displacement_y_m"])
        assert result["selected_name"] == expected["name"]
    if objective == "align_backstop":
        expected = min(
            result["candidates"],
            key=lambda row: (
                np.hypot(
                    (data.qpos[address + 1] + row["screen_displacement_y_m"] - target_y)
                    / scales[0],
                    row["screen_active_yaw_rad"] / scales[1],
                ),
                -row["screen_progress_minus_x_m"],
            ),
        )
        assert result["selected_name"] == expected["name"]
    unchecked = next(r for r in result["candidates"] if not r["eligible"])
    assert unchecked["verification_status"] == "not_force_checked"
    with pytest.raises(ValueError, match="objective"):
        evaluate_screened_batch(
            model,
            integration_state(model, data),
            candidates,
            scene,
            rates=np.ones(1),
            shortlist=1,
            objective="ignore_forces",
        )


def test_cpp_screen_never_selects_unchecked_or_guard_rejected_candidates(fixture):
    model, data, scene = fixture
    initial = integration_state(model, data)
    # All candidates undergo real native C++ simulation; a strict lower guard
    # rejects the shortlist independently of the optimistic ranking.
    candidates = [
        dict(name=f"candidate_{i}", controls=np.full((10, 1), -0.0001 * i)) for i in range(3)
    ]
    report = evaluate_screened_batch(
        model,
        initial,
        candidates,
        replace(scene, force_limit_N=0.001),
        rates=np.ones(1),
        workers=2,
        shortlist=1,
    )
    assert report["screened_count"] == 3 and report["guarded_count"] == 1
    assert report["selected_name"] is None
    assert not any(r["eligible"] for r in report["candidates"])
    assert sum(r["verification_status"] == "not_force_checked" for r in report["candidates"]) == 2
    checked = next(
        r for r in report["candidates"] if r["verification_status"] == "guarded_forecast"
    )
    assert checked["failure"] == "contact_force_guard"
    np.testing.assert_array_equal(integration_state(model, data), initial)


def test_active_backstop_force_has_its_own_guard(fixture):
    model, data, scene = fixture
    address = int(model.joint("active_joint").qposadr[0])
    data.qpos[address : address + 3] = [-0.734, 0, 0.65]
    mujoco.mj_forward(model, data)
    monitor = ContactMonitor(model, data, replace(scene, back_geom="back", force_limit_N=0.001))
    assert monitor.sample(data, elapsed=0) == "contact_force_guard"
    assert monitor.peaks["back_N"] > 0.001
    assert monitor.peaks["robot_product_N"] == 0


def test_cpp_screen_rejects_invalid_target_sequence_before_rollout(fixture):
    model, data, scene = fixture
    with pytest.raises(ValueError, match="native control"):
        evaluate_screened_batch(
            model,
            integration_state(model, data),
            [dict(name="bad", controls=np.full((5, 1), 0.2))],
            scene,
            rates=np.ones(1),
            workers=1,
            shortlist=1,
        )
