"""Bounded native forecasts choose a reference that stays square after release."""

import copy
import hashlib
import json
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from abstract_stack.contact_refinement import refine_supported_contact
from abstract_stack.contact_rollout import ContactMonitor, integration_state
from abstract_stack.mabel_controller import MabelController
from abstract_stack.native_views import retain_data_views
from abstract_stack.withdrawal import supported_withdrawal
from abstract_stack.withdrawal_feedback import WithdrawalFeedbackConfig


class ReleaseForecast:
    """Own all mutable controller/data state; share only model and frozen config.

    Task bookkeeping is deliberately absent. ContactMonitor guards every product;
    a successful forecast earns no task credit and requires fresh live execution.
    """

    def __init__(self, script):
        from abstract_stack.interactions import contact_forces

        snapshot = script.env.clone_for_planning()
        self.views = retain_data_views(snapshot.model, snapshot.data)
        self.env = SimpleNamespace(
            model=snapshot.model,
            data=snapshot.data,
            dt=snapshot.dt,
            config=script.env.config,
            update_task_state=lambda dt: None,
        )
        self.config, self.product = script.config, script.product
        self.lateral_goal_y = float(script.lateral_goal_y)
        live = script.controller
        settings = dict(
            finger=live.finger, tilt_degrees=live.tilt_degrees, wrist_posture=live.wrist_posture
        )
        self.controller = type(live)(snapshot, script.config, **settings)
        hand = self.controller
        hand.OFFSET = live.OFFSET.copy()
        hand.ik.offset = live.ik.offset.copy()
        hand.command = live.command.copy()
        hand.last_target = live.last_target.copy()
        hand.active_rotation = live.active_rotation.copy()
        hand.abduction_targets = dict(live.abduction_targets)
        hand.open_posture_overrides = dict(live.open_posture_overrides)
        if hasattr(live, "tip_body_references"):
            hand.tip_body_references = {k: v.copy() for k, v in live.tip_body_references.items()}
        hand.ROTATION = live.ROTATION.copy()
        hand.ik.lower, hand.ik.upper = live.ik.lower.copy(), live.ik.upper.copy()
        hand.ik.reference_ranges = dict(live.ik.reference_ranges)
        hand.ik.posture_constraints = dict(live.ik.posture_constraints)
        self.tracker = MabelController(
            snapshot.model,
            snapshot.data,
            script.tracker.config,
            nominal_references=dict(zip(script.tracker.NAMES, script.tracker.nominal, strict=True)),
        )
        self.tracker.applied = script.tracker.applied.copy()
        self.events, self.controls, self.nominal = [], [], []
        self.hand_bodies = set(script.hand_bodies)
        self._forces = contact_forces
        self.last_forces = self._forces(self.env, self.product.name, self.hand_bodies)
        self.env.products = tuple(script.env.products)
        self.env._product_bodies = np.array(
            [snapshot.model.body(p.name).id for p in self.env.products]
        )
        self.env._product_joints = [
            snapshot.model.joint(f"{p.name}_joint") for p in self.env.products
        ]
        self.env._boards = np.array(
            [snapshot.model.geom(f"shelf_board_{p.level}").id for p in self.env.products]
        )

    def _stage(self, phase):
        self.events.append(
            dict(phase=phase, product=self.product.name, sim_time=float(self.env.data.time))
        )

    def _sample(self, env):
        self.last_forces = self._forces(env, self.product.name, self.hand_bodies)
        self.controls.append(env.data.ctrl.copy())
        self.nominal.append(self.controller.command.copy())


def choose_release_reference(script, monitor, dwell):
    """Try at most seven native square/unload/retreat forecasts on one fixed state.

    A small bounded yaw reference anticipates elastic/contact unloading drift.
    Every candidate must satisfy the same .15 degree gate before AND after
    release. No criterion, force guard, collision permission or native gain changes.
    """
    from abstract_stack.contact_execution import shelf_exit_distance
    from abstract_stack.env import ShelfStockingEnv
    from abstract_stack.interactions import InteractionFailure

    initial = integration_state(script.env.model, script.env.data)
    script._stage("release_forecast")
    event = script.events[-1]
    event.update(
        maximum_candidates=7,
        wall_budget_seconds=60.0,
        fresh_task_credit=0,
        unchanged_yaw_gate_deg=0.15,
        candidates=[],
    )
    directory = script.directory / "release-forecasts"
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    plans = [
        (0.0, None),
        (0.13, None),
        (-0.13, None),
        (0.10, None),
        (-0.10, None),
        (0.0, 0.6),
        (0.0, 0.4),
    ]
    for index, (target, preload) in enumerate(plans):
        if time.monotonic() - started > 60:
            break
        forecast = ReleaseForecast(script)
        env, hand = forecast.env, forecast.controller
        if not np.array_equal(initial, integration_state(env.model, env.data)):
            raise InteractionFailure("release_forecast: clone initialization changed native state")
        guarded = ContactMonitor(env.model, env.data, monitor.scene)
        guarded.restore_state(monitor.state_dict(), env.data)
        square = copy.deepcopy(dwell)
        report = None
        failure = None
        try:
            if target or preload is not None:
                refine_supported_contact(
                    forecast, guarded, square, yaw_target_deg=target, normal_target_N=preload
                )
            report, _, _ = supported_withdrawal(
                hand,
                hand.command.copy(),
                guarded,
                square,
                clear_distance_m=shelf_exit_distance(env, hand),
                clear_speed_m_s=0.02,
                feedback=WithdrawalFeedbackConfig(
                    yaw_feedback=True,
                    yaw_reference_deg=target,
                    paired_plane_feedback=bool(getattr(hand, "fingers", ())),
                ),
                on_physics_step=forecast._sample,
            )
            failure = report["failure"]
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failure = str(error)
        body = env.data.xmat[guarded.active].reshape(3, 3)
        yaw = float(np.degrees(np.arctan2(body[1, 0], body[0, 0])))
        artifact = directory / f"{script.product.name}-{index:02}.npz"
        np.savez_compressed(
            artifact,
            initial_state=initial,
            state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
            terminal_state=integration_state(env.model, env.data),
            controls=np.asarray(forecast.controls).reshape(-1, env.model.nu),
            nominal=np.asarray(forecast.nominal).reshape(-1, len(hand.command)),
        )
        required = set(script.observer.completed) | {script.product.name}
        good = ShelfStockingEnv._placement_conditions(env)
        placements = {
            p.name: bool(good[i]) for i, p in enumerate(env.products) if p.name in required
        }
        eligible = bool(report and report["release_retreat_verified"] and all(placements.values()))
        registration_budget = None
        task_lateral_error = abs(float(env.data.xpos[guarded.active, 1]) - script.product.target[1])
        registration_error = abs(float(env.data.xpos[guarded.active, 1]) - script.lateral_goal_y)
        if env.config.row_free_space is not None and script.config.paired_finger_insertion:
            count = sum(p.level == script.product.level for p in env.products)
            registration_budget = min(0.001, env.config.row_free_space / count)
            eligible = eligible and registration_error <= registration_budget
        if (
            eligible
            and env.config.row_free_space is not None
            and len(required) == len(env.products)
        ):
            from abstract_stack.packing import row_geometry

            eligible = row_geometry(env)["envelopes_fit"]
        if report and report["release_retreat_verified"] and not eligible:
            failure = "forecast_required_placement_or_dense_envelope"
        row = dict(
            yaw_reference_deg=target,
            normal_target_N=preload,
            eligible=eligible,
            required_placement_conditions=placements,
            failure=failure,
            terminal_yaw_deg=yaw,
            active_lateral_error_m=registration_error,
            active_task_target_lateral_error_m=task_lateral_error,
            controller_lateral_goal_world_y_m=script.lateral_goal_y,
            controller_lateral_admission_budget_m=registration_budget,
            simulated_seconds=float(env.data.time - script.env.data.time),
            artifact=str(artifact.relative_to(script.directory)),
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            fresh_task_credit=0,
            training_eligible=False,
            withdrawal=report,
            preparation_events=forecast.events,
        )
        event["candidates"].append(row)
        (artifact.with_suffix(".json")).write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n"
        )
        if not np.array_equal(initial, integration_state(script.env.model, script.env.data)):
            raise InteractionFailure("release_forecast: live state changed during planning")
        if row["eligible"]:
            selected = dict(yaw_reference_deg=target, normal_target_N=preload)
            event.update(selected=selected, wall_seconds=time.monotonic() - started)
            return selected
        if index == 0:
            # Use the measured full-withdrawal forecast error to prioritize a
            # bounded reference, instead of spending the budget on blind force
            # variants. Actual yaw gates remain +/-0.15 degrees throughout.
            correction = float(np.clip(-yaw, -0.13, 0.13))
            plans[1:3] = [(correction, None), (-correction, None)]
            event["forecast_drift_correction_deg"] = correction
            if yaw > 0:
                plans[3:5] = [(-0.10, None), (0.10, None)]
    event["wall_seconds"] = time.monotonic() - started
    raise InteractionFailure("release_forecast: no verified bounded reference/withdrawal candidate")
