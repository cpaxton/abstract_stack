"""Isolated stocking episodes with seeded resets and guarded skill actions."""

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from abstract_stack.assets import mabel_model_path
from abstract_stack.checkpoints import native_state
from abstract_stack.contact_execution import make_backstop_monitor
from abstract_stack.contact_refinement import refine_supported_contact
from abstract_stack.env import ShelfStockingEnv
from abstract_stack.front_tangential import FrontTangentialPress
from abstract_stack.grasp_skill import grasp_lift_and_hold
from abstract_stack.interactions import InteractionFailure, ScriptConfig, StockingScript
from abstract_stack.lateral_contact import LateralPushConfig, LateralStockingScript
from abstract_stack.packing import exposed_row_capacity
from abstract_stack.provenance import policy_provenance
from abstract_stack.skill_fixtures import sample_skill_fixture

ROW_ACTIONS = ("center", "balanced-front", "exposed-side", "side-centered", "side-wall")
ROW_SKILLS = ("row-compaction", "row-press")

ACTIONS = {
    "grasp-lift": ("guarded",),
    "insert": ("paired", "single"),
    "push-to-contact": ("paired", "single"),
    "row-compaction": ROW_ACTIONS,
    "row-press": (*ROW_ACTIONS, "side-square"),
}

SIDE_ACTIONS = ("exposed-side", "side-centered", "side-wall", "side-square")


class _FixtureEnv(ShelfStockingEnv):
    def __init__(self, fixture, *, seed, mabel_root):
        self.fixture = fixture
        super().__init__(fixture.config, seed=seed, mabel_root=mabel_root)

    def _initial_qpos(self):
        state = super()._initial_qpos()
        for name, (x, y, z, yaw) in self.fixture.poses.items():
            start = int(self.model.joint(f"{name}_joint").qposadr[0])
            state[start : start + 7] = (x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        return state


class _FixtureScript(StockingScript):
    ALLOW_OFFSET_START = True

    def _sample(self, env):
        super()._sample(env)
        if self.observer.incidents:
            raise InteractionFailure(f"{self.phase}: product tipped or dropped")
        if np.any(env.data.warning.number):
            raise InteractionFailure(f"{self.phase}: native simulation warning")


class _FixtureRowScript(FrontTangentialPress, _FixtureScript):
    pass


class _FixtureSideScript(_FixtureScript, LateralStockingScript):
    row_alignment_y = None

    @property
    def lateral_goal_y(self):
        if self.row_alignment_y is not None:
            return self.row_alignment_y[self.product.name]
        return super().lateral_goal_y


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class StockingSkillEnv:
    """One policy decision runs a bounded native skill and returns measured evidence.

    ``reset`` returns observation/info and ``step`` returns the five-value
    convention used by the shelf environment. Actions select a guarded native
    primitive or preshape. Observations expose privileged MuJoCo state.
    """

    def __init__(self, skill, output, *, mabel_root=None):
        if skill not in ACTIONS:
            raise ValueError(f"Unknown skill: {skill}")
        self.skill = skill
        self.actions = ACTIONS[skill]
        self.output = Path(output)
        self.mabel_root = mabel_root
        self.env = None
        self.script = None
        self._episode = 0
        self._ready = False

    def reset(self, *, seed):
        fixture = sample_skill_fixture(self.skill, seed)
        self.close()
        self.fixture = fixture
        directory = self.output / f"episode-{self._episode:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        self._episode += 1
        self.directory = directory
        path = mabel_model_path(self.mabel_root)
        revision = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        _write(
            directory / "fixture.json",
            dict(
                kind="seeded_isolated_skill_fixture",
                skill=self.skill,
                seed=seed,
                fixture=asdict(fixture),
                actions=self.actions,
                source=policy_provenance(),
                robot_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                mujoco_version=mujoco.__version__,
            ),
        )
        try:
            self.env = _FixtureEnv(fixture, seed=seed, mabel_root=self.mabel_root)
            self.env.validate_robot_fixture_clearance()
            script = self._script(self.actions[0])
            self.script = script
            script.phase = "fixture_settle"
            for _ in range(round(self.env.config.stable_seconds / self.env.dt) + 1):
                self.env.step(on_physics_step=script._sample)
                if script.observer.incidents:
                    raise InteractionFailure("fixture_settle: product tipped or dropped")
            self.env.validate_robot_fixture_clearance()
            self._check_entry(script.last_forces)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            _write(directory / "result.json", self._result("reset_rejected", failure=str(error)))
            self.close()
            raise

        initial = native_state(self.env.model, self.env.data)
        np.savez_compressed(directory / "initial.npz", integration=initial)
        _write(
            directory / "reset.json",
            dict(
                status="ready",
                model_xml_sha256=hashlib.sha256(self.env.xml.encode()).hexdigest(),
                native_initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                physics_dt_s=self.env.config.timestep,
                control_dt_s=self.env.dt,
                staged=script.observer.snapshot(),
                initial_forces=script.last_forces,
                row=self._row() if self.skill in ROW_SKILLS else None,
            ),
        )
        self._ready = True
        return self.observe(), self._result("ready")

    def _script(self, action):
        config = ScriptConfig(
            grasp_skill=self.skill == "grasp-lift",
            head_tracking=False,
            base_reposition=True,
            ungrasped_insertion=self.skill in ("insert", "push-to-contact")
            or action in SIDE_ACTIONS,
            paired_finger_insertion=action == "paired",
        )
        kind = _FixtureScript
        if self.skill in ROW_SKILLS:
            kind = _FixtureSideScript if action in SIDE_ACTIONS else _FixtureRowScript
        script = kind(self.env, self.directory, config)
        script.product = next(p for p in self.env.products if p.name == self.fixture.active)
        script.observer = script._new_observer()
        script.observer.completed.update(self.fixture.supported_neighbors)
        script.tracker = script._new_tracker()
        script.base_start = self.env.data.body("mabel_chassis").xpos.copy()
        return script

    def _check_entry(self, forces):
        env = self.env
        if max(forces["robot_product"], forces["other_product"]) > 0.2:
            raise InteractionFailure("fixture_settle: expected an unloaded hand")
        if self.skill == "grasp-lift":
            if forces["table"] <= 0.2:
                raise InteractionFailure("fixture_settle: active carton lacks table support")
            return
        if forces["shelf"] <= 0.2:
            raise InteractionFailure("fixture_settle: active carton lacks shelf support")
        body = env.data.body(self.fixture.active)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
        )
        if body.xmat[8] < np.cos(np.radians(5)) or np.linalg.norm(velocity[3:]) >= 0.04:
            raise InteractionFailure("fixture_settle: active carton is not upright and settled")
        placed = dict(zip((p.name for p in env.products), env._placement_conditions(), strict=True))
        if not all(placed[name] for name in self.fixture.supported_neighbors):
            raise InteractionFailure("fixture_settle: a prefilled neighbor lacks stable placement")
        if self.skill in ROW_SKILLS:
            row = self._row()
            if (
                row["exposed_product"] != self.fixture.active
                or not row["envelopes_fit"]
                or row["available_m"] >= row["required_m"]
            ):
                raise InteractionFailure(
                    "fixture_settle: expected a supported row with a span deficit"
                )

    def _row(self):
        return exposed_row_capacity(
            self.env, self.fixture.supported_neighbors, self.fixture.direction
        )

    def _run(self, script, action):
        if self.skill == "grasp-lift":
            grasp_lift_and_hold(script, script.product)
            return True
        if self.skill == "insert":
            script._insert_from_front(
                partial_stop_x=self.fixture.intermediate_x_m, front_squaring=True
            )
            return any(e.get("verified_front_squaring") for e in script.events)
        if self.skill == "push-to-contact":
            script._insert_from_front(retreat=False)
            monitor, dwell = make_backstop_monitor(script)
            refine_supported_contact(script, monitor, dwell)
            return dwell.verified
        row = self._row()
        direction = self.fixture.direction
        if action in SIDE_ACTIONS:
            product_ids = [self.env.model.body(name).id for name in row["product_order"]]
            pairs = [
                tuple(sorted(pair)) for pair in zip(product_ids, product_ids[1:], strict=False)
            ]
            wall_product = product_ids[-1 if direction > 0 else 0]
            wall_press = action == "side-wall"

            def row_ready():
                current = self._row()
                return (
                    current["available_m"] >= current["required_m"]
                    and all(
                        script.carton_pair_forces.get(pair, 0) >= script.config.contact_threshold
                        for pair in pairs
                    )
                    and (
                        not wall_press
                        or script.carton_wall_forces.get(wall_product, 0)
                        >= script.config.contact_threshold
                    )
                )

            script.side_base_dx = -0.075
            event = script.push_laterally(
                direction=direction,
                distance_m=row["closable_slack_m"] + (0.002 if wall_press else 0),
                near_front_corner=True,
                row_ready=row_ready,
                front_contact_fraction=0.65 if action == "exposed-side" else 0.0,
                settings=LateralPushConfig(
                    speed_m_s=0.006,
                    force_target_N=16.0 if wall_press else 12.0,
                    timeout_s=24.0,
                    position_tolerance_m=0.0008,
                    height_m=0.018,
                ),
            )
            if action == "side-square":
                self._square_compacted_row(script)
            return event["retreat_verified"]
        # Offset against the lateral press direction so the depth preload
        # counters the yaw moment from translating along the front face.
        offset = -direction * 0.006 if action == "balanced-front" else 0.0
        base_y = self.env.data.body("mabel_chassis").xpos[1]
        product_y = self.env.data.body(script.product.name).xpos[1]
        shift_y = float(product_y + 0.12 - base_y)
        if abs(shift_y) > 0.09:
            script._move_base([0, shift_y], reason="row_press_workspace")
        event = script.push_front_tangentially(
            direction=direction,
            distance_m=row["closable_slack_m"],
            contact_offset_m=offset,
            force_guided=True,
            row_products=self.fixture.supported_neighbors,
            staged_approach=True,
        )
        return event["retreat_verified"]

    def _square_compacted_row(self, script):
        """Try the existing strict backstop finish when the compacted row is near it."""
        from abstract_stack.contact_execution import hand_clears_shelf, shelf_exit_distance
        from abstract_stack.finger_pusher import FoldedFingerPusher

        entry = self._row_finish(script)
        script._stage("row_square_admission")
        script.events[-1]["row_compaction"] = entry
        if not entry["staging_compaction_verified"]:
            raise InteractionFailure("row_square_admission: compacted staging gate failed")
        env = self.env
        # Check the depth and clearance of the entire phase before changing
        # hand posture. A row with deliberate rear clearance needs another
        # alignment strategy to preserve its depth target.
        products = [p for p in env.products if p.name in self.fixture.supported_neighbors]
        back = env.config.front_x - env.config.depth
        for product in products:
            body = env.data.body(product.name)
            extent = np.abs(body.xmat.reshape(3, 3)) @ product.half_size
            if body.xpos[0] - extent[0] - back > 0.005:
                raise InteractionFailure("row_square_admission: row has rear clearance")
        if not hand_clears_shelf(env, script.hand_bodies):
            # The independent lateral exit only requires the hand to clear
            # the carton. Continue through a checked free-shelf path here.
            distance = shelf_exit_distance(env, script.controller)
            script._cartesian_move(
                "row_square_clear_hand", script.controller.position() + [distance, 0, 0.04]
            )
            if not hand_clears_shelf(env, script.hand_bodies):
                raise InteractionFailure("row_square_admission: hand did not clear the shelf")
        script.config = replace(script.config, paired_finger_insertion=True)
        script.row_alignment_y = {p.name: float(env.data.body(p.name).xpos[1]) for p in products}
        script.refinement_horizon_s = 0.4
        script.refinement_pad_angles = True
        script.events[-1].update(
            square_settings=asdict(script.config),
            measured_row_alignment_y_m=script.row_alignment_y,
        )
        script.PAIRED_PUSHER_CLASS = FoldedFingerPusher
        for product in products:
            script.product = product
            script._restore_grasp_hand()
            script._settle_pusher_preshape()
            script._insert_from_front(
                staged_preshape=True, use_lateral_reference=True, precontact_standoff_m=0.038
            )

    def _row_finish(self, script):
        row = self._row()
        order = row["product_order"]
        pairs = {
            frozenset((first, second)) for first, second in zip(order, order[1:], strict=False)
        }
        start = next(
            (
                e["sim_time"]
                for e in script.events
                if e["phase"] in ("front_tangential_push", "lateral_push")
            ),
            np.inf,
        )
        end = next(
            (
                e["sim_time"]
                for e in script.events
                if e["phase"] in ("tangential_unload", "lateral_unload")
            ),
            np.inf,
        )
        touched = {
            frozenset((first, second))
            for time, first, second, force in script.object_pair_bins.values()
            if start < time <= end and force >= script.config.contact_threshold
        }
        yaws = {
            name: float(
                np.degrees(
                    np.arctan2(self.env.data.body(name).xmat[3], self.env.data.body(name).xmat[0])
                )
            )
            for name in order
        }
        evidence = dict(
            final_geometry=row,
            force_verified_neighbor_pairs=sorted(sorted(pair) for pair in touched & pairs),
            final_yaw_deg=yaws,
            required_neighbor_contacts=len(pairs),
        )
        required = set(order)
        evidence["all_row_products_supported"] = all(
            ok
            for product, ok in zip(self.env.products, self.env._placement_conditions(), strict=True)
            if product.name in required
        )
        evidence["both_arms_unloaded"] = (
            max(
                script.last_forces.get("robot_product", 0),
                script.last_forces.get("other_product", 0),
            )
            <= 0.2
        )
        # Reserve room for the entire projected overlap to rebound toward the
        # opening. This certifies a staging handoff with room for realignment.
        evidence["opening_after_overlap_reserve_m"] = row["available_m"] - row["overlap_sum_m"]
        evidence["staging_compaction_verified"] = bool(
            evidence["opening_after_overlap_reserve_m"] >= row["required_m"]
            and evidence["all_row_products_supported"]
            and evidence["both_arms_unloaded"]
            and max(row["gaps_including_walls_m"][1:-1]) <= 0.001
            and pairs <= touched
        )
        evidence["alignment_and_contact_verified"] = bool(
            evidence["staging_compaction_verified"]
            and row["envelopes_fit"]
            and max(abs(yaw) for yaw in yaws.values()) <= 0.15
        )
        return evidence

    def _result(self, outcome, **extra):
        return dict(
            skill=self.skill,
            outcome=outcome,
            active=self.fixture.active,
            fresh_task_credit=0,
            training_eligible=False,
            **extra,
        )

    def observe(self):
        return dict(
            self.env.observe(),
            active_product=self.fixture.active,
            row_direction=self.fixture.direction,
            intermediate_x_m=self.fixture.intermediate_x_m,
            row_capacity=self._row() if self.skill in ROW_SKILLS else None,
        )

    def step(self, action):
        if not self._ready:
            raise RuntimeError("Reset the skill environment before taking a decision")
        if action not in self.actions:
            raise ValueError(f"Choose one of {self.actions}")
        self._ready = False
        script = self._script(action)
        if self.script.execution_history is not None:
            script.execution_history = self.script.execution_history
            script._agent_skill = self.script._agent_skill
            script._agent_branch = self.script._agent_branch
            script._agent_artifacts = self.script._agent_artifacts
        self.script = script
        start = float(self.env.data.time)
        verified = False
        failure = None
        try:
            verified = self._run(script, action) and not script.observer.incidents
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failure = str(error)
        row = self._row_finish(script) if self.skill in ROW_SKILLS else None
        if row is not None:
            gate = (
                "staging_compaction_verified"
                if self.skill == "row-compaction"
                else "alignment_and_contact_verified"
            )
            verified = verified and row[gate]
        if not verified and failure is None:
            failure = "measured skill gate did not pass"
        elapsed = float(self.env.data.time) - start
        truncated = bool(self.env.data.time >= self.env.config.time_limit)
        result = self._result(
            "verified" if verified else "timeout" if truncated else "failed",
            action=action,
            success=bool(verified),
            failure=failure,
            elapsed_sim_s=elapsed,
            controller=asdict(script.config),
            tracking_config=asdict(script.tracker.config),
            incidents=script.observer.incidents,
            events=script.events,
            contact=script.episode_evidence(),
            last_forces=script.last_forces,
            row_compaction=row,
        )
        np.savez_compressed(
            self.directory / "trace.npz",
            controls=np.asarray(script.controls).reshape(-1, self.env.model.nu),
            samples=np.asarray(script.rows),
            terminal=native_state(self.env.model, self.env.data),
        )
        _write(self.directory / "result.json", result)
        observation = self.observe()
        return observation, float(verified), not truncated, truncated, result

    def close(self):
        self._ready = False
        self.script = None
        if self.env is not None:
            self.env.close()
            self.env = None
