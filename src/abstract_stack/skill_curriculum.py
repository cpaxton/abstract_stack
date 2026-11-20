"""Gymnasium episodes for guarded stocking primitives and measured reset states."""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from abstract_stack.agent_history import AgentHistory
from abstract_stack.checkpoint_manifest import sha256_file
from abstract_stack.checkpoints import load_checkpoint, native_state
from abstract_stack.contact_execution import make_backstop_monitor, squared_contact_retreat
from abstract_stack.contact_refinement import refine_supported_contact
from abstract_stack.continuation import CheckpointContinuationEnv
from abstract_stack.finger_pusher import FingerPusher
from abstract_stack.grasp_skill import grasp_lift_and_hold
from abstract_stack.interactions import ScriptConfig
from abstract_stack.interleaved_stocking import PressInterleavedStockingScript
from abstract_stack.mabel_controller import MabelController
from abstract_stack.provenance import policy_provenance
from abstract_stack.skill_checkpoint import restore_skill_controller
from abstract_stack.skill_env import StockingSkillEnv, _FixtureScript, _write
from abstract_stack.tuning import load_control_config

ACTIONS = {
    "pick": ("guarded",),
    "place": ("seat",),
    "insert": ("supported-release", "wall-center", "wall-front-corner", "wall-clearance"),
    "push": ("recorded", "paired", "single", "release-reapproach"),
    "slide": ("side-centered", "balanced-front"),
    "retreat": ("recorded", "live"),
}

PROCEDURAL = {
    "pick": ("grasp-lift", ("guarded",)),
    "push": ("push-to-contact", ("single", "paired")),
    "slide": ("row-compaction", ACTIONS["slide"]),
}

CAPTURED_ENTRIES = {
    "pick": {"grasp_lift", "motion_failure"},
    "place": {"transfer", "motion_failure"},
    "insert": {"front_release", "free_landing", "motion_failure"},
    "push": {
        "contact_finish",
        "front_insertion",
        "front_squaring_complete",
        "motion_failure",
        "release_reapproach",
    },
    "slide": {"row_press_entry"},
    "retreat": {"contact_refinement", "release_retreat", "withdrawal_planning", "motion_failure"},
}

FORCES = ("hand", "grip_normal", "robot_product", "other_product", "table", "shelf", "back")


class MeasuredResetRejected(ValueError):
    """An authenticated native start failed the skill's entry gate."""

    def __init__(self, sim_time_s):
        super().__init__("Measured skill reset preconditions failed")
        self.sim_time_s = sim_time_s


@dataclass(frozen=True)
class CurriculumCase:
    id: str
    skill: str
    role: str
    checkpoint: Path
    manifest_sha256: str
    lineage: str
    original_failure: str | None = None
    historical_recovery: str | None = None
    replan_released_tool: bool = False
    rewind_case_id: str | None = None


def load_curriculum(path, *, artifact_root):
    """Read a corpus of immutable checkpoint references relative to an artifact root."""
    manifest = json.loads(Path(path).read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported curriculum manifest version")
    root = Path(artifact_root).resolve()
    cases = []
    for entry in manifest["cases"]:
        relative = Path(entry["checkpoint"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Case checkpoint must live under the explicit artifact root")
        cases.append(CurriculumCase(**{**entry, "checkpoint": root / relative}))
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate curriculum case ID")
    if any(
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", case.id)
        or case.skill not in ACTIONS
        or case.role not in {"captured", "recovery"}
        or not case.lineage
        or (case.replan_released_tool and case.skill != "push")
        or (case.role == "recovery" and not case.original_failure)
        for case in cases
    ):
        raise ValueError("Invalid curriculum skill, role, lineage or recovery evidence")
    by_id = {case.id: case for case in cases}
    for case in cases:
        if case.rewind_case_id is None:
            continue
        rewind = by_id.get(case.rewind_case_id)
        if (
            rewind is None
            or rewind.rewind_case_id is not None
            or case.role != "recovery"
            or rewind.role != "recovery"
            or (case.lineage, case.skill) != (rewind.lineage, rewind.skill)
        ):
            raise ValueError("A rewind must name an independent earlier case in the same lineage")
    return tuple(cases)


def _sample_case(rng, choices):
    """Give each episode lineage equal weight regardless of its checkpoint count."""
    if len(choices) == 1:
        return choices[0]
    by_lineage = {}
    for case in sorted(choices, key=lambda c: (c.lineage, c.id)):
        by_lineage.setdefault(case.lineage, []).append(case)
    groups = list(by_lineage.values())
    group = groups[int(rng.integers(len(groups)))]
    return group[int(rng.integers(len(group)))]


def _failed_phase(entry):
    return (
        (entry.get("case") or {}).get("failed_phase")
        if entry["skill"] == "motion_failure"
        else None
    )


def _loaded_route_stop(entry):
    return (
        _failed_phase(entry) == "base_reposition_search"
        and entry["case"].get("failure") == "loaded_stage: no checked shelf approach candidate"
    )


def _wall_registration_entry(entry):
    return entry["skill"] == "free_landing" or _failed_phase(entry) in {
        "lateral_front_exit",
        "lateral_plan",
        "lateral_push",
    }


def _admit(script, entry, skill):
    grip, carry, forces = script.active_grip, script.carry_reference, script.last_forces
    if entry["skill"] not in CAPTURED_ENTRIES[skill] or script.observer.incidents:
        raise ValueError("Captured start needs the matching skill and an incident-free history")
    if skill == "pick":
        failed_phase = _failed_phase(entry)
        return (
            grip is None
            and carry is None
            and forces["table"] > 0.2
            and (
                entry["skill"] == "grasp_lift"
                or (
                    failed_phase in {"grasp_candidate", "pregrasp_route", "close"}
                    and forces["hand"] <= 0.2
                )
            )
        )
    if skill == "place":
        return (
            grip is not None
            and carry is not None
            and forces["opposing_contacts"]
            and (entry["skill"] == "transfer" or _loaded_route_stop(entry))
        )
    if skill in {"insert", "push", "retreat"}:
        env = script.env
        body = env.data.body(script.product.name)
        extent = np.abs(body.xmat.reshape(3, 3)) @ script.product.half_size
        gap = body.xpos[2] - extent[2] - env.config.levels[script.product.level]
        held_at_board = (
            grip is not None
            and forces["opposing_contacts"]
            and forces["table"] <= 0.15
            and env.config.front_x - env.config.depth <= body.xpos[0] <= env.config.front_x
            and abs(body.xpos[1] - env.config.center_y) <= env.config.width / 2
            and -0.002 <= gap <= 0.003
        )
        supported = forces["shelf"] > 0.2
        failed_phase = _failed_phase(entry)
        if skill == "insert":
            if _wall_registration_entry(entry):
                return (
                    supported
                    and grip is None
                    and carry is None
                    and forces["hand"] <= 0.2
                    and script._wall_front_staging_y() != script.product.target[1]
                )
            return (
                held_at_board
                and supported
                and carry is None
                and (
                    entry["skill"] == "front_release"
                    or failed_phase in {"stage_align", "release_unload", "release_front"}
                )
            )
        if skill == "push":
            return (supported if grip is None else held_at_board) and (
                entry["skill"] != "motion_failure"
                or failed_phase == "push"
                or (
                    failed_phase in {"finger_push", "pusher_approach"}
                    and grip is None
                    and carry is None
                )
            )
        if isinstance(script.controller, FingerPusher):
            if entry["skill"] == "motion_failure":
                if not (supported and grip is None and carry is None):
                    return False
                if failed_phase == "unloaded_clear_forecast":
                    return forces["hand"] <= 0.2
                return (
                    failed_phase == "contact_refine"
                    and entry["case"].get("failure")
                    in (
                        "contact_refine: no fresh guarded candidate",
                        "contact_refine: bounded release-goal search exhausted",
                    )
                    and forces["hand"] > 0.2
                )
            return supported
        if entry["skill"] == "motion_failure":
            return False
        return held_at_board and forces["back"] >= script.config.contact_threshold
    if skill == "slide":
        return grip is None and carry is None and len(script.observer.completed) == 2
    raise ValueError(f"Unknown captured skill {skill}")


def _run_captured(script, skill, action, entry):
    if skill == "pick":
        grasp_lift_and_hold(script, script.product)
        return True
    if skill == "place":
        script._transfer_and_place_front(
            script.active_grip, already_clear=_loaded_route_stop(entry)
        )
        return script.last_forces["shelf"] > 0.2
    if skill == "insert":
        if _wall_registration_entry(entry):
            if action == "wall-clearance":
                script._clear_registered_wall_hand()
            else:
                script._register_front_laterally(near_front_corner=action == "wall-front-corner")
            script._stage("front_registered")
        else:
            script._release_and_insert_front(script.active_grip, push=False)
        return (
            script.phase == "front_registered"
            and script.active_grip is None
            and script.last_forces["shelf"] > 0.2
            and script.last_forces["robot_product"] <= 0.2
        )
    if skill == "slide":
        evidence = {}
        script._press_stocked_pair(evidence)
        return evidence["interleaved_row_press"]["verified_staging_press"]
    if skill == "push":
        if action == "release-reapproach":
            script._release_for_finger_push()
        if script.active_grip is not None:
            script._push_grasped(script.active_grip, retreat=False)
            return bool(script.events[-1].get("back_contact_verified"))
        if action != "recorded":
            script.config = replace(
                script.config,
                ungrasped_insertion=True,
                paired_finger_insertion=action != "single",
            )
        if entry["skill"] == "front_squaring_complete":
            script._insert_from_front(
                retreat=False, precontact_standoff_m=script.config.post_square_pusher_standoff_m
            )
        else:
            script._insert_from_front(retreat=False)
        monitor, dwell = make_backstop_monitor(script)
        refine_supported_contact(script, monitor, dwell)
        return dwell.verified
    return _retreat_captured(
        script,
        live=action == "live",
        unloaded_entry=_failed_phase(entry) == "unloaded_clear_forecast",
    )


def _retreat_captured(script, *, live=False, unloaded_entry=False):
    if isinstance(script.controller, FingerPusher):
        if live:
            script.config = replace(script.config, live_guarded_withdrawal=True)
        push_event = getattr(script, "_active_push_event", None)
        squared_contact_retreat(
            script, push_event if push_event is not None else {}, unloaded_entry=unloaded_entry
        )
    else:
        script._retreat_grasped()
    return bool(script.observer.retreat_results[script.product.name]["verified"])


class StockingCurriculumEnv(gym.Env):
    """One option decision per reset, with guarded native physics inside the step."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        skill,
        output,
        *,
        cases=(),
        source=None,
        max_products=10,
        mabel_root=None,
        render_mode=None,
        control_config=None,
        verify_retreat=False,
    ):
        if skill not in ACTIONS or type(max_products) is not int or max_products < 1:
            raise ValueError("Choose a curriculum skill and positive product capacity")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Use rgb_array or no rendering")
        self.skill = skill
        self.actions = ACTIONS[skill]
        self.cases = tuple(case for case in cases if case.skill == skill)
        self.source = source or ("procedural" if skill in PROCEDURAL else "captured")
        self.output = Path(output)
        self.max_products = max_products
        self.render_mode = render_mode
        self.control_config = control_config
        self.control_overrides = load_control_config(control_config)
        if verify_retreat and skill != "push":
            raise ValueError("Retreat handoff verification requires a push case")
        self.verify_retreat = verify_retreat
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict(
            {
                "products": spaces.Box(-np.inf, np.inf, (max_products, 20), np.float64),
                "present": spaces.MultiBinary(max_products),
                "active": spaces.MultiBinary(max_products),
                "stocked": spaces.MultiBinary(max_products),
                "joints": spaces.Box(-np.inf, np.inf, (len(MabelController.NAMES), 3), np.float64),
                "base_pose": spaces.Box(-np.inf, np.inf, (7,), np.float64),
                "tool_position": spaces.Box(-np.inf, np.inf, (3,), np.float64),
                "forces": spaces.Box(-np.inf, np.inf, (len(FORCES),), np.float64),
            }
        )
        self.fixture = None
        if skill in PROCEDURAL:
            self.fixture = StockingSkillEnv(
                PROCEDURAL[skill][0], self.output / "procedural", mabel_root=mabel_root
            )
        self.env = self.script = None
        self._episode = 0
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        source = options.pop("source", self.source)
        case_id = options.pop("case_id", None)
        if options or source not in {"procedural", "captured", "recovery"}:
            raise ValueError("Reset options are source and case_id")
        self.close()
        self._source = source
        self._case = None
        try:
            if source == "procedural":
                if self.verify_retreat:
                    raise ValueError("Retreat handoff verification requires a captured start")
                if self.control_overrides:
                    raise ValueError("Controller overrides require a captured curriculum case")
                if self.fixture is None or case_id is not None:
                    raise ValueError("This reset requires a captured handoff")
                actual_seed = seed if seed is not None else int(self.np_random.integers(2**63))
                self.fixture.reset(seed=actual_seed)
                self.env, self.directory = self.fixture.env, self.fixture.directory
                self.script = self.fixture.script
            else:
                choices = [
                    c
                    for c in self.cases
                    if c.role == source and (case_id is None or c.id == case_id)
                ]
                if not choices:
                    raise ValueError(f"No matching {source} case for {self.skill}")
                self._case = _sample_case(self.np_random, choices)
                self._restore(self._case)
            if len(self.env.products) > self.max_products:
                raise ValueError("Scene exceeds the fixed observation product capacity")
            observation = self._observe()
        except (OSError, RuntimeError, ValueError, FloatingPointError):
            self.close()
            raise
        self._ready = True
        return observation, self._info()

    def _restore(self, case):
        if sha256_file(case.checkpoint / "manifest.json") != case.manifest_sha256:
            raise ValueError("Curriculum checkpoint manifest changed")
        checkpoint = load_checkpoint(case.checkpoint)
        self.env = env = CheckpointContinuationEnv(checkpoint)
        env.restore_task_memory(checkpoint.context["task_memory"])
        directory = self.output / f"{case.role}-{self._episode:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self._episode += 1
        config = replace(
            ScriptConfig(**checkpoint.controller["settings"]),
            record_skill_starts=False,
            head_tracking=False,
        )
        kind = PressInterleavedStockingScript if self.skill == "slide" else _FixtureScript
        if self.skill == "slide":
            config = replace(config, ungrasped_insertion=True)
        self.script = kind(env, directory, config)
        controller_initialization = restore_skill_controller(
            self.script, checkpoint, replan_released_tool=case.replan_released_tool
        )
        if self.control_overrides:
            self.script.config = replace(self.script.config, **self.control_overrides)
            self.script.controller.config = replace(
                self.script.controller.config, **self.control_overrides
            )
        entry = checkpoint.context["skill_entry"]
        self._entry = entry
        initial = native_state(env.model, env.data)
        np.savez_compressed(directory / "initial.npz", integration=initial)
        admitted = bool(
            env.data.time < env.config.time_limit and _admit(self.script, entry, self.skill)
        )
        _write(
            directory / "reset.json",
            dict(
                case_id=case.id,
                lineage=case.lineage,
                checkpoint=str(case.checkpoint),
                manifest_sha256=case.manifest_sha256,
                seed=env._seed,
                skill=self.skill,
                original_failure=case.original_failure,
                historical_recovery=case.historical_recovery,
                controller_initialization=controller_initialization,
                controller_overrides=self.control_overrides,
                status="ready" if admitted else "reset_rejected",
                admitted=admitted,
                source=policy_provenance(),
                recorder_source=checkpoint.manifest.get("source_capture"),
                physics_dt_s=env.config.timestep,
                control_dt_s=env.dt,
                fresh_task_credit=0,
            ),
        )
        if not admitted:
            raise MeasuredResetRejected(float(env.data.time))

    def _action_mask(self):
        mask = np.ones(len(self.actions), dtype=np.int8)
        if self.skill == "insert" and self._case is not None:
            wall_registration = _wall_registration_entry(self._entry)
            if _failed_phase(self._entry) == "lateral_front_exit":
                return np.array([0, 0, 0, 1], dtype=np.int8)
            if wall_registration:
                return np.array([0, 1, 1, 0], dtype=np.int8)
            return np.array([1, 0, 0, 0], dtype=np.int8)
        if self.skill == "push":
            if self._case is None:
                mask[2:] = 0
            elif self.script.active_grip is not None:
                mask[1:3] = 0
            else:
                mask[-1] = 0
                if self.script.config.wall_packing or self.script.config.center_last:
                    mask[self.actions.index("single")] = 0
            return mask
        if (
            self._case is not None
            and len(mask) > 1
            and (
                self.skill == "slide"
                or (
                    self.skill == "retreat"
                    and (
                        not isinstance(self.script.controller, FingerPusher)
                        or self._entry["skill"] == "motion_failure"
                    )
                )
            )
        ):
            mask[1] = 0
        return mask

    def _info(self):
        return dict(
            skill=self.skill,
            source=self._source,
            case_id=None if self._case is None else self._case.id,
            lineage=None if self._case is None else self._case.lineage,
            active_product=self.script.product.name,
            action_mask=self._action_mask(),
            scene_seed=self.env._seed,
            fresh_task_credit=0,
        )

    def _observe(self):
        model, data = self.env.model, self.env.data
        products = np.zeros((self.max_products, 20))
        present = np.zeros(self.max_products, dtype=np.int8)
        active, stocked = present.copy(), present.copy()
        velocity = np.empty(6)
        for i, product in enumerate(self.env.products):
            body = data.body(product.name)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0)
            products[i] = np.r_[
                body.xpos,
                body.xquat,
                product.target,
                product.half_size,
                velocity,
                model.body_mass[body.id],
            ]
            present[i], active[i], stocked[i] = (
                1,
                product == self.script.product,
                self.env._stocked[i],
            )
        ids = [model.actuator(name).id for name in MabelController.NAMES]
        joints = np.column_stack(
            (data.actuator_length[ids], data.actuator_velocity[ids], data.ctrl[ids])
        )
        base = data.body("mabel_chassis")
        return dict(
            products=products,
            present=present,
            active=active,
            stocked=stocked,
            joints=joints,
            base_pose=np.r_[base.xpos, base.xquat],
            tool_position=self.script.controller.position().copy(),
            forces=np.array([self.script.last_forces.get(k, 0.0) for k in FORCES]),
        )

    def step(self, action):
        if not self._ready:
            raise gym.error.ResetNeeded("Reset the curriculum before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the declared skill action space")
        self._ready = False
        info = self._info()
        name = self.actions[action]
        start = float(self.env.data.time)
        if not info["action_mask"][action]:
            info.update(action=name, success=False, failure="Action inadmissible for captured tool")
            _write(
                self.directory / "result.json",
                {**info, "action_mask": info["action_mask"].tolist()},
            )
            return self._observe(), 0.0, True, False, info
        if self._case is None:
            _, reward, terminated, truncated, result = self.fixture.step(
                PROCEDURAL[self.skill][1][action]
            )
            self.script = self.fixture.script
            result_keys = ("success", "failure", "elapsed_sim_s")
            info.update(action=name, **{k: result[k] for k in result_keys})
            return self._observe(), reward, terminated, truncated, info

        script = self.script
        initial = native_state(self.env.model, self.env.data)
        success, failure = False, None
        verified_skills = []
        try:
            success = bool(
                _run_captured(script, self.skill, name, self._entry)
                and not script.observer.incidents
            )
            if success:
                verified_skills.append(self.skill)
            if success and self.verify_retreat:
                script._record_agent_event(
                    "skill_finished", skill="push", skill_step=1, outcome="verified"
                )
                script._agent_skill = "retreat"
                script._record_agent_event("skill_started", skill="retreat", skill_step=2)
                success = False
                success = bool(_retreat_captured(script) and not script.observer.incidents)
                if success:
                    verified_skills.append("retreat")
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failure = str(error)
        truncated = bool(self.env.data.time >= self.env.config.time_limit)
        if not success and failure is None:
            failure = "Measured skill gate did not pass"
        if self.verify_retreat and "push" in verified_skills:
            script._record_agent_event(
                "skill_finished",
                skill="retreat",
                skill_step=2,
                outcome="verified" if success else "failed",
                failure=failure,
            )
        info.update(
            action=name,
            success=success,
            failure=failure,
            elapsed_sim_s=float(self.env.data.time) - start,
            verified_skills=verified_skills,
        )
        _write(
            self.directory / "result.json",
            dict(
                **{**info, "action_mask": info["action_mask"].tolist()},
                controller=vars(script.config),
                incidents=script.observer.incidents,
                last_forces=script.last_forces,
                events=script.events,
                contact=script.episode_evidence(),
                stage_evaluation=script.observer.report(script.events),
            ),
        )
        np.savez_compressed(
            self.directory / "continuation.npz",
            controls=np.asarray(script.controls).reshape(-1, self.env.model.nu),
            samples=np.asarray(script.rows),
            initial_state=initial,
            terminal_state=native_state(self.env.model, self.env.data),
        )
        return self._observe(), float(success), not truncated, truncated, info

    def render(self):
        return (
            self.env.render() if self.render_mode == "rgb_array" and self.env is not None else None
        )

    def close(self):
        self._ready = False
        if self.fixture is not None and self.fixture.env is not None:
            self.fixture.close()
        elif self.env is not None:
            self.env.close()
        self.env = self.script = None


def attempt_skill(
    env,
    *,
    seed,
    options=None,
    preferred=None,
    budget=1,
    before_sim_time=None,
    history=None,
    branch="initial",
):
    """Try distinct guarded actions from the same physical reset, with a retained ledger."""
    if not 1 <= budget <= 4:
        raise ValueError("A skill attempt budget must be between one and four")
    attempts = []
    seen = set()
    digest = None
    initial_time = None
    ledger = env.output / "attempts.json"
    case_id = (options or {}).get("case_id")

    def note(event, **fields):
        if history is not None:
            history.record(event, branch=branch, case_id=case_id, skill=env.skill, **fields)

    for number in range(1, budget + 1):
        note("reset_requested", attempt=number, source=(options or {}).get("source", env.source))
        try:
            observation, entry = env.reset(seed=seed, options=options)
        except (OSError, RuntimeError, ValueError, FloatingPointError) as error:
            result = dict(outcome="reset_rejected", failure=str(error), attempts=attempts)
            if isinstance(error, MeasuredResetRejected):
                result["initial_sim_time_s"] = error.sim_time_s
            note(
                "reset_rejected",
                attempt=number,
                failure=str(error),
                sim_time_s=result.get("initial_sim_time_s"),
            )
            return result
        state = native_state(env.env.model, env.env.data)
        initial_time = float(env.env.data.time)
        if before_sim_time is not None and initial_time >= before_sim_time:
            note("reset_rejected", attempt=number, failure="Rewind is not earlier than the stop")
            return dict(
                outcome="reset_rejected",
                failure="Rewind is not earlier than the stop",
                attempts=attempts,
            )
        reset_digest = hashlib.sha256(state.tobytes()).hexdigest()
        if digest is not None and reset_digest != digest:
            note("reset_rejected", attempt=number, failure="Retry reset differs from initial state")
            return dict(
                outcome="reset_rejected",
                failure="Retry reset differs from initial state",
                attempts=attempts,
            )
        digest = reset_digest
        available = [name for index, name in enumerate(env.actions) if entry["action_mask"][index]]
        if env.skill == "push" and env._case is not None and env.script.active_grip is None:
            alias = "paired" if env.script.config.paired_finger_insertion else "single"
            if preferred == alias:
                available.remove("recorded")
            else:
                available.remove(alias)
        remaining = [name for name in available if name not in seen]
        if not remaining:
            return dict(
                outcome="failed", failure="No remaining admissible action", attempts=attempts
            )
        action = preferred if not attempts and preferred in remaining else remaining[0]
        product = entry["active_product"]
        if history is not None:
            env.script.execution_history = history
            env.script._agent_skill = env.skill
            env.script._agent_branch = branch
            env.script._agent_artifacts = (
                env.directory.resolve().relative_to(history.path.parent.resolve()).as_posix()
            )
        note(
            "attempt_started",
            attempt=number,
            product=product,
            action=action,
            available_actions=remaining,
            previously_tried=sorted(seen),
            initial_state_sha256=digest,
            sim_time_s=initial_time,
            checkpoint=str(env._case.checkpoint) if env._case else None,
            checkpoint_manifest_sha256=env._case.manifest_sha256 if env._case else None,
            artifacts=env.script._agent_artifacts,
        )
        _write(ledger, dict(initial_state_sha256=digest, next_action=action, attempts=attempts))
        action_index = env.actions.index(action)
        np.savez_compressed(
            env.directory / "decision.npz",
            **{f"observation.{name}": value for name, value in observation.items()},
            action=np.asarray(action_index),
            action_mask=entry["action_mask"],
        )
        next_observation, reward, terminated, truncated, info = env.step(action_index)
        np.savez_compressed(
            env.directory / "outcome.npz",
            **{f"next_observation.{name}": value for name, value in next_observation.items()},
            reward=np.asarray(reward),
            terminated=np.asarray(terminated),
            truncated=np.asarray(truncated),
        )
        info["action_mask"] = info["action_mask"].tolist()
        info["outcome"] = "verified" if info["success"] else "timeout" if truncated else "failed"
        info["artifacts"] = str(env.directory.relative_to(env.output))
        info["lineage"] = env._case.lineage if env._case else f"procedural-{env.skill}-{seed}"
        attempts.append(info)
        seen.add(action)
        note(
            "attempt_finished",
            attempt=number,
            product=product,
            action=action,
            outcome=info["outcome"],
            failure=info["failure"],
            phase=env.script.phase,
            sim_time_s=float(env.env.data.time),
            artifacts=env.script._agent_artifacts,
        )
        _write(ledger, dict(initial_state_sha256=digest, next_action=None, attempts=attempts))
        if info["success"] or not set(available) - seen:
            break
    return dict(
        **attempts[-1],
        initial_state_sha256=digest,
        initial_sim_time_s=initial_time,
        attempts=attempts,
    )


def attempt_case(env, *, seed, case, preferred=None, budget=1, rewind=False, rewind_preferred=None):
    """Keep stopped-state retries and an optional earlier-boundary search distinct."""
    options = {"source": case.role, "case_id": case.id} if case is not None else None
    history = AgentHistory(
        env.output,
        run_kind="skill_attempts",
        seed=seed,
        skill=env.skill,
        case_id=case.id if case else None,
        attempt_budget_per_branch=budget,
    )
    initial = attempt_skill(
        env, seed=seed, options=options, preferred=preferred, budget=budget, history=history
    )
    if (
        not rewind
        or case is None
        or not case.rewind_case_id
        or initial["outcome"] == "verified"
        or "initial_sim_time_s" not in initial
    ):
        history.record("run_finished", outcome=initial["outcome"], failure=initial.get("failure"))
        return initial
    history.record(
        "branch_started",
        branch="rewind",
        skill=env.skill,
        from_case_id=case.id,
        case_id=case.rewind_case_id,
        reason=initial["outcome"],
        previous_failure=initial.get("failure"),
        preferred_action=rewind_preferred or preferred,
    )
    with StockingCurriculumEnv(
        env.skill,
        env.output / "rewind",
        cases=env.cases,
        control_config=env.control_config,
        verify_retreat=env.verify_retreat,
    ) as branch:
        earlier = attempt_skill(
            branch,
            seed=seed,
            options={"source": "recovery", "case_id": case.rewind_case_id},
            preferred=rewind_preferred or preferred,
            budget=budget,
            before_sim_time=initial["initial_sim_time_s"],
            history=history,
            branch="rewind",
        )
    history.record(
        "run_finished",
        outcome=earlier["outcome"],
        failure=earlier.get("failure"),
        branch="rewind",
        initial_outcome=initial["outcome"],
    )
    return {
        **initial,
        "outcome": earlier["outcome"],
        "failure": earlier.get("failure"),
        "success": earlier["outcome"] == "verified",
        "initial_attempt_outcome": initial["outcome"],
        "initial_attempt_failure": initial.get("failure"),
        "rewind": {"case_id": case.rewind_case_id, **earlier},
    }
