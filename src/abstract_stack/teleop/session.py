"""Owner-thread engagement, freshness and final native command gate (#20).

This is a simulation policy. Measured servo references do not establish loaded
retention, and lifecycle qualification must come from an independently tested
provider rather than the raw probe's user_present flag alone.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from uuid import UUID

import mujoco
import numpy as np

from abstract_stack.control import validate_targets
from abstract_stack.scene import ShelfConfig, build_scene_xml
from abstract_stack.teleop.calibration import Calibration
from abstract_stack.teleop.sim import BASE_DRIVE, BASE_STEER, MujocoTeleopAdapter
from abstract_stack.teleop.transport import validate_tracking

GROUPS = {
    "head": ("Neck Yaw", "Neck Pitch", "Neck Roll"),
    "arms": tuple(f"{side} Arm {index}" for side in ("Left", "Right") for index in range(1, 8))
    + ("Left Wrist", "Right Wrist"),
    "fingers": tuple(
        f"{side} {finger} {joint}"
        for side in ("Left", "Right")
        for finger, joints in (
            ("Thumb", ("MCP", "Abd", "PIP", "DIP")),
            ("Index", ("Abd", "MCP", "PIP")),
            ("Middle", ("Abd", "MCP", "PIP")),
            ("Ring", ("Abd", "MCP", "PIP")),
            ("Pinky", ("Abd", "MCP", "PIP")),
        )
        for joint in joints
    ),
    "body": ("Lift", "Torso"),
}


class SessionState(str, Enum):
    DISCONNECTED = "Disconnected"
    READY = "Ready"
    ENGAGED = "Engaged"
    PAUSED = "Paused"
    FAULT = "Fault"


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integer(value, name):
    if type(value) is not int or not 0 <= value < 2**53:
        raise ValueError(f"{name} must be an exact nonnegative integer below2^53")
    return value


@dataclass(frozen=True)
class GateSettings:
    enabled_groups: tuple[str, ...] = ("head", "arms", "fingers")
    input_age_s: float = 0.15
    video_age_s: float = 0.3
    clock_age_s: float = 10
    clock_uncertainty_s: float = 0.025
    max_velocity_rad_s: float = 1.0
    max_acceleration_rad_s2: float = 5.0
    max_lift_velocity_m_s: float = 0.05
    max_lift_acceleration_m_s2: float = 0.25
    # Body loss is optional only in this explicit head/arms/fingers mode.
    body_required: bool = False

    def __post_init__(self):
        groups = tuple(self.enabled_groups)
        if not groups or len(set(groups)) != len(groups) or set(groups) - GROUPS.keys():
            raise ValueError("select distinct supported actuator groups")
        if type(self.body_required) is not bool or (("body" in groups) != self.body_required):
            raise ValueError("body commands require explicit body-required mode")
        for name in self.__dataclass_fields__:
            if (
                name not in ("enabled_groups", "body_required")
                and finite(getattr(self, name), name) <= 0
            ):
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "enabled_groups", groups)


@dataclass(frozen=True)
class LifecycleEvidence:
    """Trusted independent provider; qualification defaults to false.

    The caller must test its actual remove/re-wear detection before setting
    removal_detection_qualified. A raw worn:true observation is insufficient.
    """

    session_id: str
    observed_host_monotonic_s: float
    active: bool
    focused: bool
    worn: bool
    removal_detection_qualified: bool = False


@dataclass(frozen=True)
class VideoEvidence:
    """Only construct verified evidence after #21 sent-pair/clock validation.

    Both original capture time and receipt time are retained; repeating an old
    frame can never renew capture age. This is app submission, not optical time.
    """

    session_id: str
    calibration_generation: int
    pair_id: int
    capture_host_monotonic_s: float
    feedback_host_monotonic_s: float
    uncertainty_s: float
    verified: bool


@dataclass(frozen=True)
class CommandProposal:
    session_id: str
    calibration_generation: int
    tracking_sequence: int
    observed_host_monotonic_s: float
    targets: Mapping[str, float]
    valid: bool = True
    reason: str = ""

    def __post_init__(self):
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))


@dataclass
class GateTelemetry:
    state: str
    reason: str
    session_id: str | None
    calibration_generation: int
    command_sequence: int
    last_tracking_sequence: int | None
    sim_time_s: float
    reset_generation: int
    video_pair_id: int | None
    observation_age_s: dict
    hold_policy: str
    hold_error: str | None
    limiter_channels: list[str] = field(default_factory=list)
    nominal_targets: dict = field(default_factory=dict)
    applied_targets: dict = field(default_factory=dict)
    measured_lengths: dict = field(default_factory=dict)
    measured_forces: dict = field(default_factory=dict)


class EngagementGate:
    """All methods stay on the simulator owner thread; networking uses a mailbox.

    Modules propose disjoint named groups. The gate composes a complete native
    vector and makes one atomic apply call per tick. It never writes live qpos,
    changes model limits or steps the simulator itself. The #14 owner steps after
    tick, including during pause; only explicit reset constructs reset state.
    """

    def __init__(
        self, sim: MujocoTeleopAdapter, *, settings=None, mabel_root=None, composed_guard=None
    ):
        self.sim, self.settings = sim, settings or GateSettings()
        self.owner = threading.current_thread()
        self.composed_guard = composed_guard
        self.provenance = sim.describe()
        xml = build_scene_xml(
            ShelfConfig(**self.provenance["config"]),
            seed=self.provenance["seed"],
            mabel_root=mabel_root,
        )
        if hashlib.sha256(xml.encode()).hexdigest() != self.provenance["model_xml_sha256"]:
            raise ValueError("gate command model does not match simulator")
        self.model = mujoco.MjModel.from_xml_string(xml)  # private limits model only
        state = sim.read_state()
        self.names = state.actuator_names
        if self.names != tuple(self.model.actuator(i).name for i in range(self.model.nu)):
            raise ValueError("actuator order mismatch")
        self.indices = {name: index for index, name in enumerate(self.names)}
        self.base_ids = np.array([self.indices[name] for name in (*BASE_STEER, *BASE_DRIVE)])
        self.base_reference = state.applied_targets[self.base_ids].copy()
        self._last_applied = validate_targets(
            self.model, state.applied_targets, state.applied_targets
        )
        self._hold = self._last_applied.copy()
        self._previous_velocity = np.zeros(self.model.nu)
        self._reset_generation = state.reset_generation
        self._last_tick = None
        self._last_step_index = None
        self.state, self.reason = SessionState.DISCONNECTED, "no_session"
        self.session_id = None
        self.calibration = None
        self._seen = set()
        self._events = OrderedDict()
        self._last_event = -1
        self._last_tracking = -1
        self._tracking = None
        self._observations = {}
        self._hand_highwater = {}
        self._clock = None
        self._lifecycle = None
        self._video = None
        self._resume_required = False
        self.command_sequence = 0
        self.hold_error = None
        self.hold_policy = "latched measured position-servo lengths; reserved base references"
        self.events = deque(maxlen=128)
        self._last_nominal = self._last_applied.copy()
        self._limiter_channels = []
        self._check_state(state)

    @property
    def generation(self):
        return 0 if self.calibration is None else self.calibration.generation

    def _owner(self):
        if threading.current_thread() is not self.owner:
            raise RuntimeError("engagement gate belongs to simulator owner thread")

    def _check_state(self, state):
        for name in (
            "qpos",
            "qvel",
            "actuator_lengths",
            "actuator_velocities",
            "actuator_forces",
            "applied_targets",
            "root_position_m",
            "root_quaternion_wxyz",
        ):
            if not np.isfinite(getattr(state, name)).all():
                raise ValueError(f"nonfinite simulator {name}")
        if state.terminated or state.truncated:
            raise ValueError("episode finished; explicit reset required")
        if state.reset_generation != self._reset_generation:
            raise ValueError("simulator reset outside command owner")
        validate_targets(self.model, state.applied_targets, state.applied_targets)
        if not np.array_equal(state.applied_targets, self._last_applied):
            raise ValueError("actuator references changed outside final command owner")
        if not np.array_equal(state.applied_targets[self.base_ids], self.base_reference):
            raise ValueError("reserved base command changed")

    def _transition(self, state, reason, now):
        if (self.state, self.reason) != (state, reason):
            self.events.append({"host_monotonic_s": now, "state": state.value, "reason": reason})
        self.state, self.reason = state, reason

    def _clear_observations(self):
        self._tracking, self._clock, self._lifecycle, self._video = None, None, None, None
        self._observations.clear()

    def _invalidate(self, reason, *, always=False):
        if self.calibration is not None and (always or self.calibration.artifact is not None):
            self.calibration.invalidate(reason)
        self._clear_observations()

    def _latch_hold(self):
        state = self.sim.read_state()
        self._check_state(state)
        hold = np.zeros(self.model.nu)
        position = self.model.actuator_biasprm[:, 1] < 0
        hold[position] = state.actuator_lengths[position]
        hold[self.base_ids] = self.base_reference
        # A measured posture may be outside native ctrlrange. Never silently
        # clip it into a claimed hold; retain the last valid reference and Fault.
        self._hold = validate_targets(self.model, self._last_applied, hold)
        self.hold_error = None

    def _pause(self, reason, now):
        self._resume_required |= self.state == SessionState.ENGAGED
        self._invalidate(reason)
        if self.state == SessionState.FAULT:
            return
        try:
            self._latch_hold()
        except (ValueError, RuntimeError) as error:
            self._fault(f"measured_hold_unavailable: {error}", now)
            return
        self._transition(
            SessionState.PAUSED if self.session_id else SessionState.DISCONNECTED, reason, now
        )

    def _fault(self, reason, now):
        if self.state == SessionState.FAULT:
            return
        self._invalidate(reason)
        self.hold_error = reason
        self._hold = self._last_applied.copy()
        self._previous_velocity[:] = 0
        self._transition(SessionState.FAULT, reason, now)

    def connect(self, session_id, *, now):
        self._owner()
        finite(now, "now")
        if not isinstance(session_id, str) or str(UUID(session_id)) != session_id:
            raise ValueError("canonical session UUID required")
        if self.session_id is not None or session_id in self._seen or len(self._seen) >= 10000:
            raise ValueError("fresh exclusive session required")
        self._seen.add(session_id)
        self.session_id = session_id
        self.calibration = Calibration(session_id)
        self._events.clear()
        self._last_event = self._last_tracking = -1
        self._hand_highwater.clear()
        self._resume_required = False
        self._clear_observations()
        if self.state != SessionState.FAULT:
            self._pause("awaiting_inputs", now)
        return self.generation

    def disconnect(self, session_id, *, now):
        self._owner()
        if session_id != self.session_id:
            return
        self.session_id = None
        self._pause("control_disconnected", now)

    def observe(self, tracking, clock, lifecycle: LifecycleEvidence, video: VideoEvidence, *, now):
        """Receive a detached, canonical observation plus trusted provider evidence."""
        self._owner()
        finite(now, "now")
        if tracking.get("session_id") != self.session_id or self.session_id is None:
            return False
        if tracking.get("calibration_generation") != self.generation:
            return False
        try:
            if (
                type(tracking.get("schema_version")) is not int
                or tracking["schema_version"] != 1
                or tracking.get("type") != "tracking"
            ):
                raise ValueError("tracking envelope invalid")
            validate_tracking(tracking)
            # These independent providers may report a focus/removal/video loss
            # even while pose acquisition has stopped. They never renew pose age.
            self._lifecycle = copy.deepcopy(lifecycle)
            self._video = copy.deepcopy(video)
            if tracking["sequence"] <= self._last_tracking:
                reason = self._input_reason(now)
                if reason is not None and self.state == SessionState.ENGAGED:
                    self._pause(reason, now)
                return False
            self._last_tracking = tracking["sequence"]
            offset = finite(clock.offset_s, "clock offset")
            uncertainty = finite(clock.uncertainty_s, "clock uncertainty")
            established = finite(clock.established_monotonic_s, "clock establishment")
            if (
                uncertainty < 0
                or uncertainty > self.settings.clock_uncertainty_s
                or not 0 <= now - established <= self.settings.clock_age_s
            ):
                raise ValueError("clock unavailable/uncertain/expired")
            self._clock = (offset, uncertainty, established)
            self._tracking = copy.deepcopy(tracking)
            self._lifecycle = copy.deepcopy(lifecycle)
            self._video = copy.deepcopy(video)
            for name in ("head", "left_hand", "right_hand", "body"):
                source = tracking[name]
                observed = finite(source["observed_monotonic_s"], "observation") + offset
                if observed > now + uncertainty:
                    raise ValueError("future tracking observation")
                if name == "body":
                    valid = all(
                        any(
                            j["name"] == required and j["position_valid"] and j["orientation_valid"]
                            for j in source["joints"]
                        )
                        for required in ("Body_Chest", "Body_Hips")
                    )
                else:
                    valid = source["pose"]["position_valid"] and source["pose"]["orientation_valid"]
                if not valid:
                    self._observations.pop(name, None)
                    continue
                if name in ("left_hand", "right_hand"):
                    stamp = source["source_time_s"]
                    if source["timestamp_kind"] != "sensor_sample" or stamp is None or stamp <= 0:
                        self._observations.pop(name, None)
                        continue
                    if stamp <= self._hand_highwater.get(name, 0):
                        continue
                    self._hand_highwater[name] = stamp
                if observed > self._observations.get(name, -math.inf):
                    self._observations[name] = observed
            reason = self._input_reason(now)
            if reason is not None:
                if self.state == SessionState.ENGAGED:
                    self._pause(reason, now)
            elif self.state == SessionState.PAUSED and self.reason == "awaiting_inputs":
                self._transition(SessionState.READY, "calibration_required", now)
            return True
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            self._pause(f"invalid_observation: {error}", now)
            return False

    def _input_reason(self, now):
        if self.session_id is None:
            return "no_session"
        if self._tracking is None or self._clock is None:
            return "tracking_unavailable"
        _, uncertainty, established = self._clock
        if not 0 <= now - established <= self.settings.clock_age_s:
            return "clock_expired"
        if not self._tracking["worn"] or not self._tracking["focused"]:
            return "tracking_lifecycle_inactive"
        lifecycle = self._lifecycle
        if lifecycle is None or lifecycle.session_id != self.session_id:
            return "independent_lifecycle_unavailable"
        if any(
            type(getattr(lifecycle, field)) is not bool
            for field in ("active", "focused", "worn", "removal_detection_qualified")
        ):
            return "invalid_lifecycle_evidence"
        if not lifecycle.removal_detection_qualified:
            return "removal_detection_unqualified"
        if not lifecycle.active or not lifecycle.focused or not lifecycle.worn:
            return "independent_lifecycle_inactive"
        lifecycle_age = now - finite(lifecycle.observed_host_monotonic_s, "lifecycle time")
        if not 0 <= lifecycle_age <= self.settings.input_age_s:
            return "lifecycle_stale"
        required = ("head", "left_hand", "right_hand") + (
            ("body",) if self.settings.body_required else ()
        )
        for source in required:
            age = now - self._observations.get(source, -math.inf)
            if age < -uncertainty or age + uncertainty > self.settings.input_age_s:
                return f"{source}_observation_stale"
        video = self._video
        if (
            video is None
            or video.session_id != self.session_id
            or video.calibration_generation != self.generation
            or video.verified is not True
        ):
            return "video_unverified"
        integer(video.pair_id, "video pair")
        vu = finite(video.uncertainty_s, "video clock uncertainty")
        capture_age = now - finite(video.capture_host_monotonic_s, "video capture time")
        feedback_age = now - finite(video.feedback_host_monotonic_s, "video feedback time")
        if (
            not 0 <= vu <= self.settings.clock_uncertainty_s
            or capture_age < -vu
            or capture_age + vu > self.settings.video_age_s
            or not 0 <= feedback_age <= self.settings.video_age_s
        ):
            return "video_stale"
        return None

    def _ack(self, accepted, reason):
        return {
            "accepted": accepted,
            "reason": reason,
            "state": self.state.value,
            "calibration_generation": self.generation,
        }

    def control(self, session_id, action, event_id, generation, *, now, neutral=None, specs=None):
        """Idempotent deliberate operator actions; never accepts an implicit reset."""
        self._owner()
        finite(now, "now")
        integer(event_id, "event_id")
        integer(generation, "generation")
        if session_id != self.session_id or self.session_id is None:
            return self._ack(False, "session_mismatch")
        actions = {"calibrate", "engage", "pause", "resume", "reset", "reference_space_changed"}
        if not isinstance(action, str) or action not in actions:
            raise ValueError("unknown control action")
        identity = (action, generation)
        if event_id in self._events:
            previous, ack = self._events[event_id]
            if identity != previous:
                raise ValueError("event ID reused with a changed action/generation")
            return copy.deepcopy(ack)
        if event_id <= self._last_event:
            raise ValueError("expired/reordered control event")
        self._last_event = event_id
        if generation != self.generation and action not in {"pause", "reference_space_changed"}:
            ack = self._ack(False, "calibration_generation_mismatch")
        elif action == "pause":
            self._pause("operator_pause", now)
            ack = self._ack(True, self.reason)
        elif action == "reference_space_changed":
            self._invalidate("reference_space_changed", always=True)
            self._pause("reference_space_changed", now)
            ack = self._ack(True, self.reason)
        elif action == "reset":
            ack = self.host_reset(now=now)
        elif action == "calibrate":
            if self.state not in (SessionState.READY, SessionState.PAUSED):
                ack = self._ack(False, "pause_before_calibration")
            elif self._input_reason(now) is not None:
                ack = self._ack(False, self._input_reason(now))
            else:
                try:
                    self._latch_hold()
                    self.calibration.capture(
                        neutral, specs, ready_or_paused=True, inputs_usable=True
                    )
                    self._clear_observations()
                    self._transition(SessionState.READY, "deliberate_engagement_required", now)
                    ack = self._ack(True, self.reason)
                except (ValueError, TypeError, KeyError) as error:
                    self._pause(f"calibration_failed: {error}", now)
                    ack = self._ack(False, self.reason)
        else:
            reason = self._input_reason(now)
            if self.state == SessionState.FAULT:
                ack = self._ack(False, "explicit_reset_required")
            elif self.state == SessionState.ENGAGED:
                ack = self._ack(False, "already_engaged")
            elif reason is not None:
                ack = self._ack(False, reason)
            elif self.calibration.artifact is None:
                ack = self._ack(False, "realignment_required")
            elif not callable(self.composed_guard):
                ack = self._ack(False, "composed_geometry_guard_required")
            elif self._resume_required and action != "resume":
                ack = self._ack(False, "deliberate_resume_required")
            elif not self._resume_required and action == "resume":
                ack = self._ack(False, "initial_engage_required")
            else:
                self.calibration.acknowledge_engagement(
                    session_id=self.session_id,
                    generation=self.generation,
                    deliberate=True,
                    inputs_usable=True,
                )
                self._resume_required = False
                self._transition(SessionState.ENGAGED, "operator_engaged", now)
                ack = self._ack(True, self.reason)
        self._events[event_id] = (identity, copy.deepcopy(ack))
        while len(self._events) > 64:
            self._events.popitem(last=False)
        return ack

    def host_pause(self, *, now):
        self._owner()
        self._pause("host_pause", finite(now, "now"))

    def body_hold_references(self):
        """Owner-approved measured latch for #17's bounded body-hold admission."""
        self._owner()
        return {name: float(self._hold[self.indices[name]]) for name in ("Lift", "Torso")}

    def host_reset(self, *, now, seed=None):
        self._owner()
        now = finite(now, "now")
        try:
            # Reject nonfinite/broken native state before attempting reset. An
            # exhausted episode is an explicit reset path, not a nonfinite fault.
            state = self.sim.read_state()
            if any(
                not np.isfinite(getattr(state, name)).all()
                for name in (
                    "qpos",
                    "qvel",
                    "applied_targets",
                    "actuator_lengths",
                    "actuator_velocities",
                    "actuator_forces",
                )
            ):
                raise ValueError("nonfinite state needs host diagnostics/restart")
            validate_targets(self.model, state.applied_targets, state.applied_targets)
            self._pause("reset_requested", now)
            reset = self.sim.reset(seed=seed)
            self._reset_generation = reset.reset_generation
            self.base_reference = reset.applied_targets[self.base_ids].copy()
            self._last_applied = validate_targets(
                self.model, reset.applied_targets, reset.applied_targets
            )
            self._hold = self._last_applied.copy()
            self._previous_velocity[:] = 0
            self._invalidate("reset_requires_realignment", always=True)
            self._transition(
                SessionState.PAUSED if self.session_id else SessionState.DISCONNECTED,
                "reset_requires_realignment",
                now,
            )
            self.hold_error = None
            self._last_tick = None
            self._last_step_index = None
            return self._ack(True, self.reason)
        except (ValueError, RuntimeError) as error:
            self._fault(f"reset_failed: {error}", now)
            return self._ack(False, self.reason)

    def _compose(self, proposals, now):
        targets = self._hold.copy()
        seen = set()
        for group in self.settings.enabled_groups:
            proposal = proposals.get(group)
            if proposal is None or proposal.valid is not True:
                raise ValueError(f"{group} proposal unavailable/invalid")
            if (
                proposal.session_id != self.session_id
                or proposal.calibration_generation != self.generation
            ):
                raise ValueError("proposal session/generation mismatch")
            if proposal.tracking_sequence != self._last_tracking:
                raise ValueError("proposal does not match newest observation")
            age = now - finite(proposal.observed_host_monotonic_s, "proposal observation")
            if not 0 <= age <= self.settings.input_age_s:
                raise ValueError("proposal observation stale/future")
            expected = set(GROUPS[group])
            if set(proposal.targets) != expected or seen & expected:
                raise ValueError(f"{group} must own exactly its declared channels")
            seen.update(expected)
            for name, value in proposal.targets.items():
                targets[self.indices[name]] = finite(value, name)
        if set(proposals) != set(self.settings.enabled_groups):
            raise ValueError("unexpected controller group")
        targets[self.base_ids] = self.base_reference
        return validate_targets(self.model, self._last_applied, targets)

    def _limit(self, nominal):
        dt = self.sim.control_dt
        velocity_limit = np.full(self.model.nu, self.settings.max_velocity_rad_s)
        acceleration_limit = np.full(self.model.nu, self.settings.max_acceleration_rad_s2)
        velocity_limit[self.indices["Lift"]] = self.settings.max_lift_velocity_m_s
        acceleration_limit[self.indices["Lift"]] = self.settings.max_lift_acceleration_m_s2
        delta = nominal - self._last_applied
        requested_velocity = delta / dt
        low = np.maximum(-velocity_limit, self._previous_velocity - acceleration_limit * dt)
        high = np.minimum(velocity_limit, self._previous_velocity + acceleration_limit * dt)
        for index in np.flatnonzero(self.model.actuator_ctrllimited):
            minimum, maximum = self.model.actuator_ctrlrange[index]
            a, position = acceleration_limit[index], self._last_applied[index]
            positive = math.sqrt((a * dt) ** 2 + 2 * a * max(0, maximum - position)) - a * dt
            negative = math.sqrt((a * dt) ** 2 + 2 * a * max(0, position - minimum)) - a * dt
            low[index], high[index] = max(low[index], -negative), min(high[index], positive)
        if np.any(low > high):
            raise ValueError("no feasible bounded rate/braking interval")
        velocity = np.clip(requested_velocity, low, high)
        applied = self._last_applied + velocity * dt
        applied[self.base_ids] = self.base_reference
        # Final validation AFTER rate correction, including native tendon ctrlrange.
        applied = validate_targets(self.model, self._last_applied, applied)
        self._previous_velocity = (applied - self._last_applied) / dt
        self._limiter_channels = [
            name for i, name in enumerate(self.names) if abs(applied[i] - nominal[i]) > 1e-12
        ]
        return applied

    def tick(self, proposals=None, *, now):
        """One complete apply, no stepping. Paused/Fault retain bounded policy."""
        self._owner()
        now = finite(now, "now")
        try:
            state = self.sim.read_state()
            self._check_state(state)
            if self._last_step_index is not None and state.step_index <= self._last_step_index:
                raise ValueError("only one command write is allowed per physics control step")
            if self._last_tick is not None:
                if now <= self._last_tick:
                    raise ValueError("control time did not advance")
                if (
                    now - self._last_tick > self.settings.input_age_s
                    and self.state == SessionState.ENGAGED
                ):
                    self._pause("control_tick_stalled", now)
            self._last_tick = now
            self._last_step_index = state.step_index
            if self.state == SessionState.ENGAGED:
                reason = self._input_reason(now)
                if reason is not None:
                    self._pause(reason, now)
            if self.state == SessionState.ENGAGED:
                try:
                    self._last_nominal = self._compose(proposals or {}, now)
                except (ValueError, TypeError, KeyError, AttributeError) as error:
                    self._pause(f"invalid_proposal: {error}", now)
            if self.state == SessionState.ENGAGED:
                applied = self._limit(self._last_nominal)
            else:
                applied = (
                    validate_targets(self.model, self._last_applied, self._hold)
                    if self.state == SessionState.FAULT
                    else self._limit(self._hold)
                )
                self._last_nominal = self._hold.copy()
            if callable(self.composed_guard):
                guarded = self.composed_guard(
                    state, dict(zip(self.names, applied.tolist(), strict=True))
                )
                if not isinstance(guarded, Mapping) or guarded.get("valid") is not True:
                    raise ValueError(f"composed geometry rejected: {guarded}")
            self.sim.apply_actuator_targets(applied)
            self._last_applied = applied.copy()
            self.command_sequence += 1
            return applied.copy()
        except Exception as error:
            self._fault(f"command_or_physics_fault: {error}", now)
            return None  # no unvalidated write; owner may diagnose/stop stepping

    def telemetry(self, *, now):
        self._owner()
        now = finite(now, "now")
        state = self.sim.read_state()
        uncertainty = self._clock[1] if self._clock else 0

        def measured(values):
            # Fault diagnostics must remain valid v1 JSON even when the native
            # state is nonfinite. Null is unavailable, never a fabricated zero.
            return {
                name: float(value) if np.isfinite(value) else None
                for name, value in zip(self.names, values, strict=True)
            }

        return GateTelemetry(
            self.state.value,
            self.reason,
            self.session_id,
            self.generation,
            self.command_sequence,
            self._last_tracking if self._last_tracking >= 0 else None,
            state.sim_time_s,
            state.reset_generation,
            self._video.pair_id if self._video else None,
            {name: now - observed + uncertainty for name, observed in self._observations.items()},
            self.hold_policy,
            self.hold_error,
            self._limiter_channels.copy(),
            dict(zip(self.names, self._last_nominal.tolist(), strict=True)),
            dict(zip(self.names, self._last_applied.tolist(), strict=True)),
            measured(state.actuator_lengths),
            measured(state.actuator_forces),
        )
