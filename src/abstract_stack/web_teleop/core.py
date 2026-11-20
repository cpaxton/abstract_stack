"""Simulation owner and operator session state machine for web teleoperation.

One background thread owns ``ShelfStockingEnv`` (and therefore live ``MjModel``/
``MjData``): it is the only writer of ``data.ctrl`` and the only caller of
``env.step``/``env.reset``. Network handlers call the thread-safe methods below,
which only validate, record intent and enqueue events; they never block physics.

States follow the Quest Teleop v1 contract: Disconnected, Ready, Engaged, Paused,
Fault. There is no automatic resume, input loss pauses, and reset is explicit.
"""

from __future__ import annotations

import enum
import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

from abstract_stack.env import ShelfStockingEnv
from abstract_stack.web_teleop import protocol
from abstract_stack.web_teleop.retarget import RetargetConfig, WebRetargeter
from abstract_stack.web_teleop.video import StereoVideo

log = logging.getLogger(__name__)


class State(str, enum.Enum):
    DISCONNECTED = "disconnected"
    READY = "ready"
    ENGAGED = "engaged"
    PAUSED = "paused"
    FAULT = "fault"


@dataclass(frozen=True)
class CoreConfig:
    input_timeout_s: float = 0.15  # contract required_input_age_limit_s
    status_every_ticks: int = 5  # 10 Hz status at 50 Hz control
    ack_cache: int = 256
    max_spectators: int = 8


class TeleopCore:
    def __init__(
        self,
        env: ShelfStockingEnv,
        *,
        config: CoreConfig | None = None,
        retarget: RetargetConfig | None = None,
        realtime: bool = True,
        video: StereoVideo | None = None,
    ):
        if not env.robot:
            raise ValueError("web teleop requires the MABEL robot")
        self.env = env
        self.config = config or CoreConfig()
        self.realtime = realtime
        self.retargeter = WebRetargeter(env.model, env.data, retarget)
        self.video = video
        self._lock = threading.Lock()
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._state = State.DISCONNECTED
        self._reason = "no operator connected"
        self._generation = 0
        self._operator: str | None = None
        self._spectators: set[str] = set()
        self._command: dict | None = None
        self._command_rx = 0.0
        self._command_timing: dict | None = None
        self._operator_name: str | None = None
        self._accepted_sequence: int | None = None
        self._acks: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self._episode_over = False
        self._listeners: list[Callable[[bytes | None, dict | None], None]] = []
        self._tick = 0
        self._overruns = 0
        self._rtf = 1.0
        self._step_ms = 0.0
        self._status: dict = {}
        self._engage_time = 0.0
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._publish_status()

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("teleop core already started")
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="web-teleop-sim", daemon=True)
        self._thread.start()
        if self.video is not None:
            self.video.start()

    def stop(self) -> None:
        self._running.clear()
        if self.video is not None:
            self.video.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def add_listener(self, callback: Callable[[bytes | None, dict | None], None]) -> None:
        """``callback(state_frame, status)`` runs on the sim thread; it must not block."""
        self._listeners.append(callback)

    # -- sessions (network threads) ------------------------------------------------
    def open_session(self, session_id: str, role: str, *, name: str | None = None) -> str:
        """Grant ``operator`` if the lease is free; otherwise the session spectates.

        ``name`` is the signed-in person's display name (Tailscale access), shown to
        everyone so spectators know who holds control.
        """
        with self._lock:
            if session_id == self._operator or session_id in self._spectators:
                raise protocol.ProtocolError("session_id already connected")
            if role == "operator" and self._operator is None:
                self._operator = session_id
                self._operator_name = name
                self._command = None
                self._command_timing = None
                self._accepted_sequence = None
                if self._state is State.DISCONNECTED:
                    self._set_state(State.READY, "operator connected")
                return "operator"
            if len(self._spectators) >= self.config.max_spectators:
                raise protocol.ProtocolError("spectator limit reached")
            self._spectators.add(session_id)
            return "spectator"

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._spectators.discard(session_id)
            if session_id == self._operator:
                self._operator = None
                self._operator_name = None
                self._command = None
                self._command_timing = None
                for key in [k for k in self._acks if k[0] == session_id]:
                    del self._acks[key]
                if self._state is not State.FAULT:
                    self._set_state(State.DISCONNECTED, "operator disconnected")

    def submit_command(self, session_id: str, command: dict) -> None:
        now = time.monotonic()
        with self._lock:
            if session_id != self._operator:
                raise protocol.ProtocolError("only the operator may send commands")
            if command["generation"] != self._generation:
                return  # obsolete generation: drop silently, status shows the current one
            if (
                self._accepted_sequence is not None
                and command["sequence"] <= self._accepted_sequence
            ):
                return  # duplicate or reordered
            self._accepted_sequence = command["sequence"]
            self._command = command
            self._command_rx = now

    def control(self, session_id: str, event: dict) -> Future:
        """Return a future resolving to the acknowledgement payload (idempotent per event)."""
        future: Future = Future()
        key = (session_id, event["event_id"])
        with self._lock:
            if key in self._acks:
                future.set_result(self._acks[key])
                return future
            if session_id != self._operator:
                future.set_result(self._ack_locked(key, event, False, "not the operator"))
                return future
            action = event["action"]
            # Pause is always honoured, even with an obsolete generation.
            if action == "pause":
                if self._state is State.ENGAGED:
                    self._set_state(State.PAUSED, "operator pause")
                future.set_result(self._ack_locked(key, event, True, None))
                return future
            if event["generation"] != self._generation:
                future.set_result(self._ack_locked(key, event, False, "obsolete generation"))
                return future
            if action == "engage":
                error = self._engage_error_locked()
                if error is None:
                    self._command = None  # require fresh, rebased targets after engage
                    self._set_state(State.ENGAGED, "operator engaged")
                    self._events.put(("resync", None))
                future.set_result(self._ack_locked(key, event, error is None, error))
                return future
        # reset / ready run on the sim thread; acknowledge once applied.
        self._events.put((action, (key, event, future)))
        return future

    def status(self) -> dict:
        """Latest published status with live session fields (state may change between ticks)."""
        with self._lock:
            return {**self._status, **self._session_fields_locked()}

    def _session_fields_locked(self) -> dict:
        return {
            "state": self._state.value,
            "reason": self._reason,
            "calibration_generation": self._generation,
            "accepted_sequence": self._accepted_sequence,
            "operator_connected": self._operator is not None,
            "operator_name": self._operator_name,
            "spectators": len(self._spectators),
            "episode_over": self._episode_over,
        }

    # -- internals (lock held where noted) -----------------------------------------
    def _engage_error_locked(self) -> str | None:
        if self._state is State.ENGAGED:
            return None
        if self._state is State.FAULT:
            return "fault: reset required"
        if self._episode_over:
            return "episode ended: reset required"
        if self._state not in (State.READY, State.PAUSED):
            return f"cannot engage from {self._state.value}"
        return None

    def _ack_locked(self, key, event, accepted: bool, error: str | None) -> dict:
        ack = {
            "event_id": event["event_id"],
            "action": event["action"],
            "accepted": accepted,
            "error": error,
            "calibration_generation": self._generation,
            "state": self._state.value,
        }
        self._acks[key] = ack
        while len(self._acks) > self.config.ack_cache:
            self._acks.popitem(last=False)
        return ack

    def _set_state(self, state: State, reason: str) -> None:
        if state is State.ENGAGED and self._state is not State.ENGAGED:
            self._engage_time = time.monotonic()
        if state is not self._state or reason != self._reason:
            log.info("teleop state %s -> %s (%s)", self._state.value, state.value, reason)
        self._state, self._reason = state, reason

    def _handle_event(self, action: str, payload) -> None:
        if action == "resync":
            self.retargeter.resync()
            return
        key, event, future = payload
        if action == "reset":
            try:
                self.env.reset()
                self.retargeter.reset()
                ok, error = True, None
            except Exception as exc:  # reset failure keeps the fault visible
                log.exception("reset failed")
                ok, error = False, f"reset failed: {exc}"
            with self._lock:
                self._generation += 1
                self._episode_over = False
                self._command = None
                self._accepted_sequence = None
                if ok:
                    self._set_state(
                        State.READY if self._operator else State.DISCONNECTED, "episode reset"
                    )
                else:
                    self._set_state(State.FAULT, error)
                future.set_result(self._ack_locked(key, event, ok, error))
        elif action == "ready":
            with self._lock:
                allowed = self._state in (State.READY, State.PAUSED, State.ENGAGED)
            if allowed and not self._episode_over:
                self.retargeter.ready()
                ok, error = True, None
            else:
                ok, error = False, "ready requires a live, non-faulted episode"
            with self._lock:
                future.set_result(self._ack_locked(key, event, ok, error))

    def _tick_once(self, dt: float) -> None:
        while True:
            try:
                action, payload = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(action, payload)
        now = time.monotonic()
        with self._lock:
            state = self._state
            command = self._command
            received = self._command_rx
            operator = self._operator
            if state is State.ENGAGED and (
                command is None or now - self._command_rx > self.config.input_timeout_s
            ):
                # Grace for the first command after engage is the same input budget.
                if command is not None or now - self._engage_time > self.config.input_timeout_s:
                    self._set_state(State.PAUSED, "operator input timed out")
                    state = State.PAUSED
                command = None
        if state is State.FAULT:
            self._publish(dt)
            return
        try:
            if state is State.ENGAGED and command is not None:
                self.retargeter.apply_intent(command, dt)
                timing = self._command_timing
                if timing is None or (timing["session_id"], timing["sequence"]) != (
                    operator,
                    command["sequence"],
                ):
                    # First tick that applies this command: browser-side uplink latency
                    # is this host time (clock-synced) minus the echoed send time.
                    # Sequences restart per session and sent_ms is on that session's
                    # browser clock, so the echo is keyed by (session, sequence) and
                    # only ever sent back to that session (see server._status_for).
                    self._command_timing = {
                        "session_id": operator,
                        "sequence": command["sequence"],
                        "sent_ms": command["sent_monotonic_s"] * 1e3,
                        "received_host_s": received,
                        "applied_host_s": time.monotonic(),
                    }
            # Velocity intents only count while engaged with fresh input; otherwise
            # the base brakes on its deceleration ramp (pause, timeout, disconnect).
            engaged_base = command["base"] if state is State.ENGAGED and command else None
            self.retargeter.drive(engaged_base, dt)
            targets = self.retargeter.targets(dt)
            started = time.perf_counter()
            _, _, terminated, truncated, _ = self.env.step(targets)
            self._step_ms = (time.perf_counter() - started) * 1e3
            if terminated or truncated:
                with self._lock:
                    self._episode_over = True
                    self._set_state(State.PAUSED, "episode time limit reached: reset to continue")
        except (FloatingPointError, ValueError, RuntimeError) as error:
            log.exception("teleop physics/command fault")
            with self._lock:
                self._set_state(State.FAULT, f"{type(error).__name__}: {error}")
        self._publish(dt)

    def _publish(self, dt: float) -> None:
        self._tick += 1
        data = self.env.data
        flags = protocol.FLAG_ENGAGED if self._state is State.ENGAGED else 0
        frame = protocol.pack_state(
            self._tick, float(data.time), time.monotonic(), data.xpos, data.xquat, flags
        )
        if self.video is not None:
            self.video.snapshot(data, self._generation)
        status = None
        if self._tick % self.config.status_every_ticks == 0:
            status = self._publish_status()
        for listener in self._listeners:
            try:
                listener(frame, status)
            except Exception:  # a broken listener must never stop physics
                log.exception("state listener failed")

    def _publish_status(self) -> dict:
        info = self.env.info()
        report = self.retargeter.report()
        with self._lock:
            self._status = {
                **self._session_fields_locked(),
                "tick": self._tick,
                "sim_time_s": float(self.env.data.time),
                "host_monotonic_s": time.monotonic(),
                "command_timing": self._command_timing,
                "real_time_factor": round(self._rtf, 3),
                "step_ms": round(self._step_ms, 2),
                "overruns": self._overruns,
                "task": {
                    key: info[key]
                    for key in (
                        "stocked_count",
                        "product_count",
                        "success",
                        "tipped",
                        "dropped",
                        "intervention_count",
                        "elapsed_sim_time",
                    )
                },
                **report,
                "video": self._video_status(),
            }
            return dict(self._status)

    def _video_status(self) -> dict | None:
        video = self.video
        if video is None:
            return None
        return {
            "error": video.error,
            "pair_id": video.pair_id,
            "render_ms": round(video.stats["render_ms"], 1),
            "encode_ms": round(video.stats["encode_ms"], 1),
            "pair_bytes": video.stats["bytes"],
            "simplified_meshes": video.stats["simplified_meshes"],
        }

    def _run(self) -> None:
        dt = self.env.dt
        next_time = time.perf_counter()
        window_start, window_sim = next_time, float(self.env.data.time)
        while self._running.is_set():
            self._tick_once(dt)
            if not self.realtime:
                continue
            next_time += dt
            now = time.perf_counter()
            if now - window_start >= 1.0:
                self._rtf = (float(self.env.data.time) - window_sim) / (now - window_start)
                window_start, window_sim = now, float(self.env.data.time)
            delay = next_time - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -5 * dt:
                # Fell far behind (debugger, CPU contention): resume pacing, do not burst.
                self._overruns += 1
                next_time = now
