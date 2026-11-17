"""Bounded event-loop → simulator-owner bridge, isolated from shared transport."""

from __future__ import annotations

import asyncio
import copy
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict

from abstract_stack.teleop.transport import ControlServer, SessionRegistry


class OwnerMailbox:
    """Reliable bounded controls, one newest input, urgent disconnect/pause.

    drain() is called by the simulator owner before gate.tick()/sim.step(). Its
    actual host time is used for controls, not stale enqueue timestamps. All
    model reads/calibration provider calls stay on that owner thread.
    """

    def __init__(self, *, capacity=32, calibration_provider=None):
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("mailbox capacity must be1..64")
        self.controls = queue.Queue(capacity)
        self.calibration_provider = calibration_provider
        self._lock = threading.Lock()
        self._input = None
        self._urgent_disconnects = set()
        self._pending_connects = {}
        self._urgent_pause = False
        self._status = None
        self.replaced_inputs = 0

    def submit(self, method, *args, **kwargs):
        if method not in {"connect", "control", "host_pause", "host_reset"}:
            raise ValueError("unsupported owner request")
        future = Future()
        with self._lock:
            try:
                self.controls.put_nowait(
                    (method, copy.deepcopy(args), copy.deepcopy(kwargs), future)
                )
            except queue.Full:
                self._urgent_pause = True
                future.set_exception(RuntimeError("control mailbox full; host pause requested"))
            else:
                if method == "connect":
                    self._pending_connects[future] = args[0]
        return future

    def disconnect(self, session_id):
        with self._lock:
            self._urgent_disconnects.add(session_id)
            for future, pending_session in self._pending_connects.items():
                if pending_session == session_id:
                    future.cancel()

    def request_pause(self):
        with self._lock:
            self._urgent_pause = True

    def observe(self, tracking, clock, lifecycle, video):
        observation = copy.deepcopy((tracking, clock, lifecycle, video))
        with self._lock:
            self.replaced_inputs += self._input is not None
            self._input = observation

    def status(self):
        with self._lock:
            return copy.deepcopy(self._status)

    def drain(self, gate, *, now, max_controls=8):
        gate._owner()
        with self._lock:
            observation, disconnects, pause = (
                self._input,
                self._urgent_disconnects,
                self._urgent_pause,
            )
            self._input, self._urgent_disconnects, self._urgent_pause = None, set(), False
        if gate.session_id in disconnects:
            gate.disconnect(gate.session_id, now=now)
        if pause:
            gate.host_pause(now=now)
        if observation is not None:
            gate.observe(*observation, now=now)
        for _ in range(max_controls):
            try:
                method, args, kwargs, future = self.controls.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._pending_connects.pop(future, None)
                running = future.set_running_or_notify_cancel()
            if not running:
                continue
            try:
                if method == "connect" and args[0] in disconnects:
                    raise ValueError("session disconnected before owner accepted it")
                kwargs["now"] = now
                if (
                    method == "control"
                    and len(args) >= 2
                    and args[1] == "calibrate"
                    and self.calibration_provider is not None
                ):
                    neutral, specs = self.calibration_provider(gate)
                    kwargs.update(neutral=neutral, specs=specs)
                future.set_result(getattr(gate, method)(*args, **kwargs))
            except Exception as error:
                future.set_exception(error)
        self.publish_status(gate, now=now)

    def publish_status(self, gate, *, now):
        status = asdict(gate.telemetry(now=now))
        status.update(
            host_monotonic_s=now, type="status", measured_tracking_error=None, ik_residual=None
        )
        with self._lock:
            self._status = status

    async def on_control(self, session, message):
        future = self.submit(
            "control",
            session.session_id,
            message["action"],
            message["event_id"],
            message["calibration_generation"],
        )
        try:
            result = await asyncio.wait_for(asyncio.wrap_future(future), 0.2)
        except (asyncio.TimeoutError, RuntimeError, ValueError):
            self.request_pause()
            return {"accepted": False, "reason": "owner_unavailable"}
        generation = result["calibration_generation"]
        if generation > session.calibration_generation:
            session.set_calibration_generation(generation)
        return result


class GateControlServer(ControlServer):
    """Use gate-owned status rather than the base host's fixed Paused placeholder.

    observation_provider(session) returns (LifecycleEvidence, VideoEvidence)
    backed by independently qualified lifecycle + #21 verified pair provenance.
    Without it, no human command can engage. Provider must not access live MuJoCo.
    """

    def __init__(self, mailbox, *, observation_provider=None, registry=None, **kwargs):
        registry = registry if registry is not None else SessionRegistry()
        registry.add_disconnect_callback(mailbox.disconnect)
        super().__init__(registry=registry, on_control=mailbox.on_control, **kwargs)
        self.mailbox, self.observation_provider = mailbox, observation_provider

    async def _status_loop(self, session):
        from websockets.exceptions import ConnectionClosed

        try:
            await asyncio.wait_for(
                asyncio.wrap_future(self.mailbox.submit("connect", session.session_id)), 2
            )
            while session.active:
                if self.observation_provider is not None and session.latest_tracking is not None:
                    lifecycle, video = self.observation_provider(session)
                    self.mailbox.observe(session.latest_tracking, session.clock, lifecycle, video)
                status = self.mailbox.status()
                if status is not None and status["session_id"] == session.session_id:
                    if time.monotonic() - status["host_monotonic_s"] > 0.15:
                        raise RuntimeError("owner status stale")
                    generation = status["calibration_generation"]
                    if generation < session.calibration_generation:
                        await asyncio.sleep(0.002)
                        continue
                    if generation > session.calibration_generation:
                        session.set_calibration_generation(generation)
                    await session.send(status)
                await asyncio.sleep(0.02)
        except (ConnectionClosed, asyncio.TimeoutError, ValueError, RuntimeError):
            await session.connection.close(code=1001, reason="simulator owner unavailable")
