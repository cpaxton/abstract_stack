"""Starlette/uvicorn transport for web teleoperation.

Endpoints (all under one origin):

* ``GET  /api/model``        visual-geometry manifest (JSON)
* ``GET  /api/model.bin``    vertex/index blob referenced by the manifest
* ``WS   /teleop/control``   JSON: hello → hello_ack, command, control → control_ack,
                             ping → pong; host pushes status at 10 Hz
* ``WS   /teleop/state``     binary ``WTS1`` body-pose frames at the control rate,
                             bound to a control session with ``state_bind``
* ``/``                      the built web client (``web/dist``) when present

Delivery is latest-wins: each socket has its own sender that transmits the newest
frame/status when it is ready, so a slow peer skips frames rather than queueing
them or slowing the simulation owner.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import secrets
import time
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from abstract_stack.web_teleop import protocol
from abstract_stack.web_teleop.core import TeleopCore
from abstract_stack.web_teleop.geometry import load_model_export

log = logging.getLogger(__name__)
MAX_PROTOCOL_ERRORS = 20
HELLO_TIMEOUT_S = 5.0


class Hub:
    """Fan out sim-thread frames to asyncio sockets with latest-wins semantics."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.frame: bytes | None = None
        self.status: dict | None = None
        self._frame_waiters: set[asyncio.Event] = set()
        self._status_waiters: set[asyncio.Event] = set()

    def from_sim_thread(self, frame: bytes | None, status: dict | None) -> None:
        self.loop.call_soon_threadsafe(self._publish, frame, status)

    def _publish(self, frame, status) -> None:
        if frame is not None:
            self.frame = frame
            for event in self._frame_waiters:
                event.set()
        if status is not None:
            self.status = status
            for event in self._status_waiters:
                event.set()

    @contextlib.contextmanager
    def subscribe(self, kind: str):
        event = asyncio.Event()
        waiters = self._frame_waiters if kind == "frame" else self._status_waiters
        waiters.add(event)
        try:
            yield event
        finally:
            waiters.discard(event)


class TeleopServer:
    def __init__(
        self,
        core: TeleopCore,
        *,
        token: str | None,
        static_dir: Path | None = None,
        allowed_origins: tuple[str, ...] = (),
        face_budget: int = 800,
    ):
        self.core = core
        self.token = token
        self.allowed_origins = set(allowed_origins)
        self.manifest, self.blob = load_model_export(core.env.model, face_budget)
        self.hub: Hub | None = None
        self._sessions: dict[str, str] = {}  # session_id -> role
        routes = [
            Route("/api/model", self.model_manifest),
            Route("/api/model.bin", self.model_blob),
            Route("/api/health", self.health),
            WebSocketRoute("/teleop/control", self.control_socket),
            WebSocketRoute("/teleop/state", self.state_socket),
        ]
        if static_dir is not None and static_dir.is_dir():
            routes.append(Mount("/", StaticFiles(directory=static_dir, html=True), name="web"))
        self.app = Starlette(
            routes=routes,
            middleware=[Middleware(GZipMiddleware, minimum_size=1024)],
            lifespan=self._lifespan,
        )

    @contextlib.asynccontextmanager
    async def _lifespan(self, app):
        self.hub = Hub(asyncio.get_running_loop())
        self.core.add_listener(self.hub.from_sim_thread)
        self.core.start()
        try:
            yield
        finally:
            self.core.stop()

    # -- auth -------------------------------------------------------------------------
    def _authorised(self, scope_token: str | None) -> bool:
        if self.token is None:
            return True
        return scope_token is not None and hmac.compare_digest(scope_token, self.token)

    def _origin_ok(self, websocket: WebSocket) -> bool:
        origin = websocket.headers.get("origin")
        if origin is None:
            return True  # non-browser client; the token still applies
        host = websocket.headers.get("host", "")
        if origin.split("://", 1)[-1] == host:
            return True
        return origin in self.allowed_origins

    # -- HTTP -------------------------------------------------------------------------
    async def model_manifest(self, request: Request) -> Response:
        if not self._authorised(request.query_params.get("token")):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        return JSONResponse(self.manifest, headers={"Cache-Control": "no-cache"})

    async def model_blob(self, request: Request) -> Response:
        if not self._authorised(request.query_params.get("token")):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        # The client requests ?v=<model_hash>; content for a hash never changes.
        return Response(
            self.blob,
            media_type="application/octet-stream",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    async def health(self, request: Request) -> Response:
        status = self.core.status()
        return JSONResponse(
            {
                "state": status.get("state"),
                "tick": status.get("tick"),
                "real_time_factor": status.get("real_time_factor"),
            }
        )

    # -- WebSockets -------------------------------------------------------------------
    async def _accept(self, websocket: WebSocket) -> bool:
        if not self._origin_ok(websocket):
            await websocket.close(code=1008, reason="origin not allowed")
            return False
        if not self._authorised(websocket.query_params.get("token")):
            await websocket.close(code=1008, reason="unauthorised")
            return False
        await websocket.accept()
        return True

    async def control_socket(self, websocket: WebSocket) -> None:
        if not await self._accept(websocket):
            return
        session_id = None
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_S)
            message = protocol.decode_json(raw)
            if message["type"] != "hello":
                raise protocol.ProtocolError("first message must be hello")
            hello = protocol.parse_hello(message)
            role = self.core.open_session(hello["session_id"], hello["role"])
            session_id = hello["session_id"]
            self._sessions[session_id] = role
        except (protocol.ProtocolError, asyncio.TimeoutError, WebSocketDisconnect) as error:
            with contextlib.suppress(Exception):
                await websocket.send_text(protocol.envelope("error", None, message=str(error)))
                await websocket.close(code=1008)
            return
        status = self.core.status()
        await websocket.send_text(
            protocol.envelope(
                "hello_ack",
                session_id,
                profile=protocol.PROFILE,
                role=role,
                model_hash=self.manifest["model_hash"],
                calibration_generation=status["calibration_generation"],
                budgets={
                    "control_hz": round(1 / self.core.env.dt),
                    "command_hz": 30,
                    "input_timeout_s": self.core.config.input_timeout_s,
                    "status_hz": round(
                        1 / (self.core.env.dt * self.core.config.status_every_ticks)
                    ),
                },
                status=status,
            )
        )
        sender = asyncio.create_task(self._status_sender(websocket, session_id, role))
        errors = 0
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    await self._handle_control(websocket, session_id, raw)
                except protocol.ProtocolError as error:
                    errors += 1
                    await websocket.send_text(
                        protocol.envelope("error", session_id, message=str(error))
                    )
                    if errors >= MAX_PROTOCOL_ERRORS:
                        await websocket.close(code=1008, reason="too many protocol errors")
                        break
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            self._sessions.pop(session_id, None)
            self.core.close_session(session_id)

    async def _handle_control(self, websocket: WebSocket, session_id: str, raw: str) -> None:
        message = protocol.decode_json(raw)
        kind = message["type"]
        if kind == "command":
            command = protocol.parse_command(message)
            self._check_session(command, session_id)
            self.core.submit_command(session_id, command)
        elif kind == "control":
            event = protocol.parse_control(message)
            self._check_session(event, session_id)
            ack = await asyncio.wrap_future(self.core.control(session_id, event))
            await websocket.send_text(protocol.envelope("control_ack", session_id, **ack))
        elif kind == "ping":
            ping = protocol.parse_ping(message)
            self._check_session(ping, session_id)
            await websocket.send_text(
                protocol.envelope("pong", session_id, t0=ping["t0"], t1=time.monotonic())
            )
        else:
            raise protocol.ProtocolError(f"unknown message type {kind!r}")

    @staticmethod
    def _check_session(message: dict, session_id: str) -> None:
        if message["session_id"] != session_id:
            raise protocol.ProtocolError("session_id does not match this connection")

    async def _status_sender(self, websocket: WebSocket, session_id: str, role: str) -> None:
        assert self.hub is not None
        with self.hub.subscribe("status") as event, contextlib.suppress(Exception):
            while True:
                await event.wait()
                event.clear()
                status = self.hub.status
                if status is not None:
                    await websocket.send_text(
                        protocol.envelope("status", session_id, role=role, **status)
                    )

    async def state_socket(self, websocket: WebSocket) -> None:
        if not await self._accept(websocket):
            return
        try:
            message = protocol.decode_json(
                await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_S)
            )
            if message.get("type") != "state_bind" or set(message) != {
                "schema_version",
                "type",
                "session_id",
            }:
                raise protocol.ProtocolError("first state message must be state_bind")
            if message["session_id"] not in self._sessions:
                raise protocol.ProtocolError("state_bind requires an open control session")
            session_id = message["session_id"]
        except (protocol.ProtocolError, asyncio.TimeoutError, WebSocketDisconnect):
            with contextlib.suppress(Exception):
                await websocket.close(code=1008)
            return
        assert self.hub is not None
        with self.hub.subscribe("frame") as event:
            try:
                while session_id in self._sessions:
                    try:
                        await asyncio.wait_for(event.wait(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    event.clear()
                    if self.hub.frame is not None:
                        await websocket.send_bytes(self.hub.frame)
            except (WebSocketDisconnect, RuntimeError):
                return
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="control session closed")


def generate_token() -> str:
    return secrets.token_urlsafe(18)


def serve(
    core: TeleopCore,
    *,
    host: str,
    port: int,
    token: str | None,
    static_dir: Path | None,
    allowed_origins: tuple[str, ...] = (),
    face_budget: int = 800,
) -> None:
    import uvicorn

    server = TeleopServer(
        core,
        token=token,
        static_dir=static_dir,
        allowed_origins=allowed_origins,
        face_budget=face_budget,
    )
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    query = f"?token={token}" if token else ""
    print(f"Web teleop: http://{shown}:{port}/{query}", flush=True)
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level="info",
        ws_max_size=protocol.MAX_JSON_BYTES * 2,
        ws_ping_interval=10,
        ws_ping_timeout=10,
    )
