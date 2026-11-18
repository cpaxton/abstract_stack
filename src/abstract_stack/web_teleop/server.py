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
import gzip
import hmac
import logging
import re
import secrets
import socket
import threading
import time
from collections.abc import Callable
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
LOW_DETAIL_FACE_BUDGET = 800
CLOSE_UNAUTHORISED = 4401  # application close codes (4000-4999)
CLOSE_FORBIDDEN_ORIGIN = 4403
HELLO_TIMEOUT_S = 5.0


class Hub:
    """Fan out sim-thread frames to asyncio sockets with latest-wins semantics."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.frame: bytes | None = None
        self.status: dict | None = None
        self.video: bytes | None = None
        self._frame_waiters: set[asyncio.Event] = set()
        self._status_waiters: set[asyncio.Event] = set()
        self._video_waiters: set[asyncio.Event] = set()
        self.on_video_subscribers: Callable[[int], None] | None = None

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

    def video_from_thread(self, pair: bytes) -> None:
        self.loop.call_soon_threadsafe(self._publish_video, pair)

    def _publish_video(self, pair: bytes) -> None:
        self.video = pair
        for event in self._video_waiters:
            event.set()

    @contextlib.contextmanager
    def subscribe(self, kind: str):
        event = asyncio.Event()
        waiters = {
            "frame": self._frame_waiters,
            "status": self._status_waiters,
            "video": self._video_waiters,
        }[kind]
        waiters.add(event)
        if kind == "video" and self.on_video_subscribers:
            self.on_video_subscribers(len(waiters))
        try:
            yield event
        finally:
            waiters.discard(event)
            if kind == "video" and self.on_video_subscribers:
                self.on_video_subscribers(len(waiters))


class TeleopServer:
    def __init__(
        self,
        core: TeleopCore,
        *,
        token: str | None,
        static_dir: Path | None = None,
        allowed_origins: tuple[str, ...] = (),
        face_budget: int = 0,
    ):
        self.core = core
        self.token = token
        self.allowed_origins = set(allowed_origins)
        # "full" honours --face-budget (0 = full resolution); "low" is a light
        # simplified preview for weak devices, built on first request.
        self._exports: dict[str, tuple[dict, bytes, bytes]] = {}
        self._export_budgets = {"full": face_budget, "low": face_budget or LOW_DETAIL_FACE_BUDGET}
        self._export_lock = threading.Lock()
        self.manifest = self._export("full")[0]
        self.hub: Hub | None = None
        self._sessions: dict[str, str] = {}  # session_id -> role
        routes = [
            Route("/api/model", self.model_manifest),
            Route("/api/model.bin", self.model_blob),
            Route("/api/health", self.health),
            WebSocketRoute("/teleop/control", self.control_socket),
            WebSocketRoute("/teleop/state", self.state_socket),
            WebSocketRoute("/teleop/video", self.video_socket),
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
        video = self.core.video
        if video is not None:
            video.set_listener(self.hub.video_from_thread)
            self.hub.on_video_subscribers = lambda count: video.set_active(count > 0)
        self.core.start()
        try:
            yield
        finally:
            self.core.stop()

    # -- auth -------------------------------------------------------------------------
    def _authorised(self, scope_token: str | None) -> bool:
        if self.token is None:
            return True
        return scope_token is not None and hmac.compare_digest(
            normalise_token(scope_token).encode(), normalise_token(self.token).encode()
        )

    def _origin_ok(self, websocket: WebSocket) -> bool:
        origin = websocket.headers.get("origin")
        if origin is None:
            return True  # non-browser client; the token still applies
        host = websocket.headers.get("host", "")
        if origin.split("://", 1)[-1] == host:
            return True
        return origin in self.allowed_origins

    # -- HTTP -------------------------------------------------------------------------
    def _export(self, detail: str) -> tuple[dict, bytes, bytes]:
        with self._export_lock:
            if detail not in self._exports:
                manifest, blob = load_model_export(
                    self.core.env.model, self._export_budgets[detail]
                )
                # Compress once: the full-resolution blob is tens of MB uncompressed.
                self._exports[detail] = (manifest, blob, gzip.compress(blob, 6, mtime=0))
            return self._exports[detail]

    @staticmethod
    def _detail(request: Request) -> str:
        return "low" if request.query_params.get("detail") == "low" else "full"

    async def model_manifest(self, request: Request) -> Response:
        if not self._authorised(request.query_params.get("token")):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        manifest, _, _ = await asyncio.to_thread(self._export, self._detail(request))
        return JSONResponse(manifest, headers={"Cache-Control": "no-cache"})

    async def model_blob(self, request: Request) -> Response:
        if not self._authorised(request.query_params.get("token")):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        # The client requests ?v=<model_hash>; content for a hash never changes.
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            "Vary": "Accept-Encoding",
        }
        _, blob, blob_gzip = await asyncio.to_thread(self._export, self._detail(request))
        if "gzip" in request.headers.get("accept-encoding", ""):
            headers["Content-Encoding"] = "gzip"
            return Response(blob_gzip, media_type="application/octet-stream", headers=headers)
        return Response(blob, media_type="application/octet-stream", headers=headers)

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
        # Accept, then close with an application code: a handshake-time rejection
        # reaches browsers only as an anonymous 1006, so the client could not tell
        # "wrong token" (stop and ask) from "host restarting" (retry).
        await websocket.accept()
        if not self._origin_ok(websocket):
            await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN, reason="origin not allowed")
            return False
        if not self._authorised(websocket.query_params.get("token")):
            await websocket.close(code=CLOSE_UNAUTHORISED, reason="invalid access token")
            return False
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
                video=self._video_descriptor(),
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

    def _video_descriptor(self) -> dict | None:
        video = self.core.video
        if video is None:
            return None
        return {
            "cameras": list(video.cameras),
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "codec": "jpeg",
            "vertical_fov_deg": video.fovy,
        }

    async def _bind(self, websocket: WebSocket, kind: str) -> str | None:
        """Accept a secondary socket bound to an open control session."""
        if not await self._accept(websocket):
            return None
        try:
            message = protocol.decode_json(
                await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_S)
            )
            if message.get("type") != f"{kind}_bind" or set(message) != {
                "schema_version",
                "type",
                "session_id",
            }:
                raise protocol.ProtocolError(f"first {kind} message must be {kind}_bind")
            if message["session_id"] not in self._sessions:
                raise protocol.ProtocolError(f"{kind}_bind requires an open control session")
            return message["session_id"]
        except (protocol.ProtocolError, asyncio.TimeoutError, WebSocketDisconnect):
            with contextlib.suppress(Exception):
                await websocket.close(code=1008)
            return None

    async def _stream(self, websocket: WebSocket, session_id: str, kind: str) -> None:
        """Latest-wins binary stream until the socket or its control session closes."""
        assert self.hub is not None
        attribute = "frame" if kind == "frame" else "video"
        with self.hub.subscribe(kind) as event:
            try:
                while session_id in self._sessions:
                    try:
                        await asyncio.wait_for(event.wait(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    event.clear()
                    payload = getattr(self.hub, attribute)
                    if payload is not None:
                        await websocket.send_bytes(payload)
            except (WebSocketDisconnect, RuntimeError):
                return
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="control session closed")

    async def state_socket(self, websocket: WebSocket) -> None:
        session_id = await self._bind(websocket, "state")
        if session_id is not None:
            await self._stream(websocket, session_id, "frame")

    async def video_socket(self, websocket: WebSocket) -> None:
        if self.core.video is None:
            await websocket.close(code=1008, reason="video disabled on this host")
            return
        session_id = await self._bind(websocket, "video")
        if session_id is not None:
            await self._stream(websocket, session_id, "video")


def lan_addresses() -> list[str]:
    """IPv4 addresses other devices on the local network can use to reach this host."""
    found: list[str] = []
    # The default-route interface first: connecting a UDP socket sends no packets.
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 80))  # TEST-NET-1, never routed
        found.append(probe.getsockname()[0])
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    unique = []
    for address in found:
        if address not in unique and not address.startswith(("127.", "169.254.")):
            unique.append(address)
    return unique


# No look-alike characters (0/o, 1/l/i), so a token read off a screen and typed on
# a phone survives; grouped 4-4-4 (~60 bits) and compared case-insensitively.
TOKEN_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"
TOKEN_PATTERN = re.compile(rf"^[{TOKEN_ALPHABET}]{{4}}(-[{TOKEN_ALPHABET}]{{4}}){{2}}$")


def generate_token() -> str:
    chars = "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(12))
    return f"{chars[:4]}-{chars[4:8]}-{chars[8:]}"


def normalise_token(token: str) -> str:
    """Ignore case, spaces and dashes so keyboard auto-capitalisation still matches."""
    return re.sub(r"[\s-]", "", token).lower()


def persistent_token(path: Path | None = None, *, rotate: bool = False) -> str:
    """Token kept across runs so saved links keep working; ``rotate`` issues a new one."""
    path = path or token_path()
    if not rotate:
        with contextlib.suppress(OSError):
            token = path.read_text().strip()
            if TOKEN_PATTERN.match(token):  # older look-alike tokens are replaced
                return token
    token = generate_token()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        path.write_text(token + "\n")
    except OSError as error:
        log.warning("could not store access token (%s); it will change next run", error)
    return token


def token_path() -> Path:
    import os

    root = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(root) / "abstract_stack" / "web_teleop_token"


def serve(
    core: TeleopCore,
    *,
    host: str,
    port: int,
    token: str | None,
    static_dir: Path | None,
    allowed_origins: tuple[str, ...] = (),
    face_budget: int = 0,
) -> None:
    import uvicorn

    server = TeleopServer(
        core,
        token=token,
        static_dir=static_dir,
        allowed_origins=allowed_origins,
        face_budget=face_budget,
    )
    query = f"?token={token}" if token else ""
    if host in ("0.0.0.0", "::"):
        print(f"Web teleop (this computer): http://localhost:{port}/{query}", flush=True)
        addresses = lan_addresses()
        for address in addresses:
            print(f"Web teleop (LAN, phone/tablet): http://{address}:{port}/{query}", flush=True)
        if not addresses:
            print("No LAN address found; check the network connection.", flush=True)
    else:
        print(f"Web teleop: http://{host}:{port}/{query}", flush=True)
        if host in ("127.0.0.1", "localhost", "::1"):
            print(
                "Only this computer can connect. Use --host 0.0.0.0 to allow phones "
                "and tablets on your network.",
                flush=True,
            )
    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_level="info",
        ws_max_size=protocol.MAX_JSON_BYTES * 2,
        ws_ping_interval=10,
        ws_ping_timeout=10,
    )
