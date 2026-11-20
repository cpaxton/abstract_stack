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
import sys
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
from abstract_stack.web_teleop.link import InFlightWindow, LinkController
from abstract_stack.web_teleop.video import H264_CODEC_STRING

log = logging.getLogger(__name__)
MAX_PROTOCOL_ERRORS = 20
LOW_DETAIL_FACE_BUDGET = 800
# Frames an H.264 viewer may fall behind before it skips to the next keyframe.
H264_MAX_BACKLOG = 2
# Queueing allowed beyond one round trip before a sender skips (see InFlightWindow).
# Video gets more: a keyframe is several frames' worth of bytes, and resynchronising
# H.264 costs another keyframe.
POSE_SLACK_S = 0.06
VIDEO_SLACK_S = 0.2
CLOSE_UNAUTHORISED = 4401  # application close codes (4000-4999)
CLOSE_FORBIDDEN_ORIGIN = 4403
HELLO_TIMEOUT_S = 5.0


def _status_for(session_id: str, status: dict) -> dict:
    """Status as one session may see it: command timing only for its own commands.

    ``sent_ms`` is on the sending browser's clock and sequences restart per
    session, so another session's echo would be meaningless to it (and the
    operator's session id is not shared with spectators).
    """
    timing = status.get("command_timing")
    if not timing:
        return status
    own = timing.get("session_id") == session_id
    public = {key: value for key, value in timing.items() if key != "session_id"}
    return {**status, "command_timing": public if own else None}


def _state_tick(frame: bytes) -> int:
    """Tick from a WTS1/WTS2 header (uint32 little-endian at byte 4)."""
    return int.from_bytes(frame[4:8], "little")


class Hub:
    """Fan out sim-thread frames to asyncio sockets with latest-wins semantics."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.frame: bytes | None = None
        self.frame_compact: bytes | None = None
        self._frame_compact_waiters: set[asyncio.Event] = set()
        self._ticks = 0
        self._statuses = 0
        # Send poses only every Nth tick when the link controller asks (congestion).
        self.pose_divider = 1
        self.status: dict | None = None
        self.video: bytes | None = None
        self.video_id = 0
        self._frame_waiters: set[asyncio.Event] = set()
        self._status_waiters: set[asyncio.Event] = set()
        self._video_waiters: set[asyncio.Event] = set()
        self._h264_queues: set[asyncio.Queue] = set()
        self.on_video_demand: Callable[[str, int], None] | None = None

    def from_sim_thread(self, frame: bytes | None, status: dict | None) -> None:
        self.loop.call_soon_threadsafe(self._publish, frame, status)

    def _publish(self, frame, status) -> None:
        if frame is not None:
            self._ticks += 1
            if self._ticks % self.pose_divider == 0:
                self.frame = frame
                for event in self._frame_waiters:
                    event.set()
                if self._frame_compact_waiters:
                    self.frame_compact = protocol.compact_state(frame)
                    for event in self._frame_compact_waiters:
                        event.set()
        if status is not None:
            self._statuses += 1
            # Status (10 Hz JSON) thins out with poses on a congested link.
            if self._statuses % self.pose_divider == 0:
                self.status = status
                for event in self._status_waiters:
                    event.set()

    def video_from_thread(self, codec: str, message: bytes, keyframe: bool, pair_id: int) -> None:
        self.loop.call_soon_threadsafe(self._publish_video, codec, message, keyframe, pair_id)

    def _publish_video(self, codec: str, message: bytes, keyframe: bool, pair_id: int) -> None:
        if codec == "h264":
            # Every viewer needs every H.264 frame (deltas depend on their
            # predecessors), so each gets a queue rather than a latest-wins slot.
            for queue in self._h264_queues:
                queue.put_nowait((message, keyframe, pair_id))
            return
        self.video = message
        self.video_id = pair_id
        for event in self._video_waiters:
            event.set()

    @contextlib.contextmanager
    def subscribe_h264(self):
        queue: asyncio.Queue = asyncio.Queue()
        self._h264_queues.add(queue)
        if self.on_video_demand:
            self.on_video_demand("h264", len(self._h264_queues))
        try:
            yield queue
        finally:
            self._h264_queues.discard(queue)
            if self.on_video_demand:
                self.on_video_demand("h264", len(self._h264_queues))

    @contextlib.contextmanager
    def subscribe(self, kind: str):
        event = asyncio.Event()
        waiters = {
            "frame": self._frame_waiters,
            "frame_compact": self._frame_compact_waiters,
            "status": self._status_waiters,
            "video": self._video_waiters,
        }[kind]
        waiters.add(event)
        if kind == "video" and self.on_video_demand:
            self.on_video_demand("jpeg", len(waiters))
        try:
            yield event
        finally:
            waiters.discard(event)
            if kind == "video" and self.on_video_demand:
                self.on_video_demand("jpeg", len(waiters))


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
        video = core.video
        self.link = LinkController(video.h264_max_kbps if video is not None else 1500)
        if video is not None:
            video.on_congestion = lambda: self._apply_link(self.link.video_congestion)
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
            self.hub.on_video_demand = video.set_demand
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
                    "command_hz": 60,
                    "input_timeout_s": self.core.config.input_timeout_s,
                    "status_hz": round(
                        1 / (self.core.env.dt * self.core.config.status_every_ticks)
                    ),
                },
                video=self._video_descriptor(),
                status=_status_for(session_id, status),
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
        elif kind == "link_feedback":
            feedback = protocol.parse_link_feedback(message)
            self._check_session(feedback, session_id)
            # The operator's link drives adaptation (one shared encoder and rate).
            if self._sessions.get(session_id) == "operator":
                self._apply_link(lambda: self.link.feedback(feedback["queue_ms"]))
        elif kind == "ping":
            ping = protocol.parse_ping(message)
            self._check_session(ping, session_id)
            await websocket.send_text(
                protocol.envelope("pong", session_id, t0=ping["t0"], t1=time.monotonic())
            )
        else:
            raise protocol.ProtocolError(f"unknown message type {kind!r}")

    def _apply_link(self, update) -> None:
        """Run a link-controller update, then push its levels to video and poses."""
        update()
        if self.core.video is not None:
            self.core.video.set_bitrate_kbps(self.link.kbps)
            self.core.video.frame_divider = self.link.video_divider
        if self.hub is not None:
            self.hub.pose_divider = self.link.pose_divider

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
                        protocol.envelope(
                            "status",
                            session_id,
                            role=role,
                            **_status_for(session_id, {**status, "link": self.link.report()}),
                        )
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
            # Encodings this host offers; the client lists its preference in video_bind.
            "codecs": list(video.codecs),
            "h264_codec_string": H264_CODEC_STRING if "h264" in video.codecs else None,
            "vertical_fov_deg": video.fovy,
        }

    async def _bind(self, websocket: WebSocket, kind: str) -> str | None:
        """Accept a secondary socket bound to an open control session."""
        message = await self._bind_message(websocket, kind)
        return None if message is None else message["session_id"]

    async def _bind_message(self, websocket: WebSocket, kind: str) -> dict | None:
        """Like ``_bind`` but returns the bind message (video carries ``codecs``)."""
        if not await self._accept(websocket):
            return None
        try:
            message = protocol.decode_json(
                await asyncio.wait_for(websocket.receive_text(), HELLO_TIMEOUT_S)
            )
            required = {"schema_version", "type", "session_id"}
            optional = {"video": {"codecs"}, "state": {"format"}}.get(kind, set())
            keys = set(message)
            if message.get("type") != f"{kind}_bind" or not required <= keys <= required | optional:
                raise protocol.ProtocolError(f"first {kind} message must be {kind}_bind")
            codecs = message.get("codecs", ["jpeg"])
            if not (
                isinstance(codecs, list)
                and 0 < len(codecs) <= 4
                and all(isinstance(codec, str) for codec in codecs)
            ):
                raise protocol.ProtocolError("codecs must be a short list of codec names")
            if message.get("format", "wts1") not in ("wts1", "wts2"):
                raise protocol.ProtocolError("format must be wts1 or wts2")
            if message["session_id"] not in self._sessions:
                raise protocol.ProtocolError(f"{kind}_bind requires an open control session")
            return message
        except (protocol.ProtocolError, asyncio.TimeoutError, WebSocketDisconnect):
            with contextlib.suppress(Exception):
                await websocket.close(code=1008)
            return None

    async def _stream(self, websocket: WebSocket, session_id: str, kind: str) -> None:
        """Latest-wins binary stream, flow-controlled by the client's acks.

        Poses ("frame"/"frame_compact") are acked by tick, JPEG video by pair id. A
        frame that would exceed the in-flight window is skipped (the next one is
        newer anyway) and counts as congestion for the link controller.
        """
        assert self.hub is not None
        video = kind == "video"
        window = InFlightWindow(slack_s=VIDEO_SLACK_S if video else POSE_SLACK_S)
        reader = asyncio.create_task(
            self._acks(websocket, window, "video_ack" if video else "state_ack")
        )
        with self.hub.subscribe(kind) as event:
            try:
                while session_id in self._sessions and not reader.done():
                    try:
                        await asyncio.wait_for(event.wait(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    event.clear()
                    payload = getattr(self.hub, kind)
                    if payload is None:
                        continue
                    now = time.monotonic()
                    if not window.can_send(now, self._frame_interval(kind)):
                        continue  # latest-wins: the next frame is newer anyway
                    window.sent(self.hub.video_id if video else _state_tick(payload), now)
                    await websocket.send_bytes(payload)
            except (WebSocketDisconnect, RuntimeError):
                return
            finally:
                reader.cancel()
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="control session closed")

    def _frame_interval(self, kind: str) -> float:
        if kind in ("video", "h264"):
            return self.core.video.frame_divider / self.core.video.fps
        return self.core.env.dt * (self.hub.pose_divider if self.hub else 1)

    async def _acks(self, websocket: WebSocket, window: InFlightWindow, ack_type: str) -> None:
        """Client → host on a stream socket: cumulative acks (and keyframe requests)."""
        key = "pair_id" if ack_type == "video_ack" else "tick"
        last_keyframe_request = 0.0
        while True:
            raw = await websocket.receive_text()
            with contextlib.suppress(protocol.ProtocolError, KeyError, TypeError, ValueError):
                message = protocol.decode_json(raw)
                now = time.monotonic()
                if message["type"] == ack_type:
                    window.ack(int(message[key]), now)
                elif message["type"] == "video_keyframe" and now - last_keyframe_request > 0.5:
                    last_keyframe_request = now
                    self.core.video.request_keyframe()

    async def state_socket(self, websocket: WebSocket) -> None:
        message = await self._bind_message(websocket, "state")
        if message is not None:
            compact = message.get("format") == "wts2"
            await self._stream(
                websocket, message["session_id"], "frame_compact" if compact else "frame"
            )

    async def video_socket(self, websocket: WebSocket) -> None:
        if self.core.video is None:
            await websocket.close(code=1008, reason="video disabled on this host")
            return
        message = await self._bind_message(websocket, "video")
        if message is None:
            return
        # First codec in the client's preference order that this host can encode.
        offered = self.core.video.codecs
        codec = next((c for c in message.get("codecs", ["jpeg"]) if c in offered), None)
        if codec is None:
            await websocket.close(code=1008, reason=f"no common video codec (host: {offered})")
            return
        if codec == "h264":
            await self._stream_h264(websocket, message["session_id"])
        else:
            await self._stream(websocket, message["session_id"], "video")

    async def _stream_h264(self, websocket: WebSocket, session_id: str) -> None:
        """Ordered H.264 stream, flow-controlled by the client's acks.

        A delta frame cannot be skipped without corrupting the frames after it. When
        the in-flight window is full (the link cannot keep up), or this viewer's queue
        grows beyond ``H264_MAX_BACKLOG``, the sender drops its queue, reports
        congestion, and resumes at a fresh keyframe once the window has room.
        """
        assert self.hub is not None
        video = self.core.video
        window = InFlightWindow(slack_s=VIDEO_SLACK_S)
        reader = asyncio.create_task(self._acks(websocket, window, "video_ack"))
        with self.hub.subscribe_h264() as queue:
            waiting_for_key = True
            try:
                while session_id in self._sessions and not reader.done():
                    try:
                        message, keyframe, pair_id = await asyncio.wait_for(queue.get(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    now = time.monotonic()
                    interval = self._frame_interval("h264")
                    if queue.qsize() > H264_MAX_BACKLOG or not window.can_send(now, interval):
                        while not queue.empty():
                            queue.get_nowait()
                        if not waiting_for_key:
                            self._apply_link(self.link.video_congestion)
                        waiting_for_key = True
                        continue
                    if waiting_for_key and not keyframe:
                        video.request_keyframe()  # the window has room again
                        continue
                    waiting_for_key = False
                    window.sent(pair_id, now)
                    await websocket.send_bytes(message)
            except (WebSocketDisconnect, RuntimeError):
                return
            finally:
                reader.cancel()
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="control session closed")


def print_qr(url: str) -> None:
    """Show ``url`` as a terminal QR code so a phone can open it without typing."""
    try:
        import segno
    except ImportError:
        return
    print("Scan with the phone camera:", flush=True)
    segno.make(url, error="m").terminal(compact=True, border=2)
    sys.stdout.flush()


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
    ws_deflate: bool = False,
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
        if addresses:
            print_qr(f"http://{addresses[0]}:{port}/{query}")
        else:
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
        # Frames are JPEG and float32 poses: deflate saves little and costs CPU and
        # latency on every message, so it is opt-in.
        ws_per_message_deflate=ws_deflate,
    )
