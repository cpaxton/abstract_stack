import { ClockSync } from "./clock";
import { canDecodeH264 } from "./h264";
import { LatencyMonitor, QueueProbe, type LatencySnapshot } from "./latency";
import { PoseBuffer } from "./poseBuffer";
import type { Freshness } from "./freshness";
import { pairMetaOf, StereoDecoder, videoAgeMs, type StereoPair } from "./stereo";
import {
  PROFILE,
  SCHEMA_VERSION,
  decodeStateFrame,
  type CommandBody,
  type CommandTiming,
  type ControlAck,
  type ControlAction,
  type HelloAck,
  type Role,
  type ServerMessage,
  type Status,
} from "./protocol";

// Link feedback cadence, and the page lag beyond which arrival times are untrusted.
const LINK_FEEDBACK_MS = 250;
const PAGE_STALL_MS = 40;

// At most 125 commands/s on high-refresh displays.
const MIN_COMMAND_INTERVAL_MS = 8;

// Application close codes sent by the host (see web_teleop/server.py).
const CLOSE_UNAUTHORISED = 4401;
const CLOSE_FORBIDDEN_ORIGIN = 4403;
const CLOSE_FORBIDDEN_USER = 4405; // Tailscale login not on the host's allowlist

export type ConnectionState = "idle" | "connecting" | "open" | "reconnecting" | "closed";

export interface TeleopClientOptions {
  /** HTTP(S) origin of the teleop host, e.g. `location.origin`. */
  origin: string;
  token: string | null;
  role: Role;
  /** Called at `command_hz` while operator; return null to skip a send. */
  commandProvider: () => CommandBody | null;
  /** "auto" prefers H.264 when the browser can decode it; "jpeg" forces JPEG. */
  videoCodec?: "auto" | "jpeg";
}

export interface TeleopClientEvents {
  connection: (state: ConnectionState, detail?: string) => void;
  hello: (ack: HelloAck) => void;
  status: (status: Status, role: Role) => void;
  rtt: (ms: number) => void;
  error: (message: string) => void;
  video: () => void;
  /** The host rejected our credentials; retrying cannot help until they change. */
  auth: (message: string) => void;
}

/** RFC 4122 v4 UUID that also works in insecure (plain-HTTP LAN) contexts. */
export function uuid4(): string {
  if (typeof crypto.randomUUID === "function" && globalThis.isSecureContext) {
    return crypto.randomUUID();
  }
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6]! & 0x0f) | 0x40;
  b[8] = (b[8]! & 0x3f) | 0x80;
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0"));
  return `${h.slice(0, 4).join("")}-${h.slice(4, 6).join("")}-${h.slice(6, 8).join("")}-${h
    .slice(8, 10)
    .join("")}-${h.slice(10).join("")}`;
}

/**
 * Framework-agnostic connection to a `web_teleop_v1` host: control socket (JSON),
 * state socket (binary poses into `poses`), heartbeat/RTT and the command loop.
 * Each (re)connection uses a new session UUID, per the Quest v1 contract.
 */
export class TeleopClient {
  readonly poses = new PoseBuffer(40, 16, true);
  readonly clock = new ClockSync();
  readonly latency = new LatencyMonitor();
  readonly stereo = new StereoDecoder(
    () => this.emit("video"),
    () => this.sendVideo({ schema_version: SCHEMA_VERSION, type: "video_keyframe" }),
  );
  videoAvailable = false;
  /** Codecs the host offers (from hello_ack). */
  private hostCodecs: string[] = ["jpeg"];
  sessionId = "";
  role: Role | null = null;
  generation = 0;
  private control: WebSocket | null = null;
  private state: WebSocket | null = null;
  private video: WebSocket | null = null;
  private videoWanted = true;
  private listeners: { [K in keyof TeleopClientEvents]: Set<TeleopClientEvents[K]> } = {
    connection: new Set(),
    hello: new Set(),
    status: new Set(),
    rtt: new Set(),
    error: new Set(),
    video: new Set(),
    auth: new Set(),
  };
  private pending = new Map<number, (ack: ControlAck) => void>();
  private eventId = 0;
  private sequence = 0;
  private commandTimer: number | undefined;
  private commandFrame = 0;
  private pingTimer: number | undefined;
  private burstTimers: number[] = [];
  private linkTimer: number | undefined;
  readonly queueProbe = new QueueProbe();
  readonly videoProbe = new QueueProbe();
  private lastUplinkSequence = -1;
  private shownPose: { hostTime: number; arrival: number } | null = null;
  private shownVideo: StereoPair | null = null;
  /** Longest recent main-thread stall seen by the freshness watchdog (ms). */
  pageStallMs = 0;
  private retryTimer: number | undefined;
  private retries = 0;
  private stopped = true;
  private options: TeleopClientOptions;

  constructor(options: TeleopClientOptions) {
    this.options = options;
  }

  on<K extends keyof TeleopClientEvents>(event: K, fn: TeleopClientEvents[K]): () => void {
    this.listeners[event].add(fn);
    return () => this.listeners[event].delete(fn);
  }

  private emit<K extends keyof TeleopClientEvents>(
    event: K,
    ...args: Parameters<TeleopClientEvents[K]>
  ): void {
    for (const fn of this.listeners[event]) (fn as (...a: unknown[]) => void)(...args);
  }

  get token(): string | null {
    return this.options.token;
  }

  setToken(token: string | null): void {
    this.options.token = token;
  }

  private url(path: string): string {
    const url = new URL(path, this.options.origin);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    if (this.options.token) url.searchParams.set("token", this.options.token);
    return url.toString();
  }

  start(): void {
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    window.clearTimeout(this.retryTimer);
    this.teardown();
    this.emit("connection", "closed");
  }

  private open(): void {
    this.teardown();
    this.sessionId = uuid4();
    this.sequence = 0;
    this.emit("connection", this.retries ? "reconnecting" : "connecting");
    const control = new WebSocket(this.url("/teleop/control"));
    this.control = control;
    control.onopen = () => {
      control.send(
        JSON.stringify({
          schema_version: SCHEMA_VERSION,
          type: "hello",
          session_id: this.sessionId,
          profile: PROFILE,
          role: this.options.role,
        }),
      );
    };
    control.onmessage = (event) => this.onControlMessage(event.data as string);
    control.onclose = (event) => {
      if (this.control !== control) return;
      this.teardown();
      if (
        event.code === CLOSE_UNAUTHORISED ||
        event.code === CLOSE_FORBIDDEN_ORIGIN ||
        event.code === CLOSE_FORBIDDEN_USER
      ) {
        this.stopped = true;
        this.retries = 0;
        this.emit("connection", "closed", event.reason);
        this.emit(
          "auth",
          event.code === CLOSE_UNAUTHORISED
            ? "This link's access token is not valid for the running host."
            : event.code === CLOSE_FORBIDDEN_USER
              ? event.reason || "Your Tailscale login is not allowed to operate this robot."
              : "This page's address is not allowed to control the host.",
        );
        return;
      }
      if (event.code === 1008) this.emit("error", event.reason || "connection refused by host");
      this.scheduleRetry();
    };
  }

  private scheduleRetry(): void {
    if (this.stopped) return;
    this.retries += 1;
    const delay = Math.min(5000, 250 * 2 ** Math.min(this.retries, 5));
    this.emit("connection", "reconnecting", `retrying in ${(delay / 1000).toFixed(1)} s`);
    this.retryTimer = window.setTimeout(() => this.open(), delay);
  }

  private teardown(): void {
    window.clearInterval(this.commandTimer);
    cancelAnimationFrame(this.commandFrame);
    window.clearInterval(this.pingTimer);
    window.clearInterval(this.linkTimer);
    this.queueProbe.clear();
    this.videoProbe.clear();
    this.burstTimers.forEach((timer) => window.clearTimeout(timer));
    this.burstTimers = [];
    this.clock.clear();
    this.latency.clear();
    this.lastUplinkSequence = -1; // sequences restart with the next session
    this.shownPose = null;
    this.shownVideo = null;
    for (const socket of [this.control, this.state, this.video]) {
      if (socket) {
        socket.onclose = null;
        socket.onmessage = null;
        socket.close();
      }
    }
    this.control = this.state = this.video = null;
    this.stereo.clear();
    this.role = null;
    for (const resolve of this.pending.values()) {
      resolve({
        type: "control_ack",
        event_id: -1,
        action: "pause",
        accepted: false,
        error: "disconnected",
        calibration_generation: this.generation,
        state: "disconnected",
      });
    }
    this.pending.clear();
    this.poses.clear();
  }

  private onControlMessage(raw: string): void {
    let message: ServerMessage;
    try {
      message = JSON.parse(raw) as ServerMessage;
    } catch {
      return;
    }
    switch (message.type) {
      case "hello_ack":
        this.retries = 0;
        this.role = message.role;
        this.generation = message.calibration_generation;
        this.openState();
        this.videoAvailable = message.video !== null;
        this.hostCodecs = message.video?.codecs ?? ["jpeg"];
        if (this.videoWanted) this.openVideo();
        this.startLoops(message.budgets.command_hz);
        this.emit("connection", "open");
        this.emit("hello", message);
        this.emit("status", message.status, message.role);
        break;
      case "status":
        this.generation = message.calibration_generation;
        this.recordUplink(message.command_timing);
        this.emit("status", message, message.role);
        break;
      case "control_ack":
        this.generation = message.calibration_generation;
        this.pending.get(message.event_id)?.(message);
        this.pending.delete(message.event_id);
        break;
      case "pong": {
        const now = performance.now();
        this.clock.add(message.t0, message.t1, now);
        this.emit("rtt", now - message.t0);
        break;
      }
      case "error":
        this.emit("error", message.message);
        break;
    }
  }

  private sendVideo(message: object): void {
    if (this.video?.readyState === WebSocket.OPEN) this.video.send(JSON.stringify(message));
  }

  /** Subscribe to head-camera video only while it is visible (saves host GPU and bandwidth). */
  setVideoWanted(wanted: boolean): void {
    this.videoWanted = wanted;
    if (wanted && this.role && !this.video) this.openVideo();
    if (!wanted && this.video) {
      this.video.onclose = null;
      this.video.close();
      this.video = null;
      this.stereo.clear();
      this.shownVideo = null;
    }
  }

  private openVideo(): void {
    if (!this.videoAvailable || this.video) return;
    const video = new WebSocket(this.url("/teleop/video"));
    video.binaryType = "arraybuffer";
    this.video = video;
    video.onopen = async () => {
      // H.264 is about 7x lighter than JPEG; offer it when both ends support it.
      const h264 =
        this.options.videoCodec !== "jpeg" &&
        this.hostCodecs.includes("h264") &&
        (await canDecodeH264());
      if (video.readyState !== WebSocket.OPEN) return;
      video.send(
        JSON.stringify({
          schema_version: SCHEMA_VERSION,
          type: "video_bind",
          session_id: this.sessionId,
          codecs: h264 ? ["h264", "jpeg"] : ["jpeg"],
        }),
      );
    };
    video.onmessage = (event) => {
      const buffer = event.data as ArrayBuffer;
      const arrival = performance.now();
      // Ack on arrival: the host bounds frames in flight to about one round trip.
      const meta = pairMetaOf(buffer);
      if (meta) {
        video.send(
          JSON.stringify({
            schema_version: SCHEMA_VERSION,
            type: "video_ack",
            pair_id: meta.pair_id,
          }),
        );
        if (this.clock.synced) {
          this.videoProbe.push(arrival - this.clock.toLocalMs(meta.capture_host_monotonic_s));
        }
      }
      this.stereo.push(buffer, arrival);
    };
    video.onclose = () => {
      if (this.video === video) this.video = null;
    };
  }

  private openState(): void {
    const state = new WebSocket(this.url("/teleop/state"));
    state.binaryType = "arraybuffer";
    this.state = state;
    state.onopen = () =>
      state.send(
        JSON.stringify({
          schema_version: SCHEMA_VERSION,
          type: "state_bind",
          session_id: this.sessionId,
          format: "wts2", // compact int16 poses: half the bandwidth
        }),
      );
    state.onmessage = (event) => {
      try {
        const arrival = performance.now();
        const frame = decodeStateFrame(event.data as ArrayBuffer);
        state.send(
          JSON.stringify({ schema_version: SCHEMA_VERSION, type: "state_ack", tick: frame.tick }),
        );
        this.poses.push(frame, arrival);
        if (this.clock.synced) {
          const delay = arrival - this.clock.toLocalMs(frame.hostTime);
          this.latency.poseNetwork.push(delay);
          this.queueProbe.push(delay);
        }
      } catch (error) {
        this.emit("error", String(error));
      }
    };
  }

  private startLoops(commandHz: number): void {
    const send = (payload: object) => {
      if (this.control?.readyState === WebSocket.OPEN) {
        this.control.send(
          JSON.stringify({
            schema_version: SCHEMA_VERSION,
            session_id: this.sessionId,
            ...payload,
          }),
        );
      }
    };
    const ping = () => send({ type: "ping", t0: performance.now() });
    // A short burst converges the clock offset within half a second of connecting;
    // after that one round trip per second tracks drift and route changes.
    for (let i = 0; i < 5; i++) this.burstTimers.push(window.setTimeout(ping, i * 100));
    this.pingTimer = window.setInterval(ping, 1000);
    // Report queueing delay so the host can adapt video bitrate and pose rate.
    let lastLinkTick = performance.now();
    this.linkTimer = window.setInterval(() => {
      const now = performance.now();
      const lag = now - lastLinkTick - LINK_FEEDBACK_MS;
      lastLinkTick = now;
      // Either stream queueing means the link is full (browsers may queue sockets
      // separately); report the worse of the two.
      const pose = this.queueProbe.window();
      const video = this.videoProbe.window();
      // Arrival times are taken on the main thread: while the page itself stalls
      // (long frames on a slow device), they measure the page, not the network.
      // Skip such windows; the host holds its levels without feedback.
      const pageStalled = lag > PAGE_STALL_MS || this.pageStallMs > PAGE_STALL_MS;
      if ((pose !== null || video !== null) && this.role === "operator" && !pageStalled) {
        send({ type: "link_feedback", queue_ms: Math.min(60000, Math.max(pose ?? 0, video ?? 0)) });
      }
    }, LINK_FEEDBACK_MS);
    let lastSent = Number.NEGATIVE_INFINITY;
    const sendCommand = () => {
      if (this.role !== "operator") return;
      const now = performance.now();
      if (now - lastSent < MIN_COMMAND_INTERVAL_MS) return;
      const body = this.options.commandProvider();
      if (!body) return;
      lastSent = now;
      this.sequence += 1;
      send({
        type: "command",
        sequence: this.sequence,
        calibration_generation: this.generation,
        sent_monotonic_s: now / 1000,
        ...body,
      });
    };
    // Send once per display frame, just after that frame's input handling (jog
    // loop, drags) has updated the intent: setTimeout(0) from a rAF callback runs
    // after the frame's other callbacks. A fixed timer would add up to a full
    // interval (8 ms on average at 60 Hz) between input and send.
    const pump = () => {
      this.commandFrame = requestAnimationFrame(pump);
      window.setTimeout(sendCommand, 0);
    };
    this.commandFrame = requestAnimationFrame(pump);
    // Fallback where rAF is throttled (e.g. an occluded window): never slower than
    // the host's command budget.
    const interval = 1000 / commandHz;
    this.commandTimer = window.setInterval(() => {
      if (performance.now() - lastSent >= interval) sendCommand();
    }, interval);
  }

  /** Browser send → host apply for the newest command the host echoed back. */
  private recordUplink(timing: CommandTiming | null | undefined): void {
    if (!timing || timing.sequence === this.lastUplinkSequence || !this.clock.synced) return;
    this.lastUplinkSequence = timing.sequence;
    this.latency.uplink.push(this.clock.toLocalMs(timing.applied_host_s) - timing.sent_ms);
  }

  /** Record that the 3D view just drew the pose from `poses.sample`. */
  markPoseShown(): void {
    this.shownPose = { hostTime: this.poses.sampledHostTime, arrival: this.poses.sampledArrival };
  }

  /** Record that a camera view just drew `pair`. */
  markVideoShown(pair: StereoPair): void {
    this.shownVideo = pair;
  }

  /**
   * Ages of what the operator actually sees: the pose the 3D view last drew (it
   * plays back behind the newest arrival) and the camera frame last drawn. A view
   * that has not drawn anything yet is unavailable. Host clock when synced.
   */
  freshness(now: number = performance.now()): Freshness {
    const shown = this.shownPose;
    const poseAgeMs = !shown
      ? Number.POSITIVE_INFINITY
      : this.clock.synced
        ? this.clock.ageMs(shown.hostTime, now)
        : now - shown.arrival;
    const pair = this.shownVideo;
    return {
      poseAgeMs,
      videoAgeMs: this.video && pair ? videoAgeMs(this.clock, pair, now) : null,
      videoAvailable: this.videoAvailable,
    };
  }

  /** Current per-stage latency summary (all on the browser clock). */
  latencySnapshot(): LatencySnapshot {
    return this.latency.snapshot(this.clock.rttMs, this.clock.bestRttMs, this.poses.delayMs);
  }

  /** Send a deliberate control action; resolves with the host acknowledgement. */
  sendControl(action: ControlAction, timeoutMs = 3000): Promise<ControlAck> {
    const control = this.control;
    if (!control || control.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("not connected"));
    }
    const eventId = ++this.eventId;
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.pending.delete(eventId);
        reject(new Error(`${action} timed out`));
      }, timeoutMs);
      this.pending.set(eventId, (ack) => {
        window.clearTimeout(timer);
        resolve(ack);
      });
      control.send(
        JSON.stringify({
          schema_version: SCHEMA_VERSION,
          type: "control",
          session_id: this.sessionId,
          event_id: eventId,
          calibration_generation: this.generation,
          action,
        }),
      );
    });
  }
}
