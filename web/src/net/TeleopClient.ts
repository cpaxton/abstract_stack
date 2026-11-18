import { PoseBuffer } from "./poseBuffer";
import { StereoDecoder } from "./stereo";
import {
  PROFILE,
  SCHEMA_VERSION,
  decodeStateFrame,
  type CommandBody,
  type ControlAck,
  type ControlAction,
  type HelloAck,
  type Role,
  type ServerMessage,
  type Status,
} from "./protocol";

export type ConnectionState = "idle" | "connecting" | "open" | "reconnecting" | "closed";

export interface TeleopClientOptions {
  /** HTTP(S) origin of the teleop host, e.g. `location.origin`. */
  origin: string;
  token: string | null;
  role: Role;
  /** Called at `command_hz` while operator; return null to skip a send. */
  commandProvider: () => CommandBody | null;
}

export interface TeleopClientEvents {
  connection: (state: ConnectionState, detail?: string) => void;
  hello: (ack: HelloAck) => void;
  status: (status: Status, role: Role) => void;
  rtt: (ms: number) => void;
  error: (message: string) => void;
  video: () => void;
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
  readonly poses = new PoseBuffer();
  readonly stereo = new StereoDecoder(() => this.emit("video"));
  videoAvailable = false;
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
  };
  private pending = new Map<number, (ack: ControlAck) => void>();
  private eventId = 0;
  private sequence = 0;
  private commandTimer: number | undefined;
  private pingTimer: number | undefined;
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
    window.clearInterval(this.pingTimer);
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
        if (this.videoWanted) this.openVideo();
        this.startLoops(message.budgets.command_hz);
        this.emit("connection", "open");
        this.emit("hello", message);
        this.emit("status", message.status, message.role);
        break;
      case "status":
        this.generation = message.calibration_generation;
        this.emit("status", message, message.role);
        break;
      case "control_ack":
        this.generation = message.calibration_generation;
        this.pending.get(message.event_id)?.(message);
        this.pending.delete(message.event_id);
        break;
      case "pong":
        this.emit("rtt", performance.now() - message.t0);
        break;
      case "error":
        this.emit("error", message.message);
        break;
    }
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
    }
  }

  private openVideo(): void {
    if (!this.videoAvailable || this.video) return;
    const video = new WebSocket(this.url("/teleop/video"));
    video.binaryType = "arraybuffer";
    this.video = video;
    video.onopen = () =>
      video.send(
        JSON.stringify({
          schema_version: SCHEMA_VERSION,
          type: "video_bind",
          session_id: this.sessionId,
        }),
      );
    video.onmessage = (event) => this.stereo.push(event.data as ArrayBuffer);
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
        }),
      );
    state.onmessage = (event) => {
      try {
        this.poses.push(decodeStateFrame(event.data as ArrayBuffer));
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
    this.pingTimer = window.setInterval(() => send({ type: "ping", t0: performance.now() }), 1000);
    send({ type: "ping", t0: performance.now() });
    this.commandTimer = window.setInterval(
      () => {
        if (this.role !== "operator") return;
        const body = this.options.commandProvider();
        if (!body) return;
        this.sequence += 1;
        send({
          type: "command",
          sequence: this.sequence,
          calibration_generation: this.generation,
          sent_monotonic_s: performance.now() / 1000,
          ...body,
        });
      },
      Math.round(1000 / commandHz),
    );
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
