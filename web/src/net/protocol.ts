// Wire types for the `web_teleop_v1` profile. Mirrors
// src/abstract_stack/web_teleop/protocol.py; see docs/web_teleop/README.md.
// All poses are MuJoCo conventions: metres, radians, quaternions [w, x, y, z],
// and arm targets in the `mabel_chassis` root frame.

export const SCHEMA_VERSION = 1;
export const PROFILE = "web_teleop_v1";

export type Side = "left" | "right";
export const SIDES: readonly Side[] = ["left", "right"];
export type TeleopState = "disconnected" | "ready" | "engaged" | "paused" | "fault";
export type Role = "operator" | "spectator";
export type ControlAction = "engage" | "pause" | "reset" | "ready";

export type Vec3 = [number, number, number];
export type QuatWxyz = [number, number, number, number];

export interface Pose {
  position_m: Vec3;
  quaternion_wxyz: QuatWxyz;
}

export interface ArmStatus {
  palm: Pose;
  ik: {
    reachable: boolean;
    ik_position_error_m: number;
    ik_orientation_error_rad: number;
  } | null;
}

export interface Channels {
  head: { yaw_rad: number; pitch_rad: number };
  lift_m: number;
  torso_rad: number;
  grip: Record<Side, number>;
  ranges: {
    head: { yaw_rad: [number, number]; pitch_rad: [number, number] };
    lift_m: [number, number];
    torso_rad: [number, number];
  };
}

export interface Status {
  state: TeleopState;
  reason: string;
  calibration_generation: number;
  accepted_sequence: number | null;
  operator_connected: boolean;
  spectators: number;
  episode_over: boolean;
  tick: number;
  sim_time_s: number;
  host_monotonic_s: number;
  real_time_factor: number;
  step_ms: number;
  overruns: number;
  task: {
    stocked_count: number;
    product_count: number;
    success: boolean;
    tipped: string[];
    dropped: string[];
    intervention_count: number;
    elapsed_sim_time: number;
  };
  target_epoch: number;
  arm_targets: Record<Side, Pose>;
  arms: Record<Side, ArmStatus>;
  channels: Channels;
  limits: { rate_limited: string[]; target_limited: string[]; force_limited: string[] };
}

export interface HelloAck {
  type: "hello_ack";
  session_id: string;
  role: Role;
  model_hash: string;
  calibration_generation: number;
  budgets: { control_hz: number; command_hz: number; input_timeout_s: number; status_hz: number };
  status: Status;
}

export interface ControlAck {
  type: "control_ack";
  event_id: number;
  action: ControlAction;
  accepted: boolean;
  error: string | null;
  calibration_generation: number;
  state: TeleopState;
}

export type ServerMessage =
  | HelloAck
  | ControlAck
  | ({ type: "status"; role: Role } & Status)
  | { type: "pong"; t0: number; t1: number }
  | { type: "error"; message: string };

export interface ArmIntent {
  position_m: Vec3;
  quaternion_wxyz: QuatWxyz;
  grip: number;
}

export interface CommandBody {
  /** Epoch of the host arm targets these poses were rebased on (see target_epoch). */
  target_epoch: number;
  arms: Partial<Record<Side, ArmIntent | null>>;
  head: { yaw_rad: number; pitch_rad: number } | null;
  lift_m: number | null;
  torso_rad: number | null;
}

// --- Binary state frames ("WTS1") -----------------------------------------------------
// Header: magic(4) tick u32 sim_time f64 host_time f64 nbody u16 flags u16 = 28 bytes,
// then float32 xpos[nbody*3] followed by float32 xquat[nbody*4] (little-endian).

export const STATE_HEADER_BYTES = 28;
export const FLAG_ENGAGED = 1;

export interface StateFrame {
  tick: number;
  simTime: number;
  hostTime: number;
  flags: number;
  nbody: number;
  xpos: Float32Array;
  xquat: Float32Array;
}

export function decodeStateFrame(buffer: ArrayBuffer): StateFrame {
  if (buffer.byteLength < STATE_HEADER_BYTES) throw new Error("state frame too short");
  const view = new DataView(buffer);
  const magic = String.fromCharCode(
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
  );
  if (magic !== "WTS1") throw new Error(`bad state frame magic ${magic}`);
  const nbody = view.getUint16(24, true);
  const expected = STATE_HEADER_BYTES + nbody * 7 * 4;
  if (buffer.byteLength !== expected) {
    throw new Error(`state frame has ${buffer.byteLength} bytes, expected ${expected}`);
  }
  return {
    tick: view.getUint32(4, true),
    simTime: view.getFloat64(8, true),
    hostTime: view.getFloat64(16, true),
    nbody,
    flags: view.getUint16(26, true),
    // 28 is 4-byte aligned, so typed views avoid copying.
    xpos: new Float32Array(buffer, STATE_HEADER_BYTES, nbody * 3),
    xquat: new Float32Array(buffer, STATE_HEADER_BYTES + nbody * 12, nbody * 4),
  };
}
