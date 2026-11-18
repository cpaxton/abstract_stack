import { Quaternion, Vector3 } from "three";
import type { CommandBody, Pose, Side, Status } from "../net/protocol";
import { SIDES } from "../net/protocol";

/**
 * Operator intent: the absolute targets sent to the host at `command_hz`.
 *
 * Mutable singleton updated at display rate by keyboard/touch/gizmo input; React
 * components read low-rate mirrors from the UI store instead. Arm poses are in the
 * MuJoCo `mabel_chassis` root frame (metres, Z up, robot forward = -X).
 */
export interface ArmTarget {
  position: Vector3;
  quaternion: Quaternion; // three.js (x, y, z, w) storage of the MuJoCo rotation
  grip: number;
}

export const intent = {
  initialized: false,
  epoch: -1,
  generation: -1,
  arms: {
    left: { position: new Vector3(), quaternion: new Quaternion(), grip: 0 },
    right: { position: new Vector3(), quaternion: new Quaternion(), grip: 0 },
  } as Record<Side, ArmTarget>,
  head: { yaw: 0, pitch: 0 },
  lift: 0.3175,
  torso: 0,
  ranges: null as Status["channels"]["ranges"] | null,
};

export function poseToTarget(pose: Pose, target: ArmTarget): void {
  const [x, y, z] = pose.position_m;
  const [w, qx, qy, qz] = pose.quaternion_wxyz;
  target.position.set(x, y, z);
  target.quaternion.set(qx, qy, qz, w).normalize();
}

/** Adopt host targets when the host moved them (epoch) or the episode changed. */
export function rebaseFromStatus(status: Status): boolean {
  let changed = false;
  if (status.target_epoch !== intent.epoch) {
    for (const side of SIDES) poseToTarget(status.arm_targets[side], intent.arms[side]);
    intent.epoch = status.target_epoch;
    changed = true;
  }
  if (status.calibration_generation !== intent.generation || !intent.initialized) {
    const ch = status.channels;
    intent.head.yaw = ch.head.yaw_rad;
    intent.head.pitch = ch.head.pitch_rad;
    intent.lift = ch.lift_m;
    intent.torso = ch.torso_rad;
    for (const side of SIDES) intent.arms[side].grip = ch.grip[side];
    intent.generation = status.calibration_generation;
    changed = true;
  }
  intent.ranges = status.channels.ranges;
  intent.initialized = true;
  return changed;
}

const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v));

export function clampChannels(): void {
  const r = intent.ranges;
  if (!r) return;
  intent.head.yaw = clamp(intent.head.yaw, r.head.yaw_rad);
  intent.head.pitch = clamp(intent.head.pitch, r.head.pitch_rad);
  intent.lift = clamp(intent.lift, r.lift_m);
  intent.torso = clamp(intent.torso, r.torso_rad);
  for (const side of SIDES) intent.arms[side].grip = clamp(intent.arms[side].grip, [0, 1]);
}

const round = (v: number) => Math.round(v * 1e5) / 1e5;

export function buildCommand(): CommandBody | null {
  if (!intent.initialized) return null;
  clampChannels();
  const arms: CommandBody["arms"] = {};
  for (const side of SIDES) {
    const a = intent.arms[side];
    const q = a.quaternion.normalize();
    arms[side] = {
      position_m: [round(a.position.x), round(a.position.y), round(a.position.z)],
      // Full precision: the host requires a unit quaternion within 1e-3.
      quaternion_wxyz: [q.w, q.x, q.y, q.z],
      grip: round(a.grip),
    };
  }
  return {
    target_epoch: intent.epoch,
    arms,
    head: { yaw_rad: round(intent.head.yaw), pitch_rad: round(intent.head.pitch) },
    lift_m: round(intent.lift),
    torso_rad: round(intent.torso),
  };
}
