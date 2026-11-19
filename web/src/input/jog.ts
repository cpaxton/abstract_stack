import { Quaternion, Vector3 } from "three";
import type { Side } from "../net/protocol";
import { clampChannels, intent, takeManualHead } from "../state/intent";

/** Normalised rates in [-1, 1]; combined from keyboard and touch each frame. */
export interface JogInput {
  /** Hand translation in the viewer's ground frame: x right, y away from viewer, z up. */
  move: Vector3;
  /** Hand rotation: x pitch (about view-right), y roll (about view-forward), z yaw (about up). */
  rotate: Vector3;
  grip: number;
  headYaw: number;
  headPitch: number;
  lift: number;
  fine: boolean;
}

export const emptyJog = (): JogInput => ({
  move: new Vector3(),
  rotate: new Vector3(),
  grip: 0,
  headYaw: 0,
  headPitch: 0,
  lift: 0,
  fine: false,
});

export const RATES = {
  linear: 0.22, // m/s
  angular: 1.0, // rad/s
  grip: 1.6, // fraction/s
  head: 1.0, // rad/s
  lift: 0.08, // m/s (host lift servo rate)
  fineScale: 0.25,
};

// Conservative root-frame box keeping ghosts near MABEL's reachable workspace so an
// unreachable target cannot drift indefinitely far from the robot.
const WORKSPACE = { min: new Vector3(-1.0, -0.85, 0.15), max: new Vector3(0.35, 0.85, 1.7) };

const tmpQ = new Quaternion();
const tmpV = new Vector3();
const up = new Vector3(0, 0, 1);

/**
 * Integrate one frame of jog input into `intent`.
 * `forward`/`right` are unit vectors in the robot root frame (MuJoCo, horizontal).
 */
export function applyJog(
  dt: number,
  input: JogInput,
  side: Side,
  forward: Vector3,
  right: Vector3,
): boolean {
  const scale = input.fine ? RATES.fineScale : 1;
  const arm = intent.arms[side];
  let changed = false;
  if (input.move.lengthSq() > 0) {
    tmpV
      .set(0, 0, 0)
      .addScaledVector(right, input.move.x)
      .addScaledVector(forward, input.move.y)
      .addScaledVector(up, input.move.z)
      .multiplyScalar(RATES.linear * scale * dt);
    arm.position.add(tmpV).clamp(WORKSPACE.min, WORKSPACE.max);
    changed = true;
  }
  const r = input.rotate;
  if (r.lengthSq() > 0) {
    const step = RATES.angular * scale * dt;
    // Rotations about root-frame axes (pre-multiply), so controls stay view-aligned.
    if (r.x) arm.quaternion.premultiply(tmpQ.setFromAxisAngle(right, r.x * step));
    if (r.y) arm.quaternion.premultiply(tmpQ.setFromAxisAngle(forward, r.y * step));
    if (r.z) arm.quaternion.premultiply(tmpQ.setFromAxisAngle(up, r.z * step));
    arm.quaternion.normalize();
    changed = true;
  }
  if (input.grip) {
    arm.grip += input.grip * RATES.grip * dt;
    changed = true;
  }
  if (input.headYaw || input.headPitch) {
    takeManualHead();
    intent.head.yaw += input.headYaw * RATES.head * scale * dt;
    intent.head.pitch += input.headPitch * RATES.head * scale * dt;
    changed = true;
  }
  if (input.lift) {
    intent.lift += input.lift * RATES.lift * scale * dt;
    changed = true;
  }
  if (changed) clampChannels();
  return changed;
}
