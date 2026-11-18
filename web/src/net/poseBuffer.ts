import type { StateFrame } from "./protocol";

/**
 * Jitter buffer for 50 Hz body-pose frames, sampled at display rate.
 *
 * Frames are placed on the local clock through a sim-time→local-time offset that
 * tracks the *fastest* recent arrival (minimum of `arrival - simTime`), so network
 * jitter only delays playback rather than making it stutter. Playback runs
 * `delayMs` behind that estimate and interpolates between the two bracketing frames
 * (lerp positions, nlerp quaternions — adequate at 20 ms spacing).
 */
export class PoseBuffer {
  private frames: { frame: StateFrame; arrival: number }[] = [];
  private offset = Number.POSITIVE_INFINITY;
  private offsetAge = 0;
  readonly capacity: number;
  delayMs: number;
  /** Scratch output buffers reused by `sample` (no per-frame allocation). */
  xpos = new Float32Array(0);
  xquat = new Float32Array(0);
  lastArrival = 0;

  constructor(delayMs = 40, capacity = 16) {
    this.delayMs = delayMs;
    this.capacity = capacity;
  }

  clear(): void {
    this.frames = [];
    this.offset = Number.POSITIVE_INFINITY;
  }

  push(frame: StateFrame, arrival: number = performance.now()): void {
    const last = this.frames[this.frames.length - 1];
    if (last && frame.simTime < last.frame.simTime) {
      // Episode reset (sim time went backwards): restart playback timeline.
      this.clear();
    } else if (last && frame.simTime === last.frame.simTime) {
      return;
    }
    const candidate = arrival - frame.simTime * 1000;
    // Follow improvements immediately; let the estimate drift up slowly (≈1 ms/s)
    // so clock skew or a route change cannot pin it to a stale minimum.
    if (candidate < this.offset || arrival - this.offsetAge > 1000) {
      this.offset = Math.min(candidate, this.offset + 1);
      this.offsetAge = arrival;
      if (!Number.isFinite(this.offset)) this.offset = candidate;
    }
    this.frames.push({ frame, arrival });
    if (this.frames.length > this.capacity) this.frames.shift();
    this.lastArrival = arrival;
    if (this.xpos.length !== frame.xpos.length) {
      this.xpos = new Float32Array(frame.xpos.length);
      this.xquat = new Float32Array(frame.xquat.length);
    }
  }

  get latest(): StateFrame | undefined {
    return this.frames[this.frames.length - 1]?.frame;
  }

  /** Age of the newest frame in milliseconds (for stale indicators). */
  ageMs(now: number = performance.now()): number {
    return this.frames.length ? now - this.lastArrival : Number.POSITIVE_INFINITY;
  }

  /** Interpolate poses for local time `now`; returns false when no data exists. */
  sample(now: number = performance.now()): boolean {
    const n = this.frames.length;
    if (n === 0) return false;
    const simMs = now - this.offset - this.delayMs;
    let i = n - 1;
    while (i > 0 && this.frames[i - 1]!.frame.simTime * 1000 > simMs) i--;
    const b = this.frames[i]!.frame;
    const a = i > 0 ? this.frames[i - 1]!.frame : b;
    const span = (b.simTime - a.simTime) * 1000;
    const t = span > 0 ? Math.min(1, Math.max(0, (simMs - a.simTime * 1000) / span)) : 1;
    // If playback is past the newest frame, hold it (no extrapolation of contacts).
    const alpha = simMs >= b.simTime * 1000 && i === n - 1 ? 1 : t;
    for (let k = 0; k < this.xpos.length; k++) {
      this.xpos[k] = a.xpos[k]! + (b.xpos[k]! - a.xpos[k]!) * alpha;
    }
    for (let k = 0; k < this.xquat.length; k += 4) {
      // Shortest-arc nlerp.
      const dot =
        a.xquat[k]! * b.xquat[k]! +
        a.xquat[k + 1]! * b.xquat[k + 1]! +
        a.xquat[k + 2]! * b.xquat[k + 2]! +
        a.xquat[k + 3]! * b.xquat[k + 3]!;
      const sign = dot < 0 ? -1 : 1;
      let norm = 0;
      for (let c = 0; c < 4; c++) {
        const v = a.xquat[k + c]! + (sign * b.xquat[k + c]! - a.xquat[k + c]!) * alpha;
        this.xquat[k + c] = v;
        norm += v * v;
      }
      norm = Math.sqrt(norm) || 1;
      for (let c = 0; c < 4; c++) this.xquat[k + c] = this.xquat[k + c]! / norm;
    }
    return true;
  }
}
