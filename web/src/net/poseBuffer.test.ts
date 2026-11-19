import { describe, expect, it } from "vitest";
import { PoseBuffer } from "./poseBuffer";
import { decodeStateFrame, STATE_HEADER_BYTES, type StateFrame } from "./protocol";

function frame(simTime: number, x: number, quat = [1, 0, 0, 0]): StateFrame {
  return {
    tick: Math.round(simTime / 0.02),
    simTime,
    hostTime: simTime,
    flags: 0,
    nbody: 1,
    xpos: new Float32Array([x, 0, 0]),
    xquat: new Float32Array(quat),
  };
}

describe("PoseBuffer", () => {
  it("interpolates between bracketing frames behind the newest arrival", () => {
    const buffer = new PoseBuffer(20);
    buffer.push(frame(0.0, 0), 1000);
    buffer.push(frame(0.02, 1), 1020);
    buffer.push(frame(0.04, 2), 1040);
    expect(buffer.sample(1050)).toBe(true); // playback sim time = 30 ms
    expect(buffer.xpos[0]).toBeCloseTo(1.5, 5);
  });

  it("smooths jitter: a late frame does not shift playback", () => {
    const buffer = new PoseBuffer(20);
    buffer.push(frame(0.0, 0), 1000);
    buffer.push(frame(0.02, 1), 1060); // 40 ms late
    buffer.sample(1030); // sim 10 ms
    expect(buffer.xpos[0]).toBeCloseTo(0.5, 5);
  });

  it("holds the newest frame instead of extrapolating", () => {
    const buffer = new PoseBuffer(0);
    buffer.push(frame(0.0, 0), 1000);
    buffer.push(frame(0.02, 1), 1020);
    buffer.sample(2000);
    expect(buffer.xpos[0]).toBeCloseTo(1, 5);
  });

  it("normalises interpolated quaternions along the short arc", () => {
    const buffer = new PoseBuffer(0);
    const h = Math.SQRT1_2;
    buffer.push(frame(0.0, 0, [1, 0, 0, 0]), 1000);
    buffer.push(frame(0.02, 0, [-h, 0, 0, -h]), 1020); // same as [h,0,0,h]
    buffer.sample(1010);
    const q = Array.from(buffer.xquat);
    expect(Math.hypot(...q)).toBeCloseTo(1, 5);
    expect(q[0]).toBeGreaterThan(0.9); // short arc: 45° from identity
  });

  it("restarts on episode reset", () => {
    const buffer = new PoseBuffer(0);
    buffer.push(frame(5.0, 9), 1000);
    buffer.push(frame(0.02, 1), 1020);
    buffer.sample(1020);
    expect(buffer.xpos[0]).toBeCloseTo(1, 5);
  });
});

describe("decodeStateFrame", () => {
  it("decodes the WTS1 layout", () => {
    const buffer = new ArrayBuffer(STATE_HEADER_BYTES + 7 * 4);
    const view = new DataView(buffer);
    "WTS1".split("").forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
    view.setUint32(4, 42, true);
    view.setFloat64(8, 1.5, true);
    view.setUint16(24, 1, true);
    view.setUint16(26, 1, true);
    new Float32Array(buffer, STATE_HEADER_BYTES).set([1, 2, 3, 1, 0, 0, 0]);
    const decoded = decodeStateFrame(buffer);
    expect(decoded.tick).toBe(42);
    expect(decoded.simTime).toBe(1.5);
    expect(Array.from(decoded.xpos)).toEqual([1, 2, 3]);
    expect(() => decodeStateFrame(buffer.slice(0, 30))).toThrow();
  });
});
