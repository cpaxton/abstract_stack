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

describe("adaptive PoseBuffer", () => {
  const feed = (buffer: PoseBuffer, n: number, jitter: (i: number) => number) => {
    for (let i = 0; i < n; i++) buffer.push(frame(i * 0.02, i), 1000 + i * 20 + jitter(i));
  };

  it("settles near one frame period on a clean link", () => {
    const buffer = new PoseBuffer(40, 16, true);
    feed(buffer, 600, () => 0);
    expect(buffer.delayMs).toBeGreaterThan(20);
    expect(buffer.delayMs).toBeLessThan(26);
  });

  it("grows to cover jitter and still interpolates between frames", () => {
    const buffer = new PoseBuffer(40, 16, true);
    feed(buffer, 300, (i) => (i % 5 === 0 ? 30 : 0)); // every 5th frame 30 ms late
    expect(buffer.delayMs).toBeGreaterThan(50);
    // Mid-way between arrivals playback still has a newer frame: no hold.
    buffer.sample(1000 + 299 * 20 + 10);
    expect(buffer.xpos[0]! % 1).not.toBe(0);
  });

  it("rises fast and falls slowly", () => {
    const buffer = new PoseBuffer(24, 16, true);
    feed(buffer, 10, (i) => (i === 5 ? 60 : 0));
    const peak = buffer.delayMs;
    expect(peak).toBeGreaterThan(30);
    buffer.push(frame(0.2, 10), 1000 + 10 * 20);
    expect(peak - buffer.delayMs).toBeLessThanOrEqual(0.25 + 1e-9);
  });
});

describe("compact WTS2 frames", () => {
  it("decodes int16 positions and quaternions", () => {
    const buffer = new ArrayBuffer(STATE_HEADER_BYTES + 7 * 2);
    const view = new DataView(buffer);
    "WTS2".split("").forEach((c, i) => view.setUint8(i, c.charCodeAt(0)));
    view.setUint32(4, 7, true);
    view.setFloat64(8, 2.0, true);
    view.setUint16(24, 1, true);
    new Int16Array(buffer, STATE_HEADER_BYTES).set([4096, -2048, 11059, 32767, 0, 0, 0]);
    const decoded = decodeStateFrame(buffer);
    expect(decoded.tick).toBe(7);
    expect(Array.from(decoded.xpos).map((v) => +v.toFixed(3))).toEqual([1, -0.5, 2.7]);
    expect(decoded.xquat[0]).toBeCloseTo(1, 6);
    expect(() => decodeStateFrame(buffer.slice(0, buffer.byteLength - 2))).toThrow();
  });
});
