import { describe, expect, it } from "vitest";
import { ClockSync } from "./clock";

// Host clock = local clock + 5000 ms, with asymmetric network delays.
const HOST_AHEAD_MS = 5000;
function roundTrip(clock: ClockSync, t0: number, up: number, down: number) {
  const hostSeconds = (t0 + up + HOST_AHEAD_MS) / 1000;
  clock.add(t0, hostSeconds, t0 + up + down);
}

describe("ClockSync", () => {
  it("recovers the offset exactly for a symmetric round trip", () => {
    const clock = new ClockSync();
    roundTrip(clock, 1000, 4, 4);
    expect(clock.offsetMs).toBeCloseTo(HOST_AHEAD_MS, 6);
    expect(clock.toLocalMs((2000 + HOST_AHEAD_MS) / 1000)).toBeCloseTo(2000, 6);
    expect(clock.ageMs((2000 + HOST_AHEAD_MS) / 1000, 2030)).toBeCloseTo(30, 6);
  });

  it("keeps the tightest (minimum round trip) sample", () => {
    const clock = new ClockSync();
    roundTrip(clock, 1000, 60, 2); // congested: offset error 29 ms
    roundTrip(clock, 2000, 3, 3); // clean
    roundTrip(clock, 3000, 2, 80);
    expect(clock.bestRttMs).toBe(6);
    expect(clock.rttMs).toBe(82);
    expect(clock.offsetMs).toBeCloseTo(HOST_AHEAD_MS, 6);
  });

  it("forgets old samples after the window", () => {
    const clock = new ClockSync(2);
    roundTrip(clock, 0, 1, 1);
    roundTrip(clock, 100, 10, 30);
    roundTrip(clock, 200, 10, 30);
    expect(clock.bestRttMs).toBe(40); // the 2 ms sample aged out
  });

  it("is unsynced until a valid sample arrives", () => {
    const clock = new ClockSync();
    expect(clock.synced).toBe(false);
    clock.add(10, 1, 5); // negative round trip: ignored
    expect(clock.synced).toBe(false);
    expect(Number.isNaN(clock.ageMs(1, 0))).toBe(true);
  });
});
