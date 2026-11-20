/**
 * Host↔browser clock offset from ping/pong round trips (NTP-style).
 *
 * The host stamps frames with `time.monotonic()` seconds; the browser uses
 * `performance.now()` milliseconds. For a ping sent at t0 and answered at t2 with
 * host time t1, the host clock reads t1 at roughly local (t0 + t2) / 2, with an
 * error bounded by half the round trip. The sample with the smallest round trip
 * among the recent window is the tightest bound, so it sets the offset.
 */
export class ClockSync {
  private samples: { rtt: number; offset: number; at: number }[] = [];
  /** host_ms − local_ms for the best recent sample; null until the first pong. */
  offsetMs: number | null = null;
  /** Round trip of the latest pong (ms). */
  rttMs: number | null = null;
  /** Round trip of the sample that set the offset (ms): the offset is within ±half. */
  bestRttMs: number | null = null;
  readonly window: number;

  constructor(window = 16) {
    this.window = window;
  }

  clear(): void {
    this.samples = [];
    this.offsetMs = this.rttMs = this.bestRttMs = null;
  }

  /** Record one round trip: local send `t0`, host reply `hostSeconds`, local receive `t2`. */
  add(t0: number, hostSeconds: number, t2: number): void {
    const rtt = t2 - t0;
    if (!(rtt >= 0) || !Number.isFinite(hostSeconds)) return;
    this.rttMs = rtt;
    this.samples.push({ rtt, offset: hostSeconds * 1000 - (t0 + t2) / 2, at: t2 });
    if (this.samples.length > this.window) this.samples.shift();
    let best = this.samples[0]!;
    for (const sample of this.samples) if (sample.rtt < best.rtt) best = sample;
    this.offsetMs = best.offset;
    this.bestRttMs = best.rtt;
  }

  get synced(): boolean {
    return this.offsetMs !== null;
  }

  /** Local `performance.now()` time (ms) at which the host clock read `hostSeconds`. */
  toLocalMs(hostSeconds: number): number {
    return hostSeconds * 1000 - (this.offsetMs ?? Number.NaN);
  }

  /** Age (ms) at local time `now` of an event the host stamped at `hostSeconds`. */
  ageMs(hostSeconds: number, now: number = performance.now()): number {
    return now - this.toLocalMs(hostSeconds);
  }
}
