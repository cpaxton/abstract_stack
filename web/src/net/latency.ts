/** Fixed-size window of recent samples with percentile summaries (no allocation per push). */
export class Rolling {
  private values: Float64Array;
  private count = 0;
  private next = 0;
  last = Number.NaN;

  constructor(size = 120) {
    this.values = new Float64Array(size);
  }

  push(value: number): void {
    if (!Number.isFinite(value)) return;
    this.values[this.next] = value;
    this.next = (this.next + 1) % this.values.length;
    this.count = Math.min(this.count + 1, this.values.length);
    this.last = value;
  }

  clear(): void {
    this.count = this.next = 0;
    this.last = Number.NaN;
  }

  get size(): number {
    return this.count;
  }

  /** Percentile `q` in [0, 1] of the window, or null when empty. */
  quantile(q: number): number | null {
    if (!this.count) return null;
    const sorted = Array.from(this.values.subarray(0, this.count)).sort((a, b) => a - b);
    return sorted[Math.min(this.count - 1, Math.round(q * (this.count - 1)))]!;
  }
}

export interface LatencySummary {
  p50: number;
  p95: number;
}

export interface LatencySnapshot {
  /** Control-socket round trip (ms) and the clock-offset uncertainty (± half the best round trip). */
  rttMs: number | null;
  clockErrorMs: number | null;
  /** Host publish → browser arrival of state frames. */
  poseNetwork: LatencySummary | null;
  /** Host physics → pose drawn in the 3D view (network + jitter buffer). */
  poseShown: LatencySummary | null;
  /** Camera capture → message arrival (host render + encode + network). */
  videoArrival: LatencySummary | null;
  /** Arrival → both eyes decoded (browser JPEG decode, incl. waiting behind a decode). */
  videoDecode: LatencySummary | null;
  /** Camera capture → frame drawn (render, encode, network, decode, next display frame). */
  videoShown: LatencySummary | null;
  /** Browser send → host applies the command to the arms. */
  uplink: LatencySummary | null;
  /** Current jitter-buffer playback delay (ms). */
  bufferMs: number;
}

const summary = (rolling: Rolling): LatencySummary | null =>
  rolling.size ? { p50: rolling.quantile(0.5)!, p95: rolling.quantile(0.95)! } : null;

/** Per-stage latency samples, all on the browser clock via `ClockSync`. */
export class LatencyMonitor {
  readonly poseNetwork = new Rolling(100); // ~2 s of 50 Hz frames
  readonly poseShown = new Rolling(120); // ~2 s of display frames
  readonly videoArrival = new Rolling(40);
  readonly videoDecode = new Rolling(40);
  readonly videoShown = new Rolling(40);
  readonly uplink = new Rolling(20); // 10 Hz status echoes

  clear(): void {
    this.poseNetwork.clear();
    this.poseShown.clear();
    this.videoArrival.clear();
    this.videoDecode.clear();
    this.videoShown.clear();
    this.uplink.clear();
  }

  snapshot(rttMs: number | null, bestRttMs: number | null, bufferMs: number): LatencySnapshot {
    return {
      rttMs,
      clockErrorMs: bestRttMs === null ? null : bestRttMs / 2,
      poseNetwork: summary(this.poseNetwork),
      poseShown: summary(this.poseShown),
      videoArrival: summary(this.videoArrival),
      videoDecode: summary(this.videoDecode),
      videoShown: summary(this.videoShown),
      uplink: summary(this.uplink),
      bufferMs,
    };
  }
}

/**
 * Queueing delay on the host → browser path, from one-way pose delays (synced
 * clock). Each window's median delay minus the lowest window median of the last
 * `history` windows is time spent waiting in buffers; the host's link controller
 * trades video bitrate and pose rate for it.
 */
export class QueueProbe {
  private samples: number[] = [];
  private medians: number[] = [];
  readonly history: number;

  constructor(history = 40) {
    this.history = history;
  }

  clear(): void {
    this.samples = [];
    this.medians = [];
  }

  push(delayMs: number): void {
    if (Number.isFinite(delayMs)) this.samples.push(delayMs);
  }

  /** Close the current window; returns its queueing delay (ms) or null if it was empty. */
  window(): number | null {
    if (!this.samples.length) return null;
    const sorted = this.samples.sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)]!;
    this.samples = [];
    this.medians.push(median);
    if (this.medians.length > this.history) this.medians.shift();
    return Math.max(0, median - Math.min(...this.medians));
  }
}
