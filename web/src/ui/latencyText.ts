import type { LatencySnapshot, LatencySummary } from "../net/latency";
import type { MainView } from "../state/store";

/** Latency of the view being steered by: capture → shown for video, physics → shown for 3D. */
export function viewLatency(
  latency: LatencySnapshot | null,
  mainView: MainView,
  rttMs: number | null,
): { label: string; ms: number | null } {
  if (mainView === "camera" && latency?.videoShown) {
    return { label: "Video", ms: latency.videoShown.p50 };
  }
  if (latency?.poseShown) return { label: "3D", ms: latency.poseShown.p50 };
  return { label: "RTT", ms: rttMs };
}

const ms = (summary: LatencySummary | null) =>
  summary ? `${Math.round(summary.p50)} ms (p95 ${Math.round(summary.p95)})` : "—";

/** One line per pipeline stage, for tooltips and diagnostics. */
export function latencyLines(latency: LatencySnapshot | null): string[] {
  if (!latency) return [];
  return [
    `Round trip: ${latency.rttMs === null ? "—" : `${latency.rttMs.toFixed(1)} ms`}` +
      (latency.clockErrorMs === null ? "" : ` (clock ±${latency.clockErrorMs.toFixed(1)} ms)`),
    `Command → robot: ${ms(latency.uplink)}`,
    `3D view: ${ms(latency.poseShown)}, buffer ${Math.round(latency.bufferMs)} ms`,
    `Camera: ${ms(latency.videoShown)}`,
  ];
}
