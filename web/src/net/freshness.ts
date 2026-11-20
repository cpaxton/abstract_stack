import type { MainView } from "../state/store";

/** Data older than this cannot be steered by (ms, host capture → now). */
export const STALE_POSE_MS = 250;
export const STALE_VIDEO_MS = 300;

export interface Freshness {
  /** Age of the robot pose the 3D view last drew (ms); Infinity before it draws one. */
  poseAgeMs: number;
  /** Age of the camera frame last drawn (ms); null before a frame is drawn. */
  videoAgeMs: number | null;
  /** Whether the host offers video at all (`--no-video` hosts do not). */
  videoAvailable: boolean;
}

/**
 * Discount a page stall from the ages: frames that arrived while the page was
 * blocked (a long frame on a slow device) may still be queued, unprocessed.
 */
export function discountStall(freshness: Freshness, stallMs: number): Freshness {
  return {
    ...freshness,
    poseAgeMs: freshness.poseAgeMs - stallMs,
    videoAgeMs: freshness.videoAgeMs === null ? null : freshness.videoAgeMs - stallMs,
  };
}

/**
 * Why the operator should not be steering right now, or null when the view they
 * are steering by is fresh. The 3D view needs fresh poses; the head-camera view
 * needs fresh video (when the host has video). Steering from an old picture is
 * the failure this guards against: commands would still flow, so the host's
 * input timeout would never fire.
 */
export function staleReason(freshness: Freshness, mainView: MainView): string | null {
  const age = (ms: number) => (Number.isFinite(ms) ? `${Math.round(ms)} ms old` : "not shown yet");
  if (mainView === "camera" && freshness.videoAvailable) {
    const video = freshness.videoAgeMs ?? Number.POSITIVE_INFINITY;
    if (!(video <= STALE_VIDEO_MS)) return `camera video is ${age(video)}`;
    return null;
  }
  if (!(freshness.poseAgeMs <= STALE_POSE_MS)) return `robot pose is ${age(freshness.poseAgeMs)}`;
  return null;
}
