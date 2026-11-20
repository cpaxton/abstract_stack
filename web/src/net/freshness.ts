import type { MainView } from "../state/store";

/** Data older than this cannot be steered by (ms, host capture → now). */
export const STALE_POSE_MS = 250;
export const STALE_VIDEO_MS = 300;

export interface Freshness {
  /** Age of the newest robot pose (ms); Infinity before the first frame. */
  poseAgeMs: number;
  /** Age of the newest camera frame (ms); null when no video is subscribed. */
  videoAgeMs: number | null;
  /** Whether the host offers video at all (`--no-video` hosts do not). */
  videoAvailable: boolean;
}

/**
 * Why the operator should not be steering right now, or null when the view they
 * are steering by is fresh. The 3D view needs fresh poses; the head-camera view
 * needs fresh video (when the host has video). Steering from an old picture is
 * the failure this guards against: commands would still flow, so the host's
 * input timeout would never fire.
 */
export function staleReason(freshness: Freshness, mainView: MainView): string | null {
  const age = (ms: number) => (Number.isFinite(ms) ? `${Math.round(ms)} ms old` : "not arriving");
  if (mainView === "camera" && freshness.videoAvailable) {
    const video = freshness.videoAgeMs ?? Number.POSITIVE_INFINITY;
    if (!(video <= STALE_VIDEO_MS)) return `camera video is ${age(video)}`;
    return null;
  }
  if (!(freshness.poseAgeMs <= STALE_POSE_MS)) return `robot pose is ${age(freshness.poseAgeMs)}`;
  return null;
}
