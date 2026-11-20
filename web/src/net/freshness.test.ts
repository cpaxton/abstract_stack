import { describe, expect, it } from "vitest";
import { staleReason, STALE_POSE_MS, STALE_VIDEO_MS } from "./freshness";

const fresh = { poseAgeMs: 30, videoAgeMs: 40, videoAvailable: true };

describe("staleReason", () => {
  it("is null when the view being steered by is fresh", () => {
    expect(staleReason(fresh, "twin")).toBeNull();
    expect(staleReason(fresh, "camera")).toBeNull();
  });

  it("checks poses for the 3D view and video for the camera view", () => {
    const oldVideo = { ...fresh, videoAgeMs: STALE_VIDEO_MS + 120 };
    expect(staleReason(oldVideo, "twin")).toBeNull();
    expect(staleReason(oldVideo, "camera")).toBe("camera video is 420 ms old");
    const oldPose = { ...fresh, poseAgeMs: STALE_POSE_MS + 1 };
    expect(staleReason(oldPose, "twin")).toBe("robot pose is 251 ms old");
    expect(staleReason(oldPose, "camera")).toBeNull();
  });

  it("treats missing data as stale", () => {
    expect(staleReason({ ...fresh, videoAgeMs: null }, "camera")).toBe(
      "camera video is not arriving",
    );
    expect(staleReason({ ...fresh, poseAgeMs: Number.POSITIVE_INFINITY }, "twin")).toBe(
      "robot pose is not arriving",
    );
  });

  it("falls back to poses when the host has no video", () => {
    const noVideo = { poseAgeMs: 30, videoAgeMs: null, videoAvailable: false };
    expect(staleReason(noVideo, "camera")).toBeNull();
  });
});
