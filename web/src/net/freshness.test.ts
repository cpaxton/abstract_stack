import { describe, expect, it } from "vitest";
import { discountStall, staleReason, STALE_POSE_MS, STALE_VIDEO_MS } from "./freshness";

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

  it("treats a view that has not drawn yet as stale", () => {
    expect(staleReason({ ...fresh, videoAgeMs: null }, "camera")).toBe(
      "camera video is not shown yet",
    );
    expect(staleReason({ ...fresh, poseAgeMs: Number.POSITIVE_INFINITY }, "twin")).toBe(
      "robot pose is not shown yet",
    );
  });

  it("falls back to poses when the host has no video", () => {
    const noVideo = { poseAgeMs: 30, videoAgeMs: null, videoAvailable: false };
    expect(staleReason(noVideo, "camera")).toBeNull();
  });
});

describe("discountStall", () => {
  it("subtracts a page stall from both ages and keeps missing data missing", () => {
    const stalled = discountStall({ poseAgeMs: 400, videoAgeMs: 500, videoAvailable: true }, 200);
    expect(stalled).toEqual({ poseAgeMs: 200, videoAgeMs: 300, videoAvailable: true });
    const missing = discountStall(
      { poseAgeMs: Number.POSITIVE_INFINITY, videoAgeMs: null, videoAvailable: true },
      200,
    );
    expect(missing.poseAgeMs).toBe(Number.POSITIVE_INFINITY);
    expect(missing.videoAgeMs).toBeNull();
  });
});
