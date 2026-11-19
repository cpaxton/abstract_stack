import { beforeEach, describe, expect, it } from "vitest";
import { intent } from "../state/intent";
import { lookAt, panBy } from "./headLook";

const geometry = { verticalFovDeg: 83.3, aspect: 4 / 3 };

describe("head look", () => {
  beforeEach(() => {
    intent.head.yaw = 0;
    intent.head.pitch = 0;
    intent.ranges = null;
  });

  it("clicking the image centre keeps the head still", () => {
    lookAt(0.5, 0.5, geometry);
    expect(intent.head.yaw).toBeCloseTo(0);
    expect(intent.head.pitch).toBeCloseTo(0);
  });

  it("clicking the right edge turns right by half the horizontal FOV", () => {
    lookAt(1, 0.5, geometry);
    const halfH = Math.atan(Math.tan((83.3 * Math.PI) / 360) * (4 / 3));
    expect(intent.head.yaw).toBeCloseTo(-halfH); // right = negative yaw
  });

  it("clicking low looks down; dragging down looks up", () => {
    lookAt(0.5, 0.75, geometry);
    expect(intent.head.pitch).toBeGreaterThan(0);
    const before = intent.head.pitch;
    panBy(0, 0.1, geometry);
    expect(intent.head.pitch).toBeLessThan(before);
  });

  it("dragging the image right turns the head left", () => {
    panBy(0.25, 0, geometry);
    expect(intent.head.yaw).toBeGreaterThan(0);
  });
});
