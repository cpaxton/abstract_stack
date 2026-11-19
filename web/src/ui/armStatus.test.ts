import { describe, expect, it } from "vitest";
import { armStatus } from "./armStatus";

const ik = (over: Partial<NonNullable<Parameters<typeof armStatus>[0]>> = {}) => ({
  reachable: true,
  ik_position_error_m: 0.004,
  ik_orientation_error_rad: 0.01,
  ...over,
});

describe("armStatus", () => {
  it("reports a reached pose with its residual", () => {
    expect(armStatus(ik())).toMatchObject({ tone: "ok", label: "IK 4 mm" });
  });

  it("warns when orientation was relaxed to reach the point", () => {
    const status = armStatus(
      ik({ reachable: false, orientation_relaxed: true, ik_orientation_error_rad: Math.PI / 4 }),
    );
    expect(status).toMatchObject({ tone: "warn", label: "Twisted 45°" });
  });

  it("names what blocked the arm, ahead of reach", () => {
    expect(armStatus(ik({ reachable: false, blocked_by: ["supply", "self"] }))).toMatchObject({
      tone: "bad",
      label: "Blocked: table, own body",
    });
    expect(armStatus(ik({ blocked_by: ["shelf_board_1"] })).label).toBe("Blocked: shelf board 1");
  });

  it("falls back to out of reach, and a dash with no data", () => {
    expect(armStatus(ik({ reachable: false })).label).toBe("Out of reach");
    expect(armStatus(null).label).toBe("—");
  });
});
