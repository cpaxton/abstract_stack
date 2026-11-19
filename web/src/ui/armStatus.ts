import type { ArmStatus } from "../net/protocol";

const OBSTACLE_NAMES: Record<string, string> = {
  supply: "table",
  self: "own body",
};

/** One-line arm reach summary: on target, twisted to reach, blocked, or out of reach. */
export function armStatus(ik: ArmStatus["ik"]): {
  tone: "ok" | "warn" | "bad";
  label: string;
  title: string;
} {
  if (!ik) return { tone: "ok", label: "—", title: "" };
  const blocked = ik.blocked_by ?? [];
  if (blocked.length) {
    const names = blocked.map((n) => OBSTACLE_NAMES[n] ?? n.replace(/_/g, " ")).join(", ");
    return { tone: "bad", label: `Blocked: ${names}`, title: "Stopped short to avoid a collision" };
  }
  if (ik.reachable) {
    const mm = (ik.ik_position_error_m * 1000).toFixed(0);
    return { tone: "ok", label: `IK ${mm} mm`, title: "Pose reached" };
  }
  if (ik.orientation_relaxed) {
    const deg = Math.round((ik.ik_orientation_error_rad * 180) / Math.PI);
    return {
      tone: "warn",
      label: `Twisted ${deg}°`,
      title: "Hand is on target but rotated to reach it",
    };
  }
  return { tone: "bad", label: "Out of reach", title: "Closest reachable pose applied" };
}
