import type { JogInput } from "./jog";

/**
 * Shared touch-control state written by on-screen joysticks and sliders (pointer
 * events) and read once per frame by the jog loop. Values are in [-1, 1].
 */
export const touch = {
  /** Left stick: hand in the ground plane (x right, y away). */
  planar: { x: 0, y: 0 },
  /** Right stick: x = yaw, y = up/down. */
  vertical: { x: 0, y: 0 },
  grip: 0,
  fine: false,
};

const deadzone = (v: number) => (Math.abs(v) < 0.12 ? 0 : (v - Math.sign(v) * 0.12) / 0.88);

export function readTouch(out: JogInput): void {
  out.move.x += deadzone(touch.planar.x);
  out.move.y += deadzone(touch.planar.y);
  out.move.z += deadzone(touch.vertical.y);
  out.rotate.z += -deadzone(touch.vertical.x);
  out.grip += touch.grip;
  out.fine ||= touch.fine;
}
