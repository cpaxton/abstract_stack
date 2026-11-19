import { clampChannels, intent, takeManualHead } from "../state/intent";

/**
 * Head aiming from the head-camera image. Uses the symmetric pinhole model of the
 * stream (vertical FOV, aspect) and MuJoCo's neck conventions: yaw positive turns to
 * the robot's left, pitch positive looks down. A small-angle model that ignores the
 * camera's offset from the neck axes; repeated clicks converge on the point.
 */
export interface LookGeometry {
  verticalFovDeg: number;
  aspect: number; // width / height
}

const rad = (deg: number) => (deg * Math.PI) / 180;

/** Turn the head toward a clicked point given in normalised image coords [0,1]. */
export function lookAt(u: number, v: number, geometry: LookGeometry): void {
  takeManualHead();
  const tanV = Math.tan(rad(geometry.verticalFovDeg) / 2);
  const tanH = tanV * geometry.aspect;
  intent.head.yaw -= Math.atan((2 * u - 1) * tanH);
  intent.head.pitch += Math.atan((2 * v - 1) * tanV);
  clampChannels();
}

/** Pan the view by a drag of (du, dv) image fractions; the image follows the finger. */
export function panBy(du: number, dv: number, geometry: LookGeometry): void {
  takeManualHead();
  const vfov = rad(geometry.verticalFovDeg);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * geometry.aspect);
  intent.head.yaw += du * hfov;
  intent.head.pitch -= dv * vfov;
  clampChannels();
}

export function recentreHead(): void {
  takeManualHead();
  intent.head.yaw = 0;
  intent.head.pitch = 0;
}
