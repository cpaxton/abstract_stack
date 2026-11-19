import { intent } from "./intent";
import { useUi } from "./store";

/** Copy current intent values into the UI store so sliders and rails re-render. */
export function syncMirror() {
  const set = useUi.getState().set;
  set({
    mirror: {
      grip: { left: intent.arms.left.grip, right: intent.arms.right.grip },
      lift: intent.lift,
      torso: intent.torso,
      yaw: intent.head.yaw,
      pitch: intent.head.pitch,
    },
  });
}
