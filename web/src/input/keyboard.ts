import type { JogInput } from "./jog";

/** Physical key codes (layout independent) currently held. */
const held = new Set<string>();

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) &&
  !(target instanceof HTMLInputElement && target.type === "range");

export type ShortcutHandler = (code: string, event: KeyboardEvent) => boolean;

/** Install global listeners; `onShortcut` returns true when it consumed a key press. */
export function installKeyboard(onShortcut: ShortcutHandler): () => void {
  const down = (event: KeyboardEvent) => {
    if (isTyping(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
    if (!event.repeat && onShortcut(event.code, event)) {
      event.preventDefault();
      return;
    }
    if (JOG_KEYS.has(event.code)) {
      held.add(event.code);
      event.preventDefault();
    }
  };
  const up = (event: KeyboardEvent) => held.delete(event.code);
  // Losing focus must release every key so the hand never keeps moving.
  const clear = () => held.clear();
  window.addEventListener("keydown", down);
  window.addEventListener("keyup", up);
  window.addEventListener("blur", clear);
  document.addEventListener("visibilitychange", clear);
  return () => {
    window.removeEventListener("keydown", down);
    window.removeEventListener("keyup", up);
    window.removeEventListener("blur", clear);
    document.removeEventListener("visibilitychange", clear);
    held.clear();
  };
}

const axis = (pos: string, neg: string) => (held.has(pos) ? 1 : 0) - (held.has(neg) ? 1 : 0);

export const KEYMAP = {
  move: { right: "KeyD", left: "KeyA", forward: "KeyW", back: "KeyS", up: "KeyE", down: "KeyQ" },
  rotate: {
    pitchUp: "KeyI",
    pitchDown: "KeyK",
    yawLeft: "KeyJ",
    yawRight: "KeyL",
    rollLeft: "KeyU",
    rollRight: "KeyO",
  },
  grip: { close: "BracketRight", open: "BracketLeft" },
  head: { left: "ArrowLeft", right: "ArrowRight", up: "ArrowUp", down: "ArrowDown" },
  lift: { up: "PageUp", down: "PageDown" },
  fine: ["ShiftLeft", "ShiftRight"],
} as const;

const JOG_KEYS = new Set<string>([
  ...Object.values(KEYMAP.move),
  ...Object.values(KEYMAP.rotate),
  ...Object.values(KEYMAP.grip),
  ...Object.values(KEYMAP.head),
  ...Object.values(KEYMAP.lift),
  ...KEYMAP.fine,
]);

/** Drive-mode turn from the keyboard: Q turns left (+), E turns right. */
export function keyboardTurn(): number {
  return axis(KEYMAP.move.down, KEYMAP.move.up);
}

/** Accumulate held keys into `out` (adds to any existing touch input). */
export function readKeyboard(out: JogInput): void {
  const m = KEYMAP.move;
  out.move.x += axis(m.right, m.left);
  out.move.y += axis(m.forward, m.back);
  out.move.z += axis(m.up, m.down);
  const r = KEYMAP.rotate;
  out.rotate.x += axis(r.pitchDown, r.pitchUp);
  out.rotate.y += axis(r.rollRight, r.rollLeft);
  out.rotate.z += axis(r.yawLeft, r.yawRight);
  out.grip += axis(KEYMAP.grip.close, KEYMAP.grip.open);
  // MuJoCo neck yaw is positive to the robot's left; pitch positive looks down.
  out.headYaw += axis(KEYMAP.head.left, KEYMAP.head.right);
  out.headPitch += axis(KEYMAP.head.down, KEYMAP.head.up);
  out.lift += axis(KEYMAP.lift.up, KEYMAP.lift.down);
  out.fine ||= KEYMAP.fine.some((k) => held.has(k));
}
