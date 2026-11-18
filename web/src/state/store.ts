import { create } from "zustand";
import type { ConnectionState } from "../net/TeleopClient";
import type { HelloAck, Role, Side, Status } from "../net/protocol";

export type GizmoMode = "translate" | "rotate";
export type CameraPreset = "operator" | "side" | "top" | "front";
export type Quality = "high" | "low";
export type MainView = "twin" | "camera";
export type Eyes = "left" | "right" | "both";
/** follow: head tracks the active hand (or looks ahead while driving); point: holds a
 * clicked spot; free: manual angles from drags, arrows or sliders. */
export type HeadMode = "follow" | "point" | "free";

const initialQuality = (): Quality => {
  if (typeof window === "undefined") return "high";
  const forced = new URLSearchParams(window.location.search).get("quality");
  if (forced === "low" || forced === "high") return forced;
  return "high";
};

export interface Toast {
  id: number;
  tone: "info" | "warn" | "error" | "ok";
  text: string;
}

interface UiState {
  connection: ConnectionState;
  connectionDetail: string | null;
  role: Role | null;
  budgets: HelloAck["budgets"] | null;
  status: Status | null;
  rttMs: number | null;
  busyAction: string | null;
  activeArm: Side;
  /** Drive mode: movement input drives the base instead of a hand. */
  driveMode: boolean;
  gizmoMode: GizmoMode;
  camera: CameraPreset;
  cameraNonce: number;
  panelOpen: boolean;
  /** Render quality; drops to "low" automatically when the frame rate declines. */
  quality: Quality;
  /** What fills the screen; the other view shows as picture-in-picture. */
  mainView: MainView;
  eyes: Eyes;
  videoAvailable: boolean;
  pipVisible: boolean;
  headMode: HeadMode;
  /** Chassis-frame point for headMode "point". */
  lookPoint: [number, number, number] | null;
  /** Set when the host rejects the access token; shows the access screen. */
  authError: string | null;
  helpOpen: boolean;
  toasts: Toast[];
  /** Mirrors of intent values shown by sliders (updated at ≤ 15 Hz). */
  mirror: { grip: Record<Side, number>; lift: number; torso: number; yaw: number; pitch: number };
  set: (patch: Partial<UiState>) => void;
  toast: (tone: Toast["tone"], text: string) => void;
  dismiss: (id: number) => void;
  setCamera: (camera: CameraPreset) => void;
  /** Cycle the controlled part: left hand → right hand → drive → left hand. */
  cycleControl: () => void;
}

let toastId = 0;

export const useUi = create<UiState>((set, get) => ({
  connection: "idle",
  connectionDetail: null,
  role: null,
  budgets: null,
  status: null,
  rttMs: null,
  busyAction: null,
  activeArm: "right",
  driveMode: false,
  gizmoMode: "translate",
  camera: "operator",
  cameraNonce: 0,
  // Open by default only on wide pointer-driven screens; on phones/tablets it would
  // cover the joysticks and picture-in-picture.
  panelOpen:
    typeof window === "undefined"
      ? true
      : window.innerWidth >= 900 && !window.matchMedia?.("(pointer: coarse)").matches,
  helpOpen: false,
  quality: initialQuality(),
  mainView: "twin",
  eyes: "left",
  videoAvailable: false,
  pipVisible: true,
  authError: null,
  headMode: "follow",
  lookPoint: null,
  toasts: [],
  mirror: { grip: { left: 0, right: 0 }, lift: 0.3175, torso: 0, yaw: 0, pitch: 0 },
  set: (patch) => set(patch),
  toast: (tone, text) => {
    const id = ++toastId;
    // Collapse repeats of the same message.
    const toasts = get().toasts.filter((t) => t.text !== text);
    set({ toasts: [...toasts.slice(-3), { id, tone, text }] });
    window.setTimeout(() => get().dismiss(id), tone === "error" ? 6000 : 3500);
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  setCamera: (camera) => set({ camera, cameraNonce: get().cameraNonce + 1 }),
  cycleControl: () => {
    const { driveMode, activeArm } = get();
    if (driveMode) set({ driveMode: false, activeArm: "left" });
    else if (activeArm === "left") set({ activeArm: "right" });
    else set({ driveMode: true });
  },
}));
