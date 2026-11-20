import type { TeleopClient } from "../net/TeleopClient";
import type { ControlAction } from "../net/protocol";
import { useUi } from "../state/store";

const LABEL: Record<ControlAction, string> = {
  engage: "Engaged",
  pause: "Paused",
  reset: "Episode reset",
  ready: "Moving to ready pose",
};

export function haptic(pattern: number | number[]): void {
  try {
    navigator.vibrate?.(pattern);
  } catch {
    // Vibration is optional (unsupported on iOS Safari / desktop).
  }
}

/** Send a control action with busy state, toasts and haptic feedback. */
export async function runAction(client: TeleopClient, action: ControlAction): Promise<boolean> {
  const ui = useUi.getState();
  if (ui.busyAction && action !== "pause") return false;
  if (action === "engage") {
    // The watchdog's verdict (persistent staleness), not one instantaneous sample.
    const reason = ui.stale;
    if (reason) {
      ui.toast("warn", `Can't engage: ${reason}`);
      haptic([60, 40, 60]);
      return false;
    }
  }
  ui.set({ busyAction: action });
  try {
    const ack = await client.sendControl(action);
    if (ack.accepted) {
      if (action !== "pause") ui.toast("ok", LABEL[action]);
      haptic(action === "engage" ? [20, 40, 20] : 15);
      return true;
    }
    ui.toast(
      "warn",
      `${action[0]!.toUpperCase()}${action.slice(1)} refused: ${ack.error ?? "unknown"}`,
    );
    haptic([60, 40, 60]);
    return false;
  } catch (error) {
    ui.toast("error", error instanceof Error ? error.message : String(error));
    return false;
  } finally {
    useUi.getState().set({ busyAction: null });
  }
}

/** Primary safety toggle: pause when engaged, otherwise engage. */
export function togglePrimary(client: TeleopClient): void {
  const state = useUi.getState().status?.state;
  void runAction(client, state === "engaged" ? "pause" : "engage");
}
