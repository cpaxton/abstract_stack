import { staleReason } from "./net/freshness";
import { useEffect, useMemo, useState } from "react";
import { recentreHead } from "./input/headLook";
import { installKeyboard } from "./input/keyboard";
import { TeleopClient } from "./net/TeleopClient";
import type { Role } from "./net/protocol";
import { loadRobotModel, type RobotModel } from "./scene/robotModel";
import { TwinView } from "./scene/TwinView";
import { buildCommand, intent, rebaseFromStatus } from "./state/intent";
import { useUi, type CameraPreset } from "./state/store";
import { runAction, togglePrimary } from "./ui/actions";
import { storeToken, takeToken } from "./state/token";
import { AccessScreen } from "./ui/AccessScreen";
import { BodyRails } from "./ui/BodyRails";
import { Guide } from "./ui/Guide";
import { HelpOverlay, Hud, KeyHints, StatusBanner, Toasts } from "./ui/Overlays";
import { syncMirror } from "./state/mirror";
import { SidePanel } from "./ui/SidePanel";
import { TopBar } from "./ui/TopBar";
import { TouchControls } from "./ui/TouchControls";
import { ViewLayers } from "./ui/ViewLayers";

const CAMERA_KEYS: Record<string, CameraPreset> = {
  Digit1: "operator",
  Digit2: "side",
  Digit3: "top",
  Digit4: "front",
};

// Freshness watchdog cadence and how long staleness must persist to pause (ms).
const WATCHDOG_MS = 50;
const STALE_PERSIST_MS = 150;

export function App() {
  const client = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const role: Role = params.get("role") === "spectator" ? "spectator" : "operator";
    return new TeleopClient({
      origin: window.location.origin,
      token: takeToken(),
      role,
      commandProvider: buildCommand,
    });
  }, []);
  const [loaded, setLoaded] = useState<{ hash: string; model?: RobotModel; error?: string }>();
  const [modelHash, setModelHash] = useState<string | null>(null);
  const state = useUi((s) => s.status?.state ?? "disconnected");

  // Read-only inspection hook for end-to-end tests and field debugging.
  useEffect(() => {
    (window as unknown as { __teleop: object }).__teleop = {
      status: () => useUi.getState().status,
      intent: () => intent,
      poseAgeMs: () => client.poses.ageMs(),
      videoPairs: () => client.stereo.decodedPairs,
      latency: () => client.latencySnapshot(),
    };
  }, [client]);

  // Connection lifecycle and event wiring.
  useEffect(() => {
    const ui = useUi.getState();
    const off = [
      client.on("connection", (connection, detail) =>
        ui.set({ connection, connectionDetail: detail ?? null }),
      ),
      client.on("hello", (ack) => {
        ui.set({ role: ack.role, budgets: ack.budgets, videoAvailable: ack.video !== null });
        setModelHash(ack.model_hash);
        if (ack.role === "spectator")
          ui.toast("info", "Another operator is in control — viewing only");
      }),
      client.on("status", (status, role) => {
        const previous = useUi.getState().status;
        if (rebaseFromStatus(status)) syncMirror();
        ui.set({ status, role });
        if (previous && previous.state !== status.state) {
          if (status.state === "paused" && previous.state === "engaged") {
            const own = useUi.getState().autoPauseReason;
            ui.toast("warn", `Paused: ${own ?? status.reason}`);
            ui.set({ autoPauseReason: null });
          }
          if (status.state === "fault") ui.toast("error", status.reason);
        }
      }),
      client.on("rtt", (ms) => {
        const prev = useUi.getState().rttMs;
        ui.set({ rttMs: prev === null ? ms : prev * 0.7 + ms * 0.3 });
      }),
      client.on("error", (message) => ui.toast("error", message)),
      client.on("auth", (message) => {
        storeToken(null);
        ui.set({ authError: message });
      }),
    ];
    client.start();
    // Freshness watchdog and latency summary. Steering from a stale view is unsafe
    // even though commands still flow (so the host's input timeout never fires):
    // pause from the client and say why.
    let lastLatency = 0;
    let lastCheck = performance.now();
    let staleSince: number | null = null;
    const watchdog = window.setInterval(() => {
      const now = performance.now();
      // If the page itself was just blocked (a long frame on a slow device), frames
      // that arrived meanwhile may still be queued behind this timer: discount the
      // stall from the data age. A truly frozen stream keeps ageing past it.
      const lag = Math.max(0, now - lastCheck - WATCHDOG_MS);
      lastCheck = now;
      const fresh = client.freshness(now);
      const state = useUi.getState();
      const reason = staleReason(
        {
          ...fresh,
          poseAgeMs: fresh.poseAgeMs - lag,
          videoAgeMs: fresh.videoAgeMs === null ? null : fresh.videoAgeMs - lag,
        },
        state.mainView,
      );
      staleSince = reason ? (staleSince ?? now) : null;
      // A momentary gap is not a frozen view: require it to persist.
      const stale = reason && now - staleSince! >= STALE_PERSIST_MS ? reason : null;
      if (stale !== state.stale) ui.set({ stale });
      if (
        stale &&
        state.role === "operator" &&
        state.status?.state === "engaged" &&
        !state.autoPauseReason
      ) {
        ui.set({ autoPauseReason: stale });
        void client.sendControl("pause").catch(() => ui.set({ autoPauseReason: null }));
      }
      if (now - lastLatency >= 250) {
        lastLatency = now;
        ui.set({ latency: client.latencySnapshot() });
      }
    }, WATCHDOG_MS);
    return () => {
      window.clearInterval(watchdog);
      off.forEach((fn) => fn());
      client.stop();
    };
  }, [client]);

  // Load (or reload) the visual model when the host's model hash changes.
  useEffect(() => {
    if (!modelHash) return;
    const abort = new AbortController();
    let result: RobotModel | undefined;
    // Mesh detail is chosen once per load: full resolution unless low quality was requested.
    const detail = useUi.getState().quality === "low" ? "low" : "full";
    loadRobotModel(window.location.origin, client.token, abort.signal, detail)
      .then((model) => {
        result = model;
        setLoaded({ hash: modelHash, model });
      })
      .catch((error: unknown) => {
        if (!abort.signal.aborted) {
          setLoaded({
            hash: modelHash,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      });
    return () => {
      abort.abort();
      result?.dispose();
    };
  }, [client, modelHash]);
  const current = loaded && loaded.hash === modelHash ? loaded : undefined;
  const model = current?.model ?? null;
  const modelError = current?.error ?? null;

  // Keep slider mirrors in step with keyboard/touch jogging (≤ 10 Hz React updates).
  useEffect(() => {
    const timer = window.setInterval(() => {
      const m = useUi.getState().mirror;
      if (
        m.lift !== intent.lift ||
        m.torso !== intent.torso ||
        m.yaw !== intent.head.yaw ||
        m.pitch !== intent.head.pitch ||
        m.grip.left !== intent.arms.left.grip ||
        m.grip.right !== intent.arms.right.grip
      ) {
        syncMirror();
      }
    }, 100);
    return () => window.clearInterval(timer);
  }, []);

  // Global shortcuts. Pause is always honoured; motion keys are read per frame.
  useEffect(
    () =>
      installKeyboard((code, event) => {
        const ui = useUi.getState();
        const engaged = ui.status?.state === "engaged";
        switch (code) {
          case "Escape":
            if (ui.guideOpen) ui.set({ guideOpen: false });
            else if (ui.helpOpen) ui.set({ helpOpen: false });
            else if (engaged) void runAction(client, "pause");
            return true;
          case "Space":
            if (engaged) void runAction(client, "pause");
            return true;
          case "Enter":
          case "NumpadEnter":
            if (!engaged) togglePrimary(client);
            return true;
          case "Tab":
            ui.cycleControl();
            return true;
          case "Home":
            recentreHead();
            syncMirror();
            return true;
          case "KeyF":
            ui.set({ headMode: "follow" });
            ui.toast("info", "Head follows the hand");
            return true;
          case "KeyG": {
            if (ui.driveMode) return true;
            const arm = intent.arms[ui.activeArm];
            arm.grip = arm.grip > 0.5 ? 0 : 1;
            syncMirror();
            return true;
          }
          case "KeyT":
            ui.set({ gizmoMode: "translate" });
            return true;
          case "KeyY":
            ui.set({ gizmoMode: "rotate" });
            return true;
          case "KeyR":
            void runAction(client, "ready");
            return true;
          case "KeyV":
            if (ui.videoAvailable) ui.set({ mainView: ui.mainView === "twin" ? "camera" : "twin" });
            return true;
          case "KeyB": {
            const order = ["left", "both", "right"] as const;
            ui.set({ eyes: order[(order.indexOf(ui.eyes) + 1) % order.length]! });
            return true;
          }
          case "KeyP":
            ui.set({ panelOpen: !ui.panelOpen });
            return true;
          case "Slash":
            if (event.shiftKey) {
              ui.set({ helpOpen: !ui.helpOpen });
              return true;
            }
            return false;
          default:
            if (code in CAMERA_KEYS) {
              ui.setCamera(CAMERA_KEYS[code]!);
              return true;
            }
            return false;
        }
      }),
    [client],
  );

  // Leaving the tab pauses immediately (the host would also pause on input timeout).
  useEffect(() => {
    const onHide = () => {
      if (document.hidden && useUi.getState().status?.state === "engaged")
        void runAction(client, "pause");
    };
    document.addEventListener("visibilitychange", onHide);
    return () => document.removeEventListener("visibilitychange", onHide);
  }, [client]);

  return (
    <div className={`app app-${state}`}>
      <ViewLayers client={client} twin={<TwinView model={model} client={client} />} />
      <div className="engaged-frame" aria-hidden />
      {!model && (
        <div className="loading">
          {modelError ? (
            <>
              <strong>Could not load the robot model</strong>
              <span className="muted">{modelError}</span>
            </>
          ) : (
            <>
              <div className="spinner" />
              <span className="muted">Loading robot model…</span>
            </>
          )}
        </div>
      )}
      <TopBar client={client} />
      <StatusBanner client={client} />
      <Toasts />
      <SidePanel />
      <Hud client={client} />
      <KeyHints />
      <TouchControls />
      <BodyRails />
      <HelpOverlay />
      <Guide />
      <AccessScreen />
    </div>
  );
}
