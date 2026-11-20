import { useEffect, useRef, useState } from "react";
import type { TeleopClient } from "../net/TeleopClient";
import type { TeleopState } from "../net/protocol";
import { useUi } from "../state/store";
import { runAction, togglePrimary } from "./actions";
import { latencyLines, viewLatency } from "./latencyText";
import { Icon } from "./Icon";

const STATE_LABEL: Record<TeleopState, string> = {
  disconnected: "Offline",
  ready: "Ready",
  engaged: "Engaged",
  paused: "Paused",
  fault: "Fault",
};

function ConnectionChip() {
  const connection = useUi((s) => s.connection);
  const detail = useUi((s) => s.connectionDetail);
  const rtt = useUi((s) => s.rttMs);
  const role = useUi((s) => s.role);
  const latency = useUi((s) => s.latency);
  const mainView = useUi((s) => s.mainView);
  const stale = useUi((s) => s.stale);
  const view = viewLatency(latency, mainView, rtt);
  const tone =
    connection === "open"
      ? stale
        ? "error"
        : view.ms !== null && view.ms > 150
          ? "warn"
          : "ok"
      : connection === "closed"
        ? "error"
        : "warn";
  const text =
    connection === "open"
      ? stale
        ? "Stale"
        : `${view.label} ${view.ms === null ? "—" : Math.round(view.ms)} ms`
      : connection === "reconnecting"
        ? "Reconnecting"
        : connection === "connecting"
          ? "Connecting"
          : "Offline";
  const title =
    detail ??
    [stale ? `Stale: ${stale}` : null, ...latencyLines(latency), role ? `Role: ${role}` : null]
      .filter(Boolean)
      .join("\n");
  return (
    <div className={`chip chip-${tone}`} title={title}>
      <span className="dot" />
      <span className="mono">{text}</span>
      {role === "spectator" && <span className="chip-tag">View only</span>}
    </div>
  );
}

function ResetButton({ client, disabled }: { client: TeleopClient; disabled: boolean }) {
  const [armed, setArmed] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const click = () => {
    if (!armed) {
      setArmed(true);
      timer.current = window.setTimeout(() => setArmed(false), 2500);
      return;
    }
    window.clearTimeout(timer.current);
    setArmed(false);
    void runAction(client, "reset");
  };
  return (
    <button
      className={`btn btn-ghost ${armed ? "btn-danger-armed" : ""}`}
      onClick={click}
      disabled={disabled}
      title="Reset the episode (click twice)"
    >
      <Icon name="reset" />
      <span className="hide-sm">{armed ? "Confirm reset" : "Reset"}</span>
    </button>
  );
}

export function TopBar({ client }: { client: TeleopClient }) {
  const status = useUi((s) => s.status);
  const role = useUi((s) => s.role);
  const busy = useUi((s) => s.busyAction);
  const panelOpen = useUi((s) => s.panelOpen);
  const set = useUi((s) => s.set);
  const state: TeleopState = status?.state ?? "disconnected";
  const operator = role === "operator";
  const engaged = state === "engaged";
  const canEngage = operator && (state === "ready" || state === "paused") && !status?.episode_over;

  return (
    <header className="topbar glass">
      <div className="brand">
        <span className="brand-mark" aria-hidden>
          ◎◎
        </span>
        <span className="brand-name hide-sm">MABEL Teleop</span>
      </div>

      <div className={`state-pill state-${state}`} role="status" aria-live="polite">
        <span className="state-dot" />
        <span className="state-label">{STATE_LABEL[state]}</span>
        {status?.reason && state !== "engaged" && (
          <span className="state-reason hide-sm">{status.reason}</span>
        )}
      </div>

      <div className="topbar-actions">
        <ConnectionChip />
        {operator && (
          <>
            <button
              className="btn btn-ghost"
              onClick={() => void runAction(client, "ready")}
              disabled={!!busy || state === "fault" || state === "disconnected"}
              title="Move both arms to the ready pose (R)"
            >
              <Icon name="ready" />
              <span className="hide-sm">Ready pose</span>
            </button>
            <ResetButton client={client} disabled={!!busy && busy !== "pause"} />
            <button
              className={`btn btn-primary ${engaged ? "btn-pause" : "btn-engage"}`}
              onClick={() => togglePrimary(client)}
              disabled={!engaged && (!canEngage || !!busy)}
              title={engaged ? "Pause (Esc or Space)" : "Engage (Enter)"}
            >
              <Icon name={engaged ? "pause" : "play"} />
              <span>{engaged ? "Pause" : "Engage"}</span>
            </button>
          </>
        )}
        <button
          className="btn btn-icon btn-ghost"
          onClick={() => set({ guideOpen: true })}
          title="How to control MABEL"
          aria-label="Help"
        >
          <Icon name="help" />
        </button>
        <button
          className="btn btn-icon btn-ghost"
          onClick={() => set({ panelOpen: !panelOpen })}
          aria-pressed={panelOpen}
          title="Controls panel (P)"
        >
          <Icon name="panel" />
        </button>
      </div>
    </header>
  );
}
