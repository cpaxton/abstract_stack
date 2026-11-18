import { useEffect, useRef, useState } from "react";
import type { TeleopClient } from "../net/TeleopClient";
import { useUi } from "../state/store";
import { togglePrimary } from "./actions";
import { ControlChip } from "./ControlChip";
import { LookChip } from "./LookChip";
import { Icon } from "./Icon";

export function Toasts() {
  const toasts = useUi((s) => s.toasts);
  const dismiss = useUi((s) => s.dismiss);
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => (
        <button key={t.id} className={`toast toast-${t.tone}`} onClick={() => dismiss(t.id)}>
          {t.text}
        </button>
      ))}
    </div>
  );
}

/** Display FPS and pose-stream age, refreshed 4×/s without re-rendering the canvas. */
export function Hud({ client }: { client: TeleopClient }) {
  const [stats, setStats] = useState({ fps: 0, age: Number.POSITIVE_INFINITY });
  const frames = useRef(0);
  useEffect(() => {
    let raf = 0;
    const count = () => {
      frames.current += 1;
      raf = requestAnimationFrame(count);
    };
    raf = requestAnimationFrame(count);
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      setStats({ fps: (frames.current * 1000) / (now - last), age: client.poses.ageMs(now) });
      frames.current = 0;
      last = now;
    }, 250);
    return () => {
      cancelAnimationFrame(raf);
      window.clearInterval(timer);
    };
  }, [client]);
  const stale = stats.age > 300;
  return (
    <div className={`hud glass mono ${stale ? "hud-stale" : ""}`}>
      <span title="Display frame rate">{stats.fps.toFixed(0)} fps</span>
      <span title="Age of the newest robot state">
        {Number.isFinite(stats.age) ? `${stats.age.toFixed(0)} ms` : "no data"}
      </span>
      {stale && <span className="hud-flag">STALE</span>}
    </div>
  );
}

export function StatusBanner({ client }: { client: TeleopClient }) {
  const status = useUi((s) => s.status);
  const role = useUi((s) => s.role);
  const connection = useUi((s) => s.connection);
  if (connection !== "open") {
    return (
      <div className="banner glass banner-warn">
        <strong>
          {connection === "reconnecting" ? "Reconnecting to host…" : "Connecting to host…"}
        </strong>
        <span className="muted">The robot holds its pose while no operator is connected.</span>
      </div>
    );
  }
  if (!status || status.state === "engaged") return null;
  if (role === "spectator") {
    return (
      <div className="banner glass">
        <strong>Viewing only</strong>
        <span className="muted">Another operator holds control.</span>
      </div>
    );
  }
  const tone =
    status.state === "fault" ? "banner-error" : status.state === "paused" ? "banner-warn" : "";
  const title =
    status.state === "fault"
      ? "Fault — reset required"
      : status.episode_over
        ? "Episode finished"
        : status.state === "paused"
          ? "Paused"
          : "Ready to engage";
  const canEngage = !status.episode_over && (status.state === "ready" || status.state === "paused");
  return (
    <div className={`banner glass ${tone}`}>
      <strong>{title}</strong>
      <span className="muted">{status.reason}</span>
      {canEngage && (
        <button className="btn btn-primary btn-engage" onClick={() => togglePrimary(client)}>
          <Icon name="play" />
          Engage <kbd>Enter</kbd>
        </button>
      )}
    </div>
  );
}

const SHORTCUTS: [string, string][] = [
  ["Enter", "Engage"],
  ["Esc / Space", "Pause (always available)"],
  ["W A S D", "Move hand (relative to view)"],
  ["Q / E", "Hand down / up"],
  ["I K · J L · U O", "Pitch · yaw · roll hand"],
  ["[ / ]", "Open / close grip"],
  ["G", "Toggle grip open/closed"],
  ["Shift", "Fine (slow) motion"],
  ["Tab", "Cycle left hand → right hand → drive"],
  ["WASD · Q/E (drive)", "Drive relative to the robot · turn"],
  ["T / Y", "Gizmo move / rotate"],
  ["← → ↑ ↓", "Head yaw / pitch"],
  ["Home", "Re-centre the head"],
  ["F", "Head follows the hand again"],
  ["Click the 3D view", "Look at that spot"],
  ["Drag / click video", "Look around / look at a point"],
  ["PgUp / PgDn", "Height (lift) up / down"],
  ["R", "Ready pose"],
  ["1 – 4", "Camera presets"],
  ["P", "Toggle panel"],
  ["?", "This help"],
];

export function HelpOverlay() {
  const open = useUi((s) => s.helpOpen);
  const set = useUi((s) => s.set);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={() => set({ helpOpen: false })}>
      <div
        className="modal glass"
        role="dialog"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-head">
          <span>Keyboard shortcuts</span>
          <button
            className="btn btn-icon btn-ghost"
            onClick={() => set({ helpOpen: false })}
            title="Close"
          >
            <Icon name="close" />
          </button>
        </div>
        <dl className="shortcuts">
          {SHORTCUTS.map(([keys, what]) => (
            <div key={keys}>
              <dt>
                <kbd>{keys}</kbd>
              </dt>
              <dd>{what}</dd>
            </div>
          ))}
        </dl>
        <p className="muted small">
          Drag the glowing target with the gizmo, or jog it with the keyboard. In drive mode the
          same keys drive the base; it brakes as soon as you let go or pause. The robot follows
          through inverse kinematics; a red target is out of reach. Input stops → the robot pauses
          and holds.
        </p>
      </div>
    </div>
  );
}

export function KeyHints() {
  const role = useUi((s) => s.role);
  const engaged = useUi((s) => s.status?.state === "engaged");
  const driveMode = useUi((s) => s.driveMode);
  const set = useUi((s) => s.set);
  if (role !== "operator") return null;
  return (
    <div className="key-hints glass hide-touch">
      <ControlChip />
      {engaged && <LookChip />}
      {engaged ? (
        driveMode ? (
          <>
            <span>
              <kbd>WASD</kbd> drive
            </span>
            <span>
              <kbd>Q</kbd> <kbd>E</kbd> turn
            </span>
            <span>
              <kbd>Shift</kbd> slow
            </span>
            <span>
              <kbd>Esc</kbd> stop
            </span>
          </>
        ) : (
          <>
            <span>
              <kbd>WASD</kbd> <kbd>QE</kbd> move
            </span>
            <span>
              <kbd>[</kbd> <kbd>]</kbd> grip
            </span>
            <span>
              <kbd>Tab</kbd> hand/drive
            </span>
            <span>
              <kbd>Esc</kbd> pause
            </span>
          </>
        )
      ) : (
        <span>
          <kbd>Enter</kbd> engage
        </span>
      )}
      <button className="link" onClick={() => set({ helpOpen: true })}>
        <kbd>?</kbd> all shortcuts
      </button>
    </div>
  );
}
