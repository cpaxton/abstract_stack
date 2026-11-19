import { useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { clampChannels, intent } from "../state/intent";
import { useUi } from "../state/store";
import { haptic } from "./actions";
import { syncMirror } from "../state/mirror";

/**
 * Vertical, pointer-driven rail (custom so it behaves the same on every mobile
 * browser and offers a large touch target). Up always means "more".
 */
function Rail({
  label,
  value,
  min,
  max,
  format,
  onChange,
  disabled,
  step,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  disabled: boolean;
  step: number;
}) {
  const track = useRef<HTMLDivElement>(null);
  const fraction = max > min ? (value - min) / (max - min) : 0;
  const setFromPointer = (e: ReactPointerEvent) => {
    const rect = track.current!.getBoundingClientRect();
    const f = 1 - (e.clientY - rect.top) / rect.height;
    onChange(min + Math.min(1, Math.max(0, f)) * (max - min));
  };
  const onKey = (e: KeyboardEvent) => {
    const delta = { ArrowUp: step, ArrowDown: -step, PageUp: step * 5, PageDown: -step * 5 }[e.key];
    if (delta === undefined || disabled) return;
    e.preventDefault();
    e.stopPropagation(); // keep PageUp/PageDown from also jogging the lift globally
    onChange(Math.min(max, Math.max(min, value + delta)));
  };
  return (
    <div className={`rail ${disabled ? "disabled" : ""}`}>
      <span className="rail-label">{label}</span>
      <div
        ref={track}
        className="rail-track"
        role="slider"
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={format(value)}
        aria-orientation="vertical"
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
        onKeyDown={onKey}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          haptic(8);
          setFromPointer(e);
        }}
        onPointerMove={(e) => {
          if (!disabled && e.currentTarget.hasPointerCapture(e.pointerId)) setFromPointer(e);
        }}
      >
        <div className="rail-fill" style={{ height: `${fraction * 100}%` }} />
        <div className="rail-thumb" style={{ bottom: `${fraction * 100}%` }} />
      </div>
      <span className="rail-value mono">{format(value)}</span>
    </div>
  );
}

const deg = (rad: number) => Math.round((rad * 180) / Math.PI);

export function formatLean(rad: number): string {
  const d = deg(rad);
  if (Math.abs(d) < 2) return "upright";
  // MABEL's torso joint is negative when bending forward.
  return d < 0 ? `${-d}° fwd` : `${d}° back`;
}

/**
 * Posture controls kept at the screen edge: Height (lift) and Lean (torso pitch).
 * MABEL's torso cannot rotate; the Turn link explains that and switches to driving.
 */
export function BodyRails() {
  const role = useUi((s) => s.role);
  const engaged = useUi((s) => s.status?.state === "engaged");
  const ranges = useUi((s) => s.status?.channels.ranges);
  const mirror = useUi((s) => s.mirror);
  const set = useUi((s) => s.set);
  if (role !== "operator" || !ranges) return null;
  const update = (apply: () => void) => {
    apply();
    clampChannels();
    syncMirror();
  };
  return (
    <div className="body-rails glass" aria-label="Body posture">
      <Rail
        label="Height"
        value={mirror.lift}
        min={ranges.lift_m[0]}
        max={ranges.lift_m[1]}
        step={0.01}
        format={(v) => `${Math.round(v * 100)} cm`}
        onChange={(v) => update(() => (intent.lift = v))}
        disabled={!engaged}
      />
      <Rail
        label="Lean"
        value={mirror.torso}
        min={ranges.torso_rad[0]}
        max={ranges.torso_rad[1]}
        step={0.02}
        format={formatLean}
        onChange={(v) => update(() => (intent.torso = v))}
        disabled={!engaged}
      />
      <button
        className="rail-turn"
        onClick={() => {
          set({ driveMode: true });
          useUi
            .getState()
            .toast("info", "MABEL turns by driving: Q/E or the right stick turns the robot");
        }}
        title="MABEL's torso only leans; turning the robot uses drive mode"
        disabled={!engaged}
      >
        Turn → Drive
      </button>
    </div>
  );
}
