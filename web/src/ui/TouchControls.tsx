import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { touch } from "../input/touch";
import { useUi } from "../state/store";
import { haptic } from "./actions";
import { ControlChip } from "./ControlChip";

/** Pointer-captured virtual joystick writing normalised [-1, 1] values into `target`. */
function Joystick({
  target,
  label,
  hint,
}: {
  target: { x: number; y: number };
  label: string;
  hint: string;
}) {
  const base = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState({ x: 0, y: 0, active: false });
  const pointer = useRef<number | null>(null);

  const move = (event: ReactPointerEvent) => {
    const rect = base.current!.getBoundingClientRect();
    const radius = rect.width / 2;
    let dx = (event.clientX - (rect.left + radius)) / radius;
    let dy = (event.clientY - (rect.top + radius)) / radius;
    const length = Math.hypot(dx, dy);
    if (length > 1) {
      dx /= length;
      dy /= length;
    }
    target.x = dx;
    target.y = -dy; // screen up = positive
    setKnob({ x: dx, y: dy, active: true });
  };
  const release = () => {
    pointer.current = null;
    target.x = target.y = 0;
    setKnob({ x: 0, y: 0, active: false });
  };
  return (
    <div
      ref={base}
      className={`joystick ${knob.active ? "active" : ""}`}
      aria-label={label}
      onPointerDown={(e) => {
        if (pointer.current !== null) return;
        pointer.current = e.pointerId;
        e.currentTarget.setPointerCapture(e.pointerId);
        haptic(8);
        move(e);
      }}
      onPointerMove={(e) => e.pointerId === pointer.current && move(e)}
      onPointerUp={(e) => e.pointerId === pointer.current && release()}
      onPointerCancel={(e) => e.pointerId === pointer.current && release()}
      onLostPointerCapture={release}
    >
      <span className="joystick-hint">{hint}</span>
      <div
        className="joystick-knob"
        style={{ transform: `translate(${knob.x * 38}px, ${knob.y * 38}px)` }}
      />
    </div>
  );
}

function HoldButton({ label, value }: { label: string; value: number }) {
  const [held, setHeld] = useState(false);
  const set = (on: boolean) => {
    setHeld(on);
    touch.grip = on ? value : 0;
    if (on) haptic(10);
  };
  return (
    <button
      className={`hold-btn ${held ? "held" : ""}`}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        set(true);
      }}
      onPointerUp={() => set(false)}
      onPointerCancel={() => set(false)}
      onLostPointerCapture={() => set(false)}
    >
      {label}
    </button>
  );
}

/** On-screen controls for touch devices, shown only while engaged. */
export function TouchControls() {
  const engaged = useUi((s) => s.status?.state === "engaged" && s.role === "operator");
  const driveMode = useUi((s) => s.driveMode);
  const [fine, setFine] = useState(false);
  if (!engaged) return null;
  return (
    <div className="touch-controls show-touch">
      <Joystick
        target={touch.planar}
        label={driveMode ? "Drive the robot" : "Move hand in the ground plane"}
        hint={driveMode ? "Drive" : "Move"}
      />
      <div className="touch-center">
        <ControlChip />
        {!driveMode && (
          <div className="touch-row">
            <HoldButton label="Open" value={-1} />
            <HoldButton label="Close" value={1} />
          </div>
        )}
        <button
          className={`chip ${fine ? "chip-ok" : ""}`}
          onClick={() => {
            touch.fine = !fine;
            setFine(!fine);
          }}
        >
          Fine {fine ? "on" : "off"}
        </button>
      </div>
      <Joystick
        target={touch.vertical}
        label={driveMode ? "Turn the robot" : "Raise, lower and turn the hand"}
        hint={driveMode ? "Turn" : "Up · Turn"}
      />
    </div>
  );
}
