import { useUi } from "../state/store";

/** Which part the movement input controls; tap/click (or Tab) cycles it. */
export function ControlChip({ interactive = true }: { interactive?: boolean }) {
  const driveMode = useUi((s) => s.driveMode);
  const activeArm = useUi((s) => s.activeArm);
  const base = useUi((s) => s.status?.base);
  const cycle = useUi((s) => s.cycleControl);
  const speed = base ? Math.hypot(base.forward_mps, base.left_mps) : 0;
  const label = driveMode
    ? `Drive${speed > 0.01 ? ` · ${speed.toFixed(2)} m/s` : ""}`
    : `${activeArm === "left" ? "Left" : "Right"} hand`;
  const tone = driveMode ? "drive" : activeArm;
  const content = (
    <>
      {label}
      {base?.tilt_guard && driveMode && (
        <span className="chip-warn-dot" title="Tilt guard braking" />
      )}
    </>
  );
  return interactive ? (
    <button
      className={`arm-chip arm-${tone}`}
      onClick={cycle}
      title="Switch between left hand, right hand and driving (Tab)"
    >
      {content} ⇄
    </button>
  ) : (
    <span className={`arm-chip arm-${tone}`}>{content}</span>
  );
}
