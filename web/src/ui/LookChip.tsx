import { useUi } from "../state/store";
import { Icon } from "./Icon";

/** Shows how the head is being aimed; tap to return to following the hand. */
export function LookChip() {
  const mode = useUi((s) => s.headMode);
  const driveMode = useUi((s) => s.driveMode);
  const set = useUi((s) => s.set);
  const label =
    mode === "follow"
      ? driveMode
        ? "Looking ahead"
        : "Following hand"
      : mode === "point"
        ? "Looking at point"
        : "Free look";
  return (
    <button
      className={`look-chip look-${mode}`}
      onClick={() => set({ headMode: "follow" })}
      title={
        mode === "follow"
          ? "The head keeps your hand in view. Drag the camera view or use the arrow keys to look freely."
          : "Tap to make the head follow your hand again (F)"
      }
      aria-pressed={mode === "follow"}
    >
      <Icon name="eye" size={14} />
      {label}
      {mode !== "follow" && <span className="look-chip-action">Follow</span>}
    </button>
  );
}
