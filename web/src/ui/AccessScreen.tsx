import { useUi } from "../state/store";

/** Shown when the host rejects the access token; reconnecting cannot help until it changes. */
export function AccessScreen() {
  const authError = useUi((s) => s.authError);
  if (!authError) return null;
  return (
    <div className="modal-backdrop access">
      <div className="modal glass access-card" role="alertdialog" aria-label="Access needed">
        <h2>Access needed</h2>
        <p>{authError}</p>
        <p className="muted small">
          Open the URL printed by <code>abstract-stack teleop</code> on the host computer.
        </p>
      </div>
    </div>
  );
}
