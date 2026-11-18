import { SIDES, type Side } from "../net/protocol";
import { clampChannels, intent, takeManualHead } from "../state/intent";
import { useUi, type CameraPreset } from "../state/store";
import { Icon } from "./Icon";

const deg = (rad: number) => (rad * 180) / Math.PI;

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
  accent,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
  accent?: string;
}) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{label}</span>
        <span className="mono muted">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ["--pct" as string]: `${pct}%`, ["--track-accent" as string]: accent }}
      />
    </label>
  );
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? "active" : ""}
          onClick={() => onChange(o.value)}
          title={o.hint}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function syncMirror() {
  const set = useUi.getState().set;
  set({
    mirror: {
      grip: { left: intent.arms.left.grip, right: intent.arms.right.grip },
      lift: intent.lift,
      torso: intent.torso,
      yaw: intent.head.yaw,
      pitch: intent.head.pitch,
    },
  });
}

function update(fn: () => void) {
  fn();
  clampChannels();
  syncMirror();
}

function ArmCard({ side, active, editable }: { side: Side; active: boolean; editable: boolean }) {
  const grip = useUi((s) => s.mirror.grip[side]);
  const ik = useUi((s) => s.status?.arms[side]?.ik ?? null);
  const set = useUi((s) => s.set);
  const reachable = ik?.reachable ?? true;
  return (
    <div className={`arm-card ${active ? "active" : ""} arm-${side}`}>
      <button
        className="arm-card-head"
        onClick={() => set({ activeArm: side })}
        aria-pressed={active}
      >
        <span className="arm-swatch" />
        <span className="arm-name">{side === "left" ? "Left arm" : "Right arm"}</span>
        <span className={`badge ${reachable ? "badge-ok" : "badge-bad"}`}>
          {ik
            ? reachable
              ? `IK ${(ik.ik_position_error_m * 1000).toFixed(0)} mm`
              : "Out of reach"
            : "—"}
        </span>
      </button>
      <Slider
        label="Grip"
        value={grip}
        min={0}
        max={1}
        step={0.01}
        format={(v) => `${Math.round(v * 100)}%`}
        disabled={!editable}
        accent={side === "left" ? "var(--left)" : "var(--right)"}
        onChange={(v) => update(() => (intent.arms[side].grip = v))}
      />
    </div>
  );
}

function BaseCard() {
  const base = useUi((s) => s.status?.base);
  const driveMode = useUi((s) => s.driveMode);
  const set = useUi((s) => s.set);
  if (!base) return null;
  const speed = Math.hypot(base.forward_mps, base.left_mps);
  return (
    <section className="card">
      <div className="card-title">
        <span>Base</span>
        <button
          className={`btn btn-ghost small ${driveMode ? "btn-drive-on" : ""}`}
          onClick={() => set({ driveMode: !driveMode })}
          aria-pressed={driveMode}
          title="Drive mode: movement input drives the robot (Tab)"
        >
          {driveMode ? "Driving" : "Drive"}
        </button>
      </div>
      <div className="task-row">
        <span className="mono">{speed.toFixed(2)} m/s</span>
        <span className="mono">{deg(base.turn_radps).toFixed(0)}°/s</span>
        <span className={`badge ${base.tilt_guard ? "badge-bad" : "badge-ok"}`}>
          tilt {base.tilt_deg.toFixed(1)}°
        </span>
      </div>
      {base.tilt_guard && <span className="small error-text">Tilt guard is braking the base.</span>}
    </section>
  );
}

function TaskCard() {
  const task = useUi((s) => s.status?.task);
  if (!task) return null;
  const pct = task.product_count ? (task.stocked_count / task.product_count) * 100 : 0;
  return (
    <section className="card">
      <div className="card-title">
        <span>Shelf task</span>
        <span className="mono muted">{task.elapsed_sim_time.toFixed(0)} s</span>
      </div>
      <div className="progress" aria-label="Stocked products">
        <div className="progress-bar" style={{ width: `${pct}%` }} />
      </div>
      <div className="task-row">
        <span>
          <strong>{task.stocked_count}</strong>/{task.product_count} stocked
        </span>
        {task.success && <span className="badge badge-ok">Complete</span>}
      </div>
      {(task.tipped.length > 0 || task.dropped.length > 0) && (
        <div className="task-row">
          {task.tipped.length > 0 && (
            <span className="badge badge-warn">{task.tipped.length} tipped</span>
          )}
          {task.dropped.length > 0 && (
            <span className="badge badge-bad">{task.dropped.length} dropped</span>
          )}
        </div>
      )}
    </section>
  );
}

function Diagnostics() {
  const status = useUi((s) => s.status);
  if (!status) return null;
  const limited = [...new Set([...status.limits.rate_limited, ...status.limits.target_limited])];
  return (
    <details className="card diagnostics">
      <summary className="card-title">Diagnostics</summary>
      <dl>
        <dt>Real-time factor</dt>
        <dd className="mono">{status.real_time_factor.toFixed(2)}×</dd>
        <dt>Physics tick</dt>
        <dd className="mono">{status.step_ms.toFixed(1)} ms</dd>
        <dt>Sim time</dt>
        <dd className="mono">{status.sim_time_s.toFixed(1)} s</dd>
        <dt>Generation</dt>
        <dd className="mono">{status.calibration_generation}</dd>
        <dt>Spectators</dt>
        <dd className="mono">{status.spectators}</dd>
        <dt>Limited joints</dt>
        <dd className="mono small">{limited.length ? limited.join(", ") : "none"}</dd>
      </dl>
    </details>
  );
}

export function SidePanel() {
  const open = useUi((s) => s.panelOpen);
  const role = useUi((s) => s.role);
  const status = useUi((s) => s.status);
  const activeArm = useUi((s) => s.activeArm);
  const gizmoMode = useUi((s) => s.gizmoMode);
  const camera = useUi((s) => s.camera);
  const mirror = useUi((s) => s.mirror);
  const set = useUi((s) => s.set);
  const setCamera = useUi((s) => s.setCamera);
  const ranges = status?.channels.ranges;
  const editable = role === "operator" && status?.state === "engaged";

  return (
    <aside className={`side-panel glass ${open ? "open" : ""}`} aria-hidden={!open}>
      <div className="panel-head">
        <span>Controls</span>
        <button
          className="btn btn-icon btn-ghost"
          onClick={() => set({ panelOpen: false })}
          title="Close"
        >
          <Icon name="close" />
        </button>
      </div>
      <div className="panel-scroll">
        <section className="card">
          <div className="card-title">
            <span>Hands</span>
            <Segmented
              label="Gizmo mode"
              value={gizmoMode}
              onChange={(v) => set({ gizmoMode: v })}
              options={[
                { value: "translate", label: "Move", hint: "Drag to move the target (T)" },
                { value: "rotate", label: "Rotate", hint: "Drag to rotate the target (Y)" },
              ]}
            />
          </div>
          {SIDES.map((side) => (
            <ArmCard key={side} side={side} active={activeArm === side} editable={editable} />
          ))}
        </section>

        {ranges && (
          <section className="card">
            <div className="card-title">Body &amp; head</div>
            <Slider
              label="Lift"
              value={mirror.lift}
              min={ranges.lift_m[0]}
              max={ranges.lift_m[1]}
              step={0.005}
              format={(v) => `${(v * 100).toFixed(1)} cm`}
              disabled={!editable}
              onChange={(v) => update(() => (intent.lift = v))}
            />
            <Slider
              label="Torso"
              value={mirror.torso}
              min={ranges.torso_rad[0]}
              max={ranges.torso_rad[1]}
              step={0.01}
              format={(v) => `${deg(v).toFixed(0)}°`}
              disabled={!editable}
              onChange={(v) => update(() => (intent.torso = v))}
            />
            <Slider
              label="Head yaw"
              value={mirror.yaw}
              min={ranges.head.yaw_rad[0]}
              max={ranges.head.yaw_rad[1]}
              step={0.01}
              format={(v) => `${deg(v).toFixed(0)}°`}
              disabled={!editable}
              onChange={(v) =>
                update(() => {
                  takeManualHead();
                  intent.head.yaw = v;
                })
              }
            />
            <Slider
              label="Head pitch"
              value={mirror.pitch}
              min={ranges.head.pitch_rad[0]}
              max={ranges.head.pitch_rad[1]}
              step={0.01}
              format={(v) => `${deg(v).toFixed(0)}°`}
              disabled={!editable}
              onChange={(v) =>
                update(() => {
                  takeManualHead();
                  intent.head.pitch = v;
                })
              }
            />
          </section>
        )}

        <section className="card">
          <div className="card-title">View</div>
          <Segmented<CameraPreset>
            label="Camera preset"
            value={camera}
            onChange={setCamera}
            options={[
              { value: "operator", label: "Behind", hint: "1" },
              { value: "side", label: "Side", hint: "2" },
              { value: "top", label: "Top", hint: "3" },
              { value: "front", label: "Front", hint: "4" },
            ]}
          />
        </section>

        <BaseCard />
        <TaskCard />
        <Diagnostics />
      </div>
    </aside>
  );
}

export { syncMirror };
