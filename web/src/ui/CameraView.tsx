import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { lookAt, panBy, recentreHead } from "../input/headLook";
import { videoAgeMs } from "../net/stereo";
import type { TeleopClient } from "../net/TeleopClient";
import { useUi, type Eyes } from "../state/store";
import { syncMirror } from "../state/mirror";

const DRAG_START_PX = 6;
const DOUBLE_TAP_MS = 320;

interface Gesture {
  id: number;
  x: number;
  y: number;
  lastX: number;
  lastY: number;
  dragging: boolean;
}

/**
 * Pointer gestures on a head-camera image: drag pans the head (image follows the
 * finger), tap/click looks at that point, double-tap re-centres. Active only while
 * the operator is engaged; otherwise a hint explains why nothing moves.
 */
function useHeadGestures(fovDeg: number, aspect: number) {
  const gesture = useRef<Gesture | null>(null);
  const lastTap = useRef({ t: 0, x: 0, y: 0 });
  const [rings, setRings] = useState<{ id: number; x: number; y: number }[]>([]);
  const ringId = useRef(0);
  const geometry = { verticalFovDeg: fovDeg, aspect };
  const allowed = () => {
    const ui = useUi.getState();
    if (ui.role === "operator" && ui.status?.state === "engaged") return true;
    ui.toast("info", ui.role === "operator" ? "Engage to aim the head" : "View only");
    return false;
  };
  const frac = (e: ReactPointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { u: (e.clientX - r.left) / r.width, v: (e.clientY - r.top) / r.height, r };
  };
  return {
    rings,
    handlers: {
      onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => {
        if (gesture.current) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        gesture.current = {
          id: e.pointerId,
          x: e.clientX,
          y: e.clientY,
          lastX: e.clientX,
          lastY: e.clientY,
          dragging: false,
        };
      },
      onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => {
        const g = gesture.current;
        if (!g || g.id !== e.pointerId) return;
        if (!g.dragging && Math.hypot(e.clientX - g.x, e.clientY - g.y) > DRAG_START_PX) {
          if (!allowed()) {
            gesture.current = null;
            return;
          }
          g.dragging = true;
        }
        if (g.dragging) {
          const { r } = frac(e);
          panBy((e.clientX - g.lastX) / r.width, (e.clientY - g.lastY) / r.height, geometry);
          g.lastX = e.clientX;
          g.lastY = e.clientY;
        }
      },
      onPointerUp: (e: ReactPointerEvent<HTMLDivElement>) => {
        const g = gesture.current;
        gesture.current = null;
        if (!g || g.id !== e.pointerId) return;
        if (g.dragging) {
          syncMirror();
          return;
        }
        if (!allowed()) return;
        const { u, v } = frac(e);
        const now = performance.now();
        const t = lastTap.current;
        if (now - t.t < DOUBLE_TAP_MS && Math.hypot(e.clientX - t.x, e.clientY - t.y) < 30) {
          recentreHead();
          lastTap.current = { t: 0, x: 0, y: 0 };
          useUi.getState().toast("info", "Head re-centred");
        } else {
          lookAt(u, v, geometry);
          lastTap.current = { t: now, x: e.clientX, y: e.clientY };
          const id = ++ringId.current;
          setRings((list) => [...list.slice(-2), { id, x: u, y: v }]);
          window.setTimeout(() => setRings((list) => list.filter((ring) => ring.id !== id)), 650);
        }
        syncMirror();
      },
      onPointerCancel: () => {
        gesture.current = null;
      },
    },
  };
}

const STALE_MS = 300; // contract video_stale_age_s

/**
 * Robot head-camera view (zed_left / zed_right, 63 mm stereo baseline). Draws the
 * newest decoded pair on display frames; both eyes always come from the same pair.
 */
export function CameraView({
  client,
  eyes,
  compact = false,
}: {
  client: TeleopClient;
  eyes: Eyes;
  compact?: boolean;
}) {
  const leftRef = useRef<HTMLCanvasElement>(null);
  const rightRef = useRef<HTMLCanvasElement>(null);
  const [info, setInfo] = useState({ stale: true, fps: 0, waiting: true });
  const [aspect, setAspect] = useState(4 / 3);
  const [fov, setFov] = useState(83.3);
  const { rings, handlers } = useHeadGestures(fov, aspect);

  useEffect(() => {
    let raf = 0;
    let drawnPair = -1;
    let frames = 0;
    let since = performance.now();
    let lastInfo = 0;
    const draw = (canvas: HTMLCanvasElement | null, bitmap: ImageBitmap) => {
      if (!canvas) return;
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext("2d", { alpha: false })?.drawImage(bitmap, 0, 0);
    };
    // Draw each pair as soon as it decodes rather than on the next animation
    // frame: the canvas then makes the very next composite (saves up to a frame).
    const show = () => {
      const pair = client.stereo.latest;
      if (!pair || pair.meta.pair_id === drawnPair) return;
      const now = performance.now();
      drawnPair = pair.meta.pair_id;
      const ratio = pair.meta.width / pair.meta.height;
      setAspect((prev) => (Math.abs(prev - ratio) < 1e-3 ? prev : ratio));
      const vfov = pair.meta.vertical_fov_deg[0];
      setFov((prev) => (Math.abs(prev - vfov) < 1e-3 ? prev : vfov));
      draw(leftRef.current, pair.left);
      draw(rightRef.current, pair.right);
      client.markVideoShown(pair);
      frames += 1;
      if (client.clock.synced) {
        const captured = client.clock.toLocalMs(pair.meta.capture_host_monotonic_s);
        client.latency.videoArrival.push(pair.arrivedAt - captured);
        client.latency.videoDecode.push(pair.receivedAt - pair.arrivedAt);
        client.latency.videoShown.push(now - captured);
      }
    };
    const off = client.on("video", show);
    const tick = () => {
      const pair = client.stereo.latest;
      const now = performance.now();
      show(); // first pair after mount, or one decoded while unmounted
      if (now - lastInfo > 250) {
        const fps = (frames * 1000) / (now - since);
        frames = 0;
        since = now;
        lastInfo = now;
        const stale = !pair || videoAgeMs(client.clock, pair, now) > STALE_MS;
        setInfo((prev) =>
          prev.stale === stale && Math.abs(prev.fps - fps) < 0.5 && prev.waiting === !pair
            ? prev
            : { stale, fps, waiting: !pair },
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      off();
      cancelAnimationFrame(raf);
    };
  }, [client]);

  return (
    <div
      className={`camera-view eyes-${eyes} ${compact ? "compact" : ""}`}
      style={{ ["--aspect" as string]: aspect }}
    >
      {(["left", "right"] as const).map((eye) => (
        <div key={eye} className="eye" hidden={eyes === (eye === "left" ? "right" : "left")}>
          <div className={`eye-frame ${compact ? "" : "aimable"}`} {...(compact ? {} : handlers)}>
            <canvas
              ref={eye === "left" ? leftRef : rightRef}
              aria-label={`Robot head camera, ${eye} eye`}
            />
            {eyes === "both" && <span className="eye-tag">{eye === "left" ? "L" : "R"}</span>}
            {!compact && <span className="crosshair" aria-hidden />}
            {!compact &&
              rings.map((ring) => (
                <span
                  key={ring.id}
                  className="look-ring"
                  style={{ left: `${ring.x * 100}%`, top: `${ring.y * 100}%` }}
                />
              ))}
          </div>
        </div>
      ))}
      <div className="camera-badges">
        {info.waiting ? (
          <span className="badge">Connecting video…</span>
        ) : info.stale ? (
          <span className="badge badge-bad">VIDEO STALE</span>
        ) : (
          !compact && <span className="badge mono">{info.fps.toFixed(0)} fps</span>
        )}
      </div>
    </div>
  );
}
