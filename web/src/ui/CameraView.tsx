import { useEffect, useRef, useState } from "react";
import type { TeleopClient } from "../net/TeleopClient";
import type { Eyes } from "../state/store";

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
    const tick = () => {
      const pair = client.stereo.latest;
      const now = performance.now();
      if (pair && pair.meta.pair_id !== drawnPair) {
        drawnPair = pair.meta.pair_id;
        const ratio = pair.meta.width / pair.meta.height;
        setAspect((prev) => (Math.abs(prev - ratio) < 1e-3 ? prev : ratio));
        draw(leftRef.current, pair.left);
        draw(rightRef.current, pair.right);
        frames += 1;
      }
      if (now - lastInfo > 250) {
        const fps = (frames * 1000) / (now - since);
        frames = 0;
        since = now;
        lastInfo = now;
        const stale = !pair || now - pair.receivedAt > STALE_MS;
        setInfo((prev) =>
          prev.stale === stale && Math.abs(prev.fps - fps) < 0.5 && prev.waiting === !pair
            ? prev
            : { stale, fps, waiting: !pair },
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [client]);

  return (
    <div
      className={`camera-view eyes-${eyes} ${compact ? "compact" : ""}`}
      style={{ ["--aspect" as string]: aspect }}
    >
      <div className="eye" hidden={eyes === "right"}>
        <div className="eye-frame">
          <canvas ref={leftRef} aria-label="Robot head camera, left eye" />
          {eyes === "both" && <span className="eye-tag">L</span>}
        </div>
      </div>
      <div className="eye" hidden={eyes === "left"}>
        <div className="eye-frame">
          <canvas ref={rightRef} aria-label="Robot head camera, right eye" />
          {eyes === "both" && <span className="eye-tag">R</span>}
        </div>
      </div>
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
