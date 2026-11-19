import { useEffect, type ReactNode } from "react";
import type { TeleopClient } from "../net/TeleopClient";
import { useUi, type Eyes } from "../state/store";
import { CameraView } from "./CameraView";
import { Icon } from "./Icon";

const EYE_OPTIONS: { value: Eyes; label: string }[] = [
  { value: "left", label: "Left" },
  { value: "both", label: "Both" },
  { value: "right", label: "Right" },
];

/**
 * Main view + picture-in-picture: the 3D twin and the robot head camera swap places.
 * The twin canvas is the same element in both roles (no second WebGL context).
 */
export function ViewLayers({ client, twin }: { client: TeleopClient; twin: ReactNode }) {
  const mainView = useUi((s) => s.mainView);
  const eyes = useUi((s) => s.eyes);
  const videoAvailable = useUi((s) => s.videoAvailable);
  const pipVisible = useUi((s) => s.pipVisible);
  const set = useUi((s) => s.set);
  const cameraMain = videoAvailable && mainView === "camera";
  const showCamera = videoAvailable && (cameraMain || pipVisible);

  // Only pull video from the host while it is on screen.
  useEffect(() => client.setVideoWanted(showCamera), [client, showCamera]);

  return (
    <>
      <div
        className={`view-layer twin-layer ${cameraMain ? "pip" : "main"}`}
        hidden={cameraMain && !pipVisible}
      >
        {twin}
        {cameraMain && (
          <button
            className="pip-label"
            onClick={() => set({ mainView: "twin" })}
            title="Swap views (V)"
          >
            3D view <Icon name="move" size={14} />
          </button>
        )}
      </div>
      {showCamera && (
        <div
          className={`view-layer camera-layer ${cameraMain ? "main" : "pip"}`}
          onClick={cameraMain ? undefined : () => set({ mainView: "camera" })}
          title={cameraMain ? undefined : "Show head camera (V)"}
        >
          <CameraView client={client} eyes={cameraMain ? eyes : "both"} compact={!cameraMain} />
          {!cameraMain && <span className="pip-label">Head camera</span>}
        </div>
      )}
      {videoAvailable && (
        <div className="view-switch glass">
          <div className="segmented" role="radiogroup" aria-label="Main view">
            <button
              role="radio"
              aria-checked={!cameraMain}
              className={!cameraMain ? "active" : ""}
              onClick={() => set({ mainView: "twin" })}
              title="3D view (V)"
            >
              3D
            </button>
            <button
              role="radio"
              aria-checked={cameraMain}
              className={cameraMain ? "active" : ""}
              onClick={() => set({ mainView: "camera" })}
              title="Head camera (V)"
            >
              Camera
            </button>
          </div>
          {cameraMain && (
            <div className="segmented" role="radiogroup" aria-label="Camera eye">
              {EYE_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  role="radio"
                  aria-checked={eyes === o.value}
                  className={eyes === o.value ? "active" : ""}
                  onClick={() => set({ eyes: o.value })}
                  title="Cycle eyes (B)"
                >
                  {o.label}
                </button>
              ))}
            </div>
          )}
          <button
            className="btn btn-icon btn-ghost pip-toggle"
            onClick={() => set({ pipVisible: !pipVisible })}
            aria-pressed={pipVisible}
            title={pipVisible ? "Hide picture-in-picture" : "Show picture-in-picture"}
          >
            <Icon name={pipVisible ? "close" : "camera"} size={16} />
          </button>
        </div>
      )}
    </>
  );
}
