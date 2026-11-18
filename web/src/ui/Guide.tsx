import { useEffect, useRef } from "react";
import { useUi } from "../state/store";
import { Icon } from "./Icon";

const SEEN_KEY = "teleop-guide-v1";

const isTouch = () => window.matchMedia?.("(pointer: coarse)").matches ?? false;

interface Card {
  icon: string;
  title: string;
  desktop: string[];
  touch: string[];
}

const CARDS: Card[] = [
  {
    icon: "hand",
    title: "Hands",
    desktop: [
      "Drag the glowing target, or move it with W A S D and Q / E.",
      "[ and ] open and close the grip.",
      "Tab switches between left hand, right hand and Drive.",
    ],
    touch: [
      "Left stick moves the hand; right stick raises it and turns it.",
      "Hold Open or Close to grip.",
      "Tap the chip to switch hand or Drive.",
    ],
  },
  {
    icon: "eye",
    title: "Look",
    desktop: [
      "The head follows the hand you control, automatically.",
      "Arrow keys, or drag the Camera view, to look freely; click the 3D view to look at a spot.",
      "Press F or tap “Free look” to follow again.",
    ],
    touch: [
      "The head follows the hand you control, automatically.",
      "Drag the Camera view to look around; tap the 3D view to look at a spot.",
      "Tap “Free look” to follow the hand again.",
    ],
  },
  {
    icon: "move",
    title: "Drive",
    desktop: [
      "Press Tab until the chip says Drive.",
      "W A S D drive relative to the robot; Q / E turn it.",
      "Let go and it stops. Shift drives slowly.",
    ],
    touch: [
      "Tap the chip until it says Drive.",
      "Left stick drives; right stick turns the robot.",
      "Let go and it stops.",
    ],
  },
  {
    icon: "ready",
    title: "Body",
    desktop: [
      "Height and Lean rails at the edge raise the body and lean it forward.",
      "MABEL turns by driving — its torso only leans.",
      "Ready pose (R) brings both arms to a comfortable start.",
    ],
    touch: [
      "Height and Lean rails at the edge raise the body and lean it forward.",
      "MABEL turns by driving — its torso only leans.",
      "Ready pose brings both arms to a comfortable start.",
    ],
  },
];

export function hasSeenGuide(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === "1";
  } catch {
    return false;
  }
}

function markSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, "1");
  } catch {
    // Without storage the guide simply shows again next visit.
  }
}

/** One-time introduction to the four control groups; reopen from the “?” button. */
export function Guide() {
  const open = useUi((s) => s.guideOpen);
  const role = useUi((s) => s.role);
  const set = useUi((s) => s.set);
  const confirm = useRef<HTMLButtonElement>(null);
  // Focus "Got it" for keyboard users without scrolling the title out of view.
  useEffect(() => {
    if (open) confirm.current?.focus({ preventScroll: true });
  }, [open]);
  // First visit as operator: show once the host has connected.
  useEffect(() => {
    if (role === "operator" && !hasSeenGuide()) set({ guideOpen: true });
  }, [role, set]);
  if (!open) return null;
  const touch = isTouch();
  const close = () => {
    markSeen();
    set({ guideOpen: false });
  };
  return (
    <div className="modal-backdrop guide-backdrop" onClick={close}>
      <div
        className="modal glass guide"
        role="dialog"
        aria-label="How to control MABEL"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="guide-head">
          <h2>How to control MABEL</h2>
          <button className="btn btn-icon btn-ghost" onClick={close} title="Close">
            <Icon name="close" />
          </button>
        </div>
        <div className="guide-grid">
          {CARDS.map((card) => (
            <section key={card.title} className="guide-card">
              <h3>
                <span className="guide-icon">
                  <Icon name={card.icon} size={16} />
                </span>
                {card.title}
              </h3>
              <ul>
                {(touch ? card.touch : card.desktop).map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>
          ))}
        </div>
        <div className="guide-foot">
          <span className="muted small">
            Press <strong>Engage</strong> to take control. <strong>Pause</strong>
            {touch ? "" : " (Esc)"} stops everything at once.
          </span>
          <div className="guide-actions">
            {!touch && (
              <button
                className="btn btn-ghost"
                onClick={() => {
                  close();
                  set({ helpOpen: true });
                }}
              >
                All shortcuts
              </button>
            )}
            <button ref={confirm} className="btn btn-primary btn-engage" onClick={close}>
              Got it
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
