// Small inline icon set (stroke icons, 24px grid) — no icon-font download.
const PATHS: Record<string, string> = {
  play: "M7 5v14l11-7z",
  pause: "M8 5v14M16 5v14",
  reset: "M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5",
  ready: "M12 3v6m0 0l-3-3m3 3l3-3M5 21h14M7 17h10",
  panel: "M4 5h16v14H4zM14 5v14",
  help: "M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14m0 3h.01",
  camera: "M4 8h3l2-2h6l2 2h3v11H4zM12 17a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z",
  move: "M12 3v18M3 12h18M12 3l-3 3m3-3l3 3M12 21l-3-3m3 3l3-3M3 12l3-3m-3 3l3 3M21 12l-3-3m3 3l-3 3",
  rotate: "M20 12a8 8 0 1 1-3-6.2M20 4v5h-5",
  close: "M6 6l12 12M18 6L6 18",
  hand: "M8 13V6a1.5 1.5 0 0 1 3 0v5m0-6a1.5 1.5 0 0 1 3 0v6m0-4a1.5 1.5 0 0 1 3 0v7a6 6 0 0 1-6 6h-1a6 6 0 0 1-5-2.7L3.5 14a1.5 1.5 0 0 1 2.5-1.6L8 15",
};

export function Icon({ name, size = 18 }: { name: keyof typeof PATHS | string; size?: number }) {
  const filled = name === "play";
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={PATHS[name] ?? ""} />
    </svg>
  );
}
