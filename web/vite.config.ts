import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// `npm run dev` proxies API/WebSocket traffic to a host started with
// `abstract-stack teleop` (default http://127.0.0.1:8080), so the dev page and
// the host share one origin and no CORS/origin allow-list is needed.
const host = process.env.TELEOP_HOST ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api": host,
      "/teleop": { target: host.replace(/^http/, "ws"), ws: true },
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        // Long-lived vendor chunks cache across client releases.
        manualChunks: (id: string) =>
          id.includes("node_modules/three/")
            ? "three"
            : id.includes("node_modules/@react-three/")
              ? "r3f"
              : undefined,
      },
    },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
