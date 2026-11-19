import { defineConfig, devices } from "@playwright/test";

// Starts a real teleop host (MuJoCo + MABEL) that serves the built client.
// Run `npm run build` first; requires the Python env with the `teleop` extra.
const PORT = Number(process.env.E2E_PORT ?? 8799);

// Headless Chromium on Linux rasterises WebGL on the CPU (SwiftShader, ~1–10 fps): the
// main thread stalls between frames, so tests use the low render-quality path, a modest
// viewport and a generous host input timeout (input-loss pausing has its own test and
// is covered natively in tests/test_web_teleop.py). On macOS the GPU (Metal) is used
// unless E2E_GPU=0, which reproduces the CPU-rendered Linux conditions.
const useGpu = process.platform === "darwin" && process.env.E2E_GPU !== "0";
const gpuArgs = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"];
const cpuArgs = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"];

export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    launchOptions: { args: useGpu ? gpuArgs : cpuArgs },
  },
  projects: [
    {
      name: "desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1024, height: 640 } },
    },
    { name: "phone", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `../.venv/bin/abstract-stack teleop --port ${PORT} --token e2e --web-dir dist --input-timeout 2.0`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
  },
});
