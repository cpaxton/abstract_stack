// Measure per-stage teleop latency against a running host, in a real browser.
//
//   node scripts/measure-latency.mjs "http://127.0.0.1:8000/?token=…" [seconds]
//
// Engages, moves the left hand continuously (so every command changes) with the
// head camera picture-in-picture visible, and prints the median of each stage's
// per-second p50/p95 as JSON. All numbers are on the browser clock via ClockSync.
import { chromium } from "@playwright/test";

const url = process.argv[2];
const seconds = Number(process.argv[3] ?? 20);
if (!url) {
  console.error("usage: node scripts/measure-latency.mjs <teleop url with token> [seconds]");
  process.exit(2);
}
const gpu =
  process.platform === "darwin"
    ? ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]
    : ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"];
const browser = await chromium.launch({ args: gpu });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
await page.addInitScript(() => {
  localStorage.setItem("teleop-guide-v1", "1");
  // Input → send: age of the newest hand-target update when each command leaves.
  window.__sendAges = [];
  const send = WebSocket.prototype.send;
  WebSocket.prototype.send = function (data) {
    if (typeof data === "string" && data.includes('"type":"command"') && window.__intentAt) {
      window.__sendAges.push(performance.now() - window.__intentAt);
    }
    return send.call(this, data);
  };
});
await page.goto(url);
await page.locator(".loading").waitFor({ state: "hidden", timeout: 60_000 });
await page.waitForTimeout(1500);
await page.keyboard.press("KeyR");
await page.waitForTimeout(2500);
await page.keyboard.press("Enter");
await page.waitForTimeout(500);
await page.evaluate(() => {
  const intent = window.__teleop.intent();
  const start = intent.arms.left.position.clone();
  const t0 = performance.now();
  const move = () => {
    const t = (performance.now() - t0) / 1000;
    window.__intentAt = performance.now();
    intent.arms.left.position.set(
      start.x + 0.05 * Math.sin(t * Math.PI),
      start.y,
      start.z + 0.05 * Math.sin(t * Math.PI * 0.7),
    );
    requestAnimationFrame(move);
  };
  move();
});
await page.waitForTimeout(2000); // warm-up: clock burst, buffers fill
const samples = [];
for (let i = 0; i < seconds; i++) {
  await page.waitForTimeout(1000);
  samples.push(await page.evaluate(() => window.__teleop.latency()));
}
const { state, video, sendAges } = await page.evaluate(() => {
  const status = window.__teleop.status();
  return { state: status?.state, video: status?.video, sendAges: window.__sendAges.slice(-600) };
});
await browser.close();

const median = (values) => {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? Math.round(v[Math.floor(v.length / 2)] * 10) / 10 : null;
};
const stage = (key) => ({
  p50: median(samples.map((s) => s[key]?.p50)),
  p95: median(samples.map((s) => s[key]?.p95)),
});
console.log(
  JSON.stringify(
    {
      state,
      rttMs: median(samples.map((s) => s.rttMs)),
      clockErrorMs: median(samples.map((s) => s.clockErrorMs)),
      bufferMs: median(samples.map((s) => s.bufferMs)),
      inputToSend: (() => {
        const v = [...sendAges].sort((a, b) => a - b);
        const q = (p) => Math.round(v[Math.floor(p * (v.length - 1))] * 10) / 10;
        return v.length ? { p50: q(0.5), p95: q(0.95) } : null;
      })(),
      uplink: stage("uplink"),
      poseNetwork: stage("poseNetwork"),
      poseShown: stage("poseShown"),
      videoArrival: stage("videoArrival"),
      videoDecode: stage("videoDecode"),
      videoShown: stage("videoShown"),
      hostVideo: video,
    },
    null,
    2,
  ),
);
