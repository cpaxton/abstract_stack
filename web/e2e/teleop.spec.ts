import { expect, test, type Page } from "@playwright/test";

interface Debug {
  status: () => {
    state: string;
    target_epoch: number;
    base: { forward_mps: number };
    channels: { head: { yaw_rad: number }; head_mode: string; lift_m: number };
    arms: Record<string, { palm: { position_m: [number, number, number] } }>;
  } | null;
  poseAgeMs: () => number;
}

// The first-run guide is covered by its own test; mark it seen everywhere else.
test.beforeEach(async ({ page }, info) => {
  if (!info.title.includes("first-run guide")) {
    await page.addInitScript(() => localStorage.setItem("teleop-guide-v1", "1"));
  }
  // E2E_CPU_THROTTLE=4 emulates a slower CI runner (Chromium DevTools CPU throttling).
  const throttle = Number(process.env.E2E_CPU_THROTTLE ?? 0);
  if (throttle > 1) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: throttle });
  }
});

const status = (page: Page) =>
  page.evaluate(() => (window as unknown as { __teleop: Debug }).__teleop.status());

// Software rendering (E2E_GPU=0) decodes H.264 into bitmaps on the CPU, which
// stalls an already slow page; real browsers decode in hardware. Use the JPEG
// fallback there; GPU runs exercise H.264.
const VIDEO = process.env.E2E_GPU === "0" ? "&video=jpeg" : "";

async function open(page: Page) {
  await page.goto(`/?token=e2e&quality=low${VIDEO}`);
  await expect(page.getByRole("status")).toContainText(/Ready|Paused/);
  await expect(page.locator(".loading")).toBeHidden({ timeout: 60_000 });
  await expect
    .poll(() =>
      page.evaluate(() => (window as unknown as { __teleop: Debug }).__teleop.poseAgeMs()),
    )
    .toBeLessThan(500);
}

test("connects, renders the robot twin and strips the token from the URL", async ({ page }) => {
  await open(page);
  expect(page.url()).not.toContain("token=");
  await expect(page.locator(".twin-layer canvas")).toBeVisible();
  await page.screenshot({ path: `test-results/ready-${test.info().project.name}.png` });
});

test("engage, keyboard jog moves the hand, pause holds", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await open(page);
  await page.keyboard.press("KeyR"); // ready pose
  await page.waitForTimeout(2500);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Engaged");
  const before = (await status(page))!.arms.right!.palm.position_m;
  await page.keyboard.down("KeyQ"); // hand down
  await page.waitForTimeout(1500);
  await page.keyboard.up("KeyQ");
  await page.waitForTimeout(1200);
  const after = (await status(page))!.arms.right!.palm.position_m;
  expect(before[2]! - after[2]!).toBeGreaterThan(0.04);
  await expect(page.getByRole("status")).toContainText("Engaged");
  await page.screenshot({ path: "test-results/engaged-desktop.png" });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("status")).toContainText("Paused");
});

test("input loss pauses the robot", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "same host behaviour on all devices");
  await open(page);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Engaged");
  // Simulate a stalled client: stop every timer and animation frame (the command
  // pump runs on both), so commands stop while the sockets stay open.
  await page.evaluate(() => {
    const hi = window.setInterval(() => {}, 1e6);
    for (let i = 0; i <= hi; i++) window.clearInterval(i);
    const frame = requestAnimationFrame(() => {});
    for (let i = 0; i <= frame; i++) cancelAnimationFrame(i);
  });
  await expect(page.getByRole("status")).toContainText("Paused", { timeout: 5000 });
});

test("touch layout shows joysticks when engaged", async ({ page }, info) => {
  test.skip(info.project.name !== "phone", "touch only");
  await open(page);
  await page
    .getByRole("button", { name: /Engage/ })
    .first()
    .tap();
  await expect(page.getByRole("status")).toContainText("Engaged");
  await expect(page.locator(".joystick")).toHaveCount(2);
  await page.screenshot({ path: "test-results/engaged-phone.png" });
});

test("head camera streams both stereo eyes", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "same stream on all devices");
  await open(page);
  await page.getByRole("radio", { name: "Camera" }).click();
  await page.getByRole("radio", { name: "Both" }).click();
  const pairs = () =>
    page.evaluate(() =>
      (window as unknown as { __teleop: { videoPairs: () => number } }).__teleop.videoPairs(),
    );
  await expect.poll(pairs).toBeGreaterThan(5);
  const before = await pairs();
  await page.waitForTimeout(2000);
  // The host sends ~20 pairs/s; a CPU-rendered headless page decodes fewer.
  expect((await pairs()) - before).toBeGreaterThan(5);
  const eyes = page.locator(".camera-layer.main canvas");
  await expect(eyes).toHaveCount(2);
  for (const eye of await eyes.all()) {
    expect(await eye.evaluate((c: HTMLCanvasElement) => c.width)).toBe(640);
  }
  await page.screenshot({ path: "test-results/camera-desktop.png" });
});

test("a wrong token says so instead of retrying silently", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "same flow on all devices");
  await page.goto("/?token=stale-token-from-an-old-run&quality=low");
  await expect(page.getByRole("heading", { name: "Access needed" })).toBeVisible();
});

/**
 * Engage (again, if a slow CPU-rendered frame let input go stale) right before
 * acting. Engage is refused while the view is stale, as it can be for a moment on a
 * software-rendered page, so press again until it takes.
 */
async function ensureEngaged(page: Page) {
  const pill = page.getByRole("status");
  await expect(async () => {
    if (!/Engaged/.test((await pill.textContent()) ?? "")) await page.keyboard.press("Enter");
    await expect(pill).toContainText("Engaged", { timeout: 2000 });
  }).toPass({ timeout: 20_000 });
}

test("drive mode moves the base and stops on release", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await open(page);
  await ensureEngaged(page);
  // Follow-hand is the default: the host tracks a gaze point, not fixed angles.
  await expect(page.locator(".key-hints .look-chip")).toContainText("Following hand");
  await expect.poll(async () => (await status(page))!.channels.head_mode).toBe("gaze");
  await page.keyboard.press("Tab"); // right hand → drive
  await expect(page.locator(".key-hints")).toContainText("Drive");
  await ensureEngaged(page);
  await page.keyboard.down("KeyW");
  await expect
    .poll(async () => (await status(page))!.base.forward_mps, { timeout: 15_000 })
    .toBeGreaterThan(0.05);
  await page.keyboard.up("KeyW");
  await expect
    .poll(async () => (await status(page))!.base.forward_mps, { timeout: 15_000 })
    .toBeLessThan(0.01);
});

test("clicking the head camera looks there and switches to free look", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "pointer flow");
  await open(page);
  await page.getByRole("radio", { name: "Camera" }).click();
  const frame = page.locator(".camera-layer.main .eye-frame.aimable").first();
  await expect(frame).toBeVisible();
  // Click only once real video is on screen, so the click lands on a live image.
  const pairs = () =>
    page.evaluate(() =>
      (window as unknown as { __teleop: { videoPairs: () => number } }).__teleop.videoPairs(),
    );
  await expect.poll(pairs, { timeout: 20_000 }).toBeGreaterThan(0);
  await ensureEngaged(page);
  const box = (await frame.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5); // look right
  await expect(page.locator(".key-hints .look-chip")).toContainText("Free look");
  await expect
    .poll(async () => (await status(page))!.channels.head.yaw_rad, { timeout: 15_000 })
    .toBeLessThan(-0.15);
});

test("body rails raise the lift and Turn switches to drive", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "pointer flow");
  await open(page);
  await ensureEngaged(page);
  const height = page.locator(".body-rails").getByRole("slider", { name: "Height" });
  const box = (await height.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.6);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.1, { steps: 4 });
  await page.mouse.up();
  await expect
    .poll(async () => (await status(page))!.channels.lift_m, { timeout: 8000 })
    .toBeGreaterThan(0.45);
  await page.getByRole("button", { name: "Turn → Drive" }).click();
  await expect(page.locator(".key-hints")).toContainText("Drive");
});

test("first-run guide introduces the controls once", async ({ page }) => {
  await page.goto("/?token=e2e&quality=low");
  const guide = page.getByRole("dialog", { name: "How to control MABEL" });
  await expect(guide).toBeVisible({ timeout: 20000 });
  for (const title of ["Hands", "Look", "Drive", "Body"]) {
    await expect(guide.getByRole("heading", { name: title })).toBeVisible();
  }
  await guide.getByRole("button", { name: "Got it" }).click();
  await expect(guide).toBeHidden();
  await page.reload();
  await expect(page.getByRole("status")).toContainText(/Ready|Paused/);
  await expect(guide).toBeHidden(); // not shown again
  await page.getByRole("button", { name: "Help" }).click();
  await expect(guide).toBeVisible(); // reopened on demand
});

/**
 * Control robot-pose frames from the test while commands keep flowing:
 * `window.__dropState` drops them, `window.__delayStateMs` delivers each one late.
 */
async function canFreezePoses(page: Page) {
  await page.addInitScript(() => {
    type Hooks = { __dropState?: boolean; __delayStateMs?: number };
    const hooks = window as unknown as Hooks;
    const Native = window.WebSocket;
    const replayed = new WeakSet<Event>();
    class Freezable extends Native {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        if (!String(url).includes("/teleop/state")) return;
        this.addEventListener(
          "message",
          (event) => {
            if (replayed.has(event)) return;
            const delay = hooks.__delayStateMs ?? 0;
            if (hooks.__dropState || delay > 0) event.stopImmediatePropagation();
            if (!hooks.__dropState && delay > 0) {
              const copy = new MessageEvent("message", { data: event.data });
              replayed.add(copy);
              window.setTimeout(() => this.dispatchEvent(copy), delay);
            }
          },
          { capture: true },
        );
      }
    }
    // Route `onmessage` through addEventListener so the capture listener runs first.
    Object.defineProperty(Freezable.prototype, "onmessage", {
      set(handler: ((event: MessageEvent) => void) | null) {
        if (handler) this.addEventListener("message", handler);
      },
    });
    window.WebSocket = Freezable as typeof WebSocket;
  });
}

test("a frozen 3D view refuses engage and pauses an engaged session", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await canFreezePoses(page);
  await open(page);
  const pill = page.getByRole("status");
  // Commands would still flow, so only the client can tell its picture is old.
  await page.evaluate(() => ((window as unknown as { __dropState: boolean }).__dropState = true));
  await expect(page.locator(".topbar .chip-error")).toContainText("Stale", { timeout: 5000 });
  await page.keyboard.press("Enter");
  await expect(
    page.locator(".toast").filter({ hasText: "Can't engage: robot pose is" }),
  ).toHaveCount(1);
  await expect(pill).not.toContainText("Engaged");

  await page.evaluate(() => ((window as unknown as { __dropState: boolean }).__dropState = false));
  await expect
    .poll(() =>
      page.evaluate(() => (window as unknown as { __teleop: Debug }).__teleop.poseAgeMs()),
    )
    .toBeLessThan(100);
  await ensureEngaged(page);
  await page.evaluate(() => ((window as unknown as { __dropState: boolean }).__dropState = true));
  await expect(pill).toContainText("Paused", { timeout: 5000 });
  await expect(
    page.locator(".toast").filter({ hasText: /Paused: robot pose is \d+ ms old/ }),
  ).toHaveCount(1);
  expect((await status(page))!.state).toBe("paused");
});

test("the latency chip reports the view being steered by", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "same on all devices");
  await open(page);
  const chip = page.locator(".topbar .chip-ok, .topbar .chip-warn").first();
  await expect(chip).toContainText(/3D \d+ ms/);
  const latency = await page.evaluate(() =>
    (
      window as unknown as {
        __teleop: { latency: () => { poseShown: { p50: number } | null; clockErrorMs: number } };
      }
    ).__teleop.latency(),
  );
  expect(latency.poseShown!.p50).toBeGreaterThan(0);
  // Software rendering (E2E_GPU=0) has long frames, so the adaptive buffer grows.
  expect(latency.poseShown!.p50).toBeLessThan(process.env.E2E_GPU === "0" ? 1000 : 250);
  expect(latency.clockErrorMs).toBeLessThan(25);
});

test("the gate uses the pose the 3D view draws, not the newest one", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await canFreezePoses(page);
  await open(page);
  // Review repro: a steady 235 ms state delay keeps the newest pose under the
  // 250 ms limit, but the jitter buffer draws an older one.
  await page.evaluate(
    () => ((window as unknown as { __delayStateMs: number }).__delayStateMs = 235),
  );
  await expect(page.locator(".topbar .chip-error")).toContainText("Stale", { timeout: 5000 });
  await page.keyboard.press("Enter");
  await expect(
    page.locator(".toast").filter({ hasText: /Can't engage: robot pose is \d+ ms old/ }),
  ).toHaveCount(1);
  await expect(page.getByRole("status")).not.toContainText("Engaged");
});

test("engage checks a just-selected camera view that has no frame yet", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await open(page);
  // Hide the picture-in-picture so no camera frame is subscribed or drawn.
  await page.getByTitle("Hide picture-in-picture").click();
  await page.waitForTimeout(300);
  await page.keyboard.press("KeyV"); // main view → head camera
  await page.keyboard.press("Enter"); // immediately, before its first frame
  await expect(
    page.locator(".toast").filter({ hasText: "Can't engage: camera video is not shown yet" }),
  ).toHaveCount(1);
  await expect(page.getByRole("status")).not.toContainText("Engaged");
  // Once the camera draws, engaging works.
  await page.waitForTimeout(800);
  await ensureEngaged(page);
});

test("camera video negotiates H.264 where supported and falls back to JPEG", async ({
  page,
}, info) => {
  test.skip(info.project.name !== "desktop", "same on all devices");
  type Hooks = { __teleop: { videoCodec: () => string | null; videoPairs: () => number } };
  const codec = () => page.evaluate(() => (window as unknown as Hooks).__teleop.videoCodec());
  const pairs = () => page.evaluate(() => (window as unknown as Hooks).__teleop.videoPairs());
  await open(page);
  const h264 = await page.evaluate(async () =>
    typeof VideoDecoder === "undefined"
      ? false
      : (await VideoDecoder.isConfigSupported({ codec: "avc1.42E01F" })).supported === true,
  );
  await expect.poll(codec).toBe(h264 && !VIDEO ? "h264" : "jpeg");
  await expect.poll(pairs, { timeout: 10_000 }).toBeGreaterThan(5);

  await page.goto("/?token=e2e&quality=low&video=jpeg");
  await expect(page.locator(".loading")).toBeHidden({ timeout: 60_000 });
  await expect.poll(codec).toBe("jpeg");
  await expect.poll(pairs, { timeout: 10_000 }).toBeGreaterThan(5);
});
