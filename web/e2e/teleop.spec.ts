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

const status = (page: Page) =>
  page.evaluate(() => (window as unknown as { __teleop: Debug }).__teleop.status());

async function open(page: Page) {
  await page.goto("/?token=e2e&quality=low");
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
  // Simulate a stalled client: stop the command timer by freezing JS timers.
  await page.evaluate(() => {
    const hi = window.setInterval(() => {}, 1e6);
    for (let i = 0; i <= hi; i++) window.clearInterval(i);
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

test("drive mode moves the base and the head follows clicks on the video", async ({
  page,
}, info) => {
  test.skip(info.project.name !== "desktop", "keyboard flow");
  await open(page);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Engaged");
  // Follow-hand is the default: the host tracks a gaze point, not fixed angles.
  await expect(page.locator(".key-hints .look-chip")).toContainText("Following hand");
  await expect.poll(async () => (await status(page))!.channels.head_mode).toBe("gaze");
  await page.keyboard.press("Tab"); // right hand → drive
  await expect(page.locator(".key-hints")).toContainText("Drive");
  await page.keyboard.down("KeyW");
  await expect
    .poll(async () => (await status(page))!.base.forward_mps, { timeout: 5000 })
    .toBeGreaterThan(0.05);
  await page.keyboard.up("KeyW");
  await expect.poll(async () => (await status(page))!.base.forward_mps).toBeLessThan(0.01);

  await page.getByRole("radio", { name: "Camera" }).click();
  const frame = page.locator(".camera-layer.main .eye-frame.aimable").first();
  await expect(frame).toBeVisible();
  const box = (await frame.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5); // look right
  await expect.poll(async () => (await status(page))!.channels.head.yaw_rad).toBeLessThan(-0.2);
});

test("body rails raise the lift and Turn switches to drive", async ({ page }, info) => {
  test.skip(info.project.name !== "desktop", "pointer flow");
  await open(page);
  await page.keyboard.press("Enter");
  await expect(page.getByRole("status")).toContainText("Engaged");
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
