import { expect, test, type Page } from "@playwright/test";

interface Debug {
  status: () => {
    state: string;
    target_epoch: number;
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
  await expect(page.locator("canvas")).toBeVisible();
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
