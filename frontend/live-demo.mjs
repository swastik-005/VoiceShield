import puppeteer from "puppeteer-core";
import path from "path";
import { mkdirSync } from "fs";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(dir, "demo-shots");
mkdirSync(out, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickText(page, includes) {
  return page.evaluate((needle) => {
    const b = [...document.querySelectorAll("button")].find((el) =>
      el.textContent.replace(/\s+/g, " ").includes(needle),
    );
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, includes);
}

const origin = process.env.DEMO_ORIGIN || "http://127.0.0.1:5173";

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: false,
  slowMo: 60,
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
    "--window-size=1100,900",
    "--no-first-run",
  ],
});

const page = await browser.newPage();
await browser.defaultBrowserContext().overridePermissions(origin, [
  "microphone",
  "notifications",
]);
await page.setViewport({ width: 1100, height: 860 });
await page.goto(origin, { waitUntil: "networkidle2", timeout: 20000 });

await page.click('[data-testid="app-icon"]');
await sleep(400);
await clickText(page, "Allow beep and notifications");
await sleep(500);
await clickText(page, "Allow access to contacts");
await sleep(400);
await clickText(page, "Continue to live monitor");
await sleep(2200);
await page.screenshot({ path: path.join(out, "demo-incoming.png") });
await clickText(page, "Answer");
await sleep(400);
await clickText(page, "Allow speaker and analyze");
await sleep(5000);
await page.screenshot({ path: path.join(out, "demo-in-call.png") });

const phase = await page.evaluate(() => document.body.innerText.slice(0, 400));
console.log("page text snippet:\n", phase.replace(/\s+/g, " ").slice(0, 300));
console.log("Leaving the demo window open for ~25s so you can watch the live call.");
await sleep(25000);
await browser.close();
