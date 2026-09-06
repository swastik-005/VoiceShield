import puppeteer from "puppeteer-core";
import path from "path";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const out = path.join(dir, "demo-shots");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function clickText(page, includes) {
  const ok = await page.evaluate((needle) => {
    const b = [...document.querySelectorAll("button")].find((el) =>
      el.textContent.replace(/\s+/g, " ").includes(needle),
    );
    if (!b || b.disabled) return false;
    b.click();
    return true;
  }, includes);
  if (!ok) console.log("click missed:", includes);
  return ok;
}

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
    "--window-size=900,1000",
    "--no-first-run",
  ],
});

const page = await browser.newPage();
await browser.defaultBrowserContext().overridePermissions("http://127.0.0.1:5173", [
  "microphone",
  "notifications",
]);
await page.setViewport({ width: 900, height: 1000 });

await page.goto("http://127.0.0.1:5173", { waitUntil: "networkidle2", timeout: 20000 });
await sleep(500);
await page.screenshot({ path: path.join(out, "0-home.png"), fullPage: true });

await page.click('[data-testid="app-icon"]');
await sleep(500);
await page.screenshot({ path: path.join(out, "1-permissions.png"), fullPage: true });

await clickText(page, "Allow beep and notifications");
await sleep(600);
await clickText(page, "Allow access to contacts");
await sleep(600);
await page.screenshot({ path: path.join(out, "2-permissions-asked.png"), fullPage: true });

await clickText(page, "Continue to live monitor");
await sleep(2500);
await page.screenshot({ path: path.join(out, "3-incoming.png"), fullPage: true });

await clickText(page, "Answer");
await sleep(500);
await page.screenshot({ path: path.join(out, "4-speaker.png"), fullPage: true });

await clickText(page, "Allow speaker and analyze");
await sleep(2000);
await page.screenshot({ path: path.join(out, "5-in-call.png"), fullPage: true });

await browser.close();
console.log("wrote", out);
