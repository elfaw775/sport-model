// 无头 Chrome 端到端冒烟测试：打开静态站 -> 上传图片 -> 等待浏览器内 ONNX 推理 -> 截图
import puppeteer from "puppeteer-core";
import path from "node:path";
import fs from "node:fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.TEST_URL || "http://127.0.0.1:8100/";
const IMG = process.argv[2] || path.resolve(process.cwd(), "../datasets/sporthorse/images/test/56-正位.jpg");

console.log("测试图片:", IMG, fs.existsSync(IMG) ? "✓" : "✗ 不存在");

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--no-sandbox"],
  defaultViewport: { width: 1280, height: 1000 },
});

const page = await browser.newPage();
const netLog = [];
page.on("console", (m) => {
  const t = m.text();
  if (t.includes("onnx") || t.includes("wasm") || t.includes("model") || m.type() === "error")
    console.log("[console]", m.type(), t.slice(0, 200));
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("requestfailed", (r) => console.log("[reqfail]", r.url().slice(-70), r.failure()?.errorText));
page.on("response", (r) => {
  const u = r.url();
  if (u.includes("model.onnx") || u.endsWith(".wasm") || u.includes("/ort/"))
    netLog.push(`${r.status()} ${u.slice(-55)}`);
});

try {
  await page.goto(URL, { waitUntil: "networkidle2", timeout: 60000 });
  await new Promise((r) => setTimeout(r, 1500));

  const input = await page.$('input[type="file"]');
  if (!input) throw new Error("找不到上传控件");
  await input.uploadFile(IMG);

  await page.waitForFunction(
    () => document.body.innerText.includes("检测结果") || document.body.innerText.includes("未检出"),
    { timeout: 180000 }
  );
  await new Promise((r) => setTimeout(r, 800));

  console.log("\n----- 网络请求 -----");
  console.log(netLog.join("\n") || "(无)");

  const text = await page.evaluate(() => document.body.innerText);
  console.log("\n----- 页面文本 -----");
  console.log(text.slice(0, 1500));

  fs.mkdirSync("scripts", { recursive: true });
  await page.screenshot({ path: "scripts/smoke.png" });
  console.log("\n✅ 截图已保存: scripts/smoke.png");
} finally {
  await browser.close();
}
