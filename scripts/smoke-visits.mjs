import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const root = fileURLToPath(new URL("../", import.meta.url));
const qaRoot = path.join(root, "work", "qa");
await mkdir(qaRoot, { recursive: true });
const db = await mkdtemp(path.join(qaRoot, "visit-smoke-"));
const reserve = net.createServer();
await new Promise(resolve => reserve.listen(0, "127.0.0.1", resolve));
const port = reserve.address().port;
await new Promise(resolve => reserve.close(resolve));
const base = "http://127.0.0.1:" + port;
const env = {...process.env, SPORT_MODEL_DATA_DIR: db, SPORT_MODEL_SECURE_COOKIE: "0"};
for (const key of ["SPORT_MODEL_USERNAME", "SPORT_MODEL_PASSWORD", "SPORT_MODEL_DISPLAY_NAME"]) delete env[key];
const service = spawn(process.env.PYTHON || path.join(root, ".venv/bin/python"),
  ["-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", String(port)], {
    cwd: root, stdio: "inherit", env
  });
const closed = new Promise(resolve => { service.once("close", resolve); service.once("error", resolve); });
let browser;
try {
  for (let attempt = 0; attempt < 50; attempt++) {
    if (service.exitCode !== null) throw new Error("test service exited");
    if (await fetch(base).then(res => res.ok).catch(() => false)) break;
    await delay(150);
  }
  browser = await puppeteer.launch({executablePath: process.env.CHROME_BIN ||
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true,
    defaultViewport: {width: 1440,height: 960}});
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.setDefaultTimeout(9000);
  const visible = selector => page.waitForSelector(selector, {visible:true});
  const click = async selector => { await visible(selector); await page.click(selector); };
  const findText = text => page.waitForFunction(value => document.body.innerText.includes(value), {}, text);
  const formSubmit = selector => click(selector + ' button[type="submit"]');
  const records = () => page.evaluate(async () => (await fetch("/api/cases")).json());
  const home = async () => { await page.goto(base + "/#dashboard"); await visible("#new-visit"); };
  const create = async (name, comment) => {
    await home(); await click("#new-visit"); await visible("#visit-form");
    await page.type('#visit-form [name="horse_name"]', name);
    await page.type('#visit-form [name="comment"]', comment);
    await formSubmit("#visit-form"); await visible("#edit-visit");
    return (await records()).find(record => record.horse_name === name).id;
  };
  await page.goto(base);
  const loginMember = async username => {
    await visible("select[name=username]"); await page.select("[name=username]", username);
    await page.type("[name=password]", username + "sportsHorse"); await formSubmit("#login-form");
    await visible("#new-visit");
  };
  await loginMember("wangyiyi");
  await visible("#new-visit");
  const first = await create("归档用马", "初诊备注");
  await click("#edit-visit"); await visible('#edit-form [name="comment"]');
  await page.locator('#edit-form [name="comment"]').fill("复查备注");
  await formSubmit("#edit-form"); await findText("复查备注");
  await click("#archive-visit");
  await page.waitForFunction(async id => !!(await (await fetch("/api/cases/" + id)).json()).archived_at, {}, first);
  const second = await create("阅片用马", "候选区域核对");
  const attach = async file => {
    await click("#add-images");
    await visible("#attach-form");
    const uploader = await page.waitForSelector('#attach-form input[type="file"]');
    await uploader.uploadFile(path.join(root, "training/data/sporthorse_v2/images/test", file));
    await page.select('#attach-form [name="projection"]', file.includes("侧位") ? "侧位" : "正位");
    const check = '#attach-form [name="quality_confirmed"]';
    if (!await page.$eval(check, el => el.checked)) await click(check);
    await formSubmit("#attach-form");
    await findText(file);
    await page.waitForFunction(async (id, name) =>
      (await (await fetch("/api/cases/" + id)).json()).images.some(image => image.filename === name), {}, second, file);
  };
  await attach("16-侧位.jpg");
  await click("#logout"); await loginMember("zhushijia");
  assert.equal(await page.$eval("#display-name", el => el.textContent), "朱诗嘉");
  assert.deepEqual((await records()).map(record => record.id).sort(), [first, second].sort());
  await click(`tr[data-id="${second}"] .horse-cell`);
  await findText("16-侧位.jpg");
  const primaryImage = await page.$eval(".image-tile", async link => {
    const response = await fetch(link.href);
    const image = await createImageBitmap(await response.blob());
    return [response.status, image.width, image.height];
  });
  const sharedRecord = (await records()).find(record => record.id === second);
  assert.deepEqual(primaryImage, [200, sharedRecord.width, sharedRecord.height]);
  await click("#edit-visit"); await visible('#edit-form [name="comment"]');
  await page.locator('#edit-form [name="comment"]').fill("同事接诊，补片待复核");
  await formSubmit("#edit-form"); await findText("同事接诊，补片待复核");
  await attach("23-LF-正位.jpg");
  await formSubmit("#analyze-form"); await visible("#review-note");
  let entry = (await records()).find(record => record.id === second);
  assert.equal(entry.detections.length, 1);
  assert.equal(entry.images.length, 2);
  await page.type("#review-note", "测试复核完毕"); await formSubmit("#review-form");
  await findText("测试复核完毕"); await page.reload(); await findText("测试复核完毕");
  assert.equal((await records()).find(record => record.id === second).reviewer, "朱诗嘉");
  await click("#logout"); await loginMember("wangyiyi");
  await click(`tr[data-id="${second}"] .horse-cell`);
  await findText("同事接诊，补片待复核"); await findText("复核人：朱诗嘉");
  const signed = (await records()).find(record => record.id === second);
  assert.ok(signed.archived_at); assert.equal(signed.images.length, 2);
  await home(); await page.type("#case-search", "复查备注"); await visible(".row-select");
  await page.waitForFunction(() => document.querySelectorAll(".row-select").length === 1);
  await page.locator("#case-search").fill("");
  await page.select("#status-filter","reviewed");
  await page.waitForFunction(() => document.querySelectorAll(".row-select").length === 1 && document.querySelector("tbody").innerText.includes("阅片用马"));
  await page.select("#status-filter","all");
  await page.waitForFunction(() => document.querySelectorAll(".row-select").length === 2);
  await page.screenshot({path: path.join(qaRoot, "visit-dashboard.png"),fullPage:true});
  await page.setViewport({width:390,height:844}); await visible("#case-search");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth),390);
  await page.setViewport({width:1440,height:960});
  await click("#logout"); await loginMember("liqihang");
  const selectBoth = async () => {
    assert.equal(await page.$$eval(".row-select", picks => picks.length), 2);
    if (!await page.$eval("#select-page", item => item.checked)) await click("#select-page");
    await page.waitForFunction(() => document.querySelectorAll(".row-select:checked").length === 2);
  };
  await selectBoth(); await click("#batch-delete"); await click("#cancel-delete");
  assert.equal((await records()).length, 2);
  await selectBoth(); await click("#batch-delete"); await click("#confirm-delete");
  await page.waitForFunction(async () => (await (await fetch("/api/cases")).json()).length === 0);
  await page.reload(); await visible("#new-visit"); assert.equal((await records()).length,0);
  await click("#logout"); await loginMember("wangyiyi"); assert.deepEqual(await records(), []);
  await click("#new-visit"); await visible("#visit-form");
  await page.type('#visit-form [name="horse_name"]', "初诊随附原图");
  const picker = await page.waitForSelector('#visit-form [name="image"]');
  await picker.uploadFile(path.join(root, "training/data/sporthorse_v2/images/test/16-侧位.jpg"));
  await page.select('#visit-form [name="projection"]', "侧位");
  await click('#visit-form [name="quality_confirmed"]'); await formSubmit("#visit-form");
  await visible("#edit-visit"); assert.equal((await records())[0].images.length, 1);
  await home(); await click(".row-delete"); await click("#confirm-delete");
  await page.waitForFunction(async () => (await (await fetch("/api/cases")).json()).length === 0);
  assert.deepEqual(errors, []);
  console.log("PASS: shared team visits, cross-member edit and image access, supplemental image, live inference, signed review, archive, search, mobile layout and shared deletion.");
} finally {
  try { await browser?.close(); }
  finally { service.kill(); await closed; await rm(db, {recursive:true,force:true}); }
}
