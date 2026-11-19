import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { checkCaseViewer } from "./check-case-viewer.mjs";

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
  const detail = id => page.evaluate(async key => (await fetch("/api/cases/" + key)).json(), id);
  const home = async () => {
    await click("#nav-archive");
    await visible("#case-search");
  };
  const fixture = name => path.join(root, "training/data/sporthorse_v2/images/test", name);
  const screenshot = name => page.screenshot({path: path.join(qaRoot, name + ".png")});
  const waitForArchive = id => page.waitForFunction(async key =>
    Boolean((await (await fetch("/api/cases/" + key)).json()).archived_at), {}, id);
  const openReport = async () => {
    await click("#open-report");
    await visible("#report-form");
  };
  const writeReport = async (findings, conclusion) => {
    await page.locator('#report-form [name="findings"]').fill(findings);
    await page.locator('#report-form [name="conclusion"]').fill(conclusion);
  };

  await page.goto(base);
  const loginMember = async username => {
    await visible("select[name=username]");
    await page.select("[name=username]", username);
    await page.type("[name=password]", username + "sportsHorse");
    await formSubmit("#login-form");
    await visible("#visit-form");
    assert.equal(new URL(page.url()).hash, "#new");
  };
  const uploadPrimary = async (name, filename) => {
    const form = "#visit-form";
    const picker = await page.waitForSelector(form + ' [name="image"]');
    await picker.uploadFile(fixture(filename));
    await visible("#image-preview");
    await page.type(form + ' [name="horse_name"]', name);
    await page.select(form + ' [name="projection"]', filename.includes("侧位") ? "侧位" : "正位");
    const quality = form + ' [name="quality_confirmed"]';
    if (!await page.$eval(quality, el => el.checked)) await click(quality);
    await formSubmit(form);
    await visible("#analyze-form");
    assert.equal(await page.$("#archive-visit"), null);
    return (await records()).find(record => record.horse_name === name).id;
  };
  await loginMember("wangyiyi");
  assert.deepEqual(await page.$$eval("#navigation nav a", links => links.map(link => link.textContent.trim())),
    ["新建检查","待分析","人工复核","病例归档","历史记录"]);
  await screenshot("clinical-new");
  await page.setViewport({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);
  await screenshot("clinical-new-mobile");
  await page.setViewport({width:1440,height:960});

  await formSubmit("#visit-form");
  assert.deepEqual(await records(), []);
  await click("#manual-visit");
  await visible("#manual-form");
  await page.type('#manual-form [name="horse_name"]', "归档用马");
  await page.type('#manual-form [name="comment"]', "初诊备注");
  await formSubmit("#manual-form");
  await visible("#edit-visit");
  const first = (await records())[0].id;
  await click("#edit-visit"); await visible('#edit-form [name="comment"]');
  await page.locator('#edit-form [name="comment"]').fill("复查备注");
  await formSubmit("#edit-form"); await findText("复查备注");
  await click("#archive-visit"); await waitForArchive(first);
  await click("#nav-work"); await visible("#visit-form");
  const second = await uploadPrimary("阅片用马", "16-侧位.jpg");
  const attach = async file => {
    await click("#viewer-upload");
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

  await click("#logout"); await loginMember("zhushijia");
  assert.equal(await page.$eval("#display-name", el => el.textContent), "朱诗嘉");
  await home();
  assert.deepEqual((await records()).map(record => record.id).sort(), [first, second].sort());
  await click("#nav-analysis");
  await click(`#queue-list a[href="#case/${second}"]`);
  await findText("16-侧位.jpg");
  const sharedRecord = (await records()).find(record => record.id === second);
  const primaryImage = await page.evaluate(async url => {
    const response = await fetch(url);
    const image = await createImageBitmap(await response.blob());
    return [response.status, image.width, image.height];
  }, sharedRecord.images[0].url);
  assert.deepEqual(primaryImage, [200, sharedRecord.width, sharedRecord.height]);
  await click("#edit-visit"); await visible('#edit-form [name="comment"]');
  await page.locator('#edit-form [name="comment"]').fill("同事接诊，补片待复核");
  await formSubmit("#edit-form"); await findText("同事接诊，补片待复核");
  await attach("23-LF-正位.jpg");
  const measurements = await checkCaseViewer(page, second);
  await formSubmit("#analyze-form");
  await visible("#open-report");
  const result = (await records()).find(record => record.id === second);
  assert.equal(result.detections.length, 2);
  assert.equal(result.threshold, 0.20);
  assert.match(result.model, /^YOLO11s-fetlock-640@/);
  assert.equal(result.images.length, 2);
  assert.equal(result.status, "analyzed");
  assert.equal(result.archived_at, null);
  assert.equal(await page.$("#archive-visit"), null);
  assert.match(await page.$eval("#analysis-summary.has-detections", el => el.innerText), /最高模型分数/);
  await screenshot("clinical-result");
  await click("#nav-reviews");
  await click(`#queue-list a[href="#report/${second}"]`);
  await visible("#report-form");
  const findings = "已逐张核对主片与附片，距离和角度已记录。";
  const conclusion = "记录本次运动马球节复查结果。";
  await writeReport(findings, conclusion);
  await click("#save-report-draft");
  await page.waitForFunction(async (id, text) =>
    (await (await fetch("/api/cases/" + id)).json()).report.findings === text, {}, second, findings);
  await page.reload();
  await visible("#report-form");
  assert.equal(await page.$eval('#report-form [name="findings"]', input => input.value), findings);
  assert.equal(await page.$eval('#report-form [name="conclusion"]', input => input.value), conclusion);
  const draft = await detail(second);
  assert.equal(draft.report.author, "朱诗嘉");
  assert.ok(draft.report.revision > 0); assert.equal(draft.archived_at, null);
  assert.equal(draft.status, "analyzed");
  await click("#logout"); await loginMember("wangyiyi");
  await home(); await click(`tr[data-id="${second}"] .horse-cell`);
  await openReport();
  await click("#confirm-report");
  await visible("#archive-visit");
  await page.reload();
  await visible("#archive-visit");
  await findText(conclusion);
  const reviewed = await detail(second);
  assert.equal(reviewed.report.author, "朱诗嘉");
  assert.equal(reviewed.reviewer, "王一伊");
  assert.ok(reviewed.reviewed_at);
  assert.deepEqual(reviewed.images[0].measurements, [measurements.distance]);
  assert.deepEqual(reviewed.images[1].measurements, [measurements.angle]);
  assert.equal(reviewed.status, "reviewed");
  assert.equal(reviewed.archived_at, null);
  await click("#nav-filing");
  await visible(`#queue-list a[href="#report/${second}"]`);
  await click('[data-filed="yes"]');
  await visible(`#queue-list a[href="#case/${first}"]`);
  assert.equal(await page.$(`#queue-list a[href="#report/${second}"]`), null);
  await click('[data-filed="no"]');
  await click(`#queue-list a[href="#report/${second}"]`);
  await click("#archive-visit");
  await waitForArchive(second);

  await click("#logout"); await loginMember("wangyiyi");
  await home(); await click(`tr[data-id="${second}"] .horse-cell`);
  await findText("同事接诊，补片待复核");
  await click("#open-report");
  await findText("朱诗嘉"); await findText("王一伊"); await findText(conclusion);
  const signed = await detail(second);
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
  await home(); await selectBoth(); await click("#batch-delete"); await click("#cancel-delete");
  assert.equal((await records()).length, 2);
  await selectBoth(); await click("#batch-delete"); await click("#confirm-delete");
  await page.waitForFunction(async () => (await (await fetch("/api/cases")).json()).length === 0);
  await page.reload(); await visible("#new-visit"); assert.equal((await records()).length,0);
  await click("#logout"); await loginMember("wangyiyi"); assert.deepEqual(await records(), []);
  const noFindingId = await uploadPrimary("零候选复核", "23-LF-正位.jpg");
  await formSubmit("#analyze-form");
  await visible("#open-report");
  const noFinding = (await records()).find(record => record.id === noFindingId);
  assert.equal(noFinding.detections.length, 0);
  assert.equal(await page.$eval("#analysis-summary.no-detections", el => {
    const bounds = el.getBoundingClientRect();
    return el.innerText.includes("未检出") && bounds.top >= 0 && bounds.bottom <= innerHeight
      && bounds.bottom <= document.querySelector("#viewer-host").getBoundingClientRect().top;
  }), true);
  await openReport(); await writeReport("　", " ");
  await click("#confirm-report"); await visible("#sign-empty-review");
  assert.equal((await detail(noFindingId)).status, "analyzed");
  await click("#cancel-empty-review");
  assert.equal(await page.$eval("#record-dialog", element => element.open), false);
  assert.equal((await detail(noFindingId)).reviewed_at, null);
  await click("#confirm-report"); await visible("#sign-empty-review"); await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.querySelector("#record-dialog").open);
  assert.equal((await detail(noFindingId)).report.revision, 0);
  await click("#confirm-report"); await click("#sign-empty-review"); await visible("#archive-visit");
  await findText("复核人已确认留空提交");
  assert.equal((await detail(noFindingId)).empty_review_confirmed, true);
  await click("#archive-visit"); await waitForArchive(noFindingId);
  await home(); await click(".row-delete"); await click("#confirm-delete");
  await page.waitForFunction(async () => (await (await fetch("/api/cases")).json()).length === 0);
  await click("#nav-work"); await visible("#visit-form");
  const singleFieldId = await uploadPrimary("单项复核", "23-LF-正位.jpg");
  await formSubmit("#analyze-form"); await openReport();
  await writeReport("", "需核对侧位片。");
  await click("#confirm-report"); await visible("#archive-visit");
  assert.equal((await detail(singleFieldId)).empty_review_confirmed, false);
  assert.equal((await detail(singleFieldId)).review_note, "需核对侧位片。");
  assert.deepEqual(errors, []);
  console.log("PASS: shared visits, two-film comparison, transformed pixel and angle measurements, live inference, report draft recovery, author/reviewer signatures, archive, search, mobile layout and shared deletion.");
} finally {
  try { await browser?.close(); }
  finally { service.kill(); await closed; await rm(db, {recursive:true,force:true}); }
}
