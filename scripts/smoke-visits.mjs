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
  const waitForAnalysis = () => page.waitForSelector("#open-report", {visible:true, timeout:60000});
  const click = async selector => { await visible(selector); await page.click(selector); };
  const findText = text => page.waitForFunction(value => document.body.innerText.includes(value), {}, text);
  const formSubmit = selector => click(selector + ' button[type="submit"]');
  const records = () => page.evaluate(async () => (await fetch("/api/cases")).json());
  const detail = id => page.evaluate(async key => (await fetch("/api/cases/" + key)).json(), id);
  const checkScreening = async (record, selector) => {
    const screening = record.screening;
    assert.match(screening.model, /^DINOv2-Small-refit84@/);
    assert.ok(Number.isFinite(screening.score) && Number.isFinite(screening.threshold));
    assert.equal(screening.predicted_positive, screening.score >= screening.threshold);
    const title = await page.$eval(selector + " .reading-title", element => element.innerText);
    const boxes = record.detections.length;
    assert.equal(title, (screening.predicted_positive ? "整片提示可疑" : "整片未触发提示") + "，" +
      (boxes ? `定位到 ${boxes} 处候选区域` : "未定位到候选区域"));
    assert.equal(await page.$eval(selector + " details", node => node.open), false);
    await click(selector + " summary");
    const evidence = await page.$eval(selector + " details", element => element.innerText);
    for (const value of [screening.score.toPrecision(5), screening.threshold.toPrecision(5), screening.model, record.model]) {
      assert.ok(evidence.includes(value));
    }
    await click(selector + " summary");
  };
  const checkReviewButton = async (count) => {
    await click("#go-to-primary");
    const view = await page.$eval(".film-stage.active-pane", el => {
      const bounds = el.getBoundingClientRect();
      return {id:el.dataset.imageId, focused:el === document.activeElement,
        inView:bounds.top < innerHeight && bounds.bottom > 0, boxes:el.querySelectorAll(".detection-boxes rect").length,
        labels:[...el.querySelectorAll(".box-label")].map(item=>item.innerText),
        original:document.querySelector("#viewer-original").getAttribute("aria-pressed")};
    });
    assert.deepEqual(view, {id:"primary", focused:true, inView:true, boxes:count,
      labels:Array.from({length:count}, (_,i)=>"AI 候选 " + String(i+1).padStart(2,"0")), original:String(!count)});
  };
  const checkBoxLabels = async () => {
    const errors = await page.$eval('.film-stage[data-image-id="primary"]', stage => {
      const labels = [...stage.querySelectorAll(".box-label")];
      return [...stage.querySelectorAll(".detection-boxes rect")].map((box,i) => {
        const region = box.getBoundingClientRect(), label = labels[i].getBoundingClientRect();
        return Math.max(Math.abs(label.left - region.left),Math.abs(label.bottom - region.top));
      });
    });
    assert.ok(errors.length && errors.every(delta => delta < 2), "Candidate numbers must line up with their own boxes");
  };
  const home = async () => {
    await click("#nav-archive");
    await visible("#case-search");
  };
  const fixture = name => path.join(root, "training/data/sporthorse_v2/images/test", name);
  const screenshot = name => page.screenshot({path: path.join(qaRoot, name + ".png")});
  const waitForArchive = id => page.waitForFunction(async key =>
    Boolean((await (await fetch("/api/cases/" + key)).json()).archived_at), {}, id);
  const openReport = async () => {
    await waitForAnalysis();
    await click("#open-report");
    await visible("#report-form");
  };
  const writeReport = async (findings, conclusion) => {
    await page.locator('#report-form [name="findings"]').fill(findings);
    await page.locator('#report-form [name="conclusion"]').fill(conclusion);
  };

  await page.goto(base);
  const branches = await page.evaluate(async () => {
    const {readingGuide} = await import("/static/reading-guide.js");
    const fixture = document.createElement("section");
    const outputs = [];
    for (const flag of [true,false,null]) {
      for (const count of [1,0]) {
        const screening = flag === null ? null : {predicted_positive:flag, score:flag ? .8 : .1, threshold:.5, model:"screen-test"};
        fixture.innerHTML = readingGuide({screening,model:"local-test",detections:count ? [{score:.7}] : []},String);
        outputs.push([fixture.querySelector("h3").textContent, fixture.querySelector(".reading-next").textContent,
          fixture.querySelector("dd").textContent.includes("本次历史分析未记录整片筛查")]);
      }
    }
    return outputs;
  });
  const expectedBranches = [
    ["整片提示可疑，定位到 1 处候选区域", /先核对主图上编号的 AI 候选框/, false],
    ["整片提示可疑，未定位到候选区域", /具体位置需要人工查找/, false],
    ["整片未触发提示，定位到 1 处候选区域", /定位模型单独给出了提示/, false],
    ["整片未触发提示，未定位到候选区域", /常规复核整张主图/, false],
    ["历史记录：定位到 1 处候选区域", /本次仅保存了定位结果/, true],
    ["历史记录：未定位到候选区域", /请查看整张原片/, true],
  ];
  branches.forEach(([title,next,historical],i) => {
    assert.equal(title,expectedBranches[i][0]);assert.match(next,expectedBranches[i][1]);assert.equal(historical,expectedBranches[i][2]);
  });
  const loginMember = async username => {
    await visible("select[name=username]");
    await page.select("[name=username]", username);
    await page.type("[name=password]", username + "sportsHorse");
    await formSubmit("#login-form");
    await visible("#visit-form");
    assert.equal(new URL(page.url()).hash, "#new");
  };
  const uploadPrimary = async (name, filename, sample = false) => {
    const form = "#visit-form";
    if (sample) {
      await click(`[data-sample="${filename}"]`);
    } else {
      const picker = await page.waitForSelector(form + ' [name="image"]');
      await picker.uploadFile(fixture(filename));
    }
    await visible("#image-preview");
    await page.locator(form + ' [name="horse_name"]').fill(name);
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
  await waitForAnalysis();
  const result = (await records()).find(record => record.id === second);
  assert.equal(result.detections.length, 2);
  assert.equal(result.threshold, 0.20);
  assert.match(result.model, /^YOLO11s-fetlock-640@/);
  assert.equal(result.images.length, 2);
  assert.equal(result.status, "analyzed");
  assert.equal(result.archived_at, null);
  assert.equal(await page.$("#archive-visit"), null);
  await checkScreening(result, "#analysis-summary");
  await click('.film-card[data-image-id="1"]');
  assert.equal(await page.$eval(".film-stage.active-pane", el=>el.dataset.imageId), "1");
  await click("#viewer-original");
  await click("#viewer-compare");
  await checkReviewButton(result.detections.length);
  assert.equal(await page.$eval(".film-stage.active-pane", el=>el.dataset.pane), "right");
  await checkBoxLabels();
  await click("#viewer-compare");
  await checkReviewButton(result.detections.length);
  await findText("主片复核区域");
  await click("[data-rename]");
  assert.equal(await page.$eval("#go-to-primary", el=>el.disabled), true);
  await click("#cancel-annotation");
  await click("#viewer-original");
  assert.equal(await page.$eval(".manual-marks",el=>getComputedStyle(el).display),"none");
  assert.equal(await page.$eval(".detection-boxes",el=>getComputedStyle(el).display),"none");
  assert.equal(await page.$eval(".measured-lines",el=>getComputedStyle(el).display),"none");
  await click("#viewer-original");
  assert.notEqual(await page.$eval(".detection-boxes",el=>getComputedStyle(el).display),"none");
  await checkBoxLabels();
  await page.$eval("#analysis-summary", el=>el.scrollIntoView({block:"start"}));
  await page.waitForSelector("#toast:empty");
  await screenshot("clinical-result");
  await click("#nav-reviews");
  await click(`#queue-list a[href="#report/${second}"]`);
  await visible("#report-form");
  await checkScreening(result, ".model-summary");
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
  const changeCase = (id, update) => page.evaluate(async (key, fields) =>
    (await fetch("/api/cases/" + key, {method:"PATCH", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(fields)})).status, id, update);
  assert.equal(await changeCase(second, {comment: draft.comment + "（复核前已更新）"}), 200);
  await click("#confirm-report");
  await click("#refresh-report");
  await findText("复核前已更新");
  assert.equal(await page.$eval('#report-form [name="conclusion"]', input => input.value), conclusion);
  await click("#confirm-report");
  await visible("#archive-visit");
  const downloadedReport = await page.$eval("#download-report", async link => {
    const response = await fetch(link.href);
    return {status:response.status, attachment:response.headers.get("content-disposition"), text:await response.text()};
  });
  assert.equal(downloadedReport.status, 200);
  assert.ok(downloadedReport.attachment.includes(second + "-report.html"));
  assert.ok(downloadedReport.text.includes(conclusion) && downloadedReport.text.includes("复核前已更新"));
  assert.equal(await changeCase(second, {comment: draft.comment}), 200);
  await page.reload();
  await findText("复核前已更新");
  await visible("#archive-visit");
  await findText(conclusion);
  const reviewed = await detail(second);
  assert.equal(reviewed.report.author, "朱诗嘉");
  assert.deepEqual(reviewed.screening, result.screening);
  await checkScreening(reviewed, ".model-summary");
  assert.match(await page.$eval(".reading-next",node=>node.innerText), /报告已签署/);
  await findText("人工标注 · 主片复核区域");
  await screenshot("clinical-reviewed");
  assert.equal(reviewed.reviewer, "王一伊");
  assert.ok(reviewed.reviewed_at);
  assert.deepEqual(reviewed.images[0].measurements, [measurements.distance]);
  assert.deepEqual(reviewed.images[1].measurements, [measurements.angle]);
  assert.equal(reviewed.status, "reviewed");
  assert.equal(reviewed.archived_at, null);
  await click(`a[href="#case/${second}"]`);
  await visible("[data-annotation-id]");
  assert.equal(await page.$("#go-to-primary"),null);
  assert.equal(await page.$eval('button[data-tool=rectangle]',el=>el.disabled),true);
  assert.equal(await page.$("[data-delete-annotation]"),null);
  assert.equal(await page.$("[data-rename]"),null);
  await page.setViewport({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
  await page.screenshot({path:path.join(qaRoot,"clinical-annotations-mobile.png"),fullPage:true});
  await page.setViewport({width:1440,height:960});
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
  const noFindingId = await uploadPrimary("零候选复核", "23-LF-正位.jpg", true);
  await formSubmit("#analyze-form");
  await waitForAnalysis();
  const noFinding = (await records()).find(record => record.id === noFindingId);
  assert.equal(noFinding.detections.length, 0);
  await checkScreening(noFinding, "#analysis-summary");
  console.log("Verified persisted screening and detections:", JSON.stringify([result,noFinding].map(record => ({
    film:record.images[0].filename, screening:record.screening, detector:record.model, boxes:record.detections.length
  }))));
  assert.equal(await page.$eval("#analysis-summary.no-detections", el => {
    const bounds = el.getBoundingClientRect();
    return el.innerText.includes("未定位到候选区域") && bounds.top >= 0 && bounds.bottom <= innerHeight
      && bounds.bottom <= document.querySelector("#viewer-host").getBoundingClientRect().top;
  }), true);
  await page.waitForSelector("#toast:empty");
  await screenshot("clinical-screening-disagreement");
  await page.setViewport({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),390);
  await screenshot("clinical-screening-disagreement-mobile");
  await checkReviewButton(0);
  await page.setViewport({width:1440,height:960});
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
  assert.match(await page.$eval(".reading-next",node=>node.textContent), /已确认本次人工所见与结论留空/);
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
  console.log("PASS: shared visits, primary-film review with and without candidate boxes, all four reading-guide branches and legacy records, two-film comparison, annotations, live inference, signatures, archive, search and mobile layout.");
} finally {
  try { await browser?.close(); }
  finally { service.kill(); await closed; await rm(db, {recursive:true,force:true}); }
}
