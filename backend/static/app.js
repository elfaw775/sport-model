import { mountViewer } from "./viewer.js";
import { mountArchiveImages } from "./archive-images.js";
import { readingGuide } from "./reading-guide.js";

const page = document.querySelector("#page");
const browserDemo = document.documentElement.dataset.storage === "browser";
const byId = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const icon = (name) => '<svg aria-hidden="true"><use href="#' + name + '"/></svg>';
const statusNames = {registered:"仅登记",uploaded:"待分析", analyzed:"待复核", reviewed:"已复核"};
const statusBadge = (status) => '<span class="badge ' + (status in statusNames ? status : "") + '">' + escapeHTML(statusNames[status]) + '</span>';
const dateText = (value) => value ? new Intl.DateTimeFormat("zh-CN",{dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Shanghai",hour12:false}).format(new Date(value)) : "—";
const casePath = (id) => "./api/cases/" + encodeURIComponent(id);
const caseLink = (id) => "#case/" + encodeURIComponent(id);
const json = (data) => ({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
let user = null;
let revision = 0;
let noticeTimer;
let loginOptions = {};
let unmountViewer;

function notify(message) {
  clearTimeout(noticeTimer);
  byId("toast").textContent = message;
  noticeTimer = setTimeout(() => { byId("toast").textContent = ""; }, 3800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {cache:"no-store", ...options});
  const data = await response.json().catch(() => null);
  if (response.ok) return data;
  if (response.status === 401 && user) {
    user = null;
    render();
    notify("登录已过期，请重新登录。");
  }
  const detail = typeof data?.detail === "string" ? data.detail : (response.status === 422 ? "请检查就诊字段、投照方向及图片格式。" : "");
  throw Object.assign(new Error(detail || ("请求未完成（" + response.status + "）")), {status:response.status});
}

function formatError(error) {
  return error instanceof TypeError ? "无法连接工作台，请检查服务后重试。" : error.message;
}

function heading(kicker, title, desc, action = "") {
  return `<div class="heading"><div><p class="eyebrow">${kicker}</p><h1>${title}</h1><p class="sub">${desc}</p></div>${action}</div>`;
}

function action(form, pending, run) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (form.getAttribute("aria-busy") === "true") return;
    const ticket = revision;
    const error = form.querySelector(".error");
    if (error) error.hidden = true;
    const button = form.querySelector('button[type="submit"]');
    const oldText = button.innerHTML;
    button.disabled = true;
    form.setAttribute("aria-busy","true");
    button.textContent = pending;
    try {
      await run(() => ticket === revision);
    } catch (error) {
      if (ticket === revision) {
        let output=form.querySelector(".error");
        if(!output){output=document.createElement("p");output.className="error";output.setAttribute("role","alert");form.append(output);}
        output.hidden=false;output.textContent=formatError(error);
      }
    } finally {
      form.removeAttribute("aria-busy");
      button.disabled = false;
      button.innerHTML = oldText;
    }
  });
}

function login() {
  byId("crumb").textContent = "登录";
  const members = Object.entries(loginOptions);
  const usernameField = members.length
    ? '<select name="username" autocomplete="username" required><option value="">请选择成员账号</option>' +
      members.map(([account,name]) => '<option value="' + escapeHTML(account) + '">' + escapeHTML(name) + ' · ' + escapeHTML(account) + '</option>').join("") + '</select>'
    : '<input name="username" autocomplete="username" required placeholder="请输入账号">';
  page.innerHTML = `<div class="login-grid">
    <section class="intro"><p class="eyebrow">EQUINE IMAGING</p><h1>运动马<br><span>影像工作台</span></h1><p>上传球节 X 线片，<br>查看整片筛查和候选区域，完成人工复核。</p>
      <div class="intro-steps"><span><b>01</b>上传影像</span><span><b>02</b>模型检测</span><span><b>03</b>复核归档</span></div>
    </section>
    <section class="card login-card"><span class="tile-icon">${icon("mark")}</span><p class="eyebrow">WORKSPACE ACCESS</p><h2>登录工作台</h2><p class="sub">使用分配的账号进入影像诊疗与病例档案。</p>
      <a class="secondary casebook-entry" href="./static/examples.html">免登录查看示例病例 →</a>
      <form id="login-form"><label>账号${usernameField}</label><label>密码<input name="password" type="password" autocomplete="current-password" required placeholder="请输入密码"></label><p class="error" role="alert" hidden></p><button class="primary wide" type="submit">登录工作台 ${icon("arrow")}</button></form>
      ${members.length ? '<p class="credentials">团队试用密码：小写拼音账号后加 <code>sportsHorse</code>，注意大写 H。<br>' + (browserDemo ? '本机浏览器中的四个演示账号共享病例。<a href="./legacy/">查看旧版检测记录</a>' : '四位成员共享就诊记录，复核按登录姓名署名。') + '</p>' : ''}
    </section></div>`;
  const form = byId("login-form");
  action(form,"正在登录…",async (active) => {
    await api("./api/login",json(Object.fromEntries(new FormData(form))));
    const identity = await api("./api/me");
    if (active()) {
      user = identity;
      Object.assign(dashboardState, {query:"", filter:"all", page:1});
      uploadMessage = null;
      if (!location.hash) history.replaceState(null, "", "#new");
      render();
    }
  });
}

const escape = escapeHTML;
const dialog = byId("record-dialog");
const errorBox = '<p class="error" role="alert" hidden></p>';
const dashboardState = {query:"", filter:"all", page:1};
let uploadMessage = null;
const stateNames = {all:"全部状态",active:"未归档",archived:"已归档",registered:"仅登记",uploaded:"待分析",analyzed:"待复核",reviewed:"已复核"};
const archiveBadge = record => `<span class="badge ${record.archived_at ? "filed" : "open"}">${record.archived_at ? "已归档" : "未归档"}</span>`;
const newLink = '<a class="primary" href="#new" id="new-visit">' + icon("plus") + '新建检查</a>';

function visitFields(record={}) {
  return `<label>马匹名称 / 编号<input name="horse_name" id="horse-name" required maxlength="80" placeholder="例如：追风 或 H-001" value="${escape(record.horse_name)}"></label>
    <label>就诊备注 <span class="optional">选填</span><textarea name="comment" rows="4" maxlength="4000" placeholder="记录就诊原因、观察或后续安排">${escape(record.comment)}</textarea></label>`;
}
const projectionOptions = '<option value="">请选择投照方向</option><option>正位</option><option>侧位</option><option>斜位</option><option>其他</option>';
const fileTypes = "image/png,image/jpeg,image/webp";
const stages = ["上传影像", "模型分析", "查看结果", "人工复核", "确认复核", "病例归档"];
let previewURL = "";

function progressBar(current) {
  return `<ol class="flow-progress" aria-label="影像检查流程">${stages.map((label, index) => {
    const done = index < current;
    return `<li class="${done ? "done" : index === current ? "current" : ""}" ${index === current ? 'aria-current="step"' : ""}>
      <span class="flow-dot" aria-hidden="true">${done ? "✓" : index + 1}</span><span>${label}</span></li>`;
  }).join("")}</ol>`;
}

function clearPreview() {
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = "";
}

function imageFields() {
  return `<div class="file-field"><span class="file-symbol">${icon("upload")}</span><div><strong class="file-title">选择或拖入影像</strong><small>可多选 · JPG / PNG / WebP · 每张最多 10 MB</small></div>
    <input type="file" name="image" aria-label="补充影像文件" accept="${fileTypes}" multiple required></div>
    <div class="upload-options"><label>本批影像投照方向<select name="projection" required>${projectionOptions}</select></label>
    <label class="check"><input type="checkbox" name="quality_confirmed" value="true" required>我已确认影像清晰，且属于同一次球节检查。</label>
    <small class="dim">主图用于分析，同次检查的其他影像作为附图保存。</small></div>`;
}

function wireImages(form) {
  const input = form.elements.image;
  const preview = form.querySelector("#image-preview");
  preview?.addEventListener("error", () => {
    if (!input.files.length || !preview.hasAttribute("src")) return;
    preview.hidden = true;
    preview.closest(".image-drop").classList.remove("has-preview");
    input.setCustomValidity("无法读取所选主图，请重新选择 JPG、PNG 或 WebP 影像。");
    form.querySelector(".file-title").textContent = input.validationMessage;
  });
  input.addEventListener("change", () => {
    const files = Array.from(input.files);
    const file = files[0];
    const tooLarge = files.some(item => item.size > 10 * 1024 * 1024);
    input.setCustomValidity(tooLarge ? "每张影像最多 10 MB，请重新选择。" : "");
    const title = form.querySelector(".file-title");
    title.textContent = file ? file.name + (files.length > 1 ? " 等 " + files.length + " 张" : "") : "选择或拖入影像";
    if (preview) {
      clearPreview();
      const host = preview.closest(".image-drop");
      host.classList.toggle("has-preview", Boolean(file) && !tooLarge);
      preview.hidden = !file || tooLarge;
      if (preview.hidden) preview.removeAttribute("src");
      else {
        previewURL = URL.createObjectURL(file);
        preview.src = previewURL;
      }
      if (tooLarge) input.reportValidity();
    }
  });
}

function imagePayload(file, projection, quality) {
  const payload = new FormData();
  payload.set("image", file);
  payload.set("projection", projection);
  payload.set("quality_confirmed", String(quality));
  return payload;
}

async function attachFiles(id, files, projection, quality) {
  for (const [index, file] of files.entries()) {
    try {
      await api(casePath(id) + "/images", {method:"POST", body:imagePayload(file, projection, quality)});
    } catch (error) {
      return `未上传 ${files.length - index} 张影像。${file.name}：${formatError(error)}。请从“补充影像”重试。`;
    }
  }
  return "";
}

function createVisit() {
  const featured = [
    {id:16, file:"XR-014.jpg", alias:"16-侧位.jpg", projection:"侧位"},
    {id:23, file:"XR-027.jpg", alias:"23-LF-正位.jpg", projection:"正位", caption:"左前正位"},
    {id:56, file:"XR-057.jpg", projection:"正位"},
    {id:8, file:"XR-083.jpg", projection:"侧位", caption:"右前侧位"},
  ];
  const sampleChoices = featured.map(({id, file, alias, projection, caption}) => `
    <button type="button" class="sample-choice" data-id="${id}" data-sample="${alias || file}" data-projection="${projection}" aria-pressed="false">
      <img loading="lazy" decoding="async" alt="" src="./api/demo-images/${file}">
      <span><strong>${id} 号</strong><small>${caption || projection}</small></span>
    </button>`).join("");
  page.innerHTML = heading("新建检查", "上传检查影像", "选择下方体验样片或上传 X 线原片，核对影像后开始本次检查。") + progressBar(0) + `
    <form class="card intake" id="visit-form">
      <section class="intake-image"><div class="panel-title sample-header"><div><h2>检查影像</h2><small>首张为分析主图</small></div>
        <button type="button" class="secondary" id="draw-sample">${icon("image")}<span>随机抽取样片</span></button></div>
        <div class="image-drop">
          <div class="drop-corners" aria-hidden="true"></div>
          <img id="image-preview" alt="所选主图预览" hidden>
          <div class="drop-prompt"><span class="drop-icon">${icon("upload")}</span><strong>选择或拖入 X 线片</strong><span>JPG、PNG、WebP · 每张最多 10 MB</span></div>
          <div class="preview-hint">点击重新选择影像</div>
          <input id="image-file" type="file" name="image" aria-label="选择检查影像" accept="${fileTypes}" multiple required>
        </div>
        <div class="file-status">${icon("image")}<span class="file-title" aria-live="polite">尚未选择影像</span>
          <button type="button" class="text-button" id="clear-image" hidden>清除所选影像</button></div>
        <section id="featured-samples" aria-labelledby="sample-picker-title"><h3 id="sample-picker-title">选择体验样片</h3>
          <div class="sample-picker">${sampleChoices}</div></section>
      </section>
      <section class="intake-settings"><h2>检查信息</h2>
        <label>马匹名称 / 编号<input id="horse-name" name="horse_name" maxlength="80" required placeholder="例如：追风 或 H-001"></label>
        <label>投照方向<select name="projection" required>${projectionOptions}</select></label>
        <dl class="model-scope"><dt>检查部位</dt><dd>球节</dd>
          <dt><label for="lesion-focus">关注病变</label></dt>
          <dd><select id="lesion-focus" name="lesion_focus"><option value="骨折">骨折</option></select></dd></dl>
        ${browserDemo ? '<div class="model-preflight"><button type="button" class="text-button" id="prepare-models">提前准备模型</button><small role="status">首次需下载约 141 MB，可在上传影像前准备。</small></div>' : ""}
        <label class="check"><input type="checkbox" name="quality_confirmed" value="true" required>已核对马匹、方向和影像清晰度</label>
        ${errorBox}<button class="primary wide" type="submit"><span>上传影像</span> ${icon("arrow")}</button>
      </section>
    </form>
    <footer class="intake-foot"><span>暂时没有影像？ <a id="manual-visit" href="#manual">手动登记就诊</a></span><a href="#dashboard">${icon("book")} 查找历史记录</a></footer>`;
  const form = byId("visit-form");
  const prepare = byId("prepare-models");
  if (prepare) prepare.onclick = async () => {
    prepare.disabled = true;
    const status = prepare.nextElementSibling;
    try {
      const {prepareModels} = await import("../model-client.js");
      await prepareModels(text => { status.textContent = text; });
      prepare.textContent = "模型已就绪";
    } catch (error) {
      status.textContent = error.message;
      prepare.textContent = "重试准备模型";
      prepare.disabled = false;
    }
  };
  wireImages(form);
  const input = form.elements.image;
  const draw = byId("draw-sample");
  const recommendations = form.querySelectorAll("[data-sample]");
  const clear = byId("clear-image");
  const title = form.querySelector(".file-title");
  let catalog, currentSample, request, suggestedName = "";
  const horseName = form.elements.horse_name;
  horseName.addEventListener("input", () => { suggestedName = ""; });
  function refreshSamples() {
    const localFiles = input.files.length > 0 && !currentSample;
    draw.disabled = Boolean(request) || localFiles;
    draw.querySelector("span").textContent = currentSample ? "换一张样片" : "随机抽取样片";
    for (const button of recommendations) {
      const selected = currentSample && (button.dataset.sample === currentSample.filename ||
        button.querySelector("img").src === new URL(currentSample.url, document.baseURI).href);
      button.setAttribute("aria-pressed", String(Boolean(selected)));
    }
    clear.hidden = input.files.length === 0;
    if (!form.hasAttribute("aria-busy")) {
      const submit = form.querySelector('[type="submit"]');
      submit.disabled = Boolean(request);
      submit.querySelector("span").textContent = currentSample ? "用此样片新建检查" : "上传影像";
    }
  }
  input.addEventListener("change", () => {
    request?.abort();
    request = null;
    if (suggestedName && horseName.value === suggestedName) horseName.value = "";
    suggestedName = "";
    if (form.elements.projection.value === currentSample?.projection) form.elements.projection.value = "";
    currentSample = null;
    form.elements.quality_confirmed.checked = false;
    refreshSamples();
  });
  clear.onclick = () => {
    input.value = "";
    input.dispatchEvent(new Event("change"));
    input.focus();
  };
  async function loadSample(sample) {
    if (form.hasAttribute("aria-busy") || (!sample && input.files.length && !currentSample)) return;
    request?.abort();
    const selectionRequest = new AbortController();
    request = selectionRequest;
    const active = () => form.isConnected && request === selectionRequest;
    refreshSamples();
    title.textContent = "正在载入研究样片…";
    try {
      if (!sample) {
        catalog ??= await api("./api/demo-images", {signal:selectionRequest.signal});
        if (!active()) return;
        if (!catalog.length) throw new Error("暂无可用样片，请上传本地影像。");
        const pool = catalog.length > 1 ? catalog.filter(item => item.url !== currentSample?.url) : catalog;
        sample = pool[Math.floor(Math.random() * pool.length)];
      }
      const response = await fetch(sample.url, {signal:selectionRequest.signal});
      if (!response.ok) throw new Error(response.status === 401 ? "登录已过期，请重新登录。" : "样片载入失败，可重试或上传本地影像。");
      const blob = await response.blob();
      if (!active()) return;
      const selection = new DataTransfer();
      selection.items.add(new File([blob], sample.filename, {type:blob.type}));
      input.files = selection.files;
      input.dispatchEvent(new Event("change"));
      currentSample = {...sample, url:response.headers.get("Content-Location") || sample.url};
      form.elements.projection.value = sample.projection;
      if (!horseName.value.trim()) {
        suggestedName = sample.horse_name;
        horseName.value = suggestedName;
      }
      title.textContent = "已载入 " + sample.horse_name + " · " + sample.filename + (sample.projection ? "" : " · 请选择投照方向");
      refreshSamples();
    } catch (error) {
      if (active()) title.textContent = (input.files.length ? "已保留 " + input.files[0].name + "。" : "") + formatError(error);
    } finally {
      if (active()) {
        request = null;
        refreshSamples();
      }
    }
  }
  draw.onclick = () => loadSample();
  for (const button of recommendations) button.onclick = () => {
    const {sample: filename, projection} = button.dataset;
    loadSample({filename, projection, horse_name:"研究样片 " + button.dataset.id + " 号",
      url:"./api/demo-images/" + encodeURIComponent(filename)});
  };
  action(form, "正在上传…", async active => {
    const horseName = form.elements.horse_name.value.trim();
    if (!horseName) throw new Error("请填写马匹名称或编号。");
    const [mainImage, ...additional] = Array.from(form.elements.image.files);
    const projection = form.elements.projection.value;
    const quality = form.elements.quality_confirmed.checked;
    const payload = imagePayload(mainImage, projection, quality);
    payload.set("horse_name", horseName);
    const record = await api("./api/cases", {method:"POST", body:payload});
    const message = await attachFiles(record.id, additional, projection, quality);
    if (active()) {
      if (message) uploadMessage = {id:record.id, message};
      location.hash = caseLink(record.id);
      notify(message ? "主图已保存，请核对附图。" : "影像已上传。");
    }
  });
}

function manualVisit() {
  page.innerHTML = heading("就诊登记", "手动登记就诊", "先记录马匹和就诊备注，之后可在同一病例中补充影像。",
    '<a class="secondary" href="#new">返回影像检查</a>') + `
    <section class="card manual-card"><h2>就诊信息</h2><form id="manual-form">${visitFields()}${errorBox}
      <button type="submit" class="primary">保存登记 ${icon("arrow")}</button></form></section>`;
  const form = byId("manual-form");
  action(form, "正在登记…", async active => {
    const horse_name = form.elements.horse_name.value.trim();
    if (!horse_name) throw new Error("请填写马匹名称或编号。");
    const record = await api("./api/visits", json({horse_name, comment:form.elements.comment.value.trim()}));
    if (active()) {
      location.hash = caseLink(record.id);
      notify("就诊登记已保存。");
    }
  });
}
function openDialog(content, closeId) {
  dialog.innerHTML = content;
  dialog.showModal();
  byId(closeId).onclick = () => dialog.close();
}
dialog.addEventListener("cancel",event=>{
  if(dialog.querySelector('form[aria-busy="true"]')) event.preventDefault();
});
function editVisit(record, saved) {
  openDialog(`<p class="eyebrow">VISIT DETAILS</p><h2 id="dialog-title">编辑就诊记录</h2><p class="dim">${escape(record.id)}</p>
    <form id="edit-form">${visitFields(record)}${errorBox}<div class="dialog-actions"><button class="secondary" type="button" id="cancel-edit">取消</button><button class="primary" type="submit">保存修改</button></div></form>`,"cancel-edit");
  const form=byId("edit-form");
  action(form,"正在保存…",async active=>{
    const horse_name=form.elements.horse_name.value.trim(),comment=form.elements.comment.value.trim();
    if(!horse_name) throw new Error("请填写马匹名称或编号。");
    const cancel=byId("cancel-edit");cancel.disabled=true;
    try {
      const updated=await api(casePath(record.id),{...json({horse_name,comment}),method:"PATCH"});
      if(active()){dialog.close();saved(updated);notify("就诊信息已更新。");}
    } finally { cancel.disabled=false; }
  });
}
function deleteVisits(records, complete) {
  const single = records.length === 1;
  openDialog(`<span class="delete-symbol">${icon("trash")}</span><p class="eyebrow">DELETE RECORDS</p><h2 id="dialog-title">删除 ${records.length} 条就诊记录？</h2>
    <p class="sub">${single ? escape(records[0].horse_name) + " · " + escape(records[0].id) : escape(records.slice(0,3).map(record=>record.id).join("、")) + (records.length>3?" 等":"")}<br>所选记录的影像、备注、模型结果和复核意见将一起删除。${browserDemo ? "删除会影响本浏览器中的所有成员。" : Object.keys(loginOptions).length ? "删除对所有团队成员生效。" : ""}此操作不可撤销。</p>
    <form id="delete-form">${errorBox}<div class="dialog-actions"><button class="secondary" type="button" id="cancel-delete" autofocus>取消</button><button class="danger" type="submit" id="confirm-delete">确认删除 ${records.length} 条</button></div></form>`,"cancel-delete");
  const form = byId("delete-form");
  action(form,"正在删除…",async active=>{
    byId("cancel-delete").disabled = true;
    const removed = [], failed = [];
    for (const record of records) {
      try { await api(casePath(record.id),{method:"DELETE"});removed.push(record.id); }
      catch(error) { failed.push(`${record.id}（${formatError(error)}）`); }
    }
    if(active()) { dialog.close();complete(removed,failed); }
  });
}
const worklists = {
  analysis: {title:"待分析", icon:"scan", description:"已上传主图，等待球节病灶筛查和定位。", action:"打开分析", empty:"目前没有等待分析的影像"},
  reviews: {title:"人工复核", icon:"check", description:"核对整片筛查、定位结果与原片，签署报告。", action:"进入复核", empty:"目前没有待复核的病例"},
  filing: {title:"病例归档", icon:"folder", description:"办理归档，或查阅已经归档的病例及复核报告。", action:"办理归档", empty:"目前没有待归档的病例"}
};

function navRoute(record) {
  if (record.archived_at || record.status === "reviewed") return "filing";
  if (record.status === "uploaded") return "analysis";
  if (record.status === "analyzed") return "reviews";
  return "dashboard";
}

function selectNav(route) {
  for (const link of byId("navigation").querySelectorAll("nav a")) {
    const current = link.hash === "#" + route;
    link.classList.toggle("selected", current);
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

const reviewResultLabels = {
  all: "全部", "screen-only": "筛查可疑 · 无定位框", both: "筛查可疑 · 有定位框",
  "boxes-only": "仅定位提示", neither: "两项均未提示", legacy: "旧版无整片结果"
};

function reviewResultKey(record) {
  if (!record.screening) return "legacy";
  const hasBoxes = record.detections.length > 0;
  if (record.screening.predicted_positive) return hasBoxes ? "both" : "screen-only";
  return hasBoxes ? "boxes-only" : "neither";
}

function worklist(route, records) {
  const queue = worklists[route];
  const filing = route === "filing";
  const pending = records.filter(r => !r.archived_at && (
    filing ? r.status === "reviewed" || r.status === "registered" : r.status === (route === "analysis" ? "uploaded" : "analyzed")
  ));
  const archived = filing ? records.filter(r => r.archived_at) : [];
  let showArchived = filing && !pending.length && archived.length > 0;
  const reviewing = route === "reviews";
  let resultFilter = "all";
  const counts = {all: pending.length};
  if (reviewing) {
    for (const record of pending) {
      const key = reviewResultKey(record);
      counts[key] = (counts[key] || 0) + 1;
    }
  }
  page.innerHTML = heading("临床工作台", queue.title, queue.description, newLink) + `
    <section class="card queue-card"><header class="queue-header"><div class="queue-tabs">
      ${filing ? '<button type="button" data-filed="no">待归档 <b>'+pending.length+'</b></button><button type="button" data-filed="yes">已归档 <b>'+archived.length+'</b></button>'
        : '<h2>待处理病例 <span class="count-tag">'+pending.length+'</span></h2>'}</div>
      <label class="search-field">${icon("search")}<input type="search" id="queue-search" placeholder="搜索马匹或就诊编号" aria-label="搜索当前队列"></label></header>
      ${reviewing ? `<div class="review-filters" role="group" aria-label="按主图模型结果筛选">
        ${Object.entries(reviewResultLabels).map(([key,label]) => `<button type="button" data-review-filter="${key}" aria-pressed="${key === resultFilter}">${label} <b>${counts[key] || 0}</b></button>`).join("")}
      </div>` : ""}
      <div id="queue-list"></div></section>`;
  const query = byId("queue-search");
  const renderRows = () => {
    const term = query.value.trim().toLowerCase();
    const list = showArchived ? archived : pending;
    const shown = list.filter(r => (resultFilter === "all" || reviewResultKey(r) === resultFilter)
      && [r.id, r.horse_name].some(s => String(s).toLowerCase().includes(term)));
    const filtering = Boolean(term) || resultFilter !== "all";
    byId("queue-list").innerHTML = shown.length ? shown.map(r => {
      const openReport = route === "reviews" || (filing && r.status === "reviewed");
      const action = showArchived ? (openReport ? "查看报告" : "查看病例") : queue.action;
      const status = showArchived ? "已归档" : filing ? (r.status === "registered" ? "仅登记" : "已确认复核") : route === "reviews" && r.report?.revision ? "已有复核草稿" : statusNames[r.status];
      const time = showArchived ? r.archived_at : r.reviewed_at || r.analyzed_at || r.created_at;
      const resultKey = reviewing ? reviewResultKey(r) : null;
      const evidence = reviewing ? `<span class="queue-signal" data-signal="${resultKey}">${reviewResultLabels[resultKey]}${resultKey === "legacy" ? " · " + r.detections.length + " 个定位框" : ""}</span>` : "";
      return `<article class="queue-row"><span class="horse-avatar" aria-hidden="true">${escape(Array.from(r.horse_name)[0])}</span><div class="queue-person"><strong>${reviewing ? `<a href="${caseLink(r.id)}" title="打开病例原片">${escape(r.horse_name)}</a>` : escape(r.horse_name)}</strong><small>${escape(r.id)}</small>${evidence}</div>
        <div class="queue-meta"><span>${escape(status)}</span><small>${r.images.length} 张影像 · ${escape(dateText(time))}</small></div>
        <a class="secondary" href="${openReport ? reportLink(r.id) : caseLink(r.id)}">${action} ${icon("arrow")}</a></article>`;
    }).join("") : `<div class="empty large-empty"><span class="tile-icon">${icon(queue.icon)}</span><h3>${filtering ? "没有匹配的病例" : showArchived ? "目前没有已归档的病例" : queue.empty}</h3>${filtering ? '<button class="secondary" id="clear-queue">清除筛选</button>' : '<a class="secondary" href="#dashboard">查看历史记录</a>'}</div>`;
    byId("clear-queue")?.addEventListener("click", () => { query.value=""; resultFilter="all"; renderRows(); query.focus(); });
    for (const tab of page.querySelectorAll("[data-review-filter]")) tab.setAttribute("aria-pressed", String(tab.dataset.reviewFilter === resultFilter));
    for (const tab of page.querySelectorAll("[data-filed]")) tab.setAttribute("aria-pressed", String((tab.dataset.filed === "yes") === showArchived));
  };
  query.oninput = renderRows;
  for (const tab of page.querySelectorAll("[data-review-filter]")) tab.onclick = () => { resultFilter = tab.dataset.reviewFilter; renderRows(); };
  for (const tab of page.querySelectorAll("[data-filed]")) tab.onclick = () => { showArchived = tab.dataset.filed === "yes"; renderRows(); };
  renderRows();
}

function archive(records) {
  const selected = new Set(), perPage=8;
  const stats = [
    ["book","全部就诊",records.length,"登记在册的记录","fern"],
    ["image","已收影像",records.reduce((n,r)=> n+(r.images?.length||0),0),"就诊记录中的全部影像","slate"],
    ["clock","待人工复核",records.filter(r=>r.status==="analyzed").length,"已完成模型分析","sand"],
    ["check","归档记录",records.filter(r=>r.archived_at).length,"已完成手动或复核归档","olive"],
  ];
  const description = browserDemo ? "本浏览器保存的检查与归档报告，同一浏览器内的四位成员可共同查看。" :
    Object.keys(loginOptions).length ? "连接本服务的团队成员共享就诊记录、检查影像和归档报告。" : "集中管理就诊记录、检查影像和归档报告。";
  page.innerHTML = heading("CLINICAL WORKSPACE","就诊记录看板",description,newLink) + `
    <aside class="casebook-shelf" aria-label="公开只读报告"><strong>公开示例报告</strong><a href="./static/examples.html#18">18 号 · 球节正位片 →</a><a href="./static/examples.html#75">75 号 · 球节正位片 →</a></aside>
    <div class="metric-grid">${stats.map(([symbol,title,count,note,tone])=>`<div class="metric ${tone}"><span class="metric-icon">${icon(symbol)}</span><span class="metric-title">${title}</span><strong>${count}</strong><small>${note}</small></div>`).join("")}</div>
    <section class="card ledger"><header class="ledger-head"><div><p class="eyebrow">VISIT REGISTRY</p><h2>就诊记录 <span class="count-tag">${records.length}</span></h2></div><div class="tools"><label class="search-field">${icon("search")}<input type="search" id="case-search" placeholder="搜索马匹、编号或备注" aria-label="搜索就诊记录"></label><select id="status-filter" aria-label="筛选记录状态">${Object.entries(stateNames).map(([key,label])=>`<option value="${key}">${label}</option>`).join("")}</select></div></header>
    <div class="selection-bar"><span id="match-count"></span><div class="row-actions"><button type="button" class="text-button" id="refresh-records">刷新记录</button><button type="button" class="text-danger" id="batch-delete" disabled>${icon("trash")} 批量删除</button></div></div>
    <p id="batch-feedback" role="status" hidden></p><div id="case-list"></div><footer class="pager" id="pager"></footer></section>`;
  const query=byId("case-search"),filter=byId("status-filter");
  byId("refresh-records").onclick=render;
  query.value=dashboardState.query;filter.value=dashboardState.filter;
  function refresh() {
    const term=query.value.trim().toLowerCase(),key=filter.value;
    const matches=records.filter(r=>[r.id,r.horse_name,r.comment].some(v=>String(v??"").toLowerCase().includes(term)) && (key==="all" || (key==="archived"?Boolean(r.archived_at):key==="active"?!r.archived_at:r.status===key)));
    const pages=Math.max(1,Math.ceil(matches.length/perPage));dashboardState.page=Math.max(1,Math.min(pages,dashboardState.page));
    const shown=matches.slice((dashboardState.page-1)*perPage,dashboardState.page*perPage);
    byId("match-count").textContent=selected.size ? `已选择 ${selected.size} 条记录` : `共 ${matches.length} 条记录`;
    byId("batch-delete").disabled=!selected.size;
    byId("case-list").innerHTML=shown.length ? `<div class="table-scroll"><table><thead><tr><th><input type="checkbox" id="select-page" aria-label="全选本页记录"></th><th>马匹 / 就诊编号</th><th>就诊备注</th><th>影像</th><th>状态</th><th>登记时间</th><th>操作</th></tr></thead><tbody>${shown.map(record=>`<tr data-id="${escape(record.id)}"><td><input type="checkbox" class="row-select" aria-label="选择 ${escape(record.id)}" ${selected.has(record.id)?"checked":""}></td>
    <td><a class="horse-cell" href="${caseLink(record.id)}"><span class="horse-avatar">${escape(Array.from(record.horse_name)[0])}</span><span><strong>${escape(record.horse_name)}</strong><small>${escape(record.id)}</small></span></a></td>
    <td class="comment-cell" title="${escape(record.comment)}">${escape(record.comment||"暂无备注")}</td><td>${record.images?.length||0} 张</td><td>${archiveBadge(record)}<small>${escape(statusNames[record.status])}</small></td><td>${escape(dateText(record.created_at))}</td>
    <td><div class="row-actions"><a href="${caseLink(record.id)}">查看</a><button type="button" class="text-button row-edit">编辑</button><button type="button" class="text-danger row-delete">删除</button></div></td></tr>`).join("")}</tbody></table></div>` : `<div class="empty large-empty"><span class="tile-icon">${icon("book")}</span><h3>${records.length?"没有匹配的就诊记录":"从第一条就诊记录开始"}</h3><p>${records.length?"请更换搜索词或筛选条件。":"登记马匹和就诊备注即可保存，稍后也能补充影像。"}</p>${records.length?'<button class="secondary" id="reset-search">清除筛选</button>':'<a class="secondary" href="#new">新建第一条记录</a>'}</div>`;
    byId("pager").innerHTML=`<span>每页 ${perPage} 条 · 第 ${dashboardState.page} / ${pages} 页</span><div><button class="secondary" id="prev-page" ${dashboardState.page<=1?"disabled":""}>上一页</button><button class="secondary" id="next-page" ${dashboardState.page>=pages?"disabled":""}>下一页</button></div>`;
    if (shown.length) {
      const master=byId("select-page");
      master.checked=shown.every(r=>selected.has(r.id));master.indeterminate=!master.checked&&shown.some(r=>selected.has(r.id));
      master.onchange=()=>{shown.forEach(r=>master.checked?selected.add(r.id):selected.delete(r.id));refresh();};
    }
    for(const row of byId("case-list").querySelectorAll("tr[data-id]")) {
      const rec=shown.find(r=>r.id===row.dataset.id);
      row.querySelector(".row-select").onchange=event=>{event.target.checked?selected.add(rec.id):selected.delete(rec.id);refresh();};
      row.querySelector(".row-edit").onclick=()=>editVisit(rec,updated=>archive(records.map(r=>r.id===updated.id?updated:r)));
      row.querySelector(".row-delete").onclick=()=>remove([rec]);
    }
    byId("prev-page").onclick=()=>{dashboardState.page--;refresh();};byId("next-page").onclick=()=>{dashboardState.page++;refresh();};
    if(byId("reset-search"))byId("reset-search").onclick=()=>{query.value="";filter.value="all";change();};
  }
  function change() { selected.clear();dashboardState.query=query.value;dashboardState.filter=filter.value;dashboardState.page=1;refresh(); }
  function remove(items) {
    deleteVisits(items,(removed,failed)=>{
      archive(records.filter(r=>!removed.includes(r.id)));
      const feedback=byId("batch-feedback");
      feedback.textContent=`已删除 ${removed.length} 条记录。`+(failed.length?`删除失败：${failed.join("、")}。可以重新选择后重试。`:"");
      feedback.className=failed.length?"error":"success";feedback.hidden=false;
    });
  }
  query.oninput=change;filter.onchange=change;byId("batch-delete").onclick=()=>remove(records.filter(r=>selected.has(r.id)));
  refresh();
}

const archiveButton = '<button id="archive-visit" class="primary" type="button">' + icon("folder") + ' 归档病例</button>';
const reportLink = id => "#report/" + encodeURIComponent(id);

function archiveAction(record, refresh) {
  byId("archive-visit")?.addEventListener("click", async event => {
    const button = event.currentTarget;
    const ticket = revision;
    button.disabled = true;
    try {
      const updated = await api(casePath(record.id) + "/archive", {method:"POST"});
      if (ticket === revision) { refresh(updated); notify("病例已归档。"); }
    } catch (error) { if (ticket === revision) notify(formatError(error)); }
    finally { button.disabled = false; }
  });
}

function enableAttachment(form, record, onSaved) {
  wireImages(form);
  action(form, "正在上传…", async active => {
    const message = await attachFiles(record.id, Array.from(form.elements.image.files), form.elements.projection.value, form.elements.quality_confirmed.checked);
    const updated = await api(casePath(record.id));
    if (active()) {
      if (message) uploadMessage = {id:record.id, message};
      if (dialog.open) dialog.close();
      onSaved(updated);
      if (!message) notify("影像已保存。");
    }
  });
}

function analysisSummary(record, analyzed, signed) {
  const boxes = record.detections || [];
  const needsReview = !signed && (record.screening?.predicted_positive || boxes.length > 0);
  const action = analyzed
    ? `<nav class="reading-actions" aria-label="阅片操作">${signed ? "" : `<button type="button" id="go-to-primary" class="primary">${boxes.length ? "查看候选框" : "核对主图"}</button>`}
       <a href="${reportLink(record.id)}" id="open-report" class="${signed ? "primary" : "secondary"}">${icon("book")} ${signed ? "查看复核报告" : record.report?.revision ? "继续复核报告" : "编写复核报告"}</a></nav>`
    : `<form id="analyze-form"><button class="primary" type="submit">${icon("scan")} 开始分析</button>${errorBox}</form>`;
  return `<section class="case-result-strip ${analyzed ? boxes.length ? "has-detections" : "no-detections" : "awaiting-analysis"} ${needsReview ? "review-prompt" : ""}" id="analysis-summary" aria-label="主图阅片提示">
    <span class="tile-icon">${icon("scan")}</span><div><div class="result-meta"><b>主图阅片提示</b><span id="analysis-state">${analyzed ? "分析完成 · "+escape(dateText(record.analyzed_at)) : "待分析"}</span></div>
    ${analyzed ? readingGuide(record, escapeHTML) : '<h2 id="analysis-title">主图已上传，等待分析</h2><p id="analysis-lead">分析后可查看整片是否提示可疑，以及主图上的候选区域。</p>'}</div>${action}</section>`;
}

function detail(record) {
  unmountViewer?.();
  unmountViewer = undefined;
  const base = casePath(record.id);
  const analyzed = ["analyzed", "reviewed"].includes(record.status);
  const signed = record.status === "reviewed";
  const creator = loginOptions[record.created_by] || record.created_by;
  const hasImage = record.has_image;
  const stage = signed ? (record.archived_at ? 6 : 5) : analyzed ? 2 : 1;
  selectNav(navRoute(record));
  const notice = uploadMessage?.id === record.id ? uploadMessage.message : "";
  if (notice) uploadMessage = null;
  page.innerHTML = `<div class="clinical-page-head"><div><a class="back-link" href="#dashboard">‹ 历史记录</a>
    <div class="case-heading"><h1>${escape(record.horse_name)}</h1><span>${hasImage ? "影像工作台" : "就诊登记"}</span><code>${escape(record.id)}</code></div></div></div>
    ${notice ? `<p class="error" role="alert">${escape(notice)}</p>` : ""}
    ${record.archived_at && !signed && hasImage ? '<p class="legacy-note">历史病例已归档，人工复核尚待完成。</p>' : ""}
    ${hasImage ? progressBar(stage) + analysisSummary(record, analyzed, signed) + (record.archived_at ? '<div id="archive-images"></div>' : '') + '<div id="viewer-host"></div>' : ""}
    <section class="card visit-summary"><div class="spread"><div class="panel-title"><h2>就诊信息</h2>${archiveBadge(record)}</div>
      <div class="row-actions"><button id="edit-visit" class="secondary">编辑记录</button>${!hasImage && !record.archived_at ? archiveButton : ""}</div></div>
      <p class="visit-comment">${escape(record.comment || "暂无就诊备注")}</p><div class="detail-meta"><span>${icon("user")} 登记人：${escape(creator)}</span><span>${escape(dateText(record.created_at))}</span>${record.archived_at ? `<span>归档于 ${escape(dateText(record.archived_at))}</span>` : ""}</div>
      ${!hasImage ? '<button class="secondary" id="add-images">上传首张影像</button>' : ""}</section>`;
  const add = () => {
    openDialog(`<h2 id="dialog-title">补充就诊影像</h2><p class="dim">${escape(record.id)} · ${escape(record.horse_name)}</p>
      <form id="attach-form">${imageFields()}${errorBox}<div class="dialog-actions"><button type="button" class="secondary" id="cancel-attach">取消</button><button type="submit" class="primary">保存影像</button></div></form>`, "cancel-attach");
    enableAttachment(byId("attach-form"),record,refreshDetail);
  };
  const refreshDetail = updated => updated.archived_at ? render() : detail(updated);
  mountArchiveImages(byId("archive-images"),record,escapeHTML,openDialog);
  byId("edit-visit").onclick = () => editVisit(record,refreshDetail);
  byId("add-images")?.addEventListener("click",add);
  archiveAction(record,refreshDetail);
  if (hasImage) unmountViewer = mountViewer(byId("viewer-host"), record, {api,onUpload:add,reviewButton:byId("go-to-primary"),
    onReport:() => { if (analyzed) location.hash = reportLink(record.id); else byId("analyze-form").requestSubmit(); }});
  if (byId("analyze-form")) action(byId("analyze-form"),"模型分析中…",async active => {
    const summary = byId("analysis-summary");
    summary.classList.remove("analysis-failed");
    byId("analysis-state").textContent = "分析中";
    byId("analysis-title").textContent = "正在分析主图…";
    byId("analysis-lead").textContent = browserDemo ? "首次分析需要下载约 126 MB 模型。请保持页面打开，模型将在本机浏览器运行。" : "正在进行整片筛查和病灶定位。";
    let updated;
    try {
      if (browserDemo) {
        const {prepareModels} = await import("../model-client.js");
        const lead = byId("analysis-lead");
        await prepareModels(text => { if (active()) lead.textContent = text; });
        if (!active()) return;
        lead.textContent = "正在对主图进行整片筛查和病灶定位。";
      }
      updated = await api(base+"/analyze",{method:"POST"});
    } catch (error) {
      if (!active()) return;
      // A concurrent analysis or a lost response can leave a completed result on the server.
      updated = await api(base).catch(() => null);
      if (!active()) return;
      if (!updated || !["analyzed","reviewed"].includes(updated.status)) {
        summary.classList.add("analysis-failed");
        byId("analysis-state").textContent = "可重试";
        byId("analysis-title").textContent = "暂未取得分析结果";
        byId("analysis-lead").textContent = "已上传的主图保留在病例中，可再次点击「开始分析」。";
        throw error;
      }
    }
    if (active()) { detail(updated); window.scrollTo(0,0); notify("分析完成，请核对原片与模型结果。"); }
  });
}

function reportPage(record) {
  selectNav(navRoute(record));
  if (!["analyzed","reviewed"].includes(record.status)) {
    location.replace(caseLink(record.id));
    return;
  }
  const signed = record.status === "reviewed";
  if (signed && record.signed_evidence) {
    record = {...record, ...record.signed_evidence, archived_at:record.archived_at};
  }
  const report = record.report || {};
  const images = record.images || [];
  const priorFindings = report.findings || "";
  const priorConclusion = report.conclusion || (signed ? record.review_note : "") || "";
  const formBody = signed ? `<section class="signed-section">${record.empty_review_confirmed
    ? '<p>复核人已确认留空提交，未填写人工所见或结论。</p>'
    : `<h3>影像所见</h3><p>${escape(priorFindings || "未填写")}</p><h3>复核结论</h3><p>${escape(priorConclusion || "未填写")}</p>`}
    <div class="report-signature">${icon("check")} ${escape(record.reviewer)} 已确认 <time>${escape(dateText(record.reviewed_at))}</time></div>
    ${record.signed_evidence ? "" : '<p class="dim">历史报告依据当前留存的病例资料整理。</p>'}
    <a href="${casePath(record.id)}/report/download" class="secondary" id="download-report" download>下载签署报告（HTML）</a></section>` : `<form id="report-form">
    <label>影像所见（选填）<textarea name="findings" rows="4" maxlength="4000" placeholder="记录不同投照位观察到的影像特征、位置与测量情况">${escape(priorFindings)}</textarea></label>
    <label>复核结论（选填）<textarea name="conclusion" rows="3" maxlength="2000" placeholder="记录人工判断及需要进一步核查的内容">${escape(priorConclusion)}</textarea></label>
    <p class="dim" id="draft-status" role="status">${report.updated_at ? "最近保存 · "+escape(report.author)+" · "+escape(dateText(report.updated_at)) : "草稿可保存，下次打开时继续编辑。"}</p>
    ${errorBox}<div class="report-buttons"><button class="secondary" type="submit" id="save-report-draft">保存草稿</button><button class="primary" id="confirm-report" type="submit">${icon("check")} 确认复核</button></div></form>`;
  page.innerHTML = `<header class="clinical-page-head"><div><a class="back-link" href="${caseLink(record.id)}">‹ 返回阅片</a><div class="case-heading">
    <h1>人工复核报告</h1><code>${escape(record.id)}</code></div></div>${signed ? (record.archived_at ? '<a href="#dashboard" class="secondary">查找历史记录</a>' : archiveButton) : ""}</header>
    ${progressBar(signed ? (record.archived_at ? 6 : 5) : 3)}
    <div class="report-layout"><article class="card structured-report"><div class="report-topline"><span>EQUIVISION / 影像科</span>${record.archived_at ? archiveBadge(record) : signed ? statusBadge("reviewed") : '<span class="badge analyzed">待人工确认</span>'}</div>
      <h2>${escape(record.horse_name)} <span>· 球节影像检查</span></h2>
      <div class="report-facts"><span>检查日期 <b>${escape(dateText(record.created_at))}</b></span><span>检查影像 <b>${images.length} 张</b></span>
        ${signed && report.author && report.author !== record.reviewer ? `<span>草稿作者 <b>${escape(report.author)}</b></span>` : ""}
        <span>复核人 <b>${escape(signed ? record.reviewer : user.display_name)}</b></span></div>
      ${record.comment ? `<p class="visit-context"><strong>就诊备注</strong>　${escape(record.comment)}</p>` : ""}
      ${formBody}${record.archived_at ? `<p class="dim">归档时间 · ${escape(dateText(record.archived_at))}</p>` : ""}</article>
    <aside class="report-aside"><section class="card"><div class="panel-title"><h2>影像、标注与测量</h2><a href="${caseLink(record.id)}">返回阅片</a></div>
      ${images.map((im,i)=>`<div class="report-evidence"><img src="${escape(im.url)}" alt=""><div><b>${i===0?"主图":"附图 "+i} · ${escape(im.projection)}</b><small>${escape(im.filename)}</small><small>影像质量 · ${escape(im.quality || "未确认")}</small>
      ${(im.measurements || []).map(m=>`<span>${m.kind==="distance"?"距离":"角度"}　${Number(m.value).toFixed(1)} ${m.kind==="distance"?"px":"°"}</span>`).join("")}
      ${(im.annotations || []).map(a=>`<span>人工标注 · ${escape(a.label)}（${escape(a.updated_by)}）</span>`).join("")}</div></div>`).join("")}</section>
      <section class="card model-summary"><h2>主图 AI 辅助结果</h2>${readingGuide(record, escapeHTML)}</section></aside></div>${signed ? '<div id="archive-images"></div>' : ''}`;
  mountArchiveImages(byId("archive-images"),record,escapeHTML,openDialog);
  archiveAction(record,render);
  const form = byId("report-form");
  if (!form) return;
  let version = report.revision || 0;
  const save = async (confirm) => {
    const findings = form.elements.findings.value.trim(), conclusion = form.elements.conclusion.value.trim();
    const body = {findings,conclusion,revision:version};
    if (confirm) {
      body.annotation_revision = record.annotation_revision || 0;
      body.review_context = record.review_context;
    }
    if (!findings && !conclusion) {
      if (!confirm) throw new Error("请先填写一项草稿内容。");
      openDialog(`<h2 id="dialog-title">留空提交复核？</h2><p class="sub">影像所见和复核结论均为空。继续将以${escape(user.display_name)}的姓名签署此报告，并记录留空确认。</p>
        <div class="dialog-actions"><button class="secondary" type="button" id="cancel-empty-review" autofocus>返回填写</button><button class="primary" type="button" id="sign-empty-review">确认留空提交</button></div>`, "cancel-empty-review");
      const confirmed = await new Promise(resolve => {
        let accepted = false;
        byId("sign-empty-review").onclick = () => { accepted = true; dialog.close(); };
        dialog.addEventListener("close", () => resolve(accepted), {once:true});
      });
      if (!confirmed || !form.isConnected) return;
      body.empty_review_confirmed = true;
    }
    const updated = await api(casePath(record.id)+(confirm?"/review":"/report"),{...json(body),method:confirm?"POST":"PUT"});
    if (!form.isConnected) return;
    if (confirm) { await render(); window.scrollTo(0,0); notify("复核人及签署时间已记录。"); }
    else { version=updated.report.revision; byId("draft-status").textContent="草稿已保存 · "+updated.report.author+" · "+dateText(updated.report.updated_at); }
  };
  form.addEventListener("submit",async event => {
    event.preventDefault();
    if (form.getAttribute("aria-busy")==="true") return;
    const confirm = event.submitter?.id !== "save-report-draft";
    const error = form.querySelector(".error");
    const controls=[byId("save-report-draft"),byId("confirm-report")];
    error.hidden=true; form.setAttribute("aria-busy","true");
    controls.forEach(button => button.disabled=true);
    try { await save(confirm); }
    catch (err) {
      error.textContent=formatError(err);
      error.hidden=false;
      if (err.status === 409) {
        const refresh = document.createElement("button");
        refresh.type = "button";
        refresh.className = "secondary";
        refresh.id = "refresh-report";
        refresh.textContent = "核对最新报告";
        refresh.addEventListener("click", async () => {
          refresh.disabled = true;
          try {
            const latest = await api(casePath(record.id) + "?include_signed_evidence=true");
            if (!form.isConnected) return;
            // Keep unsubmitted text while refreshing the image evidence.
            const fields = ["findings", "conclusion"];
            const pending = fields.map(key => form.elements[key].value);
            reportPage(latest);
            const updatedForm = byId("report-form");
            if (updatedForm && latest.report.revision === version) {
              fields.forEach((key, i) => { updatedForm.elements[key].value = pending[i]; });
            } else if (updatedForm) {
              const previous = document.createElement("details");
              const summary = document.createElement("summary");
              summary.textContent = "查看本次未提交的文字";
              previous.append(summary);
              for (const [i, label] of ["影像所见", "复核结论"].entries()) {
                const text = document.createElement("p");
                text.textContent = label + " · " + (pending[i] || "未填写");
                previous.append(text);
              }
              updatedForm.prepend(previous);
            }
          } catch (reloadError) { notify(formatError(reloadError)); refresh.disabled=false; }
        });
        error.append(" ", refresh);
      }
    }
    finally { form.removeAttribute("aria-busy"); controls.forEach(button => button.disabled=false); }
  });
}

async function render() {
  const ticket = ++revision;
  clearPreview();
  unmountViewer?.();
  unmountViewer = undefined;
  if(dialog.open) dialog.close();
  byId("navigation").hidden = byId("identity").hidden = !user;
  byId("display-name").textContent = user?.display_name || "";
  page.setAttribute("aria-busy","false");
  if (!user) { login(); return; }
  if (!location.hash) history.replaceState(null, "", "#new");
  const route = location.hash.slice(1);
  const isReport = route.startsWith("report/");
  const isCase = route.startsWith("case/") || isReport;
  const historyPage = route === "dashboard" || route === "archive";
  const queue = Object.hasOwn(worklists, route);
  byId("crumb").textContent = historyPage ? "历史记录" : queue ? worklists[route].title : isReport ? "复核报告" : isCase ? "阅片工作台" : route === "manual" ? "手动登记" : "新建检查";
  selectNav(historyPage ? "dashboard" : queue ? route : isCase ? "" : "new");
  if (route === "manual") { manualVisit(); return; }
  if (!historyPage && !isCase && !queue) { createVisit(); return; }
  page.innerHTML = '<p class="sub">正在读取病例…</p>';
  page.setAttribute("aria-busy","true");
  try {
    const result = await api(historyPage || queue ? "./api/cases" : casePath(decodeURIComponent(route.slice(route.indexOf("/")+1))) + "?include_signed_evidence=true");
    if (ticket !== revision) return;
    if (historyPage) archive(result);
    else if (queue) worklist(route, result);
    else if (isReport) reportPage(result);
    else detail(result);
  } catch (error) {
    if (ticket === revision) {
      page.innerHTML = `<div class="card"><h2>暂时无法读取</h2><p class="error" role="alert">${escapeHTML(formatError(error))}</p><button class="secondary" id="retry" type="button">重试</button> <a href="#archive">返回病例档案</a></div>`;
      byId("retry").onclick = render;
    }
  } finally {
    if (ticket === revision) page.setAttribute("aria-busy","false");
  }
}

window.addEventListener("hashchange",() => { render(); window.scrollTo(0,0); });
byId("logout").addEventListener("click",async () => {
  const button = byId("logout");
  button.disabled = true;
  try {
    await api("./api/logout",{method:"POST"});
    user = null;
    history.replaceState(null,"",location.pathname);
    render();
    notify("已退出工作台。");
  } catch (error) { notify(formatError(error)); }
  finally { button.disabled = false; }
});

try { loginOptions = await api("./api/login-options"); }
catch { loginOptions = {}; }
try { user = await api("./api/me"); }
catch (error) { if (error instanceof TypeError) notify(formatError(error)); }
render();
