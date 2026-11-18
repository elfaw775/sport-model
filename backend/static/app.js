const page = document.querySelector("#page");
const byId = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const icon = (name) => '<svg aria-hidden="true"><use href="#' + name + '"/></svg>';
const statusNames = {uploaded:"待分析", analyzed:"待复核", reviewed:"已复核"};
const statusBadge = (status) => '<span class="badge ' + (status in statusNames ? status : "") + '">' + escapeHTML(statusNames[status]) + '</span>';
const dateText = (value) => value ? new Intl.DateTimeFormat("zh-CN",{dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Shanghai",hour12:false}).format(new Date(value)) : "—";
const pct = (value) => (Number(value) * 100).toFixed(1).replace(/\.0$/, "") + "%";
const casePath = (id) => "/api/cases/" + encodeURIComponent(id);
const caseLink = (id) => "#case/" + encodeURIComponent(id);
const json = (data) => ({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
let user = null;
let revision = 0;
let previewUrl = "";
let noticeTimer;

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
  const detail = typeof data?.detail === "string" ? data.detail : "";
  throw new Error(detail || ("请求未完成（" + response.status + "）"));
}

function formatError(error) {
  return error instanceof TypeError ? "无法连接工作台，请检查服务后重试。" : error.message;
}

function heading(kicker, title, desc, action = "") {
  return `<div class="heading"><div><p class="eyebrow">${kicker}</p><h1>${title}</h1><p class="sub">${desc}</p></div>${action}</div>`;
}

function showError(message) {
  const output = page.querySelector(".error");
  if (output) { output.textContent = message; output.hidden = false; }
}

function action(form, pending, run) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (form.getAttribute("aria-busy") === "true") return;
    const ticket = revision;
    const error = page.querySelector(".error");
    if (error) error.hidden = true;
    const button = form.querySelector('button[type="submit"]');
    const oldText = button.innerHTML;
    button.disabled = true;
    form.setAttribute("aria-busy","true");
    button.textContent = pending;
    try {
      await run(() => ticket === revision);
    } catch (error) {
      if (ticket === revision) showError(formatError(error));
    } finally {
      form.removeAttribute("aria-busy");
      button.disabled = false;
      button.innerHTML = oldText;
    }
  });
}

function login() {
  byId("crumb").textContent = "登录";
  page.innerHTML = `<div class="login-grid">
    <section class="intro"><p class="eyebrow">EQUINE IMAGING &nbsp; / &nbsp; 01</p><h1>运动马<br><span>影像工作台</span></h1><p>从一张球节 X 线片，<br>到一份可复核、可追溯的病例报告。</p>
      <div class="intro-steps"><span><b>01</b>影像登记</span><span><b>02</b>模型检测</span><span><b>03</b>复核归档</span></div>
    </section>
    <section class="card login-card"><span class="tile-icon">${icon("mark")}</span><p class="eyebrow">WORKSPACE ACCESS</p><h2>登录工作台</h2><p class="sub">使用分配的账号进入影像诊疗与病例档案。</p>
      <form id="login-form"><label>账号<input name="username" autocomplete="username" required placeholder="请输入账号"></label><label>密码<input name="password" type="password" autocomplete="current-password" required placeholder="请输入密码"></label><p class="error" role="alert" hidden></p><button class="primary wide" type="submit">登录工作台 ${icon("arrow")}</button></form>
      <p class="credentials">本地示例账号 <code>demo</code><br>示例密码 <code>EquiDemo2026!</code></p>
    </section></div>`;
  const form = byId("login-form");
  action(form,"正在登录…",async (active) => {
    await api("/api/login",json(Object.fromEntries(new FormData(form))));
    const identity = await api("/api/me");
    if (active()) {
      user = identity;
      if (!location.hash) history.replaceState(null, "", "#work");
      render();
    }
  });
}

function upload() {
  page.innerHTML = heading("CLINICAL WORKSPACE","影像诊疗","登记本次检查的影像，进入球节骨折辅助检测。") + `
    <div class="upload-grid"><section class="card"><header class="section-head"><span class="step-no">01</span><div><h2>新建影像检查</h2><p class="sub">每次提交生成一个独立病例编号。</p></div></header>
    <form id="upload-form">
      <label for="horse-name">马匹名称 / 编号</label><input id="horse-name" name="horse_name" required maxlength="80" placeholder="例如：追风 或 H-001">
      <div class="field-row"><label>检查部位<input value="球节" readonly></label><label>投照方向<select name="projection" required><option value="" selected disabled>请选择投照方向</option><option>正位</option><option>侧位</option><option>斜位</option><option>其他</option></select></label></div>
      <label for="image-file">检查影像</label><div class="file-field"><span class="file-symbol">${icon("upload")}</span><div><strong id="file-title">选择 X 线影像</strong><small>PNG、JPEG 或 WebP · 最大 10 MB</small></div><input id="image-file" name="image" type="file" accept="image/png,image/jpeg,image/webp" required aria-label="选择 X 线影像"></div>
      <div id="preview" hidden><img alt="待上传影像预览"></div>
      <label class="check"><input type="checkbox" name="quality_confirmed" value="true" required>我已确认影像清晰且为本次球节检查。</label>
      <p class="error" role="alert" hidden></p><button class="primary" type="submit">上传并创建病例 ${icon("arrow")}</button>
    </form></section>
    <aside class="context-panel"><p class="eyebrow">MODEL PROFILE</p><span class="large-symbol">${icon("scan")}</span><h2>球节骨折检测</h2><p>模型在 X 线片上定位疑似骨折区域，供阅片时逐一核对。</p><dl><dt>检查部位</dt><dd>马球节</dd><dt>目标病变</dt><dd>骨折</dd><dt>模型</dt><dd>YOLO11s</dd></dl><div class="model-note">仅用于研究。未检出候选区域不能排除骨折，最终结论由人工复核。</div></aside></div>`;
  const form = byId("upload-form");
  byId("image-file").addEventListener("change", (event) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = "";
    const input = event.currentTarget;
    const file = input.files[0];
    input.setCustomValidity(file && file.size > 10 * 1024 * 1024 ? "单张影像不能超过 10 MB。" : "");
    byId("file-title").textContent = file?.name || "选择 X 线影像";
    byId("preview").hidden = true;
    if (file && input.reportValidity()) {
      previewUrl = URL.createObjectURL(file);
      const image = byId("preview").querySelector("img");
      image.src = previewUrl;
      byId("preview").hidden = false;
    }
  });
  action(form,"正在上传…",async (active) => {
    const result = await api("/api/cases",{method:"POST",body:new FormData(form)});
    if (active()) { location.hash = caseLink(result.id); notify("影像已保存。"); }
  });
}

function archive(records) {
  const pending = records.filter(item => item.status === "analyzed").length;
  const reviewed = records.filter(item => item.status === "reviewed").length;
  page.innerHTML = heading("CASE ARCHIVE","病例档案","按病例编号追溯原始影像、模型结果与复核意见。",'<a class="primary" href="#work">＋ 新建检查</a>') + `
    <div class="stats"><div><span>归档病例</span><strong>${records.length}</strong></div><div><span>待人工复核</span><strong>${pending}</strong></div><div><span>已复核</span><strong>${reviewed}</strong></div></div>
    <section class="card"><div class="archive-head"><h2>全部病例</h2><input id="case-search" type="search" placeholder="搜索马匹名称或病例编号" aria-label="搜索病例"></div><div id="case-list"></div></section>`;
  function filter() {
    const query = byId("case-search").value.trim().toLowerCase();
    const matches = records.filter(item => [item.horse_name, item.id].some(value => String(value).toLowerCase().includes(query)));
    byId("case-list").innerHTML = matches.length ? `<div class="table-scroll"><table><thead><tr><th>马匹 / 病例编号</th><th>检查</th><th>状态</th><th>登记时间（北京时间）</th><th>操作</th></tr></thead><tbody>${matches.map(item => `<tr><td><strong>${escapeHTML(item.horse_name)}</strong><small>${escapeHTML(item.id)}</small></td><td>球节 · ${escapeHTML(item.projection || "—")}</td><td>${statusBadge(item.status)}</td><td>${escapeHTML(dateText(item.created_at))}</td><td><a href="${escapeHTML(caseLink(item.id))}">查看病例 →</a></td></tr>`).join("")}</tbody></table></div>` : `<div class="empty">${icon("folder")}<h3>${query ? "没有匹配的病例" : "这里将保存你的第一份病例"}</h3><p>${query ? "换个名称或编号再试。" : "上传影像后，就可以在这里继续分析、复核或回看记录。"}</p></div>`;
  }
  byId("case-search").addEventListener("input",filter);
  filter();
}

function detail(record) {
  const analyzed = record.status !== "uploaded";
  const boxes = record.detections || [];
  const base = casePath(record.id);
  const size = Number(record.width) + " " + Number(record.height);
  const boxLayer = boxes.map((box,index) => {
    const left = Number(box.x1), top = Number(box.y1);
    const tag = String(index + 1).padStart(2, "0");
    return `<rect x="${left}" y="${top}" width="${Number(box.x2)-left}" height="${Number(box.y2)-top}"/><svg x="${left}" y="${top}" width="${record.width * .06}" height="${record.width * .035}" viewBox="0 0 40 22"><rect class="tag-bg" width="40" height="22"/><text x="20" y="16" text-anchor="middle">${tag}</text></svg>`;
  }).join("");
  page.innerHTML = heading("CASE REPORT","影像诊疗报告",escapeHTML(record.id) + " &nbsp;/&nbsp; " + escapeHTML(record.horse_name),'<a href="#archive" class="secondary">查看病例档案 →</a>') + `
    <div class="report-grid"><section class="card image-card"><div class="section-head spread"><div><p class="eyebrow">RADIOGRAPH</p><h2>检查影像</h2></div>${statusBadge(record.status)}</div>
      <div class="viewer"><svg class="radiograph" viewBox="0 0 ${size}" role="img" aria-label="原始 X 线影像及模型候选区域"><image width="100%" height="100%" href="${escapeHTML(base)}/image"/><g id="boxes">${boxLayer}</g></svg></div>
      <div class="viewer-foot"><span>${escapeHTML(record.filename)} · ${Number(record.width)} × ${Number(record.height)}</span>${boxes.length ? '<label class="overlay-switch"><input id="overlay" type="checkbox" checked>候选框</label>' : ""}</div>
    </section><div class="report-side"><section class="card"><p class="eyebrow">EXAM DETAILS</p><h2>检查信息</h2><dl><dt>马匹</dt><dd>${escapeHTML(record.horse_name)}</dd><dt>部位 / 方向</dt><dd>球节 · ${escapeHTML(record.projection || "—")}</dd><dt>关注病变</dt><dd>骨折</dd><dt>登记时间</dt><dd>${escapeHTML(dateText(record.created_at))}</dd></dl></section>
    <section class="card results"><p class="eyebrow">MODEL FINDINGS</p><h2>${analyzed ? "辅助检测结果" : "模型检测"}</h2>
      ${analyzed ? `<p class="count"><strong>${boxes.length}</strong> 个候选区域</p>${boxes.length ? `<ul class="findings">${boxes.map((box,index) => `<li><span>区域 ${String(index + 1).padStart(2,"0")}</span><b>置信分数 ${pct(box.score)}</b></li>`).join("")}</ul>` : '<p class="sub">在本次阈值下未检出候选区域，不能排除骨折。</p>'}<dl><dt>模型版本</dt><dd class="mono">${escapeHTML(record.model)}</dd><dt>检测阈值</dt><dd>${pct(record.threshold)}</dd></dl>` : `<p class="sub">影像已保存。开始后将调用球节骨折检测模型。</p><form id="analyze-form"><button class="primary" type="submit">${icon("scan")} 开始 AI 分析</button></form>`}
      <p class="model-note">AI 仅提供参考。模型置信分数不是患病概率，最终结论由人工复核。</p></section></div></div>
    <section class="card review"><div><p class="eyebrow">HUMAN REVIEW</p><h2>人工复核</h2></div>
    ${record.status === "reviewed" ? `<div class="reviewed-note"><p>${escapeHTML(record.review_note)}</p><small>复核人：${escapeHTML(record.reviewer)} · ${escapeHTML(dateText(record.reviewed_at))}</small></div>` : analyzed ? `<form id="review-form"><label for="review-note">复核意见</label><textarea id="review-note" name="note" rows="3" required maxlength="2000" placeholder="结合影像记录人工判断与候选区域核对结果"></textarea><button class="primary" type="submit">确认复核并归档</button></form>` : '<p class="sub">完成模型检测后，可填写人工复核意见。</p>'}</section><p class="error" role="alert" hidden></p>`;
  byId("overlay")?.addEventListener("change",event => { byId("boxes").style.display = event.target.checked ? "" : "none"; });
  const analyzeForm = byId("analyze-form");
  if (analyzeForm) action(analyzeForm,"模型推理中…",async (active) => {
    const result = await api(base + "/analyze",{method:"POST"});
    if (active()) { detail(result); notify("模型检测完成，结果已保存。"); }
  });
  const reviewForm = byId("review-form");
  if (reviewForm) action(reviewForm,"正在归档…",async (active) => {
    const note = new FormData(reviewForm).get("note").trim();
    if (!note) throw new Error("请填写复核意见。");
    const result = await api(base + "/review",json({note}));
    if (active()) { detail(result); notify("复核意见已归档。"); }
  });
}

async function render() {
  const ticket = ++revision;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = "";
  byId("navigation").hidden = byId("identity").hidden = !user;
  byId("display-name").textContent = user?.display_name || "";
  page.setAttribute("aria-busy","false");
  if (!user) { login(); return; }
  const route = location.hash.slice(1);
  const isCase = route.startsWith("case/");
  const archived = route === "archive";
  byId("crumb").textContent = archived ? "病例档案" : isCase ? "病例详情" : "影像诊疗";
  byId("nav-work").classList.toggle("selected", !archived && !isCase);
  byId("nav-archive").classList.toggle("selected", archived || isCase);
  if (!archived && !isCase) { upload(); return; }
  page.innerHTML = '<p class="sub">正在读取病例…</p>';
  page.setAttribute("aria-busy","true");
  try {
    const result = await api(archived ? "/api/cases" : casePath(decodeURIComponent(route.slice(5))));
    if (ticket === revision) archived ? archive(result) : detail(result);
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
    await api("/api/logout",{method:"POST"});
    user = null;
    history.replaceState(null,"",location.pathname);
    render();
    notify("已退出工作台。");
  } catch (error) { notify(formatError(error)); }
  finally { button.disabled = false; }
});

try { user = await api("/api/me"); }
catch (error) { if (error instanceof TypeError) notify(formatError(error)); }
render();
