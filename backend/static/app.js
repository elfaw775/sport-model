const page = document.querySelector("#page");
const byId = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const icon = (name) => '<svg aria-hidden="true"><use href="#' + name + '"/></svg>';
const statusNames = {registered:"仅登记",uploaded:"待分析", analyzed:"待复核", reviewed:"已复核"};
const statusBadge = (status) => '<span class="badge ' + (status in statusNames ? status : "") + '">' + escapeHTML(statusNames[status]) + '</span>';
const dateText = (value) => value ? new Intl.DateTimeFormat("zh-CN",{dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Shanghai",hour12:false}).format(new Date(value)) : "—";
const pct = (value) => (Number(value) * 100).toFixed(1).replace(/\.0$/, "") + "%";
const casePath = (id) => "/api/cases/" + encodeURIComponent(id);
const caseLink = (id) => "#case/" + encodeURIComponent(id);
const json = (data) => ({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
let user = null;
let revision = 0;
let noticeTimer;
let loginOptions = {};

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
  throw new Error(detail || ("请求未完成（" + response.status + "）"));
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
    <section class="intro"><p class="eyebrow">EQUINE IMAGING</p><h1>运动马<br><span>影像工作台</span></h1><p>上传球节 X 线片，<br>查看模型候选区域并完成人工复核。</p>
      <div class="intro-steps"><span><b>01</b>上传影像</span><span><b>02</b>模型检测</span><span><b>03</b>复核归档</span></div>
    </section>
    <section class="card login-card"><span class="tile-icon">${icon("mark")}</span><p class="eyebrow">WORKSPACE ACCESS</p><h2>登录工作台</h2><p class="sub">使用分配的账号进入影像诊疗与病例档案。</p>
      <form id="login-form"><label>账号${usernameField}</label><label>密码<input name="password" type="password" autocomplete="current-password" required placeholder="请输入密码"></label><p class="error" role="alert" hidden></p><button class="primary wide" type="submit">登录工作台 ${icon("arrow")}</button></form>
      ${members.length ? '<p class="credentials">团队试用密码：小写拼音账号后加 <code>sportsHorse</code>，注意大写 H。<br>四位成员共享就诊记录，复核按登录姓名署名。</p>' : ''}
    </section></div>`;
  const form = byId("login-form");
  action(form,"正在登录…",async (active) => {
    await api("/api/login",json(Object.fromEntries(new FormData(form))));
    const identity = await api("/api/me");
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
  page.innerHTML = heading("新建检查", "上传检查影像", "选择本次检查的 X 线片，上传后即可开始模型分析。") + progressBar(0) + `
    <form class="card intake" id="visit-form">
      <section class="intake-image"><div class="panel-title"><h2>检查影像</h2><span>首张为分析主图</span></div>
        <div class="image-drop">
          <div class="drop-corners" aria-hidden="true"></div>
          <img id="image-preview" alt="所选主图预览" hidden>
          <div class="drop-prompt"><span class="drop-icon">${icon("upload")}</span><strong>选择或拖入 X 线片</strong><span>JPG、PNG、WebP · 每张最多 10 MB</span></div>
          <div class="preview-hint">点击重新选择影像</div>
          <input id="image-file" type="file" name="image" aria-label="选择检查影像" accept="${fileTypes}" multiple required>
        </div>
        <p class="file-status">${icon("image")}<span class="file-title" aria-live="polite">尚未选择影像</span></p>
      </section>
      <section class="intake-settings"><h2>检查信息</h2>
        <label>马匹名称 / 编号<input id="horse-name" name="horse_name" maxlength="80" required placeholder="例如：追风 或 H-001"></label>
        <label>投照方向<select name="projection" required>${projectionOptions}</select></label>
        <dl class="model-scope"><dt>检查部位</dt><dd>球节</dd><dt>关注病变</dt><dd>疑似骨折</dd></dl>
        <label class="check"><input type="checkbox" name="quality_confirmed" value="true" required>已核对马匹、方向和影像清晰度</label>
        ${errorBox}<button class="primary wide" type="submit">上传影像 ${icon("arrow")}</button>
      </section>
    </form>
    <footer class="intake-foot"><span>暂时没有影像？ <a id="manual-visit" href="#manual">手动登记就诊</a></span><a href="#dashboard">${icon("book")} 查找历史记录</a></footer>`;
  const form = byId("visit-form");
  wireImages(form);
  action(form, "正在上传…", async active => {
    const horseName = form.elements.horse_name.value.trim();
    if (!horseName) throw new Error("请填写马匹名称或编号。");
    const [mainImage, ...additional] = Array.from(form.elements.image.files);
    const projection = form.elements.projection.value;
    const quality = form.elements.quality_confirmed.checked;
    const payload = imagePayload(mainImage, projection, quality);
    payload.set("horse_name", horseName);
    const record = await api("/api/cases", {method:"POST", body:payload});
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
    const record = await api("/api/visits", json({horse_name, comment:form.elements.comment.value.trim()}));
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
    <p class="sub">${single ? escape(records[0].horse_name) + " · " + escape(records[0].id) : escape(records.slice(0,3).map(record=>record.id).join("、")) + (records.length>3?" 等":"")}<br>所选记录的影像、备注、模型结果和复核意见将一起删除。${Object.keys(loginOptions).length ? "删除对所有团队成员生效。" : ""}此操作不可撤销。</p>
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
function archive(records) {
  const selected = new Set(), perPage=8;
  const stats = [
    ["book","全部就诊",records.length,"登记在册的记录","fern"],
    ["image","已收影像",records.reduce((n,r)=> n+(r.images?.length||0),0),"就诊记录中的全部影像","slate"],
    ["clock","待人工复核",records.filter(r=>r.status==="analyzed").length,"已完成模型分析","sand"],
    ["check","归档记录",records.filter(r=>r.archived_at).length,"已完成手动或复核归档","olive"],
  ];
  const description = Object.keys(loginOptions).length ? "团队共享就诊记录、检查影像和归档报告。" : "集中管理就诊记录、检查影像和归档报告。";
  page.innerHTML = heading("CLINICAL WORKSPACE","就诊记录看板",description,newLink) + `
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

function imageViewer(record) {
  const boxes = record.detections || [];
  const labelWidth = Math.max(record.width / 18, record.height / 14);
  const boxLayer = boxes.map((box, index) => {
    const left = Number(box.x1), top = Number(box.y1);
    const tag = String(index + 1).padStart(2, "0");
    return `<rect x="${left}" y="${top}" width="${Number(box.x2) - left}" height="${Number(box.y2) - top}"/>
      <svg x="${left}" y="${top}" width="${labelWidth}" height="${labelWidth * .55}" viewBox="0 0 40 22"><rect class="tag-bg" width="40" height="22"/><text x="20" y="16" text-anchor="middle">${tag}</text></svg>`;
  }).join("");
  return `<section class="card image-card"><div class="panel-title"><h2>检查主图</h2><span>球节 · ${escape(record.projection)}</span></div>
    <div class="viewer"><svg class="radiograph" viewBox="0 0 ${Number(record.width)} ${Number(record.height)}" role="img" aria-label="原始 X 线影像及模型候选区域">
      <image width="100%" height="100%" href="${casePath(record.id)}/image"/><g id="boxes">${boxLayer}</g></svg></div>
    <div class="viewer-foot"><span>${escape(record.filename)} · ${Number(record.width)} × ${Number(record.height)}</span>
    ${boxes.length ? '<label class="overlay-switch"><input id="overlay" type="checkbox" checked>候选框</label>' : ""}</div></section>`;
}

const archiveButton = '<button id="archive-visit" class="primary wide" type="button">' + icon("folder") + ' 归档病例</button>';

function analysisPanel(record) {
  const analyzed = ["analyzed", "reviewed"].includes(record.status);
  const boxes = record.detections || [];
  const findings = boxes.length
    ? `<ul class="findings">${boxes.map((box,index) => `<li><span>疑似骨折 · 区域 ${String(index + 1).padStart(2,"0")}</span><b>置信分数 ${pct(box.score)}</b></li>`).join("")}</ul>`
    : '<p class="sub">当前阈值下未检出疑似骨折区域，仍需结合原片进行复核。</p>';
  const modelInfo = `<details class="model-info"><summary>模型信息</summary><dl><dt>模型版本</dt><dd class="mono">${escape(record.model)}</dd><dt>检测阈值</dt><dd>${pct(record.threshold)}</dd><dt>分析时间</dt><dd>${escape(dateText(record.analyzed_at))}</dd></dl></details>`;
  let controls;
  if (!analyzed) {
    controls = `<form id="analyze-form">${errorBox}<button class="primary wide" type="submit">${icon("scan")} 开始分析</button><p class="inference-status" role="status">模型正在处理主图，请稍候…</p></form>`;
  } else if (record.status === "analyzed") {
    controls = `<button class="primary wide" id="start-review" aria-expanded="false" aria-controls="review-form">${icon("user")} 人工复核</button>
      <form id="review-form" hidden><label for="review-note">人工复核意见</label>
      <textarea id="review-note" name="note" rows="3" required maxlength="2000" placeholder="核对原片与候选区域，记录您的判断"></textarea>
      <small class="dim">复核人 · ${escape(user.display_name)}</small>${errorBox}
      <button class="primary wide" type="submit">确认复核 ${icon("check")}</button></form>`;
  } else {
    controls = `<div class="reviewed-note"><h3>人工复核意见</h3><p>${escape(record.review_note)}</p>
      <small>复核人：${escape(record.reviewer)} · ${escape(dateText(record.reviewed_at))}</small></div>
      ${record.archived_at ? '<a class="secondary wide" href="#dashboard">查找历史记录</a>' : archiveButton}`;
  }
  return `<section class="card results" tabindex="-1"><div class="panel-title"><h2>${analyzed ? "模型分析结果" : "模型分析"}</h2>${record.archived_at && record.status === "reviewed" ? archiveBadge(record) : statusBadge(record.status)}</div>
    ${analyzed ? `<p class="count"><strong>${boxes.length}</strong> 个候选区域</p>${findings}${modelInfo}` :
      '<div class="analysis-ready"><span class="ready-icon">' + icon("scan") + '</span><div><strong>主图已就绪</strong><p>检测球节 X 线片中的疑似骨折区域。</p></div></div>'}
    <p class="model-note">AI 仅提供参考。置信分数表示模型对候选区域的评分，诊疗结论由人工复核。</p>
    ${controls}</section>`;
}

function detail(record) {
  const base = casePath(record.id);
  byId("nav-work").classList.toggle("selected", !record.archived_at);
  byId("nav-archive").classList.toggle("selected", Boolean(record.archived_at));
  const images = record.images || [];
  const warning = uploadMessage?.id === record.id ? uploadMessage.message : "";
  if (warning) uploadMessage = null;
  const reviewerConfirmed = record.status === "reviewed";
  const stage = reviewerConfirmed ? (record.archived_at ? 6 : 5) : record.status === "analyzed" ? 2 : 1;
  const creator = loginOptions[record.created_by] || (user.username === record.created_by ? user.display_name : record.created_by);
  const title = heading("检查工作台", escape(record.horse_name), escape(record.id) + "　·　" + escape(dateText(record.created_at)),
    '<a href="#dashboard" class="secondary">历史记录</a>');
  const matchesLegacy = record.has_image && record.archived_at && !reviewerConfirmed;
  page.innerHTML = title +
    (record.has_image ? progressBar(stage) : "") +
    (warning ? `<p class="error" role="alert">${escape(warning)}</p>` : "") +
    (matchesLegacy ? '<p class="legacy-note">这是一条已归档的历史病例，人工复核尚待完成。</p>' : "") +
    (record.has_image ? `<div class="report-grid">${imageViewer(record)}${analysisPanel(record)}</div>` : "") + `
    <section class="card visit-summary"><div class="spread"><div class="panel-title"><h2>就诊信息</h2>${archiveBadge(record)}</div>
      <div class="row-actions"><button id="edit-visit" class="secondary">编辑记录</button>${!record.has_image && !record.archived_at ? archiveButton : ""}</div></div>
      <p class="visit-comment">${escape(record.comment || "暂无就诊备注")}</p>
      <div class="detail-meta"><span>${icon("user")} 登记人：${escape(creator || "未记录")}</span><span>${icon("image")} ${images.length} 张影像</span>
      ${record.archived_at ? `<span>${icon("folder")} 归档于 ${escape(dateText(record.archived_at))}</span>` : ""}</div></section>
    <section class="card media-card"><div class="spread"><div class="panel-title"><h2>就诊影像</h2><span class="count-tag">${images.length}</span></div>
      <button class="secondary" id="add-images" aria-expanded="false" aria-controls="attach-form">${icon("upload")} 补充影像</button></div>
      ${images.length ? `<div class="image-tray">${images.map(image => `<a class="image-tile" href="${escape(image.url)}" target="_blank" rel="noopener" aria-label="查看原图 ${escape(image.filename)}">
        <img loading="lazy" src="${escape(image.url)}" alt=""><span><strong>${image.id === "primary" ? "主图" : "附图"} · ${escape(image.projection)}</strong><small>${escape(image.filename)}</small></span>${icon("arrow")}</a>`).join("")}</div>` :
        '<p class="dim media-placeholder">尚未上传影像，补充主图后即可开始模型分析。</p>'}
      <form id="attach-form" hidden>${imageFields()}${errorBox}<button class="primary" type="submit">保存影像</button></form></section>`;

  function displayNextStage(updated, message) {
    detail(updated);
    const results = page.querySelector(".results");
    if (results) results.focus({preventScroll:true});
    if (results && window.matchMedia("(max-width: 800px)").matches) results.scrollIntoView({block:"start"});
    else window.scrollTo(0, 0);
    notify(message);
  }
  byId("edit-visit").onclick = () => editVisit(record, detail);
  byId("archive-visit")?.addEventListener("click", async event => {
    const button = event.currentTarget;
    const ticket = revision;
    button.disabled = true;
    try {
      const updated = await api(base + "/archive", {method:"POST"});
      if (ticket === revision) displayNextStage(updated, "病例已归档。");
    } catch (error) {
      if (ticket === revision) notify(formatError(error));
    } finally { button.disabled = false; }
  });
  const attach = byId("attach-form");
  byId("add-images").onclick = event => {
    attach.hidden = !attach.hidden;
    event.currentTarget.setAttribute("aria-expanded", String(!attach.hidden));
    if (!attach.hidden) attach.elements.image.focus();
  };
  wireImages(attach);
  action(attach, "正在上传…", async active => {
    const message = await attachFiles(record.id, Array.from(attach.elements.image.files), attach.elements.projection.value, attach.elements.quality_confirmed.checked);
    const updated = await api(base);
    if (active()) {
      if (message) uploadMessage = {id:record.id, message};
      if (!record.has_image && updated.has_image) displayNextStage(updated, "主图已保存，可开始分析。");
      else {
        detail(updated);
        if (!message) notify("影像已保存。");
      }
    }
  });
  byId("overlay")?.addEventListener("change", event => { byId("boxes").style.display = event.target.checked ? "" : "none"; });
  const analyzeForm = byId("analyze-form");
  if (analyzeForm) action(analyzeForm, "模型分析中…", async active => {
    const result = await api(base + "/analyze", {method:"POST"});
    if (active()) displayNextStage(result, "模型分析完成，请核对影像与结果。");
  });
  const reviewForm = byId("review-form");
  if (!reviewForm) return;
  function showStep(index) {
    page.querySelector(".flow-progress").outerHTML = progressBar(index);
  }
  byId("start-review").onclick = event => {
    event.currentTarget.hidden = true;
    event.currentTarget.setAttribute("aria-expanded", "true");
    reviewForm.hidden = false;
    showStep(3);
    byId("review-note").focus();
  };
  byId("review-note").oninput = event => showStep(event.currentTarget.value.trim() ? 4 : 3);
  action(reviewForm, "正在确认…", async active => {
    const note = reviewForm.elements.note.value.trim();
    if (!note) throw new Error("请填写复核意见。");
    const result = await api(base + "/review", json({note}));
    if (active()) displayNextStage(result, "复核意见已确认。");
  });
}

async function render() {
  const ticket = ++revision;
  clearPreview();
  if(dialog.open) dialog.close();
  byId("navigation").hidden = byId("identity").hidden = !user;
  byId("display-name").textContent = user?.display_name || "";
  page.setAttribute("aria-busy","false");
  if (!user) { login(); return; }
  if (!location.hash) history.replaceState(null, "", "#new");
  const route = location.hash.slice(1);
  const isCase = route.startsWith("case/");
  const archived = route === "dashboard" || route === "archive";
  byId("crumb").textContent = archived ? "历史记录" : isCase ? "检查详情" : route === "manual" ? "手动登记" : "新建检查";
  byId("nav-work").classList.toggle("selected", !archived);
  byId("nav-archive").classList.toggle("selected", archived);
  if (route === "manual") { manualVisit(); return; }
  if (!archived && !isCase) { createVisit(); return; }
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

try { loginOptions = await api("/api/login-options"); }
catch { loginOptions = {}; }
try { user = await api("/api/me"); }
catch (error) { if (error instanceof TypeError) notify(formatError(error)); }
render();
