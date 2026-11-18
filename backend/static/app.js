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
  page.innerHTML = `<div class="login-grid">
    <section class="intro"><p class="eyebrow">EQUINE IMAGING &nbsp; / &nbsp; 01</p><h1>运动马<br><span>影像工作台</span></h1><p>从首次就诊登记，<br>到每一张影像与每一份归档。</p>
      <div class="intro-steps"><span><b>01</b>就诊建档</span><span><b>02</b>模型检测</span><span><b>03</b>复核归档</span></div>
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
      if (!location.hash) history.replaceState(null, "", "#dashboard");
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
const newLink = '<a class="primary" href="#new" id="new-visit">' + icon("plus") + '新建就诊记录</a>';

function visitFields(record={}) {
  return `<label>马匹名称 / 编号<input name="horse_name" id="horse-name" required maxlength="80" placeholder="例如：追风 或 H-001" value="${escape(record.horse_name)}"></label>
    <label>就诊备注 <span class="optional">选填</span><textarea name="comment" rows="4" maxlength="4000" placeholder="记录就诊原因、观察或后续安排">${escape(record.comment)}</textarea></label>`;
}
function imageFields(required=false) {
  return `<div class="file-field"><span class="file-symbol">${icon("upload")}</span><div><strong class="file-title">选择或拖入影像</strong><small>可多选 · JPG / PNG / WebP · 单张不超过 10 MB</small></div><input type="file" id="image-file" name="image" aria-label="检查影像" accept="image/png,image/jpeg,image/webp" multiple ${required?"required":""}></div>
    <div class="upload-options"${required?"":" hidden"}><label>本批影像投照方向<select name="projection"><option value="">请选择方向</option><option>正位</option><option>侧位</option><option>斜位</option><option>其他</option></select></label>
    <label class="check"><input type="checkbox" name="quality_confirmed" value="true">我已确认本批影像清晰且对应本次球节检查。</label>
    <small class="dim">方向不同的图片请分批上传。首张用于模型分析，后续图片作为本次就诊附件。</small></div>`;
}
function wireImages(form) {
  const input = form.elements.image;
  const update = () => {
    const files = Array.from(input.files);
    form.querySelector(".upload-options").hidden = !files.length;
    for (const key of ["projection", "quality_confirmed"]) form.elements[key].required = Boolean(files.length);
    input.setCustomValidity(files.some(file => file.size > 10 * 1024 * 1024) ? "单张影像不能超过 10 MB。" : "");
    form.querySelector(".file-title").textContent = files.length ? files.map(file => file.name).join("、") : "选择或拖入影像";
  };
  input.addEventListener("change",update);
  update();
}
async function attachFiles(id, form) {
  const files = Array.from(form.elements.image.files);
  let complete = 0;
  for (const file of files) {
    const payload = new FormData();
    payload.set("image",file);
    payload.set("projection",form.elements.projection.value);
    payload.set("quality_confirmed",String(form.elements.quality_confirmed.checked));
    try { await api(casePath(id) + "/images",{method:"POST",body:payload}); }
    catch (error) { return `已保存记录，成功上传 ${complete}/${files.length} 张。${file.name}：${formatError(error)}。其余未上传，请从“补充影像”重试。`; }
    complete++;
  }
  return "";
}
function createVisit() {
  page.innerHTML = heading("NEW VISIT","新建就诊记录","可以先登记马匹与备注，影像、模型分析和人工复核均可稍后补充。",'<a href="#dashboard" class="secondary">返回记录看板</a>') + `
    <div class="upload-grid"><section class="card"><form id="visit-form"><header class="section-head"><span class="step-no">01</span><h2>就诊登记</h2></header>${visitFields()}
    <header class="section-head upload-head"><span class="step-no">02</span><h2>检查影像</h2><span class="optional">选填</span></header>${imageFields()}${errorBox}
    <button class="primary" type="submit">保存就诊记录 ${icon("arrow")}</button></form></section>
    <aside class="context-panel"><p class="eyebrow">VISIT GUIDE</p><span class="large-symbol">${icon("book")}</span><h2>一份档案，连续记录</h2><ol class="guide"><li><b>登记</b>保存时生成独立的 EV 就诊编号。</li><li><b>补充</b>可随时更新备注，上传同次检查的影像。</li><li><b>归档</b>普通就诊可直接归档；模型辅助结果另行人工复核。</li></ol></aside></div>`;
  const form = byId("visit-form");
  wireImages(form);
  action(form,"正在保存…",async active=>{
    const horse_name = form.elements.horse_name.value.trim(), comment = form.elements.comment.value.trim();
    if (!horse_name) throw new Error("请填写马匹名称或编号。");
    const record = await api("/api/visits",json({horse_name,comment}));
    const message = await attachFiles(record.id,form);
    if (message) uploadMessage = {id:record.id,message};
    if(active()) { location.hash=caseLink(record.id); notify(message ? "记录已保存，部分影像尚未上传。" : "就诊记录已保存。"); }
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
    <p class="sub">${single ? escape(records[0].horse_name) + " · " + escape(records[0].id) : escape(records.slice(0,3).map(record=>record.id).join("、")) + (records.length>3?" 等":"")}<br>所选记录的影像、备注、模型结果和复核意见将一起删除，此操作不可撤销。</p>
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
    ["image","已收影像",records.reduce((n,r)=> n+(r.images?.length||0),0),"本次账户上传的影像","slate"],
    ["clock","待人工复核",records.filter(r=>r.status==="analyzed").length,"已完成模型分析","sand"],
    ["check","归档记录",records.filter(r=>r.archived_at).length,"已完成手动或复核归档","olive"],
  ];
  page.innerHTML = heading("CLINICAL WORKSPACE","就诊记录看板","集中管理马匹就诊信息、检查影像与归档记录。",newLink) + `
    <div class="metric-grid">${stats.map(([symbol,title,count,note,tone])=>`<div class="metric ${tone}"><span class="metric-icon">${icon(symbol)}</span><span class="metric-title">${title}</span><strong>${count}</strong><small>${note}</small></div>`).join("")}</div>
    <section class="card ledger"><header class="ledger-head"><div><p class="eyebrow">VISIT REGISTRY</p><h2>就诊记录 <span class="count-tag">${records.length}</span></h2></div><div class="tools"><label class="search-field">${icon("search")}<input type="search" id="case-search" placeholder="搜索马匹、编号或备注" aria-label="搜索就诊记录"></label><select id="status-filter" aria-label="筛选记录状态">${Object.entries(stateNames).map(([key,label])=>`<option value="${key}">${label}</option>`).join("")}</select></div></header>
    <div class="selection-bar"><span id="match-count"></span><button type="button" class="text-danger" id="batch-delete" disabled>${icon("trash")} 批量删除</button></div>
    <p id="batch-feedback" role="status" hidden></p><div id="case-list"></div><footer class="pager" id="pager"></footer></section>`;
  const query=byId("case-search"),filter=byId("status-filter");
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

function detail(record) {
  const analyzed = ["reviewed","analyzed"].includes(record.status);
  const boxes = record.detections || [];
  const base = casePath(record.id);
  const size = Number(record.width) + " " + Number(record.height);
  const boxLayer = boxes.map((box,index) => {
    const left = Number(box.x1), top = Number(box.y1);
    const tag = String(index + 1).padStart(2, "0");
    return `<rect x="${left}" y="${top}" width="${Number(box.x2)-left}" height="${Number(box.y2)-top}"/><svg x="${left}" y="${top}" width="${record.width * .06}" height="${record.width * .035}" viewBox="0 0 40 22"><rect class="tag-bg" width="40" height="22"/><text x="20" y="16" text-anchor="middle">${tag}</text></svg>`;
  }).join("");
  const images=record.images || [];
  const warning=uploadMessage?.id===record.id ? uploadMessage.message : "";
  if(warning)uploadMessage=null;
  page.innerHTML=heading("VISIT RECORD",escape(record.horse_name),escape(record.id)+" &nbsp;·&nbsp; 登记于 "+escape(dateText(record.created_at)),'<a href="#dashboard" class="secondary">返回记录看板</a>')+`
    <section class="card visit-summary"><div class="spread"><div><p class="eyebrow">ENCOUNTER NOTE</p><h2>就诊记录</h2></div><div class="row-actions">${archiveBadge(record)}<button id="edit-visit" class="secondary">编辑记录</button>${!record.archived_at?'<button id="archive-visit" class="primary">手动归档</button>':""}</div></div>
    <p class="visit-comment">${escape(record.comment||"暂无就诊备注，可点击「编辑记录」补充。")}</p><div class="detail-meta"><span>${icon("image")} ${images.length} 张影像</span><span>${statusBadge(record.status)}</span>${record.archived_at?`<span>归档于 ${escape(dateText(record.archived_at))}</span>`:""}</div>
    ${warning?`<p class="error" role="alert">${escape(warning)}</p>`:""}</section>
    <section class="card media-card"><div class="spread"><div><p class="eyebrow">VISIT IMAGES</p><h2>就诊影像 <span class="count-tag">${images.length}</span></h2></div><button class="secondary" id="add-images" aria-expanded="false" aria-controls="attach-form">${icon("upload")} 补充影像</button></div>
    ${images.length?`<div class="image-tray">${images.map(image=>`<a class="image-tile" href="${escape(image.url)}" target="_blank" rel="noopener" aria-label="查看原图 ${escape(image.filename)}"><img loading="lazy" src="${escape(image.url)}" alt=""><span><strong>${image.id==="primary"?"主图":"附图"} · ${escape(image.projection)}</strong><small>${escape(image.filename)}</small></span>${icon("arrow")}</a>`).join("")}</div>`:'<p class="dim media-placeholder">尚未上传影像。可直接归档，也可以上传第一张球节 X 线片开始分析。</p>'}
    <form id="attach-form" hidden>${imageFields(true)}${errorBox}<button class="primary" type="submit">保存影像</button></form></section>`;
  if(record.has_image) page.insertAdjacentHTML("beforeend",`
    <div class="report-grid"><section class="card image-card"><div class="section-head spread"><div><p class="eyebrow">RADIOGRAPH</p><h2>检查影像</h2></div>${statusBadge(record.status)}</div>
      <div class="viewer"><svg class="radiograph" viewBox="0 0 ${size}" role="img" aria-label="原始 X 线影像及模型候选区域"><image width="100%" height="100%" href="${escapeHTML(base)}/image"/><g id="boxes">${boxLayer}</g></svg></div>
      <div class="viewer-foot"><span>${escapeHTML(record.filename)} · ${Number(record.width)} × ${Number(record.height)}</span>${boxes.length ? '<label class="overlay-switch"><input id="overlay" type="checkbox" checked>候选框</label>' : ""}</div>
    </section><div class="report-side"><section class="card"><p class="eyebrow">EXAM DETAILS</p><h2>主图检查信息</h2><dl><dt>马匹</dt><dd>${escapeHTML(record.horse_name)}</dd><dt>部位 / 方向</dt><dd>球节 · ${escapeHTML(record.projection || "—")}</dd><dt>关注病变</dt><dd>骨折</dd><dt>登记时间</dt><dd>${escapeHTML(dateText(record.created_at))}</dd></dl></section>
    <section class="card results"><p class="eyebrow">MODEL FINDINGS</p><h2>${analyzed ? "辅助检测结果" : "模型检测"}</h2>
      ${analyzed ? `<p class="count"><strong>${boxes.length}</strong> 个候选区域</p>${boxes.length ? `<ul class="findings">${boxes.map((box,index) => `<li><span>区域 ${String(index + 1).padStart(2,"0")}</span><b>置信分数 ${pct(box.score)}</b></li>`).join("")}</ul>` : '<p class="sub">在本次阈值下未检出候选区域，不能排除骨折。</p>'}<dl><dt>模型版本</dt><dd class="mono">${escapeHTML(record.model)}</dd><dt>检测阈值</dt><dd>${pct(record.threshold)}</dd></dl>` : `<p class="sub">影像已保存。开始后将调用球节骨折检测模型。</p><form id="analyze-form"><button class="primary" type="submit">${icon("scan")} 开始 AI 分析</button></form>`}
      <p class="model-note">AI 仅提供参考。模型置信分数不是患病概率，最终结论由人工复核。</p></section></div></div>
    <section class="card review"><div><p class="eyebrow">HUMAN REVIEW</p><h2>人工复核</h2></div>
    ${record.status === "reviewed" ? `<div class="reviewed-note"><p>${escapeHTML(record.review_note)}</p><small>复核人：${escapeHTML(record.reviewer)} · ${escapeHTML(dateText(record.reviewed_at))}</small></div>` : analyzed ? `<form id="review-form"><label for="review-note">复核意见</label><textarea id="review-note" name="note" rows="3" required maxlength="2000" placeholder="结合影像记录人工判断与候选区域核对结果"></textarea><button class="primary" type="submit">确认复核并归档</button></form>` : '<p class="sub">完成模型检测后，可填写人工复核意见。</p>'}</section><p class="error" role="alert" hidden></p>`);
  byId("edit-visit").onclick=()=>editVisit(record,detail);
  if(byId("archive-visit"))byId("archive-visit").onclick=async event=>{
    const button=event.currentTarget, ticket=revision;
    button.disabled=true;
    try {const updated=await api(base+"/archive",{method:"POST"});if(ticket===revision){detail(updated);notify("就诊记录已归档。");}}
    catch(error){if(ticket===revision)notify(formatError(error));}finally{button.disabled=false;}
  };
  const attach=byId("attach-form");
  byId("add-images").onclick=()=>{attach.hidden=!attach.hidden;byId("add-images").setAttribute("aria-expanded",String(!attach.hidden));};
  wireImages(attach);
  action(attach,"正在上传…",async active=>{
    const message=await attachFiles(record.id,attach);
    const updated=await api(base);
    if(active()){if(message)uploadMessage={id:record.id,message};detail(updated);if(!message)notify("影像已保存。");}
  });
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
  if(dialog.open) dialog.close();
  byId("navigation").hidden = byId("identity").hidden = !user;
  byId("display-name").textContent = user?.display_name || "";
  page.setAttribute("aria-busy","false");
  if (!user) { login(); return; }
  const route = location.hash.slice(1);
  const isCase = route.startsWith("case/");
  const create = route === "new" || route === "work";
  const archived = !isCase && !create;
  byId("crumb").textContent = archived ? "就诊记录" : isCase ? "就诊详情" : "新建就诊";
  byId("nav-work").classList.toggle("selected", !archived && !isCase);
  byId("nav-archive").classList.toggle("selected", archived || isCase);
  if (create) { createVisit(); return; }
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
