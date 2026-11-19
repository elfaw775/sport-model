const escape = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
const paths = {
  pan:"M8 12V7a1.4 1.4 0 0 1 2.8 0v4-6a1.4 1.4 0 0 1 2.8 0v6-5a1.4 1.4 0 0 1 2.8 0v5-2a1.4 1.4 0 0 1 2.8 0v6c0 4-2.3 6-6.1 6h-1c-2.7 0-4.1-1.8-5.5-3.8l-3.1-4a1.5 1.5 0 0 1 2.2-2z",
  distance:"m3 16 13-13 5 5L8 21zm5-3 2 2m1-5 2 2m1-5 2 2",
  angle:"m18 4-14 15h17M11 19a7 7 0 0 0-2-5",
  rectangle:"M3 5h18v14H3z",
  polygon:"m3 16 5-12 13 3-3 13z",
  "zoom-in":"M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0zm-2 5 6 6M7 10h6m-3-3v6",
  "zoom-out":"M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0zm-2 5 6 6M7 10h6",
  rotate:"M19.5 10a8 8 0 1 0 .4 4M20 4v6h-6",
  flip:"M12 2v20M9 6 2 18h7zm6 0 7 12h-7z",
  invert:"M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0zm-9-9v18m0-18a9 9 0 0 1 0 18",
  reset:"M3 8h6M3 8V2m1 6a8 8 0 1 1-1 8",
  compare:"M3 4h18v16H3zm9 0v16",
  film:"M3 4h18v16H3zm0 4h18m-13-4v4m8-4v4",
  add:"M12 5v14m-7-7h14",
  report:"M6 3h10l3 3v15H6zm9 0v5h4M9 12h7m-7 4h5",
  remove:"M4 6h16M9 6V3h6v3M7 7l1 14h8l1-14m-7 3v7m4-7v7"
};
const icon = name => `<svg class="view-icon" aria-hidden="true" viewBox="0 0 24 24"><path d="${paths[name]}"/></svg>`;
const projections = ["未确认", "正位", "侧位", "斜位", "背内侧-掌外侧斜位", "背外侧-掌内侧斜位", "其他"];
const qualities = ["未确认", "良好", "一般", "不适合"];
const sequence = index => String(index + 1).padStart(2, "0");
const newView = () => ({zoom:1, rotation:0, flip:false, invert:false, brightness:100, contrast:100, x:0, y:0});
const choices = (values, selected) => values.map(value => `<option${value === selected ? " selected" : ""}>${escape(value)}</option>`).join("");
const pointString = points => points.map(p => `${Number(p.x)},${Number(p.y)}`).join(" ");
const annotationTool = tool => tool === "rectangle" || tool === "polygon";
const shapeLabel = kind => kind === "rectangle" ? "矩形" : "多边形";
function annotationShape(kind, points) {
  if (kind === "rectangle" && points.length === 2) {
    const [a,b] = points;
    return `<rect x="${Math.min(a.x,b.x)}" y="${Math.min(a.y,b.y)}" width="${Math.abs(a.x-b.x)}" height="${Math.abs(a.y-b.y)}"/>`;
  }
  if (kind === "polygon" && points.length >= 3) return `<polygon points="${pointString(points)}"/>`;
  return `<polyline points="${pointString(points)}"/>`;
}

function toolButton(name, label, attribute, text = false) {
  return `<button type="button" data-${attribute}="${name}" aria-label="${label}" title="${label}">${icon(name)}${text ? `<span>${label}</span>` : ""}</button>`;
}

export function mountViewer(container, record, {api, onUpload, onReport}) {
  const signals = new AbortController();
  const listen = (event, fn) => container.addEventListener(event, fn, {signal:signals.signal});
  const find = sel => container.querySelector(sel);
  const images = () => record.images || [];
  if (!images().length) return () => {};
  const imageById = id => images().find(image => image.id === id);
  const readonly = () => record.status === "reviewed";
  const rootPath = "/api/cases/" + encodeURIComponent(record.id) + "/images/";
  const panes = {left:images()[0].id, right:images()[1]?.id};
  const views = new Map();
  let active = "left", compare = false, tool = "pan", showBoxes = true, pending = false;
  let points = [], drag = null, rectangleStart = null, preview = null;
  let draftId = null, editRevision = 0, showAnnotations = true, originalOnly = false;
  const currentImage = () => imageById(panes[active]);
  const paneNode = pane => find(`.film-stage[data-pane="${pane}"]`);
  const state = pane => {
    const id = panes[pane];
    if (!views.has(id)) views.set(id, newView());
    return views.get(id);
  };
  const toolButtons = [["pan","拖动"],["distance","距离"],["angle","角度"],["rectangle","矩形"],["polygon","多边形"]].map(([k,v]) => toolButton(k,v,"tool",true)).join("");
  const viewButtons = [["rotate","旋转90度"],["flip","水平翻转"],["invert","反色"],["reset","重置视图"]].map(([k,v]) => toolButton(k,v,"view")).join("");
  container.innerHTML = `<section class="study-viewer" id="case-viewer" aria-label="病例阅片工作台">
    <aside class="film-strip"><div class="viewer-section-title">${icon("film")} <h2>检查影像</h2><span class="film-count"></span></div>
      <div class="film-list"></div><button type="button" class="add-film" id="viewer-upload">${icon("add")} 补充影像</button></aside>
    <div class="viewer-desk">
      <div class="view-toolbar" role="toolbar" aria-label="阅片工具"><div class="tool-group">${toolButtons}</div>
      <div class="tool-group zoom-controls">${toolButton("zoom-out","缩小","view")}<output class="view-zoom">100%</output>${toolButton("zoom-in","放大","view")}</div>
      <div class="tool-group view-controls">${viewButtons}</div>
      <div class="tool-group"><button type="button" id="viewer-original" title="只看原图，隐藏标注、测量和模型框" aria-pressed="false">原图</button>
      <button type="button" id="viewer-compare" title="双片对照" aria-pressed="false">${icon("compare")}<span>对照</span></button></div></div>
      <div class="pane-selectors" hidden></div><div class="view-panes"></div>
      <form id="viewer-annotation-form" class="annotation-form" hidden>
        <label>标注名称<input name="label" required maxlength="80" placeholder="例如：近节指骨可疑骨折线" autocomplete="off"></label>
        <button type="submit" id="save-annotation">保存标注</button><button type="button" id="undo-vertex">撤回一点</button>
        <button type="button" id="cancel-annotation">取消</button>
      </form>
      <div class="tone-bar"><label>亮度<input type="range" min="30" max="200" value="100" data-tone="brightness" aria-label="亮度"><output>100%</output></label>
      <label>对比度<input type="range" min="30" max="200" value="100" data-tone="contrast" aria-label="对比度"><output>100%</output></label></div>
    </div>
    <aside class="image-inspector"><div class="inspector-info"></div><div class="inspector-annotations"></div><div class="inspector-measures"></div>
      <p class="viewer-feedback" role="status" aria-live="polite" hidden></p>
      <button type="button" class="primary wide" id="viewer-report">${icon("report")} 分析与复核报告</button></aside>
  </section>`;
  const viewer = find("#case-viewer");

  function feedback(message, error = false) {
    const el = find(".viewer-feedback");
    el.textContent = message;
    el.hidden = !message;
    el.classList.toggle("failed", error);
  }
  function drawStrip() {
    find(".film-count").textContent = images().length;
    find(".film-list").innerHTML = images().map((image, index) => `<button type="button" class="film-card" data-image-id="${escape(image.id)}"
      aria-pressed="${image.id === panes[active]}" aria-label="查看第 ${index + 1} 张 ${escape(image.filename)}">
      <span class="film-thumb"><img src="${escape(image.url)}" alt=""><b>${sequence(index)}</b></span>
      <strong>${image.id === "primary" ? "主图" : "附图"} · ${escape(image.projection)}</strong><small title="${escape(image.filename)}">${escape(image.filename)}</small></button>`).join("");
    const button = find("#viewer-compare");
    button.disabled = images().length < 2;
    button.title = button.disabled ? "至少需要两张影像才能对照" : "双片对照";
    button.setAttribute("aria-pressed", String(compare));
    viewer.classList.toggle("is-comparing", compare);
    find("#viewer-upload").hidden = !onUpload;
    find("#viewer-report").hidden = !onReport;
    const reportLabel = record.status === "reviewed" ? "查看复核报告" : record.status === "analyzed" ? "编写复核报告" : "开始分析";
    find("#viewer-report").innerHTML = icon("report") + " " + reportLabel;
  }
  function drawSelectors() {
    const row = find(".pane-selectors");
    row.hidden = !compare;
    if (!compare) {row.replaceChildren();return;}
    row.innerHTML = ["left","right"].map((pane, i) => `<label>${i ? "右片" : "左片"}<select id="${i ? "viewer-compare-image" : "viewer-primary-image"}" data-pick="${pane}">
      ${images().map((image, n) => `<option value="${escape(image.id)}" ${image.id === panes[pane] ? "selected" : ""}>${sequence(n)} · ${escape(image.projection)} · ${escape(image.filename)}</option>`).join("")}</select></label>`).join("");
  }
  function drawCanvases() {
    find(".view-panes").innerHTML = (compare ? ["left","right"] : ["left"]).map(pane => {
      const image = imageById(panes[pane]);
      const sourceTag = image.id === "primary" ? "主图" : "附图";
      return `<div class="film-stage" data-pane="${pane}" data-image-id="${escape(image.id)}" tabindex="0" aria-label="${sourceTag} ${escape(image.filename)}">
        <svg class="radiograph" aria-label="原始影像与测量" role="img"><g class="image-space">
          <image href="${escape(image.url)}" width="${Number(image.width)}" height="${Number(image.height)}"/>
          <g class="detection-boxes"${image.id === "primary" ? ' id="boxes"' : ""}></g><g class="manual-marks"></g><g class="measured-lines"></g></g></svg>
        <div class="evidence-labels" aria-hidden="true"></div>
        <div class="film-caption"><span>${sourceTag} · ${escape(image.projection)}</span><span>${escape(image.filename)}</span></div>
        <span class="canvas-hint"></span></div>`;
    }).join("");
    for (const pane of compare ? ["left","right"] : ["left"]) { applyView(pane); drawOverlays(pane); }
  }
  function applyView(pane) {
    const stage = paneNode(pane);
    const image = imageById(panes[pane]), view = state(pane);
    const w = Number(image.width), h = Number(image.height);
    const rotated = view.rotation % 180 !== 0;
    const width = rotated ? h : w, height = rotated ? w : h;
    stage.querySelector(".radiograph").setAttribute("viewBox", `${-width/2} ${-height/2} ${width} ${height}`);
    stage.querySelector(".image-space").setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.zoom}) rotate(${view.rotation}) scale(${view.flip ? -1 : 1} 1) translate(${-w/2} ${-h/2})`);
    stage.querySelector("image").style.filter = `brightness(${view.brightness}%) contrast(${view.contrast}%) invert(${view.invert ? 1 : 0})`;
    positionLabels(stage);
  }
  // Use screen-space labels so their text stays upright when the film rotates or flips.
  function positionLabels(stage) {
    const matrix = stage.querySelector(".image-space").getScreenCTM();
    if (!matrix) return;
    const origin = stage.querySelector(".evidence-labels").getBoundingClientRect();
    for (const label of stage.querySelectorAll(".evidence-labels span")) {
      const point = new DOMPoint(Number(label.dataset.x), Number(label.dataset.y)).matrixTransform(matrix);
      label.style.left = (point.x - origin.left) + "px";
      label.style.top = (point.y - origin.top) + "px";
    }
  }
  function drawOverlays(pane) {
    const stage = paneNode(pane), image = imageById(panes[pane]);
    const boxes = image.id === "primary" ? (record.detections || []) : [];
    const boxLayer = stage.querySelector(".detection-boxes");
    boxLayer.style.display = showBoxes ? "" : "none";
    boxLayer.innerHTML = boxes.map((box,i) => `<rect x="${Number(box.x1)}" y="${Number(box.y1)}" width="${Number(box.x2)-Number(box.x1)}" height="${Number(box.y2)-Number(box.y1)}"><title>区域 ${i+1} · ${Math.round(box.score * 1000)/10}%</title></rect>`).join("");
    const drawing = annotationTool(tool) && pane === active;
    const draft = !drawing && pane === active && points.length ? [{points, draft:true}] : [];
    const annotations = showAnnotations ? image.annotations || [] : [];
    let draftPoints = points;
    if (drawing && tool === "rectangle" && preview && points.length < 2) {
      const first = points[0] || rectangleStart?.point;
      if (first) draftPoints = [first,preview];
    }
    const drawingMarkup = drawing && draftPoints.length ? `<g class="manual-draft">${annotationShape(tool,draftPoints)}${draftPoints.map(p=>`<circle cx="${p.x}" cy="${p.y}" r="${Math.max(.7,Math.min(image.width,image.height)/185)}"/>`).join("")}</g>` : "";
    stage.querySelector(".manual-marks").innerHTML = annotations.map(a=>`<g data-manual-id="${escape(a.id)}">${annotationShape(a.kind,a.points)}<title>${escape(a.label)}</title></g>`).join("") + drawingMarkup;
    const records = [...(image.measurements || []), ...draft];
    stage.querySelector(".measured-lines").innerHTML = records.map(m => `<g class="${m.draft ? "measure-draft" : "measure-mark"}">
      <polyline points="${pointString(m.points)}"/>${m.points.map(p => `<circle cx="${Number(p.x)}" cy="${Number(p.y)}" r="${Math.max(.6, Math.min(image.width,image.height) / 185)}"/>`).join("")}</g>`).join("");
    const marks = (image.measurements || []).map((m, i) => {
      const anchor = m.points[1];
      return `<span class="mark-label" data-x="${Number(anchor.x)}" data-y="${Number(anchor.y)}">${sequence(i)} · ${Number(m.value).toFixed(1)} ${escape(m.unit)}</span>`;
    });
    const tags = showBoxes ? boxes.map((box, i) => `<span class="box-label" data-x="${Number(box.x1)}" data-y="${Number(box.y1)}">区域 ${sequence(i)} · ${Math.round(box.score*1000)/10}%</span>`) : [];
    const manual = annotations.map(a=>`<span class="manual-label" data-x="${Number(a.points[0].x)}" data-y="${Number(a.points[0].y)}">人工 · ${escape(a.label)}</span>`);
    stage.querySelector(".evidence-labels").innerHTML = [...tags, ...marks, ...manual].join("");
    positionLabels(stage);
  }
  function inspector() {
    const image = currentImage();
    const projection = image.projection || "未确认", quality = image.quality || "未确认";
    const locked = readonly() ? " disabled" : "";
    find(".inspector-info").innerHTML = `<div class="viewer-section-title"><h2>影像信息</h2><small>${image.id === "primary" ? "分析主图" : "对照附图"}</small></div>
      <p class="inspector-file" title="${escape(image.filename)}">${escape(image.filename)}<small>${Number(image.width)} × ${Number(image.height)} px · ${escape(record.body_part || "球节")}</small></p>
      <form id="viewer-metadata"><label>投照方向<select name="projection"${locked}>${choices(projections,projection)}</select></label>
      <label>影像质量<select name="quality"${locked}>${choices(qualities,quality)}</select></label>
      ${readonly() ? "" : '<button type="submit" class="metadata-save">保存影像信息</button>'}</form>
      ${image.metadata_reviewer ? `<small class="metadata-by">核对人 · ${escape(image.metadata_reviewer)}</small>` : ""}
      ${image.id === "primary" && record.detections?.length ? `<label class="overlay-switch"><input type="checkbox" id="viewer-boxes" ${showBoxes ? "checked" : ""}> 显示模型候选框</label>` : ""}`;
    const saved = image.measurements || [];
    const annotations = image.annotations || [];
    find(".inspector-annotations").innerHTML = `<div class="viewer-section-title"><h2>人工标注</h2><span class="film-count">${annotations.length}</span></div>
      ${annotations.length ? `<label class="overlay-switch"><input id="annotation-layer" type="checkbox" ${showAnnotations ? "checked" : ""}> 显示人工标注</label>
        <ul class="annotation-list">${annotations.map(a=>`<li data-annotation-id="${escape(a.id)}"><strong>${escape(a.label)}</strong>
          <small>${shapeLabel(a.kind)} · ${escape(a.updated_by)}</small>
          ${readonly() ? "" : `<div><button type="button" data-rename="${escape(a.id)}">修改名称</button><button type="button" data-delete-annotation="${escape(a.id)}">删除</button></div>`}</li>`).join("")}</ul>
        <a class="annotation-export" id="annotation-export" download href="${rootPath}${encodeURIComponent(image.id)}/annotations.png">导出标注 PNG</a>` : 
        `<p class="measure-empty">${readonly() ? "本片无人工标注" : "选择矩形或多边形，在本片上标记关注区域。"}</p>`}`;
    find(".inspector-measures").innerHTML = `<div class="viewer-section-title"><h2>测量记录</h2><span class="film-count">${saved.length}</span></div>
      <div class="measurement-list">${saved.map((m,i) => `<div class="measurement-item" data-measure-id="${escape(m.id)}"><div><strong>${m.kind === "angle" ? "角度" : "距离"} ${sequence(i)}</strong>
        <b>${Number(m.value).toFixed(1)} ${escape(m.unit)}</b><small>${escape(m.created_by)}</small></div>${readonly() ? "" : `<button type="button" data-remove-measurement="${escape(m.id)}" aria-label="删除测量 ${i+1}" title="删除此测量">${icon("remove")}</button>`}</div>`).join("") || '<p class="measure-empty">本片暂无测量</p>'}</div>
      <p class="measure-unit">像素间距未标定，距离以 px 显示。</p><button id="viewer-cancel" type="button" hidden>取消本次测量</button>`;
  }
  function syncTools() {
    const view = state(active);
    const drawing = annotationTool(tool), draftActive = drawing && (points.length || editRevision || find("#viewer-annotation-form").elements.label.value.trim());
    const form = find("#viewer-annotation-form");
    form.hidden = !drawing;
    form.elements.label.disabled = pending;
    find("#save-annotation").disabled = pending || points.length < (tool === "polygon" ? 3 : 2);
    find("#undo-vertex").disabled = pending || !points.length || editRevision > 0;
    find("#cancel-annotation").disabled = pending;
    const download = find("#annotation-export");
    if (download) download.hidden = Boolean(draftActive || pending);
    for (const b of container.querySelectorAll("[data-tool]")) {
      b.setAttribute("aria-pressed", String(b.dataset.tool === tool));
      b.disabled = pending || Boolean(draftActive) || (readonly() && b.dataset.tool !== "pan");
    }
    for (const b of container.querySelectorAll("[data-view=flip],[data-view=invert]")) b.setAttribute("aria-pressed", String(view[b.dataset.view]));
    find(".view-zoom").textContent = Math.round(view.zoom * 100) + "%";
    for (const input of container.querySelectorAll("[data-tone]")) {
      input.value = view[input.dataset.tone];
      input.disabled = originalOnly;
      input.nextElementSibling.textContent = input.value + "%";
    }
    viewer.dataset.tool = tool;
    viewer.classList.toggle("original-only",originalOnly);
    find("#viewer-original").setAttribute("aria-pressed",String(originalOnly));
    find("#viewer-original").disabled = pending || Boolean(draftActive);
    find("[data-view=invert]").disabled = originalOnly;
    for (const b of container.querySelectorAll(".overlay-switch input")) b.disabled = originalOnly || pending || Boolean(draftActive);
    for (const stage of container.querySelectorAll(".film-stage")) {
      const selected = stage.dataset.pane === active;
      stage.classList.toggle("active-pane", selected);
      const hint = !selected ? "点选此片操作" : tool === "pan" ? "拖动平移" : pending ? "正在保存…" :
        editRevision ? "修改名称后保存" : tool === "rectangle" ? ["拖出矩形或点选两个对角","点选另一个对角","填写名称后保存"][points.length] :
        tool === "polygon" ? `依次点选边界 · 已选 ${points.length} 点，至少 3 点后可保存` :
        tool === "distance" ? `距离 · 点选${points.length ? "终点" : "起点"}` : `角度 · 点选${["第一点","顶点","第三点"][points.length] || "第三点"}`;
      stage.querySelector(".canvas-hint").textContent = hint;
    }
    find("#viewer-cancel").hidden = drawing || !points.length || pending;
    viewer.classList.toggle("viewer-saving", pending);
    for (const b of container.querySelectorAll("[data-remove-measurement],.metadata-save,#viewer-upload,#viewer-report,[data-pick],.film-card,[data-rename],[data-delete-annotation]")) b.disabled = pending || Boolean(draftActive);
    find("#viewer-compare").disabled = images().length < 2 || pending || Boolean(draftActive);
    return Boolean(draftActive);
  }
  function clearDraft() {
    points = [];draftId = null;editRevision = 0;rectangleStart = preview = null;
    find("#viewer-annotation-form").elements.label.value = "";
  }
  function startAnnotation(kind, annotation) {
    clearDraft();tool = kind;showAnnotations = true;originalOnly = false;
    draftId = annotation?.id || crypto.randomUUID();
    editRevision = annotation?.revision || 0;
    points = annotation ? [...annotation.points] : [];
    find("#viewer-annotation-form").elements.label.value = annotation?.label || "";
    redraw();feedback("");
    if (annotation) find("#viewer-annotation-form").elements.label.focus();
  }
  function redraw() { drawStrip();drawSelectors();drawCanvases();inspector();syncTools(); }

  async function write(path, method, data) {
    if (pending) return false;
    pending = true;
    syncTools();
    try {
      const options = {method};
      if (data) { options.headers = {"Content-Type":"application/json"}; options.body = JSON.stringify(data); }
      const updated = await api(path, options);
      if (signals.signal.aborted || !container.isConnected) return false;
      Object.assign(record, updated);
      feedback("已保存");
      return true;
    } catch (error) {
      if (!signals.signal.aborted && container.isConnected) feedback(error.status === 409 ? error.message + " 请刷新病例后重新核对。" : error.message || "保存失败，请重试。", true);
      return false;
    } finally {
      pending = false;
      if (!signals.signal.aborted && container.isConnected) syncTools();
    }
  }

  function activate(pane) {
    if (active === pane || pending) return;
    points = [];
    drawOverlays(active);
    active = pane;
    drawStrip();inspector();syncTools();feedback("");
  }
  function choose(pane, id) {
    if (pending || !imageById(id)) return;
    const other = pane === "left" ? "right" : "left";
    if (compare && panes[other] === id) panes[other] = panes[pane];
    active = pane;panes[pane] = id;points = [];
    redraw();feedback("");
  }
  function changeView(key) {
    const view = state(active);
    if (key === "zoom-in" || key === "zoom-out") view.zoom = Math.min(5, Math.max(.25, view.zoom * (key === "zoom-in" ? 1.25 : .8)));
    if (key === "rotate") {view.rotation = (view.rotation + 90) % 360;view.x = view.y = 0;}
    if (key === "flip" || key === "invert") view[key] = !view[key];
    if (key === "reset") Object.assign(view, newView());
    applyView(active);syncTools();
  }
  const toPoint = (event, el) => new DOMPoint(event.clientX, event.clientY).matrixTransform(el.getScreenCTM().inverse());
  listen("pointerdown", event => {
    const stage = event.target.closest(".film-stage");
    if (!stage || event.button !== 0 || !event.isPrimary || pending) return;
    if (stage.dataset.pane !== active && annotationTool(tool) && syncTools()) return;
    activate(stage.dataset.pane);
    if (tool === "rectangle" && !points.length && !editRevision) {
      const point = toPoint(event,stage.querySelector(".image-space")), image = currentImage();
      if (point.x >= 0 && point.y >= 0 && point.x < image.width && point.y < image.height)
        rectangleStart = {point:{x:point.x,y:point.y}, x:event.clientX, y:event.clientY};
    }
    if (tool !== "pan") return;
    const view = state(active);
    const point = toPoint(event,stage.querySelector(".radiograph"));
    drag = {id:event.pointerId, stage, point, x:view.x, y:view.y};
    stage.setPointerCapture(event.pointerId);
  });
  listen("pointermove", event => {
    if (tool === "rectangle" && !pending && !editRevision && (rectangleStart || points.length === 1)) {
      const stage = event.target.closest(".film-stage");
      if (stage?.dataset.pane === active) {
        const p = toPoint(event,stage.querySelector(".image-space")), im = currentImage();
        preview = {x:Math.max(0, Math.min(im.width - 1, p.x)), y:Math.max(0, Math.min(im.height - 1, p.y))};
        drawOverlays(active);
      }
    }
    if (!drag || drag.id !== event.pointerId) return;
    const point = toPoint(event, drag.stage.querySelector(".radiograph"));
    const view = state(active);
    view.x = drag.x + point.x - drag.point.x;
    view.y = drag.y + point.y - drag.point.y;
    applyView(active);
  });
  listen("lostpointercapture", () => { drag = null; });
  listen("pointercancel", () => { drag = rectangleStart = preview = null;drawOverlays(active); });
  listen("pointerup", async event => {
    const stage = event.target.closest(".film-stage");
    if (!stage || stage.dataset.pane !== active || event.button !== 0 || !event.isPrimary || pending || tool === "pan" || readonly() || editRevision) return;
    const image = currentImage();
    const p = toPoint(event, stage.querySelector(".image-space"));
    if (p.x < 0 || p.y < 0 || p.x >= image.width || p.y >= image.height) {
      rectangleStart = preview = null;drawOverlays(active);
      feedback("请在影像范围内取点。", true);return;
    }
    const point = {x:p.x, y:p.y};
    if (tool === "rectangle") {
      const start = rectangleStart;
      rectangleStart = preview = null;
      if (points.length === 2) return;
      if (!points.length && start && Math.hypot(event.clientX-start.x,event.clientY-start.y) > 4) points.push(start.point);
      if (points.length && (Math.abs(points[0].x-p.x) < .01 || Math.abs(points[0].y-p.y) < .01)) {
        feedback("矩形需要有宽度和高度。", true);drawOverlays(active);syncTools();return;
      }
    }
    if (tool === "polygon" && points.length >= 32) {feedback("每个多边形最多 32 点。", true);return;}
    if (points.length && Math.hypot(point.x-points.at(-1).x, point.y-points.at(-1).y) < .001) {
      feedback("请选取不同的位置。", true);return;
    }
    points.push(point);drawOverlays(active);syncTools();feedback("");
    if (annotationTool(tool)) return;
    if (points.length < (tool === "distance" ? 2 : 3)) return;
    const ok = await write(rootPath + encodeURIComponent(image.id) + "/measurements", "POST", {kind:tool, points:[...points]});
    if (signals.signal.aborted || !container.isConnected) return;
    if (ok) {points = [];redraw();}
    else {points.pop();drawOverlays(active);syncTools();}
  });
  listen("click", async event => {
    const button = event.target.closest("button");
    if (!button || button.disabled) return;
    if (button.id === "viewer-upload") return onUpload?.(record);
    if (button.id === "viewer-report") return onReport?.(record);
    if (button.dataset.view) return changeView(button.dataset.view);
    if (pending) return;
    if (button.id === "viewer-original") {
      originalOnly = !originalOnly;clearDraft();tool = "pan";redraw();feedback("");return;
    }
    if (button.id === "cancel-annotation") {
      clearDraft();tool = "pan";redraw();feedback("");return;
    }
    if (button.id === "undo-vertex") {
      points.pop();preview = null;drawOverlays(active);syncTools();return;
    }
    if (button.dataset.rename) {
      const a = currentImage().annotations.find(a=>a.id === button.dataset.rename);
      startAnnotation(a.kind,a);return;
    }
    if (button.dataset.deleteAnnotation) {
      const image = currentImage(), a = image.annotations.find(a=>a.id === button.dataset.deleteAnnotation);
      if (!window.confirm(`删除本片的人工标注「${a.label}」？`)) return;
      if (await write(rootPath + encodeURIComponent(image.id) + "/annotations/" + a.id + "?revision=" + a.revision,"DELETE")) redraw();
      return;
    }
    if (button.classList.contains("film-card")) return choose(active, button.dataset.imageId);
    if (button.id === "viewer-compare") {
      compare = !compare;active = "left";points = [];
      if (compare && (!panes.right || panes.right === panes.left)) panes.right = images().find(img => img.id !== panes.left).id;
      redraw();feedback("");return;
    }
    if (button.dataset.tool || button.id === "viewer-cancel") {
      if (annotationTool(button.dataset.tool)) {startAnnotation(button.dataset.tool);return;}
      if (button.dataset.tool && button.dataset.tool !== "pan") originalOnly = false;
      if (button.dataset.tool) tool = button.dataset.tool;
      points = [];drawOverlays(active);syncTools();feedback("");return;
    }
    if (button.hasAttribute("data-remove-measurement")) {
      const image = currentImage();
      if (await write(rootPath + encodeURIComponent(image.id) + "/measurements/" + encodeURIComponent(button.dataset.removeMeasurement), "DELETE")) redraw();
    }
  });
  listen("change", event => {
    const input = event.target;
    if (input.dataset.pick) return choose(input.dataset.pick, input.value);
    if (input.id === "annotation-layer") {
      showAnnotations = input.checked;
      for (const pane of compare ? ["left","right"] : ["left"]) drawOverlays(pane);
    }
    if (input.id === "viewer-boxes") {
      showBoxes = input.checked;
      for (const pane of compare ? ["left","right"] : ["left"]) drawOverlays(pane);
    }
  });
  listen("input", event => {
    const input = event.target;
    if (input.name === "label") {syncTools();return;}
    if (!input.dataset.tone) return;
    state(active)[input.dataset.tone] = Number(input.value);
    applyView(active);syncTools();
  });
  listen("submit", async event => {
    if (event.target.id === "viewer-annotation-form") {
      event.preventDefault();
      if (readonly() || pending || !annotationTool(tool)) return;
      const label = event.target.elements.label.value.trim(), image = currentImage();
      if (!label) {feedback("请输入标注名称。",true);return;}
      if (await write(rootPath + encodeURIComponent(image.id) + "/annotations/" + draftId,"PUT",{kind:tool,label,points:[...points],revision:editRevision})) {
        tool = "pan";clearDraft();redraw();
      }
      return;
    }
    if (event.target.id !== "viewer-metadata") return;
    event.preventDefault();
    const image = currentImage();
    const form = event.target;
    if (readonly() || pending) return;
    if (await write(rootPath + encodeURIComponent(image.id), "PATCH", {projection:form.elements.projection.value, quality:form.elements.quality.value})) redraw();
  });
  redraw();
  const resize = new ResizeObserver(() => {
    for (const stage of container.querySelectorAll(".film-stage")) positionLabels(stage);
  });
  resize.observe(find(".view-panes"));
  return () => {signals.abort();resize.disconnect();drag = null;};
}
