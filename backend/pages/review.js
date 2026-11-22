import {largestQuadrilateral, simplePolygon} from "./static/polygon.js";
// Synchronous edits run inside the case store's IndexedDB write transaction.
function fail(status, message) {
  throw Object.assign(new Error(message), {status});
}

function textField(value, limit) {
  if (typeof value !== "string" || value.length > limit) fail(422, "请检查字段内容与长度。");
  return value.trim();
}

function pointsOnImage(image, points, min, max) {
  if (!Array.isArray(points) || points.length < min || points.length > max) fail(422, "取点数量不符。");
  return points.map(point => {
    const {x, y} = point || {};
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= image.width || y >= image.height) {
      fail(422, "取点须在原片内。");
    }
    return {x, y};
  });
}

function shapePoints(image, {kind, points}, original) {
  const pts = pointsOnImage(image, points, 2, 32);
  if (kind === "rectangle") {
    const [a,b] = pts;
    if (pts.length !== 2 || a.x === b.x || a.y === b.y) fail(422, "请使用两个对角点画出有面积的矩形。");
    return [{x:Math.min(a.x,b.x), y:Math.min(a.y,b.y)}, {x:Math.max(a.x,b.x), y:Math.max(a.y,b.y)}];
  }
  // Keep the original boundary when only renaming a saved annotation.
  const unchanged = original?.kind === kind && JSON.stringify(original.points) === JSON.stringify(pts);
  if (kind === "polygon" && pts.length === 4 && !unchanged) {
    const ring = largestQuadrilateral(pts);
    if (ring) return ring;
  } else if (kind === "polygon" && simplePolygon(pts)) return pts;
  fail(422, "请选择互不重合、能围出面积的顶点；五点以上请沿边界依次取点。");
}

function editAnnotation(image, id, method, body, query, actor, now) {
  if (!/^[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}$/i.test(id)) fail(422, "标注编号无效。");
  const original = image.annotations.find(a => a.id === id);
  const removed = image.removed_annotations?.[id];
  if (method === "DELETE") {
    const revision = Number(query.get("revision"));
    if (!Number.isSafeInteger(revision) || revision < 1) fail(422, "缺少有效的标注版本。");
    if (removed?.revision === revision + 1 && removed.updated_by === actor) return false;
    if (!original && !removed) fail(404, "未找到此影像的标注。");
    if (!original || original.revision !== revision) fail(409, "标注已变化，请刷新影像后核对。");
    image.removed_annotations ??= {};
    image.removed_annotations[id] = {revision: revision + 1, updated_by: actor};
    image.annotations.splice(image.annotations.indexOf(original), 1);
    return true;
  }
  const {kind, revision} = body;
  const label = textField(body.label, 80), points = shapePoints(image, body, original);
  if (!label || !Number.isSafeInteger(revision) || revision < 0) fail(422, "请填写标注名称和有效的版本号。");
  if (removed || (!original && revision !== 0)) fail(409, "标注已移除或不存在，请刷新影像。");
  const same = original?.kind === kind && original.label === label && JSON.stringify(original.points) === JSON.stringify(points);
  if (original && original.revision !== revision) {
    if (same && original.revision === revision + 1 && original.updated_by === actor) return false;
    fail(409, "标注已被更新，请刷新影像后核对。");
  }
  if (same) return false;
  const changes = {kind, label, points, revision:revision + 1, updated_by:actor, updated_at:now};
  if (original) Object.assign(original, changes);
  else image.annotations.push({id, source:"manual", created_by:actor, created_at:now, ...changes});
  return true;
}

function addMeasurement(image, body, actor, now) {
  const {kind} = body;
  if (!["distance","angle"].includes(kind)) fail(422, "测量类型无效。");
  const count = kind === "distance" ? 2 : 3;
  const points = pointsOnImage(image, body.points, count, count);
  for (let i=1; i<points.length; i++) {
    if (points[i].x === points[i-1].x && points[i].y === points[i-1].y) fail(422, "相邻测量点不能重合。");
  }
  const [a,b,c] = points;
  let value = Math.hypot(a.x-b.x, a.y-b.y);
  if (kind === "angle") {
    const angle = Math.abs(Math.atan2(a.y-b.y,a.x-b.x) - Math.atan2(c.y-b.y,c.x-b.x)) * 180/Math.PI;
    value = Math.min(angle, 360-angle);
  }
  image.measurements.push({id:crypto.randomUUID(), kind, points, value, unit:kind === "angle" ? "°" : "px",
    created_by:actor, created_at:now});
}

function writeReport(record, body, sign, actor, now, publicCase) {
  if (!record.images.length || record.status === "reviewed") fail(409, "请在上传影像后、确认复核前编辑报告。");
  if (sign && record.status !== "analyzed") fail(409, "仅待复核病例可以提交复核。");
  if (sign && (body.annotation_revision ?? 0) !== record.annotation_revision) fail(409, "人工标注已变化，请重新核对。");
  if (sign && body.review_context != null && body.review_context !== String(record.evidence_revision)) {
    fail(409, "病例或影像资料已有更新，请重新核对后签署。");
  }
  const legacy = sign && Object.hasOwn(body, "note");
  const report = record.report;
  const findings = legacy ? "" : textField(body.findings, 4000);
  const conclusion = textField(legacy ? body.note : body.conclusion, 2000);
  if (report.revision !== (legacy ? 0 : body.revision)) fail(409, "报告已更新，请重新读取病例后核对。");
  const empty = !findings && !conclusion;
  if (empty && !(sign && body.empty_review_confirmed === true)) {
    fail(422, sign ? "复核内容为空，请再次确认留空提交。" : "请填写影像所见或人工结论。");
  }
  if (empty || findings !== report.findings || conclusion !== report.conclusion) {
    record.report = {findings, conclusion, author:actor, updated_at:now, revision:report.revision+1};
  }
  if (sign) {
    Object.assign(record, {status:"reviewed", review_note:conclusion, reviewed_at:now, reviewer:actor, empty_review_confirmed:empty});
    const source = publicCase(record);
    const keys = ["id","horse_name","comment","created_by","created_at","images","report","screening","detections",
      "model","threshold","analyzed_at","reviewer","reviewed_at","empty_review_confirmed"];
    record.signed_evidence = structuredClone(Object.fromEntries(keys.map(key => [key,source[key]])));
    for (const image of record.signed_evidence.images) delete image.url;
  }
}

export function editReview(record, parts, method, body, query, user, publicCase) {
  const [route,id,action,mark] = parts;
  const actor = user.display_name, now = new Date().toISOString();
  if (parts.length === 1 && ((route === "report" && method === "PUT") || (route === "review" && method === "POST"))) {
    writeReport(record, body, route === "review", actor, now, publicCase);
    return;
  }
  const image = route === "images" && record.images.find(im => im.id === id);
  if (!image) fail(404, "未找到此影像。");
  if (record.status === "reviewed") fail(409, "病例已复核，阅片记录只读。");
  if (parts.length === 2 && method === "PATCH") {
    const views = ["正位","侧位","斜位","背内侧-掌外侧斜位","背外侧-掌内侧斜位","其他","未确认"];
    const {projection, quality} = body;
    if ((projection == null && quality == null) || (projection != null && !views.includes(projection)) ||
        (quality != null && !["未确认","良好","一般","不适合"].includes(quality))) fail(422, "请选择体位或影像质量。");
    Object.assign(image, {projection:projection ?? image.projection, quality:quality ?? image.quality,
      metadata_reviewer:actor, metadata_reviewed_at:now});
  } else if (action === "measurements" && parts.length === 3 && method === "POST") {
    addMeasurement(image, body, actor, now);
  } else if (action === "measurements" && parts.length === 4 && method === "DELETE") {
    const index = image.measurements.findIndex(m => m.id === mark);
    if (index === -1) fail(404, "未找到此影像的测量记录。");
    image.measurements.splice(index,1);
  } else if (action === "annotations" && parts.length === 4 && ["PUT","DELETE"].includes(method)) {
    if (!editAnnotation(image, mark, method, body, query, actor, now)) return;
    record.annotation_revision++;
  } else fail(404, "未找到此阅片操作。");
  record.evidence_revision++;
}

function escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function field(label, value) {
  return '<p><strong>' + label + '</strong>　' + escape(value === "" || value == null ? "未记录" : value) + '</p>';
}

const coords = points => points.map(p => "(" + p.x + ", " + p.y + ")").join("、");

function reportDocument(record) {
  const data = record.signed_evidence || record;
  const film = data.screening;
  const images = data.images.map((im,i) => '<section><h3>' + (i ? "附图 " + i : "主图") + ' · ' + escape(im.filename) + '</h3>' +
    field("投照与尺寸", im.projection + " · " + im.width + " × " + im.height + " px") + field("影像质量",im.quality) +
    im.annotations.map((mark,n) => field("人工标注 " + (n+1), mark.label + " · " + (mark.kind === "rectangle" ? "矩形" : "多边形") +
      " · " + coords(mark.points) + " px · " + mark.updated_by)).join("") +
    im.measurements.map((m,n) => field("人工测量 " + (n+1), (m.kind === "angle" ? "角度" : "距离") + " · " + m.value.toFixed(2) +
      " " + m.unit + " · " + coords(m.points) + " px · " + m.created_by)).join("") + '</section>').join("");
  const boxes = data.detections.map((box,i) => field("候选 " + (i+1), "分数 " + box.score + " · " +
    coords([{x:box.x1,y:box.y1},{x:box.x2,y:box.y2}]) + " px")).join("");
  return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\';style-src \'unsafe-inline\'">' +
    '<title>' + escape(data.id) + ' 影像复核报告</title><style>body{max-width:800px;margin:auto;padding:28px;font:16px/1.6 system-ui,sans-serif;color:#203b31;overflow-wrap:anywhere}header{border-bottom:2px solid #265b46}p{white-space:pre-wrap;margin:.45em 0}h2,h3{break-after:avoid}section{margin-bottom:1.2em}@media print{body{padding:0;font-size:10.5pt}}</style>' +
    '<main><header><p>EQUIVISION · 运动马影像工作台</p><h1>影像复核报告</h1>' +
    field("病例编号",data.id) + '</header><h2>' + escape(data.horse_name) + '</h2>' +
    field("资料来源",record.signed_evidence ? "签署时保存的病例资料" : "历史报告 · 按现存病例资料整理") +
    field("登记时间",data.created_at) + field("就诊备注",data.comment) + '<h2>人工复核</h2>' +
    field("影像所见",data.report.findings) + field("复核结论",data.report.conclusion) +
    (data.empty_review_confirmed ? '<p>复核人已二次确认两项留空。</p>' : '') +
    field("复核人",data.reviewer) + field("签署时间",data.reviewed_at) + '<h2>模型辅助分析（主图）</h2>' +
    (film ? field("整片筛查模型",film.model) + field("筛查分数",film.score) + field("提示阈值",film.threshold) +
      field("筛查结果",film.predicted_positive ? "达到提示阈值" : "低于提示阈值") : field("整片筛查",null)) +
    field("定位模型",data.model) + field("候选阈值",data.threshold) + field("候选区域",data.detections.length + " 个") + boxes +
    field("分析时间",data.analyzed_at) + '<p>模型输出仅供人工阅片参考。</p><h2>检查影像与阅片记录</h2>' + images + '</main></html>';
}

// Canvas preserves the raster. Insert the same UTF-8 annotation legend as the Python PNG export.
function pngWithLegend(png, marks) {
  const legend = marks.map(({id,label,kind,points,revision}, i) => ({number:i+1,id,label,kind,points,revision}));
  const chunkData = new TextEncoder().encode("EQUIVISION.manual_annotations\0\0\0\0\0" + JSON.stringify(legend));
  const chunk = new Uint8Array(12 + chunkData.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, chunkData.length);
  chunk.set([105,84,88,116],4); // iTXt
  chunk.set(chunkData,8);
  let crc = 0xffffffff;
  for (const byte of chunk.subarray(4,chunk.length-4)) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  view.setUint32(chunk.length-4, (crc ^ 0xffffffff) >>> 0);
  return new Blob([png.slice(0,-12),chunk,png.slice(-12)], {type:"image/png"});
}

export async function renderRegions(image, marks) {
  const bitmap = await createImageBitmap(image.blob, {imageOrientation:"from-image"});
  const canvas = new OffscreenCanvas(image.width, image.height);
  const ctx = canvas.getContext("2d");
  try { ctx.drawImage(bitmap,0,0); } finally { bitmap.close(); }
  const width = Math.max(1,Math.round(Math.min(image.width,image.height)/240));
  const font = Math.max(10,Math.min(32,width*5));
  ctx.font = "bold " + font + "px sans-serif";
  ctx.textBaseline = "top";
  for (const [index, mark] of marks.entries()) {
    const [first,second] = mark.points;
    ctx.beginPath();
    if (mark.kind === "rectangle") ctx.rect(first.x,first.y,second.x-first.x,second.y-first.y);
    else {
      ctx.moveTo(first.x,first.y);
      for (const p of mark.points.slice(1)) ctx.lineTo(p.x,p.y);
      ctx.closePath();
    }
    ctx.lineWidth = width+2; ctx.strokeStyle = "#07392b"; ctx.stroke();
    ctx.lineWidth = width; ctx.strokeStyle = mark.source === "ai" ? "#ffbd58" : "#54efac"; ctx.stroke();
    const number = mark.reference || String(index+1), tagWidth = Math.ceil(ctx.measureText(number).width)+6;
    const x = Math.max(0,Math.min(first.x,image.width-tagWidth));
    const y = Math.max(0,Math.min(first.y-font-5,image.height-font-5));
    ctx.fillStyle = "#07392b"; ctx.fillRect(x,y,tagWidth,font+5);
    ctx.fillStyle = "#ffffff"; ctx.fillText(number,x+3,y+2);
  }
  return canvas.convertToBlob({type:"image/png"});
}

export async function exportReview(record, parts) {
  let payload, type, name;
  if (parts.length === 2 && parts[0] === "report" && parts[1] === "download") {
    if (record.status !== "reviewed") fail(409, "请先签署复核报告。");
    payload = reportDocument(record);
    name = record.id + "-report.html"; type = "text/html; charset=utf-8";
  } else if (parts.length === 3 && parts[0] === "images" && parts[2] === "annotations.png") {
    const image = record.images.find(im => im.id === parts[1]);
    if (!image) fail(404, "未找到此影像。");
    if (!image.annotations.length) fail(409, "此影像尚无已保存的人工标注。");
    payload = pngWithLegend(await renderRegions(image, image.annotations), image.annotations);
    name = record.id + "-" + image.id + "-annotations.png"; type = "image/png";
  } else fail(404, "未找到此下载。");
  return new Response(payload, {headers:{"Content-Type":type, "Cache-Control":"private, no-store",
    "Content-Disposition":'attachment; filename="' + name + '"'}});
}
