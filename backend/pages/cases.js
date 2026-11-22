// The Pages workbench keeps originals and their clinical record in one IndexedDB transaction.
import { editReview, exportReview } from "./review.js";
import { exportLayer } from "./layer-export.js";

const dbName = "equivision-pages-cases-v1:" + new URL(self.registration.scope).pathname;
const projections = ["正位", "侧位", "斜位", "背内侧-掌外侧斜位", "背外侧-掌内侧斜位", "其他"];
let connection;

function fail(status, message) {
  throw Object.assign(new Error(message), { status });
}

function idb(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function database() {
  if (!connection) {
    const open = indexedDB.open(dbName, 1);
    open.onupgradeneeded = () => open.result.createObjectStore("cases", { keyPath: "number", autoIncrement: true });
    connection = idb(open).then(db => {
      db.onversionchange = () => { db.close(); connection = undefined; };
      return db;
    }).catch(error => { connection = undefined; throw error; });
  }
  return connection;
}

async function transact(mode, operation) {
  const db = await database();
  const tx = db.transaction("cases", mode);
  const settled = new Promise(resolve => {
    tx.oncomplete = () => resolve(null);
    tx.onabort = () => resolve(tx.error || new Error("病例保存中断，请重试"));
  });
  try {
    const result = await operation(tx.objectStore("cases"));
    const error = await settled;
    if (error) throw error;
    return result;
  } catch (error) {
    try { tx.abort(); } catch { /* Transaction may have already rolled back. */ }
    await settled;
    if (error.name === "QuotaExceededError") fail(507, "当前浏览器存储空间不足，病例未保存");
    throw error;
  }
}

const formattedId = number => "EV-" + String(number).padStart(6, "0");

function publicCase(record, includeEvidence = false) {
  const { number, evidence_revision, signed_evidence, images, ...detail } = record;
  const id = formattedId(number);
  const primary = images[0] || {};
  const imageUrl = imageId => new URL("api/cases/" + id + "/images/" + imageId, self.registration.scope).pathname;
  const result = {
    ...detail, id, filename: primary.filename ?? null, projection: primary.projection ?? null,
    width: primary.width ?? null, height: primary.height ?? null, quality_confirmed: !!primary.quality_confirmed,
    quality: primary.quality ?? null, metadata_reviewer: primary.metadata_reviewer ?? null,
    metadata_reviewed_at: primary.metadata_reviewed_at ?? null,
    images: images.map(({ blob, removed_annotations, ...image }) => ({
      ...image, url: imageUrl(image.id)
    }))
  };
  if (includeEvidence && record.status === "analyzed") result.review_context = String(evidence_revision);
  if (includeEvidence && signed_evidence) {
    result.signed_evidence = { ...signed_evidence, images: signed_evidence.images.map(image => ({...image, url: imageUrl(image.id)})) };
  }
  return result;
}

async function readRecord(store, number) {
  const record = await idb(store.get(number));
  if (!record) fail(404, "未找到此病例");
  record.id = formattedId(number);
  return record;
}

// Read the latest record, apply a synchronous mutation, and write in the same transaction.
// Decoding an image or running the models must complete before opening this transaction.
function update(number, change) {
  return transact("readwrite", async store => {
    const record = await readRecord(store, number);
    change(record);
    await idb(store.put(record));
    return publicCase(record);
  });
}

function textField(value, maxLength, required = false) {
  if (typeof value !== "string" || value.length > maxLength || (required && !value.trim())) {
    fail(422, "请核对马匹名称、编号及备注");
  }
  return value.trim();
}

function blankRecord(horse, comment, user) {
  return {
    horse_name: textField(horse, 80, true), comment: textField(comment, 4000),
    created_by: user.username, created_at: new Date().toISOString(), archived_at: null,
    status: "registered", images: [], has_image: false, detections: [], model: null,
    screening: null, threshold: null, analyzed_at: null, reviewer: null, review_note: null,
    reviewed_at: null, empty_review_confirmed: false, annotation_revision: 0, evidence_revision: 0,
    report: { findings: "", conclusion: "", author: null, updated_at: null, revision: 0 }
  };
}

function containsAnimation(buffer, mime) {
  const bytes = new DataView(buffer);
  if (mime === "image/webp") {
    // An animated WebP has an extended VP8X header with the animation flag set.
    return bytes.byteLength >= 21 && bytes.getUint32(12) === 0x56503858 && Boolean(bytes.getUint8(20) & 2);
  }
  // APNG declares its animation control chunk (acTL) before the first IDAT.
  for (let pos = 8; pos + 12 <= bytes.byteLength; pos += 12 + bytes.getUint32(pos)) {
    const type = bytes.getUint32(pos + 4);
    if (type === 0x6163544c) return bytes.getUint32(pos + 8) > 1;
    if (type === 0x49444154) break;
  }
  return false;
}

async function inspectUpload(form) {
  if (form.get("quality_confirmed") !== "true") fail(422, "请先确认影像质量");
  const projection = form.get("projection");
  if (!projections.includes(projection)) fail(422, "请选择投照方向");
  const file = form.get("image");
  if (!(file instanceof Blob) || !file.size) fail(422, "请选择影像");
  if (file.size > 10 * 1024 * 1024) fail(413, "文件不能超过 10 MiB");
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const header = String.fromCharCode(...bytes);
  let mime;
  if (header.startsWith("\x89PNG\r\n\x1a\n")) mime = "image/png";
  else if (header.startsWith("\xff\xd8\xff")) mime = "image/jpeg";
  else if (header.startsWith("RIFF") && header.slice(8) === "WEBP") mime = "image/webp";
  else fail(422, "请上传 PNG、JPEG 或 WebP 影像");
  if (mime !== "image/jpeg" && containsAnimation(await file.arrayBuffer(), mime)) fail(422, "请上传单帧影像");
  const blob = file.slice(0, file.size, mime);
  let bitmap;
  try { bitmap = await createImageBitmap(blob); }
  catch { fail(422, "无法读取影像，请重新选择完整的原片"); }
  const { width, height } = bitmap;
  bitmap.close();
  if (Math.max(width, height) > 8000 || width * height > 24000000) fail(413, "影像超过尺寸限制");
  return {
    blob, filename: String(file.name || "影像").split(/[\\/]/).pop().slice(0, 255) || "影像",
    projection, quality_confirmed: true, width, height, quality: "未确认",
    metadata_reviewer: null, metadata_reviewed_at: null, annotations: [], measurements: []
  };
}

function attach(record, image) {
  image.id = record.images.length ? String(record.images.length) : "primary";
  record.images.push(image);
  record.has_image = true;
  record.evidence_revision++;
  if (record.status === "registered") {
    record.status = "uploaded";
    record.archived_at = null;
  }
}

const json = (data, status = 200) => Response.json(data, {status, headers: {"Cache-Control": "no-store"}});

export async function handleCases(request, route, user, runInference) {
  const path = route.split("?")[0];
  const [, , resource, identifier, ...parts] = path.split("/");
  const { method } = request;
  if (!identifier) {
    if (method === "POST" && (resource === "visits" || resource === "cases")) {
      let record;
      if (resource === "visits") {
        const data = await request.json();
        record = blankRecord(data.horse_name, data.comment ?? "", user);
      } else {
        const data = await request.formData();
        record = blankRecord(data.get("horse_name"), "", user);
        attach(record, await inspectUpload(data));
      }
      record.number = await transact("readwrite", store => idb(store.add(record)));
      return json(publicCase(record), 201);
    }
    if (resource === "cases" && method === "GET") {
      const rows = await transact("readonly", store => idb(store.getAll()));
      return json(rows.reverse().map(row => publicCase(row)));
    }
    fail(404, "未找到此操作");
  }
  const number = Number(identifier.slice(3));
  if (resource !== "cases" || !Number.isSafeInteger(number) || number <= 0 || formattedId(number) !== identifier) {
    fail(404, "未找到此病例");
  }
  const query = new URL(request.url).searchParams;
  const [action, imageId, imageAction] = parts;
  if (method === "GET") {
    const record = await transact("readonly", store => readRecord(store, number));
    if (!parts.length) return json(publicCase(record, query.get("include_signed_evidence") === "true"));
    if ((action === "image" && parts.length === 1) || (action === "images" && parts.length === 2)) {
      const image = record.images.find(im => im.id === (action === "image" ? "primary" : imageId));
      if (!image) fail(404, "未找到此影像");
      return new Response(image.blob, { headers: { "Content-Type": image.blob.type, "Cache-Control": "no-store" } });
    }
    if (action === "images" && imageAction === "layers" && parts.length === 4) {
      return exportLayer(record, imageId, parts[3]);
    }
    return exportReview(record, parts);
  }
  if (!parts.length && method === "DELETE") {
    await transact("readwrite", async store => {
      await readRecord(store, number);
      await idb(store.delete(number));
    });
    return new Response(null, { status: 204 });
  }
  if (!parts.length && method === "PATCH") {
    const data = await request.json();
    if (data.horse_name == null && data.comment == null) fail(422, "请填写需要更新的资料");
    return json(await update(number, record => {
      const horse = data.horse_name == null ? record.horse_name : textField(data.horse_name, 80, true);
      const comment = data.comment == null ? record.comment : textField(data.comment, 4000);
      if (record.horse_name !== horse || record.comment !== comment) record.evidence_revision++;
      Object.assign(record, {horse_name: horse, comment});
    }));
  }
  if (parts.length === 1 && method === "POST") {
    if (action === "archive") return json(await update(number, record => {
      if (record.has_image && record.status !== "reviewed") fail(409, "请先分析并签署复核报告");
      record.archived_at ??= new Date().toISOString();
    }));
    if (action === "images") {
      const image = await inspectUpload(await request.formData());
      return json(await update(number, record => attach(record, image)), 201);
    }
    if (action === "analyze") {
      const current = await transact("readonly", store => readRecord(store, number));
      if (!current.has_image) fail(409, "请先上传用于分析的主影像");
      if (current.status !== "uploaded") fail(409, "本病例已有分析结果");
      const result = await runInference(current.images[0].blob);
      return json(await update(number, record => {
        if (record.status !== "uploaded") fail(409, "本病例已有分析结果");
        Object.assign(record, {
          screening: result.screening, detections: result.detections, model: result.model,
          threshold: result.threshold, analyzed_at: new Date().toISOString(), status: "analyzed"
        });
        record.evidence_revision++;
      }));
    }
  }
  const body = method === "DELETE" ? null : await request.json();
  const record = await update(number, r => editReview(r, parts, method, body, query, user, publicCase));
  return json(record, method === "POST" && imageAction === "measurements" ? 201 : 200);
}
