import * as ort from "./ort/ort.wasm.min.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL("./ort/", import.meta.url).href;
const yoloDigest = "61286f939c6a6c4a591d5034e33794ba0af99b5f1f29aac279bb15974c025b0a";
const threshold = 0.20;
let models;

async function loadSession(filename, digest) {
  const bytes = Uint8Array.from(digest.match(/../g), pair => parseInt(pair, 16));
  const integrity = "sha256-" + btoa(String.fromCharCode(...bytes));
  const response = await fetch(new URL("./models/" + filename, import.meta.url), {integrity});
  if (!response.ok) throw new Error("模型文件加载失败：" + filename);
  return ort.InferenceSession.create(new Uint8Array(await response.arrayBuffer()), {executionProviders:["wasm"]});
}

async function loadModels() {
  const response = await fetch(new URL("./models/dinov2-fetlock-84.json", import.meta.url));
  if (!response.ok) throw new Error("无法读取整片筛查配置");
  const manifest = await response.json();
  if (manifest.file !== "dinov2-fetlock-84.onnx" || !/^[a-f0-9]{64}$/.test(manifest.sha256) ||
      manifest.model_id !== "DINOv2-Small-refit84@" + manifest.sha256.slice(0,12) ||
      !(manifest.threshold > 0 && manifest.threshold < 1)) throw new Error("整片筛查配置无效");
  const dino = await loadSession(manifest.file, manifest.sha256);
  try {
    const yolo = await loadSession("yolo11s-fetlock-640.onnx", yoloDigest);
    return {dino, yolo, manifest};
  } catch (error) {
    await dino.release();
    throw error;
  }
}

function roundEven(value) {
  const floor = Math.floor(value);
  return value - floor === 0.5 ? floor + floor % 2 : Math.round(value);
}

function layout(image, edge) {
  const scale = edge / Math.max(image.width, image.height);
  const width = Math.max(1, roundEven(image.width * scale));
  const height = Math.max(1, roundEven(image.height * scale));
  return {width, height, left: Math.floor((edge - width) / 2), top: Math.floor((edge - height) / 2)};
}

// Pillow uses a widened bicubic kernel for downsampling and rounds to 8 bits after each pass.
function cubicAxis(src, dst) {
  const scale = src / dst, stretch = Math.max(1, scale);
  return Array.from({length:dst}, (_, out) => {
    const center = (out + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - 2 * stretch + 0.5));
    const end = Math.min(src, Math.floor(center + 2 * stretch + 0.5));
    const weights = Array.from({length:end-start}, (_, i) => {
      const x = Math.abs((start + i + 0.5 - center) / stretch);
      if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
      return x < 2 ? -0.5 * (x - 1) * (x - 2) ** 2 : 0;
    });
    const sum = weights.reduce((a,b) => a+b, 0);
    return {start, weights:weights.map(w => Math.round(w / sum * 4194304))};
  });
}

function resample(data, offset, stride, weights) {
  let sum = 2097152;
  for (const weight of weights) { sum += weight * data[offset]; offset += stride; }
  return Math.max(0, Math.min(255, Math.floor(sum / 4194304)));
}

function resizeAxis(data, width, height, axis, horizontal) {
  const outWidth = horizontal ? axis.length : width, outHeight = horizontal ? height : axis.length;
  if (outWidth === width && outHeight === height) return data;
  const result = new Uint8Array(outWidth * outHeight);
  for (let y=0; y<outHeight; y++) {
    for (let x=0; x<outWidth; x++) {
      const {start, weights} = axis[horizontal ? x : y];
      const offset = horizontal ? y*width+start : start*width+x;
      result[y*outWidth+x] = resample(data, offset, horizontal ? 1 : width, weights);
    }
  }
  return result;
}

function screeningTensor(image) {
  const edge = 336, plane = edge * edge;
  const {width, height, top, left} = layout(image, edge);
  const gray = new Uint8Array(image.width * image.height);
  for (let p = 0; p < gray.length; p++) {
    const i = p * 4;
    gray[p] = (19595*image.data[i] + 38470*image.data[i+1] + 7471*image.data[i+2] + 32768) >> 16;
  }
  const xAxis = cubicAxis(image.width, width), yAxis = cubicAxis(image.height, height);
  // Pillow 13 changes pass order for tall images. The intermediate 8-bit rounding matters.
  const verticalFirst = image.height > height && image.height-height > 2*(image.width-width);
  const scaled = verticalFirst
    ? resizeAxis(resizeAxis(gray, image.width, image.height, yAxis, false), image.width, height, xAxis, true)
    : resizeAxis(resizeAxis(gray, image.width, image.height, xAxis, true), width, image.height, yAxis, false);
  const padded = new Uint8Array(plane);
  for (let y=0; y<height; y++) {
    padded.set(scaled.subarray(y*width,(y+1)*width),(top+y)*edge+left);
  }
  const mean = [.485,.456,.406], std = [.229,.224,.225], values = new Float32Array(3*plane);
  for (let c=0; c<3; c++) {
    const normalized = Float32Array.from({length:256}, (_,v) =>
      Math.fround(Math.fround(v/255) - Math.fround(mean[c])) / Math.fround(std[c]));
    for (let i=0; i<plane; i++) values[c*plane+i] = normalized[padded[i]];
  }
  return new ort.Tensor("float32", values, [1,3,edge,edge]);
}

function linearAxis(src, dst) {
  return Array.from({length:dst}, (_,i) => {
    const center = Math.fround((i+.5)*src/dst-.5);
    const index = Math.floor(center);
    return {a:Math.max(0, index), b:Math.min(src-1, index+1), fraction:roundEven((center-index)*2048)};
  });
}

function detectionTensor(image, box) {
  const edge = 640, plane = edge * edge;
  const values = new Float32Array(3 * plane).fill(114/255);
  const xs = linearAxis(image.width, box.width), ys = linearAxis(image.height, box.height);
  for (let y=0; y<box.height; y++) {
    const rows = ys[y];
    for (let x=0; x<box.width; x++) {
      const cols = xs[x];
      const dst = (y+box.top)*edge+x+box.left;
      for (let c=0; c<3; c++) {
        const top = (rows.a*image.width)*4+c, bottom = (rows.b*image.width)*4+c;
        const f = cols.fraction;
        const upper = (2048-f)*image.data[top+4*cols.a]+f*image.data[top+4*cols.b];
        const lower = (2048-f)*image.data[bottom+4*cols.a]+f*image.data[bottom+4*cols.b];
        // OpenCV's 8-bit linear resize uses 11-bit axis coefficients and rounding per row.
        const u = ((2048-rows.fraction)*(upper >> 4)) >> 16;
        const l = (rows.fraction*(lower >> 4)) >> 16;
        values[c*plane+dst] = ((u+l+2) >> 2)/255;
      }
    }
  }
  return new ort.Tensor("float32", values, [1,3,edge,edge]);
}

function overlap(a,b) {
  const area = box => (box.x2-box.x1)*(box.y2-box.y1);
  const w = Math.max(0,Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1));
  const h = Math.max(0,Math.min(a.y2,b.y2)-Math.max(a.y1,b.y1));
  return w*h / (area(a)+area(b)-w*h);
}

function decode(output, image, box) {
  if (output.dims.length !== 3 || output.dims[0] !== 1 || output.dims[1] !== 5) throw new Error("定位模型输出格式无效");
  const data = output.data, count = output.dims[2], candidates = [], kept = [];
  for (let i=0; i<count; i++) {
    const score = data[4*count+i];
    if (!(score >= threshold && Number.isFinite(score))) continue;
    const x = data[i], y = data[count+i], w = data[2*count+i], h = data[3*count+i];
    if (w>0 && h>0 && [x,y,w,h].every(Number.isFinite)) candidates.push({x1:x-w/2, y1:y-h/2, x2:x+w/2, y2:y+h/2, score});
  }
  candidates.sort((a,b) => b.score-a.score);
  for (const candidate of candidates) {
    if (!kept.some(prior => overlap(prior,candidate) > .7)) kept.push(candidate);
  }
  const unscale = (x, pad, length, original) => Math.max(0, Math.min(original,(x-pad)*original/length));
  return kept.map(b => ({
    x1:unscale(b.x1,box.left,box.width,image.width), x2:unscale(b.x2,box.left,box.width,image.width),
    y1:unscale(b.y1,box.top,box.height,image.height), y2:unscale(b.y2,box.top,box.height,image.height), score:b.score
  })).filter(b => b.x2>b.x1 && b.y2>b.y1);
}

async function infer(blob) {
  models ??= loadModels().catch(error => { models = undefined; throw error; });
  const {dino,yolo,manifest} = await models;
  const bitmap = await createImageBitmap(blob, {imageOrientation:"from-image", colorSpaceConversion:"none"});
  let image;
  try {
    if (Math.max(bitmap.width, bitmap.height) > 8000 || bitmap.width*bitmap.height > 24000000) throw new Error("影像尺寸超过限制");
    const context = new OffscreenCanvas(bitmap.width,bitmap.height).getContext("2d", {willReadFrequently:true});
    context.drawImage(bitmap,0,0);
    image = context.getImageData(0,0,bitmap.width,bitmap.height);
  } finally {
    bitmap.close();
  }
  const score = Number((await dino.run({pixel_values:screeningTensor(image)})).film_score.data[0]);
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error("整片模型输出无效");
  const box = layout(image,640);
  const output = await yolo.run({[yolo.inputNames[0]]:detectionTensor(image,box)});
  return {screening:{model:manifest.model_id, score, threshold:manifest.threshold, predicted_positive:score >= manifest.threshold},
    detections:decode(output[yolo.outputNames[0]],image,box), model:"YOLO11s-fetlock-640@"+yoloDigest.slice(0,12), threshold};
}

let pending = Promise.resolve();
self.onmessage = ({data:{id,blob}}) => {
  pending = pending.then(async () => {
    try { self.postMessage({id, result:await infer(blob)}); }
    catch (error) { self.postMessage({id, error: error instanceof Error ? error.message : String(error)}); }
  });
};
