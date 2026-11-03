// 浏览器端 YOLO ONNX 推理：letterbox 预处理 + 解码 + NMS
// 与训练时（ultralytics）的逻辑保持一致，已在 Python 端逐位验证过。

export const INPUT_SIZE = 640;

export type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
  cls: number;
};

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("图片加载失败"));
    img.src = src;
  });
}

/** letterbox 到 640x640，返回 CHW RGB float32（0~1）+ 还原用的缩放/偏移 */
export function letterbox(img: HTMLImageElement, size = INPUT_SIZE) {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const scale = Math.min(size / w, size / h);
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);
  const padX = Math.floor((size - nw) / 2);
  const padY = Math.floor((size - nh) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(img, padX, padY, nw, nh);

  const { data } = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const chw = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    chw[i] = data[i * 4] / 255;
    chw[n + i] = data[i * 4 + 1] / 255;
    chw[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { chw, scale, padX, padY };
}

/** 解析输出 [1, 4+nc, 8400]（已展平），返回 640 输入空间下的框 */
export function decode(output: Float32Array, numAnchors: number, confThr: number): Detection[] {
  const stride = output.length / numAnchors; // = 4 + nc
  const nc = stride - 4;
  const out: Detection[] = [];
  for (let i = 0; i < numAnchors; i++) {
    let best = 0;
    let bestCls = 0;
    for (let c = 0; c < nc; c++) {
      const s = output[(4 + c) * numAnchors + i];
      if (s > best) {
        best = s;
        bestCls = c;
      }
    }
    if (best < confThr) continue;
    const cx = output[0 * numAnchors + i];
    const cy = output[1 * numAnchors + i];
    const bw = output[2 * numAnchors + i];
    const bh = output[3 * numAnchors + i];
    out.push({
      x1: cx - bw / 2,
      y1: cy - bh / 2,
      x2: cx + bw / 2,
      y2: cy + bh / 2,
      conf: best,
      cls: bestCls,
    });
  }
  return out;
}

function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

export function nms(boxes: Detection[], iouThr = 0.7): Detection[] {
  const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
  const keep: Detection[] = [];
  const dropped = new Array(sorted.length).fill(false);
  for (let i = 0; i < sorted.length; i++) {
    if (dropped[i]) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (!dropped[j] && iou(sorted[i], sorted[j]) > iouThr) dropped[j] = true;
    }
  }
  return keep;
}

/** 把 640 空间的框映射回原图坐标 */
export function mapBack(boxes: Detection[], scale: number, padX: number, padY: number): Detection[] {
  return boxes.map((b) => ({
    ...b,
    x1: (b.x1 - padX) / scale,
    y1: (b.y1 - padY) / scale,
    x2: (b.x2 - padX) / scale,
    y2: (b.y2 - padY) / scale,
  }));
}

/** 在 canvas 上画出检测结果 */
export function draw(img: HTMLImageElement, boxes: Detection[]): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0);

  const line = Math.max(2, Math.round(img.naturalWidth / 260));
  const fs = Math.max(12, Math.round(img.naturalWidth / 40));

  for (const b of boxes) {
    ctx.strokeStyle = "#4c8dff";
    ctx.lineWidth = line;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

    const label = `lesion ${b.conf.toFixed(2)}`;
    ctx.font = `600 ${fs}px -apple-system, "PingFang SC", sans-serif`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "#4c8dff";
    ctx.fillRect(b.x1, Math.max(0, b.y1 - fs * 1.3), tw + fs * 0.5, fs * 1.3);
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "top";
    ctx.fillText(label, b.x1 + fs * 0.25, Math.max(0, b.y1 - fs * 1.3) + fs * 0.2);
  }
  return canvas;
}
