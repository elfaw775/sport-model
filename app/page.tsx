"use client";

import { useEffect, useRef, useState } from "react";
import {
  INPUT_SIZE,
  type Detection,
  loadImage,
  letterbox,
  decode,
  nms,
  mapBack,
  draw,
} from "@/lib/yolo";
import { addRecord, listRecords, clearRecords, deleteRecord, type Rec } from "@/lib/db";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
const ORT_WASM_PATH = `${BASE_PATH}/ort/`;

/** 内置两个模型：快速版（yolo11n，小/快）/ 精确版（yolo11s，大/准） */
const MODELS = {
  fast: { label: "快速版", model: "yolo11n", file: "model-fast.onnx", size: "10MB" },
  accurate: { label: "精确版", model: "yolo11s", file: "model-accurate.onnx", size: "36MB" },
} as const;
type ModelKey = keyof typeof MODELS;

type Ort = any;
let ortPromise: Promise<Ort> | null = null;
const sessionCache: Record<string, Promise<any>> = {};

async function getOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((m: any) => {
      const ort = m?.env ? m : m?.default ?? m;
      ort.env.wasm.wasmPaths = ORT_WASM_PATH;
      ort.env.wasm.numThreads = 1; // GitHub Pages 无 COOP/COEP，单线程最稳
      ort.env.logLevel = "error";
      return ort;
    });
  }
  return ortPromise;
}

async function getSession(key: ModelKey): Promise<any> {
  if (!sessionCache[key]) {
    sessionCache[key] = (async () => {
      const ort = await getOrt();
      const url = `${BASE_PATH}/models/${MODELS[key].file}`;
      return ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
    })();
  }
  return sessionCache[key];
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = () => reject(new Error("读取文件失败"));
    fr.readAsDataURL(file);
  });
}

function toThumb(canvas: HTMLCanvasElement, maxW = 420): string {
  const scale = Math.min(1, maxW / canvas.width);
  const c = document.createElement("canvas");
  c.width = Math.round(canvas.width * scale);
  c.height = Math.round(canvas.height * scale);
  c.getContext("2d")!.drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.82);
}

type ResultState = {
  url: string;
  boxes: Detection[];
  ms: number;
  filename: string;
  thumb: string;
  model: string;
};

export default function Home() {
  const [conf, setConf] = useState(0.1);
  const [modelKey, setModelKey] = useState<ModelKey>("fast");
  const [status, setStatus] = useState<"idle" | "running">("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState<ResultState | null>(null);
  const [inputUrl, setInputUrl] = useState("");
  const [history, setHistory] = useState<Rec[]>([]);
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshHistory = async () => {
    try {
      setHistory(await listRecords());
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    refreshHistory();
    if (typeof window !== "undefined" && (window as any).__warm) return;
    (window as any).__warm = true;
    // 提前预热默认（快速）模型，避免第一次上传等太久
    getSession("fast").catch(() => {});
  }, []);

  const run = async (file: File) => {
    setStatus("running");
    setError("");
    setResult(null);
    try {
      if (file.size > 16 * 1024 * 1024) throw new Error("图片超过 16MB");
      const dataUrl = await fileToDataUrl(file);
      setInputUrl(dataUrl);

      const img = await loadImage(dataUrl);
      const ort = await getOrt();
      const session = await getSession(modelKey);

      const lb = letterbox(img, INPUT_SIZE);
      const tensor = new ort.Tensor("float32", lb.chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const feeds: Record<string, any> = {};
      feeds[session.inputNames[0]] = tensor;

      const t0 = performance.now();
      const out = await session.run(feeds);
      const ms = performance.now() - t0;

      const t = out[session.outputNames[0]];
      const data = t.data as Float32Array;
      const dims = t.dims as number[];
      const numAnchors = dims[2];

      let dets = decode(data, numAnchors, conf);
      dets = nms(dets, 0.7);
      dets = mapBack(dets, lb.scale, lb.padX, lb.padY);

      const canvas = draw(img, dets);
      const url = canvas.toDataURL("image/jpeg", 0.92);
      const thumb = toThumb(canvas);

      setResult({ url, boxes: dets, ms, filename: file.name, thumb, model: MODELS[modelKey].label });

      await addRecord({
        created_at: new Date().toLocaleString("zh-CN", { hour12: false }),
        filename: file.name,
        conf,
        n: dets.length,
        duration_ms: Math.round(ms),
        thumb,
        boxes: dets,
      });
      await refreshHistory();
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setStatus("idle");
    }
  };

  const onFiles = (files: FileList | null) => {
    if (!files || !files.length) return;
    run(files[0]);
  };

  return (
    <>
      <header>
        <div className="brand">
          🐴 马球节病变检测 <span className="tag">YOLO11 · ONNX · 浏览器端</span>
        </div>
        <div className="hdr-right">
          <span>纯前端推理，图片不上传服务器</span>
        </div>
      </header>

      <main>
        <section className="card">
          <h1>上传一张 X 光片</h1>
          <p className="hint">
            支持 jpg / png / bmp / webp，单张不超过 16MB。首次使用需加载 {MODELS[modelKey].size} 模型。
          </p>

          <div className="model-switch">
            {(Object.keys(MODELS) as ModelKey[]).map((k) => (
              <button
                key={k}
                type="button"
                className={`model-btn ${modelKey === k ? "active" : ""}`}
                onClick={() => setModelKey(k)}
                disabled={status === "running"}
              >
                {k === "fast" ? "⚡" : "🎯"} {MODELS[k].label}
                <small>
                  {MODELS[k].model} · {MODELS[k].size}
                </small>
              </button>
            ))}
          </div>
          <p className="hint">
            快速版加载快、适合日常试用；精确版更大更准，首次加载约 {MODELS["accurate"].size}。
          </p>

          {error && <p className="err">⚠️ {error}</p>}

          <label
            className={`drop ${over ? "over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              onFiles(e.dataTransfer.files);
            }}
          >
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={(e) => onFiles(e.target.files)}
            />
            <strong>{status === "running" ? "推理中…" : "点击选择图片，或把图片拖到这里"}</strong>
            <small>浏览器本地推理 · 支持离线使用</small>
          </label>

          {status === "running" && (
            <div className="progress">
              <div />
            </div>
          )}

          <div className="controls">
            <label>
              置信度阈值
              <input
                type="range"
                min={0.05}
                max={0.6}
                step={0.01}
                value={conf}
                onChange={(e) => setConf(parseFloat(e.target.value))}
              />
              <input
                type="number"
                min={0.01}
                max={0.95}
                step={0.01}
                value={conf}
                onChange={(e) => setConf(parseFloat(e.target.value) || 0.1)}
              />
            </label>
            <span className="hint">建议 0.1：当前模型置信度普遍偏低</span>
          </div>
        </section>

        {result && (
          <section className="card">
            <div className="row-between">
              <h2>检测结果 <span className="muted">{result.filename}</span></h2>
              <span className="hint">推理 {result.ms.toFixed(0)} ms</span>
            </div>

            <div className="result-grid">
              <figure>
                <figcaption>检测结果</figcaption>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.url} alt="检测结果" />
              </figure>
              <figure>
                <figcaption>原图</figcaption>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={inputUrl} alt="原图" />
              </figure>
            </div>

            <div className="summary">
              {result.boxes.length > 0 ? (
                <span>
                  共检出 <b>{result.boxes.length}</b> 个候选病变，最高置信度{" "}
                  <b>{result.boxes[0].conf.toFixed(3)}</b>
                </span>
              ) : (
                <span className="warn">未检出病变（可降低阈值重试）</span>
              )}
            </div>

            {result.boxes.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>类别</th>
                    <th>置信度</th>
                    <th>框 (x1, y1, x2, y2)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.boxes.map((b, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>lesion</td>
                      <td>
                        <span style={{ color: "var(--ok)" }}>{b.conf.toFixed(3)}</span>
                      </td>
                      <td>
                        <code>
                          [{b.x1.toFixed(0)}, {b.y1.toFixed(0)}, {b.x2.toFixed(0)},{" "}
                          {b.y2.toFixed(0)}]
                        </code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}

        <section className="card">
          <div className="row-between">
            <h2>
              历史记录 <span className="muted">（共 {history.length} 条，存在本机浏览器）</span>
            </h2>
            {history.length > 0 && (
              <button
                className="btn ghost"
                onClick={async () => {
                  if (confirm("清空全部历史记录？")) {
                    await clearRecords();
                    await refreshHistory();
                  }
                }}
              >
                清空
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <p className="hint">还没有记录，上传一张试试。</p>
          ) : (
            <div className="history-grid">
              {history.map((h) => (
                <div className="hist-item" key={h.id}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={h.thumb} alt={h.filename} />
                  <div className="meta">
                    <b title={h.filename}>{h.filename}</b>
                    {h.created_at}
                    <br />
                    {h.n > 0 ? (
                      <span className="pill ok">{h.n} 个候选</span>
                    ) : (
                      <span className="pill zero">未检出</span>
                    )}{" "}
                    · {h.duration_ms}ms
                    <button
                      className="btn ghost"
                      style={{ padding: "2px 8px", fontSize: 12, marginLeft: 8 }}
                      onClick={async () => {
                        await deleteRecord(h.id!);
                        await refreshHistory();
                      }}
                    >
                      删
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <p className="hint" style={{ textAlign: "center" }}>
          ⚠️ 仅供研究演示，不能作为临床诊断依据（当前模型 test 召回率约 17%）。
        </p>
      </main>

      <footer>
        纯前端 · ONNX Runtime Web · 模型 {result?.model || MODELS[modelKey].label} · 数据存于本机 IndexedDB
      </footer>
    </>
  );
}
