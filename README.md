# 马球节病变检测 · 纯静态 Web（Next.js + ONNX Runtime Web）

**完全没有后端**：模型在浏览器里用 ONNX Runtime Web 推理，图片不上传服务器。
构建产物是纯静态文件，可直接部署到 **GitHub Pages**。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `app/` `lib/` `public/` | 网页本体（Next.js 静态站 + 浏览器端 ONNX 推理） |
| `training/` | **模型训练全过程**：数据、转换/拆分/评估脚本、训练权重、实验指标 |
| `.github/workflows/` | GitHub Pages 自动构建部署 |

训练部分详见 [training/README.md](training/README.md)。

> 为什么不能用原来的 Flask + PyTorch？GitHub Pages 只托管静态文件，跑不了 Python/PyTorch。
> 所以把模型导出成 ONNX，推理搬到浏览器。

---

## 1. 本地开发 / 预览

```bash
cd web-next

# 安装依赖（pnpm-workspace.yaml 里已放行 protobufjs 的构建脚本）
pnpm install

# 生产构建（静态导出到 out/）
pnpm build

# 本地预览构建产物
cd out && python3 -m http.server 8100
# 打开 http://127.0.0.1:8100
```

> **本机注意**：当前 DSH 运行时的 Node 二进制带签名校验，加载原生 SWC 会报
> `different Team IDs`，需要改用 wasm SWC：
>
> ```bash
> NEXT_TEST_WASM_DIR="$PWD/node_modules/@next/swc-wasm-nodejs" pnpm build
> NEXT_TEST_WASM_DIR="$PWD/node_modules/@next/swc-wasm-nodejs" \
>   node node_modules/next/dist/bin/next dev   # 开发模式
> ```
>
> 如果你换成自己安装的正常 Node（例如 `brew install node`），原生 SWC 可以加载，
> 直接 `pnpm build` 即可，不需要这个环境变量。GitHub Actions 的 ubuntu 环境也不需要。

### 一键冒烟测试（无头 Chrome 真实跑一遍推理）

```bash
NEXT_TEST_WASM_DIR="$PWD/node_modules/@next/swc-wasm-nodejs" \
  node scripts/smoke.mjs
```

会打开页面、上传测试图、等浏览器推理完成、把页面文本和截图（`scripts/smoke.png`）打出来。

---

## 2. 部署到 GitHub Pages

### 步骤

1. 新建一个 GitHub 仓库，把 **`web-next/` 目录里的内容** 作为仓库根目录推上去
   （`.github/workflows/deploy.yml` 必须位于仓库根目录的 `.github/` 下）。
2. 仓库 **Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**。
3. push 到 `main` 分支，Actions 会自动构建并部署。
4. 部署地址：`https://<用户名>.github.io/<仓库名>/`

### basePath 说明（很关键）

- 项目站点 `https://<user>.github.io/<repo>/` → 构建时必须带前缀 `/<repo>`。
  workflow 里已自动用 `NEXT_PUBLIC_BASE_PATH: /${{ github.event.repository.name }}`。
- 用户站点 `https://<user>.github.io/`（仓库名必须是 `<user>.github.io`）→ 把 workflow 里
  `NEXT_PUBLIC_BASE_PATH` 改成 `""`。

本地手动构建项目站点：

```bash
NEXT_PUBLIC_BASE_PATH=/你的仓库名 \
NEXT_TEST_WASM_DIR="$PWD/node_modules/@next/swc-wasm-nodejs" \
  node node_modules/next/dist/bin/next build
```

---

## 3. 模型（两个，页面可切换）

| 文件 | 模型 | 大小 | 说明 |
|---|---|---|---|
| `public/models/model-fast.onnx` | yolo11n（v2，含负样本） | 10MB | 默认「快速版」 |
| `public/models/model-accurate.onnx` | yolo11s（v2） | 36MB | 可选「精确版」，更准但首载慢 |

页面顶部有切换按钮；两个模型按需懒加载，选了才下载。

换模型步骤：

```bash
cd ../   # 回 stu_train
export YOLO_CONFIG_DIR="$PWD/.ultralytics" MPLCONFIGDIR="$PWD/.mplconfig"
.venv/bin/yolo export model=runs/horse/lesion_v2_yolo11s_fresh/weights/best.pt format=onnx imgsz=640 opset=12
cp runs/horse/lesion_v2_yolo11s_fresh/weights/best.onnx web-next/public/models/model-accurate.onnx
# 然后 pnpm build / git push 即可
```

## 4. 工作方式

```
上传图片
  ↓
canvas 缩放到 640×640（letterbox，灰边 114）
  ↓
onnxruntime-web（WASM SIMD）跑 model.onnx
  ↓
解码 [1, 5, 8400] 输出 → 置信度过滤 → NMS → 坐标还原到原图
  ↓
canvas 画框 + 显示；历史记录存 IndexedDB
```

- 预处理/后处理逻辑与训练时一致，已在 Python 端逐位校验。
- 历史记录（时间、文件名、阈值、检出数、耗时、缩略图）存在浏览器 **IndexedDB**，
  不上传服务器；换浏览器/清缓存会丢。
- ORT 的 WASM 文件放在 `public/ort/`，构建后随静态站一起发布。

---

## 5. 目录结构

```
web-next/
├── app/
│   ├── layout.tsx
│   ├── page.tsx          # 主页面：上传 + 推理 + 结果 + 历史
│   └── globals.css
├── lib/
│   ├── yolo.ts           # letterbox / 解码 / NMS / 画框
│   └── db.ts             # IndexedDB 历史记录
├── public/
│   ├── models/*.onnx      # 两个模型：model-fast / model-accurate
│   ├── ort/*.wasm|.mjs   # ONNX Runtime Web 运行时
│   └── .nojekyll         # GitHub Pages 必需
├── scripts/smoke.mjs     # 无头浏览器端到端测试
├── next.config.mjs       # output: export + basePath
└── .github/workflows/deploy.yml
```

---

## 6. 已知限制

- **模型本身很弱**：test 召回率约 17%，框也不够准，属流程演示级别。
- 首次打开要下载 **约 24MB**（模型 10MB + WASM 14MB），之后浏览器缓存。
- 浏览器推理速度：**约 150–300ms/张**（yolo11n，WASM；M 系列机器）。
- SQLite 换成了 IndexedDB：只能存在本机浏览器，不能多端同步、不能服务端查询。
- 如果要真正的服务端推理 + 数据库，还是得用 Flask/FastAPI 部署到有 GPU/Python 的环境
  （Fly.io / Railway / 自建服务器），GitHub Pages 做不到。
