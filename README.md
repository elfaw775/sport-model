# EQUIVISION · 马球节影像工作台

本仓库有两个入口：Python 工作台支持登录、真实检测、人工复核和持久病例档案；[已有 GitHub Pages 演示](https://elfaw775.github.io/sport-model/) 仍在浏览器推理，不会将影像上传到 Python 或共享病例库。

## 一期工作台：本地运行

在仓库根目录使用 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，本地试用账号 `demo`、密码 `EquiDemo2026!`，默认复核人王一伊。
填写马匹、手选投照方向，确认影像清晰后上传 PNG、JPEG 或 WebP；每次上传创建一个 EV 病例号。点「开始 AI 分析」时，服务才用原仓库的 YOLO11s v2 ONNX 权重和 CPU 分析该图。可以用仓库中的 [侧位示例](training/data/sporthorse_v2/images/test/16-侧位.jpg) 走一遍流程。

报告保留候选框在原图上的位置、模型置信分数、权重标识、阈值和推理时间；即使未检出也可以复核。提交意见后从「病例档案」按名称或编号搜索，复看原图、模型结果、复核人和时间。原始文件和记录均保存在默认的 `var/cases.sqlite3`（不进 Git）；重启时使用同一数据目录。模型故障会明确失败并保留待分析病例。

- `backend/app.py`：FastAPI 会话和病例接口；`backend/detector.py`：校验权重、真实 ONNX 推理及坐标还原；`backend/static/`：工作台界面。
- 本地账号可用 `SPORT_MODEL_USERNAME`、`SPORT_MODEL_PASSWORD`、`SPORT_MODEL_DISPLAY_NAME` 替换。若另行托管，使用持久化的 `SPORT_MODEL_DATA_DIR`、HTTPS、非演示凭据和 `SPORT_MODEL_SECURE_COOKIE=1`；GitHub Pages 本身不能运行此服务。
- 验证：安装运行依赖后执行 `.venv/bin/python -m pip install pytest httpx` 和 `.venv/bin/python -m pytest -q backend/tests`。测试包含仓库权重在两张影像上的输出回归、登录与跨用户隔离、病例状态和重启持久化。输出回归不能证明诊断准确性。

一期真实模型只支持马球节疑似骨折定位。0.15 是候选框筛选阈值，分数不是患病概率；未检出不能排除骨折。项目保留原 v2 权重，修正了供下轮训练使用的旋转框转换，没有重新训练或宣称准确率提升。[训练归档](training/README.md) 记录了版本和已知限制。[Qwen3‑VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 与 [DeepSeek‑VL2](https://github.com/deepseek-ai/DeepSeek-VL2) 均有视觉定位能力，官方资料未提供本任务的独立评估；本版不使用它们判片或生成结论。

以下是原静态演示的开发与发布说明：

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

- **模型仍属于研究原型**：不同版本、阈值和标签下的结果不能混用，参见训练指南的 v2 评估。
- 首次打开要下载 **约 24MB**（模型 10MB + WASM 14MB），之后浏览器缓存。
- 浏览器推理速度：**约 150–300ms/张**（yolo11n，WASM；M 系列机器）。
- SQLite 换成了 IndexedDB：只能存在本机浏览器，不能多端同步、不能服务端查询。
- 如果要真正的服务端推理 + 数据库，还是得用 Flask/FastAPI 部署到有 GPU/Python 的环境
  （Fly.io / Railway / 自建服务器），GitHub Pages 做不到。
