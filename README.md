# EQUIVISION · 马球节影像工作台

本仓库有两个入口：Python 工作台支持登录、真实检测、人工复核和持久病例档案；[已有 GitHub Pages 演示](https://elfaw775.github.io/sport-model/) 仍在浏览器推理，不会将影像上传到 Python 或共享病例库。

## 一期工作台：本地运行

在仓库根目录使用 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，选择成员或输入全小写拼音账号：王一伊 `wangyiyi`、朱诗嘉 `zhushijia`、阿里米热 `alimire`、李其杭 `liqihang`。本地演示密码为账号紧接 `sportsHorse`（大写 H），例如 `wangyiyisportsHorse`。四位成员登录同一服务后，共用一套就诊记录：任何成员都可查看、编辑、补图、分析、复核、归档或删除团队记录。记录保留原登记账号，复核署名取实际提交人的姓名。原演示账号迁入的病例也能从团队看板查看。
登录后默认进入[新建检查](http://127.0.0.1:8765/#new)：选择球节 X 线片，核对马匹、投照方向和影像质量，上传后获得 EV 编号。点击分析，查看原图上的真实候选框与模型分数，再由登录成员核对结果、填写意见并确认复核。最后单独点击归档；关闭页面或确认复核都不会自动归档。

纸面就诊仍可从次要的无影像登记入口建档、直接归档；此类记录后来补入第一张影像会重新进入待分析流程。[历史记录](http://127.0.0.1:8765/#dashboard)用于搜索、筛选、编辑、补图和单条或批量删除。重新打开已有记录可从已保存的步骤继续。

同一次就诊可以在新建时或之后补充多张 PNG、JPEG 或 WebP。第一张为主图，后续为附图；需要 AI 分析时手动确认主图是合适的球节 X 线片。只有主图用于当前 YOLO11s 检测，附图供对照阅片，不覆盖已有的模型结果和复核意见。可用仓库中的[侧位示例](training/data/sporthorse_v2/images/test/16-侧位.jpg)试用。

### 病例阅片与复核报告

上传完成后打开病例阅片页。左侧按本次检查列出主图和附图，中间显示原始像素，右侧核对体位、影像质量与逐图测量。点击「对照」可选择左右两张不同原片，切换活动窗后独立平移、缩放、旋转、翻转、反色或调节亮度与对比度。两点记录原图像素距离，三点记录以第二点为顶点的角度。未标定的 X 线图仅显示 px 和 °。候选框只叠加在实际参加检测的主图。

分析完成后从「编写复核报告」填写影像所见与复核结论，可先保存草稿再回到阅片。保存时检查草稿版本，遇到其他成员先保存会提示冲突。确认复核时记录当前登录人，已签署报告、影像核对项和测量只读。点击「归档病例」独立写入归档时间。有影像的病例可用 `#case/EV-编号` 与 `#report/EV-编号` 回到阅片与报告。

报告保留候选框在原图上的位置、模型置信分数、权重标识、阈值和推理时间；即使未检出也需复核才能归档。原图不另放上传文件夹：SQLite 的 `case_images.original` 直接保存原文件字节，`cases` 保存病例、备注、预测框、模型版本、复核人及确认和归档时间。`image_measurements` 保存每张片上的取点、数值、记录人及时间，`case_reports` 保存报告草稿和版本。默认数据库为本项目的 `var/cases.sqlite3`（不进 Git），重启时使用同一数据目录。静态演示的浏览器记录不会同步到这里。模型故障会明确失败并保留待分析病例。

- `backend/app.py`：FastAPI 会话和病例接口；`backend/detector.py`：校验权重、真实 ONNX 推理及坐标还原；`backend/static/`：工作台界面。
- 需要自定义单账户时可使用 `SPORT_MODEL_USERNAME`、`SPORT_MODEL_PASSWORD`、`SPORT_MODEL_DISPLAY_NAME`；它与默认四人演示模式分开。若另行托管，使用持久化的 `SPORT_MODEL_DATA_DIR`、HTTPS、非演示凭据和 `SPORT_MODEL_SECURE_COOKIE=1`；GitHub Pages 本身不能运行此服务。
- 验证：安装运行依赖后执行 `.venv/bin/python -m pip install pytest httpx` 和 `.venv/bin/python -m pytest -q backend/tests`。测试覆盖登录、团队成员协作、自定义单用户空间隔离、病例状态与持久化，以及仓库权重在两张影像上的输出回归。输出回归不能证明诊断准确性。
- Chrome 流程验收：`pnpm install --frozen-lockfile` 后运行 `node scripts/smoke-visits.mjs`。脚本启动隔离的临时数据库，测试建档、编辑、多图、模型推理、人工复核、查询及批量删除，不会删除实际病例。非 macOS 环境用 `CHROME_BIN` 指定 Chrome、用 `PYTHON` 指定装有后端依赖的 Python。
- 更新时自动迁移旧病例库，保留编号、原图及复核历史。现有人工复核过的旧记录沿用原复核时间作为归档时间；手动归档不会标记为已复核。

**模型对照：** [实际训练结果、候选权重和复现说明](training/results/comparison-2026-11-18/README.md)；[架构选型依据](training/docs/候选模型与对照试验.md)。两种方案均已训练，候选权重不自动上线，工作台病例不自动加入训练集。

一期真实模型只支持马球节疑似骨折定位。0.15 是在用模型的候选框筛选阈值，分数不是患病概率；未检出不能排除骨折。工作台保留原 v2 权重，本轮候选在修正标签上另行训练和评估。[训练归档](training/README.md) 记录了版本和已知限制。

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
