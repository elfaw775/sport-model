# EQUIVISION · 马球节影像工作台

**简体中文** · [English](README.en.md)

[黑边对照实验与逐片成绩](image-study/published/cv84/canvas-round2/README.md#中文) · [模型研究中文指南](image-study/README.zh-CN.md)

Python 工作台支持登录、整片筛查、疑似病灶定位、人工复核和团队病例归档。病例分析由服务所在机器用 CPU 运行 DINOv2 Small 和 YOLO11s，两个推理权重随仓库提供。独立的 [GitHub Pages 演示](https://elfaw775.github.io/sport-model/) 使用访问者浏览器里的 YOLO，病例记录存在该浏览器。

## 后端改动会影响模型吗？

病例、登录、签署和人工标注由 Python 工作台处理。服务在同一进程中读取两个 ONNX 文件并核对 SHA-256，然后用 CPU 推理。修改表单、病例逻辑或保存医生画的框，都不会写入权重。训练和导出须另行运行。文件与代码可以分别更新，部署时由同一套工作台加载。[已部署的模型文件及 DINO 门槛](backend/models) · [加载与校验](backend/onnx_models.py)

改动缩放、补边、输入顺序、门槛或候选框过滤，即使权重完全相同，也可能改变医生看到的结果。涉及这些改动时，用固定参考片对比整片分数、报警和候选框。阅片页提供「查看候选框」和「核对主图」，可以从文字提示直接回到原图。[整片推理](backend/screening.py) · [候选框推理](backend/detector.py) · [回归用例](backend/tests/test_detector.py)

[病灶框监督实验](image-study/published/cv84/box-token/README.md) · [奖励训练与公开方法借鉴](image-study/published/cv84/one-step-reward/README.md)

## 一期工作台：本地运行

在仓库根目录使用 Python 3.12：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>，选择成员或输入全小写拼音账号：王一伊 `wangyiyi`、朱诗嘉 `zhushijia`、阿里米热 `alimire`、李其杭 `liqihang`。本地演示密码为账号紧接 `sportsHorse`（大写 H），例如 `wangyiyisportsHorse`。四位成员登录同一服务后，共用一套就诊记录：任何成员都可查看、编辑、补图、分析、复核、归档或删除团队记录。记录保留原登记账号，复核署名取实际提交人的姓名。原演示账号迁入的病例也能从团队看板查看。
登录后默认进入[新建检查](http://127.0.0.1:8765/#new)：选择球节 X 线片，核对马匹、投照方向和影像质量，上传后获得 EV 编号。点击分析，查看原图上的真实候选框与模型分数，再由登录成员核对结果、填写意见并确认复核。最后单独点击归档；关闭页面或确认复核都不会自动归档。

左侧工作台提供新建检查、待分析、人工复核、病例归档和历史记录五个入口。上传后的病例出现在[待分析](http://127.0.0.1:8765/#analysis)，分析后转入[人工复核](http://127.0.0.1:8765/#reviews)。主图上方显示分析状态、候选框数量和分数，零候选时会给出实际阈值。确认报告后可从[病例归档](http://127.0.0.1:8765/#filing)继续办理。

纸面就诊可先登记、直接归档，后来补入第一张影像会进入待分析流程。[历史记录](http://127.0.0.1:8765/#dashboard)可搜索、筛选、编辑、补图和单条或批量删除。重新打开已有记录可从已保存的步骤继续。

同一次就诊可以在新建时或之后补充多张 PNG、JPEG 或 WebP。第一张为主图，后续为附图。确认主图是合适的球节 X 线片后，点击分析。DINOv2 Small 为整张主图打分，YOLO11s 给出可疑区域的坐标。附图用于对照阅片。可用仓库中的[侧位示例](training/data/sporthorse_v2/images/test/16-侧位.jpg)试用。

### 中期审查演示

在「新建检查」点「随机抽取样片」，可从项目的 84 张球节 X 线片中抽一张，预览原片并带出演示马编号。再次抽取可更换样片。遇到多体位片或方向不明的片子，请核对原图后选择投照方向。抽片和换片只更新表单，确认并点击「上传影像」才会生成 EV 病例编号。

现场体验依次点击上传、分析、人工复核和归档，最后用 EV 编号在历史记录里找回报告。样片与本地选取的影像使用同一套 DINOv2 筛查和 YOLO 定位流程。这 84 张片也用于项目模型开发，适合演示操作；评估新马的识别能力需要独立样本。

如需复现固定场景，可展开「推荐病例」。16 号侧位片的实际结果为整片提示可疑、定位到两个候选区域；23 号正位片同样提示可疑，定位模型未给出候选区域。两种结果都可以继续人工阅片、签署报告。[16 号原片](training/data/sporthorse_v2/images/test/16-侧位.jpg) · [23 号原片](training/data/sporthorse_v2/images/test/23-LF-正位.jpg)

网页版签署前会再次核对打开报告时的病例、影像、测量和标注。若同事已更新资料，可点击「核对最新报告」再决定是否签署。新报告签署时保存当时的证据，可在复核页下载离线 HTML 并通过浏览器打印。随后更正马名、修改备注或补图时，签署报告仍引用签署时的资料。旧报告缺少快照，页面与导出文件均标明按现存资料整理。

### 病例阅片与复核报告

医生可在主图或附图上绘制矩形和多边形，填写病灶区域名称，再保存人工标注。坐标相对于该张原片，看图时可以切换标注显隐。保存后重新打开同一病例或换一个团队账号，仍可复核这些区域；签署后的标注只读。可导出带编号轮廓的原分辨率 PNG，附带区域编号、名称与坐标的 PNG 文本信息。原片保持独立，人工区域和 YOLO 候选框各有来源。[展示仓库中的标注交互](https://github.com/theAtlantic-zza/sports-horse--studio--display/tree/6612f270604ae764fd212fe8fe2cc769e200b195)

上传完成后打开病例阅片页。左侧按本次检查列出主图和附图，中间显示原始像素，右侧核对体位、影像质量与逐图测量。点击「对照」可选择左右两张不同原片，切换活动窗后独立平移、缩放、旋转、翻转、反色或调节亮度与对比度。两点记录原图像素距离，三点记录以第二点为顶点的角度。未标定的 X 线图仅显示 px 和 °。候选框只叠加在实际参加检测的主图。

分析完成后可进入「编写复核报告」。影像所见和复核结论均为选填，填写任意一项可直接确认或先保存草稿。两项均为空白时须再次选择「确认留空提交」，系统会记录这一选择。返回填写或按 Esc 会保留待复核状态。提交时检查草稿版本，遇到其他成员先保存会提示冲突。确认后记录登录人和签署时间，报告、影像核对项和测量转为只读。点击「归档病例」独立写入归档时间。有影像的病例可用 `#case/EV-编号` 与 `#report/EV-编号` 回到阅片与报告。

报告保留候选框在原图上的位置、模型置信分数、权重标识、阈值和推理时间；即使未检出也需复核才能归档。原图不另放上传文件夹：SQLite 的 `case_images.original` 直接保存原文件字节，`cases` 保存病例、备注、预测框、模型版本、复核人及确认和归档时间。`image_measurements` 保存每张片上的取点、数值、记录人及时间，`case_reports` 保存报告草稿和版本。默认数据库为本项目的 `var/cases.sqlite3`（不进 Git），重启时使用同一数据目录。静态演示的浏览器记录不会同步到这里。模型故障会明确失败并保留待分析病例。

- `backend/app.py`：FastAPI 会话和病例接口；`backend/detector.py`：校验权重、真实 ONNX 推理及坐标还原；`backend/static/`：工作台界面。
- 需要自定义单账户时可使用 `SPORT_MODEL_USERNAME`、`SPORT_MODEL_PASSWORD`、`SPORT_MODEL_DISPLAY_NAME`；它与默认四人演示模式分开。若另行托管，使用持久化的 `SPORT_MODEL_DATA_DIR`、HTTPS、非演示凭据和 `SPORT_MODEL_SECURE_COOKIE=1`；GitHub Pages 本身不能运行此服务。
- 验证：安装运行依赖后执行 `.venv/bin/python -m pip install pytest httpx` 和 `.venv/bin/python -m pytest -q backend/tests`。测试覆盖登录、团队成员协作、自定义单用户空间隔离、病例状态与持久化，以及仓库权重在两张影像上的输出回归。输出回归不能证明诊断准确性。
- Chrome 流程验收：`pnpm install --frozen-lockfile` 后运行 `node scripts/smoke-visits.mjs`。脚本启动隔离的临时数据库，测试建档、编辑、多图、模型推理、人工复核、查询及批量删除，不会删除实际病例。非 macOS 环境用 `CHROME_BIN` 指定 Chrome、用 `PYTHON` 指定装有后端依赖的 Python。
- 更新时自动迁移旧病例库，保留编号、原图及复核历史。现有人工复核过的旧记录沿用原复核时间作为归档时间；手动归档不会标记为已复核。

### 一期研究模型和验收范围

平台当前支持马球节 X 光主图的疑似病灶筛查。DINOv2 Small 使用在全部 84 张片上重新拟合的分类头，输入为 336 × 336，分数达到 0.1788344393 时提示可疑。YOLO11s 为修正旋转框后重训的定位模型，输入为 640 × 640，候选框阈值为 0.20。工作台保存各自的版本与门槛，医生可复核两项结果并签署。两者可以给出不同的提示。原有病例沿用原分析与签署记录。[筛查模型登记](backend/models/dinov2-fetlock-84.json) · [定位权重](backend/models/yolo11s-fetlock-640.onnx)

在同一批 84 张片的五折验证中，DINOv2 Small 整片判对 82 张，漏检 2/68、误报 0/16。该成绩由各折对应模型给出。工作台处理新片时使用全量拟合权重及原黑色居中补边。锁定五折和门槛后，只把外围补边从黑色换成深灰色，有 1 张阳性由正确变为漏检；保留黑色改为向画布起点对齐，有 1 张原先漏检的阳性变为检出。仍需使用同采集流程且来自新马的阳性和阴性片检验泛化。[五折结果](image-study/published/cv84/three-model-summary.json) · [补边逐片对照](image-study/published/cv84/canvas-audit.json)

验收时登录工作台，上传示例主图并确认质量，运行分析，核对整片分数、筛查门槛和候选框，再保存、签署复核报告并归档。重开病例可核对已保存的结果。运行服务需要可写的持久数据目录；团队如需异地持续访问，应将 Python 工作台部署在常在线主机上。

本轮用 25 张既有参考片核对后台和训练版 DINO 的推理一致性，25 张的判断全部一致，最大绝对分数差小于 0.00000009。Chrome 端到端验收中，16 号侧位片整片分数约 0.999627、定位到 2 框；23 号左前肢正位片约 0.999955、定位到 0 框，两项提示均按实际结果展示。分数只用于与本模型阈值比较。这批参考片用于验证软件部署的一致性，独立新病例的泛化能力仍待验证。

[修正标签与训练对照](training/results/comparison-2026-11-18/README.md)说明阈值的选择方法，并提供逐片预测。另有 [960 像素重训](training/results/yolo11s-960-2026-11-18/README.md)与 [多款视觉模型的实测](training/results/vision-model-scale-2026-11-18/README.md)。这些历史报告中的“现役”指当次实验时使用的 v2 权重。

以下为原 GitHub Pages 浏览器端演示的开发与发布说明。

## 仓库结构

| 目录 | 内容 |
|---|---|
| `app/` `lib/` `public/` | 网页本体（Next.js 静态站 + 浏览器端 ONNX 推理） |
| `training/` | **模型训练全过程**：数据、转换/拆分/评估脚本、训练权重、实验指标 |
| `.github/workflows/` | GitHub Pages 自动构建部署 |

第一次了解这个模型，可以从一张侧位片读起。[选模型、修标签、训练与评估全过程](training/results/yolo-family-2026-11-19/README.md) · [训练目录](training/README.md)

> 为什么不能用原来的 Flask + PyTorch？GitHub Pages 只托管静态文件，跑不了 Python/PyTorch。
> 所以把模型导出成 ONNX，推理搬到浏览器。

---

## 1. 本地开发 / 预览

```bash
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
