# 马球节病变检测 · 训练归档

这份目录是**网页 Demo 背后的模型训练全过程**：数据转换 → 拆分 → 训练 → 评估。
网站本身（Next.js + 浏览器端 ONNX 推理）在仓库根目录。

---

## 目录结构

```
training/
├── README.md                  # 本文件
├── requirements.txt
├── scripts/
│   ├── coco2yolo.py           # CVAT COCO 导出 -> YOLO label（含越界框裁剪）
│   ├── split_dataset.py       # 按图分层拆分 train/val/test + 生成 data.yaml
│   └── eval_report.py         # 可读评估：TP/FP/FN、精度、召回、图片级命中
├── docs/
│   ├── 训练指南.md             # 环境、训练命令、超参、实验对比、风险
│   └── 使用说明.md             # 推理/使用命令
├── data/
│   ├── originals/             # 68 张原始骨折 X 光片
│   ├── healthy/               # 16 张健康马 X 光片（负样本，无标注）
│   ├── sporthorse/            # v1 数据集（仅正样本）
│   └── sporthorse_v2/         # v2 数据集（68 正 + 16 负，train 59 / val 13 / test 12）
├── source/
│   ├── coco/instances_default.json   # CVAT 导出的 COCO 原始标注
│   └── cvat_yolo_export/             # 早期不完整的 YOLO 导出（保留作对照，勿用）
├── models/
│   ├── yolo11n_baseline_best.pt       # v1
│   ├── yolo11n_freeze10_best.pt       # v1
│   ├── yolo11s_freeze10_best.pt       # v1 最佳
│   ├── yolo11n_fresh_v2_best.pt       # v2（网页快速版来源）
│   └── yolo11s_fresh_v2_best.pt       # v2 最佳（网页精确版来源）
└── results/                   # 每轮训练曲线（results.csv）
```

---

## 数据

- 来源：CVAT 任务「骨折1」（项目「x线片」），导出为 **COCO 1.0**
- 图片：68 张马球节 X 光片，**每张来自不同的马**
- 标注：110 个框 = `病变` 81 个 + `annotation mark` 29 个
- 训练只保留 **`病变` → class 0**；`annotation mark` 是部位标记，已剔除
- 拆分（按图分层，seed=42）：v1 train 48 / val 10 / test 10
- **v2 追加：16 张健康马（无病变）负样本** → 84 张，train 59 / val 13 / test 12

> ⚠️ **数据局限**：68 张全部是骨折病例，**没有任何「正常马」负样本**；
> 因此模型只能学「病变长什么样」，学不会「正常长什么样」。要做筛查，
> 必须补正常 X 光片（建议 ≥30~50 张）。

---

## 三组实验对比

同数据同拆分，只改「预训练模型 + 是否冻结骨干」：

| 实验 | 权重文件 | 最佳 epoch | val mAP50 | test mAP50 | test P | test R |
|---|---|---|---|---|---|---|
| yolo11n 全量微调（基线） | `yolo11n_baseline_best.pt` | 49 | 0.162 | 0.0077 | 0.0015 | 0.333 |
| yolo11n + `freeze=10` | `yolo11n_freeze10_best.pt` | 71 | 0.199 | 0.0214 | 0.092 | 0.083 |
| **yolo11s + `freeze=10`** | `yolo11s_freeze10_best.pt` | 93 | **0.211** | **0.089** | 0.138 | 0.25 |


---

## v2：加入健康马负样本（2026-11-04）

在 v1 的 68 张骨折片基础上，加入 **16 张健康马 X 光片**（全部无标注 → 空 label 作背景），
重新拆分为 train 59 / val 13 / test 12，训练 4 组模型：

| 实验 | 初始化 | val mAP50 | test mAP50 | test mAP50-95 | 权重 |
|---|---|---|---|---|---|
| yolo11s warm | v1 最佳模型 | 0.176 | 0.105 | 0.0335 | `runs/horse/lesion_v2_yolo11s_warm` |
| **yolo11s fresh** | COCO | 0.149 | **0.121** | **0.0545** | `models/yolo11s_fresh_v2_best.pt` |
| yolo11n warm | v1 最佳模型 | 0.176 | 0.026 | 0.0083 | — |
| yolo11n fresh | COCO | 0.230 | 0.0536 | 0.0275 | `models/yolo11n_fresh_v2_best.pt` |

**要点：**

1. **warm start（从旧模型继续训）在小数据上不如从 COCO 全新训** —— 旧模型已经偏向「见谁都报病变」，
   新增的 11 张负样本不足以纠正这种偏差。
2. 负样本让模型开始能识别「正常片」，但 **16 张太少**，误报仍然明显（conf≥0.15 时 test 有 9 个误报）。
3. test 只有 12 张，指标方差极大，**不要用单个数字下结论**。

网页端据此提供两个模型（`web-next/public/models/`）：
`model-fast.onnx`（yolo11n，10MB，默认）和 `model-accurate.onnx`（yolo11s，36MB）。

> int8 量化试过：动态量化检测数从 4 掉到 1，静态量化（`quantize_onnx.py`，带校准）mAP 直接归零，
> 所以网页仍用 fp32。

**结论：** 冻结骨干 + 更大预训练模型是当前数据量下最有效的组合（test mAP50 约提升 11 倍），
但绝对值仍低 —— **数据量才是天花板**。网页 Demo 为了体积/速度用的是 yolo11n 的 ONNX（10MB）。

---

## 快速复现

```bash
# 0) 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS 上 ultralytics 需要 numpy<2：
pip install "numpy==1.26.4"

cd training

# 1) COCO -> YOLO（只保留「病变」，越界框自动裁剪）
python scripts/coco2yolo.py source/coco/instances_default.json --out labels_coco --keep 病变

# 2) 拆分
python scripts/split_dataset.py \
    --images data/originals \
    --labels labels_coco \
    --out data/sporthorse --ratio 0.7 0.15 0.15 --seed 42

# 3) 训练（当前最佳：yolo11s + freeze=10）
yolo detect train model=yolo11s.pt data=data/sporthorse/data.yaml \
    epochs=200 imgsz=640 batch=16 device=mps seed=42 patience=50 freeze=10 \
    project=runs name=lesion_yolo11s_freeze10

# 4) 可读评估
python scripts/eval_report.py --model runs/lesion_yolo11s_freeze10/weights/best.pt \
    --data data/sporthorse --splits test val --confs 0.1 0.25 0.5
```

> **portable 注意**：`data/sporthorse/data.yaml` 里的 `path` 是原机器的绝对路径。
> 换机器后请改成本机绝对路径，或直接删掉 `path` 行并把工作目录切到
> `data/sporthorse/` 下再训练。

---

## 评估口径说明

detection 没有单一「正确率」，`eval_report.py` 用 IoU≥0.5 统计最直观的三个数：

- **命中 TP**：报出的框与真值 IoU≥0.5
- **误报 FP**：报出的框对不上任何真值
- **漏检 FN**：真值有但没报出来

当前最佳模型在 test（10 张、12 个病变）上：
conf≥0.1 时命中 2 个、误报 12 个、漏检 10 个（召回 16.7%）—— 属流程演示级别。

---

## 注意事项

1. **中文字体坑**：类别名用中文会触发 ultralytics 下载 22MB `Arial.Unicode.ttf`，
   且会写到工作区外。本仓库统一用 ASCII 类别名 `lesion`，并设
   `YOLO_CONFIG_DIR` 到工作区内。
2. **MPS 非确定性**：macOS MPS 上训练会打印 `scatter_reduce_mps does not have a
   deterministic implementation`，属无害警告。
3. **ONNX 导出与 PT 有轻微数值差异**：`yolo export format=onnx` 后置信度会和 `.pt`
   不同（导出融合导致），但网页端用的是 ONNX 自身，逻辑已对齐。
4. **不要用 test 反复挑模型**：test 只有 10 张，方差极大，仅作参考。
