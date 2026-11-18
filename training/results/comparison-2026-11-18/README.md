# 球节骨折候选框检测：两种迁移方案的实际对照

**结论：保留修正标签重训的 YOLO11s 作为第一候选，工作台仍使用原 v2。** 在现用分数阈值 0.15 下，候选 YOLO 在历史留出测试集上的误报从 9 个降至 5 个，但两者均只检出 12 个病灶中的 3 个。这个规模下还不能宣称对新马匹的检出率提高。只训练 Faster R-CNN 的末端预测层效果不好，不能推广到整套架构。

按项目方确认，每匹独立的马提供一张影像，共 84 匹：68 阳性、16 标为正常。沿用现有 59/13/12 匹训练/验证/测试划分，不重抽样。两候选使用根据 CVAT 旋转角度修正的水平外接框，三组比较都对照修正后的标签。类别只有球节疑似骨折，数据修正未替代兽医复核。旧测试片过去已用于项目模型比较，本轮只做留出回归，不作为独立外部验证。

## 固定阈值的逐病灶结果

交并比 ≥ 0.5 计为命中，按分数优先，每个标注最多匹配一次；“误报”是框数。验证集和测试集分别有 12 个标注病灶。下表按 **命中 / 误报 / 漏检** 排列。

| 模型（分数 ≥ 0.15） | 验证，13 匹 | 留出回归，12 匹 |
|---|---:|---:|
| 现役 YOLO11s v2 ONNX | 3 / 15 / 9 | 3 / 9 / 9 |
| 新 YOLO11s，第 12 轮 | 5 / 11 / 7 | 3 / 5 / 9 |
| Faster R-CNN R50-FPN v2，仅微调预测头 | 3 / 123 / 9 | 2 / 95 / 10 |

不同架构的原始分数尺度不同。另用固定候选阈值网格，在验证集限制平均每片误报 ≤ 1 个；优先选命中最多的阈值，同命中数选误报更少，再选较低阈值。将选定阈值原样应用到历史测试片：

| 模型 | 验证选定阈值 | 验证：命中 / 误报 / 漏检 | 留出：命中 / 误报 / 漏检 |
|---|---:|---:|---:|
| 现役 YOLO | 0.30 | 3 / 5 / 9 | 2 / 1 / 10 |
| 新 YOLO | 0.20 | 5 / 6 / 7 | 3 / 2 / 9 |
| Faster R-CNN | 0.90 | 0 / 0 / 12 | 0 / 0 / 12 |

这些阈值是实验结果，不是推荐的临床工作点。本轮仅比较两种有限计算预算的微调方案，不是架构优劣的证明。YOLO 的新旧版本还存在训练轮数、增强和超参差异，不能把结果完全归因于旋转框修正。

## 漏检发生在哪里

对每个已标注病灶，按原评分规则将归档框按分数降序与剩余病灶一对一匹配。四列合计均为 12 个病灶。“部分重叠”指所有得分 ≥ 0.01 的归档框与该病灶的最高 IoU 介于 0 和 0.5 之间。“无重叠”包含整张片无框、框落在别处两种情况。两模型均使用已导出的候选框，未观测到模型内部被抑制的框。

| 影像组 / 模型 | 0.15 命中 | 降至 0.01 才命中 | 仅部分重叠 | 无重叠 | 误报框数，0.15 → 0.01 |
|---|---:|---:|---:|---:|---:|
| 验证 / 现役 | 3 | 2 | 6 | 1 | 15 → 298 |
| 验证 / 重训 | 5 | 1 | 2 | 4 | 11 → 71 |
| 历史留出 / 现役 | 3 | 4 | 5 | 0 | 9 → 195 |
| 历史留出 / 重训 | 3 | 0 | 3 | 6 | 5 → 47 |

历史留出片上，重训版的 6 个无重叠病灶中，3 个所在片整片无框，另 3 个所在片有框但位置偏离。把工作阈值降到 0.01 对重训版没有增加一次留出命中。验证片上的 4 个无重叠目标，原图按长边 640 像素等比换算后，标注框短边为 35.6–64.1 像素；5 个命中目标短边为 73.2–330.8 像素。这里量的是标注框而非骨折线宽度，例数少，仅作为下一轮分辨率实验的依据。

[逐病灶原图索引](lesion-error-audit.csv)每行列出数据集分组、模型、仓库内原片路径、病灶序号（从 1 开始）、修正后水平框的原图像素坐标、等比换算的短边、最高 IoU、一对一命中的分数与整片候选框数。验证集可先看这四片，再打开预测核对：

- [26 号正位](../../data/sporthorse_v2/images/val/26-LF-正位.jpg)与[29 号正位](../../data/sporthorse_v2/images/val/29-RF-正位.jpg)各有一个病灶。重训版对前者输出 10 个低阈值候选框，全部偏离；后者整片无框。
- [57 号正侧位合片](../../data/sporthorse_v2/images/val/57-LF-正+侧，近端籽骨顶端骨折.jpg)有两处标注。重训版对第一处最高 IoU 0.246，第二处无重叠。复核时可登记各框所在投照区域。
- [41 号侧位](../../data/sporthorse_v2/images/val/41-侧位.jpg)最高 IoU 为 0.498，接近匹配边界。请两位读片人先各自框出可疑范围，再核对框边界差异。

### 下一轮的两个实验

1. **先统一标注口径。** 两名兽医在训练、验证片上先独立阅片，记录部位、投照位、可见病灶数、每处框、骨折线是否清楚及图像是否足以判断。同一部位统一“框住可见病灶范围”的尺度，正侧位合片逐投照标注。分歧经会审，保存旧框、复核框和变更原因。同步复读两组中标为正常的片，区分确认阴性与暂不能判读。用一致的口径补收独立马匹的小病灶阳性及同投照的易误报阴性，登记新马匹/检查编号、来源与原始像素尺寸。另留一批连续入组的新马匹，盲评前封存。
2. **只改变输入分辨率。** 完成口径复核后，以同一批标签、同一预训练权重、固定训练轮数和增强分别训练 640 与 960 输入的 YOLO。训练/验证继续按马隔离，分别报病灶命中数、每片误报及投照方式和框大小；工作阈值只在验证集选择，沿用每片平均误报不超过 1 个的工作预算。最后在封存的新马匹上使用固定模型和阈值做一次盲评，再决定是否替换当前权重。历史留出片已反复比较，仅供回归核查。

## 权重和复现

- [候选 YOLO11s 权重](yolo11s-epoch12.pt)：官方 COCO 初始化，冻结前 10 层，640 输入、batch 4、AdamW、轻微平移与缩放、MPS，训练 12 轮。候选使用最后一轮；[运行参数](yolo-run.json)和[训练曲线](yolo-epochs.csv)随附。
- [Faster R-CNN 预测头](frcnn-head.pth)：官方 COCO 初始化，冻结 backbone/FPN、RPN 和 ROI 特征，仅微调两类预测层的 10,250 个参数，CPU 两线程、12 轮。缓存训练片特征，并仅对训练样本增加真值附近的候选框。验证和测试是完整影像推理，不使用真值候选框。[运行参数](frcnn-run.json)。用训练脚本的 `build_model(base_weights_path, head_path)` 和指定官方基座可重建，已核对全部 404 项参数及缓冲区完全一致。
- [汇总计数](metrics.json)、[三种模型的逐图预测](predictions.zip)。归档只含输出框与分数，不包含病例数据库。

在仓库根目录使用 Python 3.12；两个训练脚本使用 seed 42。完整训练图位于 `training/data/sporthorse_v2/images`，使用稀疏克隆时须先取齐。

```sh
mkdir -p work/model-comparison/pretrained
uv venv --python 3.12 work/model-comparison/venv
uv pip install --python work/model-comparison/venv/bin/python 'torch==2.15.0' 'torchvision==0.29.2' 'ultralytics==8.4.215' 'numpy==1.26.4' 'opencv-python==4.11.0.86'
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt -o work/model-comparison/pretrained/yolo11s.pt
curl -fL https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth -o work/model-comparison/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
python3 training/scripts/prepare_comparison.py
work/model-comparison/venv/bin/python training/scripts/train_yolo_comparison.py
work/model-comparison/venv/bin/python training/scripts/train_frcnn_comparison.py
.venv/bin/python training/scripts/compare_detectors.py --shipped
.venv/bin/python training/scripts/compare_detectors.py
```

最后两步使用后端环境（含 ONNX Runtime）。训练端非 Apple 芯片时为 YOLO 指定 `--device cpu` 或可用的 CUDA 设备。若只想重计分，准备数据后将预测压缩包解压到 `work/model-comparison/`，直接执行最后一步，无须重训。

只复现上述逐病灶审计时，在根目录运行 `python3 training/scripts/prepare_comparison.py`，然后运行 `.venv/bin/python training/scripts/audit_detection_errors.py`。审计直接读取归档 ZIP、修正后的标签和原图尺寸，只需安装 Pillow，会校验 0.15 结果与归档计数一致并重新生成 CSV。
