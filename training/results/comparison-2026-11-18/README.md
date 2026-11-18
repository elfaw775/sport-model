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
