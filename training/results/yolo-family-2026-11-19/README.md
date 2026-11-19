# 马球节病灶检测 · YOLO640 发布与候选对照

Python 工作台使用校正旋转框后重训的 YOLO11s，输入 640 × 640，候选阈值 0.20。该模型在 12 张历史留出片上命中 3 个标注病灶，误报 2 框，漏掉 9 处。保留每个病例原先记录的模型版本和阈值。[ONNX 推理权重](../../../backend/models/yolo11s-fetlock-640.onnx) · [PyTorch 训练权重](../comparison-2026-11-18/yolo11s-epoch12.pt)

## 相同马片上的结果

沿用 84 张影像各来自一匹马的项目假设，59 张训练、13 张验证、12 张历史留出。后两组各有 12 个参考病灶。每个模型只在验证集选阈值，限制平均每张验证片最多一个误报框，优先命中数，其次误报更少；最后将相同阈值用于留出片。一处病灶只匹配一次，命中要求交并比至少 0.5。“命中”与“误报”均按框统计。历史留出片已用于多轮实验，结果反映本批片的表现。

| YOLO 模型 | 验证选择阈值 | 验证命中 / 12 | 验证误报框 | 留出命中 / 12 | 留出误报框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 修标 YOLO11s，标准 AdamW，当前版本 | 0.20 | 5 | 6 | 3 | 2 |
| 同一 YOLO11s，精细定位层学习率 × 1.5 | 0.25 | 6 | 11 | 3 | 5 |
| 更大的 YOLO11m | 0.075 | 4 | 12 | 1 | 6 |
| 新架构 YOLO26s | 0.075 | 7 | 11 | 3 | 13 |
| Hugging Face 骨折 ONNX，直接读片 | 0.15 | 0 | 0 | 0 | 3 |

YOLO26s 在验证片多找到了两处，优势在留出片上消失。切换到固定阈值 0.15 后，YOLO26s 在留出片上命中 0 处、误报 5 框，修标 YOLO11s 在同阈值命中 3 处、误报 5 框。工作台选用修标 YOLO11s 的依据是病灶定位与误报负担。这一小规模单种子对照不足以估算新病例上的诊断准确率。[完整计数、阈值和权重哈希](metrics.json) · [逐片预测、运行记录和训练曲线](runs.zip)

## 训练和来源

YOLO11s、11m 和 26s 均从各自的官方 COCO 检测权重起步，训练 12 轮、输入 640、batch 4、seed 42、冻结前 10 层，使用相同的几何增强，取最后一轮权重。精细定位实验保留原有 AdamW，仅将最高分辨率检测层的框回归参数分组，使其学习率目标从 0.001 提至 0.0015。分类、其余尺度、权重衰减和各组预热规则沿用原方案。它在验证片增加一个命中，在留出片没有增加命中，误报增加 3 框。[训练入口](../../scripts/train_yolo_comparison.py) · [优化器实现](../../scripts/fetlock_optimizer.py)

修标训练使用的官方 YOLO11s 初始化文件，其 SHA-256 与 Hugging Face 固定版本一致，项目此前的复现说明给出的下载源是 Ultralytics GitHub Release。项目训练指南另引用了 Jesteban247 的人类骨折模型作为冻结层数的参考。该仓库提供 YOLO11n 单类骨折 ONNX，本轮将这份权重直接应用于马片，未对它进行微调。[原项目训练指南](../../docs/训练指南.md) · [YOLO11 官方权重](https://huggingface.co/Ultralytics/YOLO11/tree/8b8ac7d1fae7468f85dbf89670dd66f41f485aab) · [YOLO26 官方权重](https://huggingface.co/Ultralytics/YOLO26/tree/070ac3c51435984d992ba92eb3f234daf4e4500d) · [骨折模型源文件](https://huggingface.co/Jesteban247/yolo11-fracture-onnx/blob/8227dbcdcf80f3cd06f052a8295ee42657989deb/best.onnx)

## 验证与复算

发布权重从归档的第 12 轮 PyTorch 权重导出，CPU、FP32、ONNX opset 17、静态 batch 1、无内置 NMS。Python 服务对 25 张验证和留出影像逐张推理。0.20 阈值下的 16 个候选与训练端逐一对应，最大坐标差低于 0.00006 像素，最大分数差低于 0.000002。后端回归测试比较全部 25 张，浏览器验收覆盖有框、无框、人工复核和归档。[回归测试](../../../backend/tests/test_detector.py)

准备数据用 `python training/scripts/prepare_comparison.py`。在仓库根目录及既有训练环境中，以下命令可重跑对照。权重文件从上述固定版本获取，SHA-256 见指标文件。

```sh
python training/scripts/train_yolo_comparison.py --weights work/model-comparison/pretrained/yolo11s.pt --optimizer-profile fetlock-box --output work/model-comparison/repro-fetlock-box
python training/scripts/train_yolo_comparison.py --weights work/model-comparison/yolo-family/pretrained/yolo26s.pt --output work/model-comparison/repro-yolo26s
python training/scripts/train_yolo_comparison.py --weights work/model-comparison/yolo-family/pretrained/yolo11m.pt --output work/model-comparison/repro-yolo11m
python training/scripts/compare_detectors.py --candidate work/model-comparison/repro-yolo26s
```

也可解压实验包，直接将含有两份逐片预测的目录传给 `compare_detectors.py --candidate` 复算，无需重训。复测单类 ONNX 使用后端环境中的 `training/scripts/evaluate_yolo_onnx.py --model 文件 --sha256 校验值 --output 新目录`。
