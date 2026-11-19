# RTX 4070 Laptop 上的 RF-DETR 对照实验

这项实验在本机比较 RF-DETR Small、修正标签后重训的 YOLO11s，以及仓库归档的 `yolo11s_fresh_v2_best.pt`。后者是网页精确版的 PyTorch 来源，本实验测的是它本身，不是网页 ONNX 的输出或浏览器速度。

[2026-11-19 实测报告](../results/comparison-2026-11-19-rfdetr/README.md)：RF-DETR 在验证集领先，但没有超过 YOLO11s 的历史留出表现，未替换网页模型。

沿用 [v2 清单](../data/sporthorse_v2/split_manifest.tsv)的 59 / 13 / 12 张划分，以原始 CVAT 标注重新生成水平外接框。影像复制到新目录，归档图片和标签不改动，已检查跨集合像素完全相同的图片。马匹独立性来自项目此前的确认，仓库没有可复核的马匹 ID 表。历史测试集已用于以往模型比较，当前也只用于回归，不能代替外部盲测。

两个候选用 COCO 权重起步、640 输入、同样的训练轮数，按各自验证集 mAP50–95 选权重。同是 640 输入，但 YOLO 保比例填充，RF-DETR 使用自己的方形缩放；优化器参数和冻结策略也不同，所以这是可复现的两套配方对比，不是只改变网络结构的消融试验。

## Windows 本机运行

在仓库根目录，用 Python 3.13、支持 BF16 的 NVIDIA 显卡和匹配的 CUDA 版 PyTorch。本机验证过的是 PyTorch 2.9.1 + CUDA 12.8；不需要改动网页依赖。

```powershell
python -m venv .venv
$trainer = '.\.venv\Scripts\python.exe'
& $trainer -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $trainer -m pip install -r training/requirements-rfdetr.txt

$weights = New-Item -ItemType Directory -Force -Path 'work/rfdetr-weights'
$yolo = Join-Path $weights.FullName 'yolo11s.pt'
$rf = Join-Path $weights.FullName 'rf-detr-small.pth'
curl.exe --fail --location 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt' --output $yolo
curl.exe --fail --location 'https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth' --output $rf

$env:PYTHONUTF8 = '1'
& $trainer -u training/scripts/benchmark_rfdetr.py --yolo-init $yolo --rf-init $rf --epochs 32
& $trainer -m unittest discover -s training/tests -v
```

脚本核查两份预训练文件的校验值后才会加载。默认两个候选各跑 32 轮；显存紧时可以加 `--rf-batch 1 --yolo-batch 2`。实验默认在已忽略的 `work/rfdetr-comparison/` 下创建新目录，也可用 `--work` 指定本机空闲位置（目录必须尚不存在）。`comparison.json` 记录环境、权重校验值、时间、显存和指标；逐片预测和模型权重保存在本机。不上传影像，不启动网页部署。

若两套本机训练权重都已保存，但评分在留出集开始前中断，可用相同初始化参数和 `--work <原目录> --score-existing` 继续，避免重训。只接受自己生成、可信的权重目录。RF-DETR 1.9.4 的最终文件可能省略分辨率配置，本脚本明确按训练时的 640 输入和位置网格加载，不会回到默认 512。

## 如何读结果

- 三组用同一个 COCO 评分器计算 AP50、AP75 和 mAP50–95。输入同一张完整影像；不会拿真值框裁图再预测。
- 命中按 IoU ≥ 0.5 一对一匹配。置信阈值只在验证集选择：先限制平均每张不超过一个误报框，再取命中最多的阈值。验证集选定的阈值不变地应用到历史测试集，也另外报告固定 0.15 的结果，供和旧实验核对。
- 测试只有 12 个标注病灶，多命中一个就是约 8.3 个百分点。原始标注和所谓正常片仍需要专家复核；不应用这批结果决定真实筛查或替换线上模型。

参考：[RF-DETR 本地训练](https://rfdetr.roboflow.com/learn/train/) · [YOLO11](https://docs.ultralytics.com/models/yolo11/) · [上一篇修正标签对照试验](../results/comparison-2026-11-18/README.md)
