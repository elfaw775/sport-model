# 84 张马球节 X 光片的整片病灶分类研究

**简体中文** · [English](README.md)

这项研究判断一张 X 光片是否至少有一处已标注病灶。84 张片中有 68 张阳性、16 张阴性，共 81 个病灶框。DINOv2 Small 给整张片打分，YOLO 给出候选区域的位置。病例工作台同时显示这两项结果。[工作台中文说明](../README.md)

## 黑边影响

固定五折与每折门槛，在同一批片上检查 10 种预先写好的补边颜色和位置。正常黑边居中时，原 DINO 方案判对 82/84；训练时为每张拟合片加入四种轻微变化的候选方案判对 83/84。要求同一张片在全部 10 种条件下都判对，两套方法均为 81/84。测试衡量的是同一批影像对补边的敏感性。工作台继续使用已经过部署验收的权重。[双语结果、逐条件表和复现命令](published/cv84/canvas-round2/README.md) · [逐片分数](published/cv84/canvas-round2/scores.csv)

## 应该怎样读这些成绩

| 相同五折、相同训练与校准样本 | 整片判对 | 漏检，分母为 68 张阳性 | 误报，分母为 16 张阴性 |
|---|---:|---:|---:|
| 最初 YOLO11s 配方重新训练五次 | 64/84 | 15 | 5 |
| 修正旋转标注框后重训的 YOLO11s | 62/84 | 20 | 2 |
| DINOv2 Small | 82/84 | 2 | 0 |

同一张片在每折中的用途对三种模型都相同。每轮用 50 或 51 张拟合模型，17 张校准，另外 16 或 17 张测试。汇总时每片只计一次。原配方如果采用历史固定门槛 0.15，也判对 64 张，漏检减至 12 张，误报增至 8 张。原始与重训 YOLO 在不同门槛下的框级定位成绩应另外解读。[三模型汇总](published/cv84/three-model-summary.json) · [逐片对照表](published/cv84/three-model-oof.csv) · [原版训练设置](published/cv84/original-yolo-recipe.json)

在另一种实验中，每一外折用四个内折从七种冻结编码器中选模型并定门槛。固定种子 42 时，所选分类器判对 83/84，漏检 1 张、误报 0 张；把 COCO 预训练 YOLO 当冻结特征提取器也能判对 80 张。种子 42、73、101 的三轮结果依次为 83、82、84 张，三轮共用这 84 张片。部署的 DINO 分类头在全部 84 张片上重新拟合，门槛 0.1788344393。它在自己训练片上的结果只说明拟合情况。[选模方案](cv84/plan.md) · [各方法结果](published/cv84/summary.json) · [折号](published/cv84/folds.json)

高度规则可判对 81/84，漏 2 张、误报 1 张。58 张阳性片能找到与病灶大小和亮度接近的对照区域，分别模糊病灶或对照区域后，这 58 张的二元判断均未改变。其余 10 张缺少合适对照。按文件名前导数字分组是对潜在重合的检查，无法确认具体马匹身份。历史上还用过 59 张训练、13 张验证、12 张测试的划分，DINO 在这 12 张上全部判对；Qwen3-VL-8B 直接读片判对 4 张，微调后判对 6 张。这些比较使用过同一批影像，换设备、换马匹后的表现仍需独立验证。[审计证据](published/evidence.zip) · [历史测试报告](published/holdout/test.json)

## 判断一张片

在仓库根目录运行。已验证的研究环境使用 Python 3.13、PyTorch 2.9.1、torchvision 0.24.1 和 CUDA 12.8。把 `$film` 指向自己的 JPG 或 PNG。若使用 CPU，安装适配本机的 PyTorch 并改用 `--device cpu`。

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r image-study/requirements.txt
$film = (Get-ChildItem training/data/sporthorse_v2/images/test -File | Select-Object -First 1).FullName
& $py image-study/cv84/predict84.py $film --device cuda
```

首次运行会获取固定版本的 Hugging Face 编码器并核对校验值。返回的 `predicted_positive` 表示分数是否达到该分类头的门槛。旧 59 张版可用 `image-study/predict_image.py` 运行。病例后台使用随仓库提供的 CPU ONNX，可离线运行。

## 复现实验

每次新实验设置独立的 `SPORT_XRAY_RUN` 输出目录，`SPORT_XRAY_CACHE` 则可跨实验复用。生成折表后提取特征，运行内折选模、外折评估和逐片校验。内折用 ROC-AUC、平均精确率和预定次序选编码器，再按平衡准确率选门槛。各折只用拟合片学习标准化和分类参数。[完整 PowerShell 命令](README.md#reproduce-the-84-film-analysis)

定位实验根据原始 COCO 标注重建旋转修正框。其余历史核查包括轻微亮度或对比度变化、JPEG 压缩、病灶区域模糊和按文件名前缀分组。模型和原图来源均可通过固定版本和校验值核对。[定位、扰动及分组复现命令](README.md#localization-perturbations-and-grouping)

复现原 YOLO 配方时，用 Ultralytics 8.4.200 和当时的标注，每折从公共 COCO 权重开始。原 Mac 的 MPS 实际使用单精度，正式 CUDA 复训也关闭了 AMP 和 TF32，五折分别完成 101、79、110、130、196 轮。若使用临时取片工具，原片在子进程结束后释放，结果应写入独立目录。

```powershell
$out = Join-Path $PWD 'work/original-yolo-cv'
& $py image-study/with_film_data.py --legacy-labels -- $py image-study/cv84/original_yolo.py --work $out
& $py image-study/cv84/compare_original.py --original "$out/original-yolo-report.json" --output $out
```
