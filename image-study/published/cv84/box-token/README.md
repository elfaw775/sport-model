# 病灶框监督的 DINOv2 图块打分 / DINOv2 scoring with lesion-box supervision

[中文](#中文) · [English](#english)

## 中文

本轮使用 68 张病灶片、16 张阴性片和 **81 个病灶框**。DINOv2 Small 的图像编码器保持冻结；在每折的 50 或 51 张拟合片上，用旋转修正的病灶外接框监督一个新的局部线性头，学习哪些 14 × 14 像素图块更像病灶。每折用另外 17 张片的整片标签选门槛，在余下 16 或 17 张上考试。推断只读图像，排除外补边上的图块中心。局部分数取最高三个图块分数的平均；主要候选再与原全局头的整片分数各占一半。该方法、评估指标和画布条件已在运行前确定。

| 模型 | 黑边居中：判对；漏检 / 误报 | 灰 32 居中：判对；漏检 / 误报 | 黑边起点对齐：判对；漏检 / 误报 |
|---|---:|---:|---:|
| 原整片 DINO | 82/84；2 / 0 | 81/84；3 / 0 | 83/84；1 / 0 |
| 新局部头 | 73/84；7 / 4 | 72/84；8 / 4 | 70/84；11 / 3 |
| 整片与局部等权融合 | **83/84；1 / 0** | 81/84；3 / 0 | 83/84；1 / 0 |

标准黑边下，融合头找回了 79 号正位阳性片，13-RF 侧位仍漏检。三种条件都正确的片数，原整片头与融合头均为 81/84。三种视图重复使用同一组 84 张片。标准黑边逐折结果如下，数字表示答对片数。

| 折 | 留出片 | 原整片头 | 局部头 | 融合头 |
|---|---:|---:|---:|---:|
| 1 | 17 | 17 | 15 | 17 |
| 2 | 17 | 16 | 14 | 16 |
| 3 | 17 | 16 | 17 | 17 |
| 4 | 17 | 17 | 12 | 17 |
| 5 | 16 | 16 | 15 | 16 |

### 指向哪里

对全部 68 张留出阳性片，每张取一个最高分图块的中心。在标准黑边下，55/68 个点落入任一病灶的修正外接框，49/68 落入更严格的原始旋转矩形。总指向图像中心分别命中 29/68 和 25/68。均匀随机有效图块的期望命中数分别为 8.84 和 6.08。灰 32 时最高分点命中外接框 / 旋转矩形的张数为 54 / 48，靠起点时为 53 / 50。即使在标准条件下，仍有 19 张阳性片的最高分点落在全部旋转矩形外。这个检查衡量一个点是否落入任一粗标注区域，不能衡量每处病灶是否检出、框的位置精度或分割质量。影像来源和患者身份仍需在独立数据上核实。

## English

We used 68 positive films, 16 negative films and 81 annotated lesion boxes. With the DINOv2 Small encoder frozen, fitting-fold boxes supervised a linear patch head. No test or calibration boxes were used in fitting, threshold selection or inference. Pooling used the mean probability of the three highest valid patch tokens. A prespecified candidate averaged the local and global film scores equally. Each of five folds reused the matched fitting (50–51 films), calibration (17) and held-out (16–17) subsets. All three input conditions shared each model's fold-specific cutoff.

| Film scorer | Centered black: correct; FN / FP | Centered gray 32: correct; FN / FP | Black origin alignment: correct; FN / FP |
|---|---:|---:|---:|
| Global baseline | 82/84; 2 / 0 | 81/84; 3 / 0 | 83/84; 1 / 0 |
| Local patch head | 73/84; 7 / 4 | 72/84; 8 / 4 | 70/84; 11 / 3 |
| Equal-weight fusion | **83/84; 1 / 0** | 81/84; 3 / 0 | 83/84; 1 / 0 |

The centered-black fusion corrected film 79 and still missed film 13-RF. Both global and fusion heads were correct on 81/84 films across all three views. The five standard-view fold totals (global / local / fusion) were 17/15/17, 16/14/16, 16/17/17, 17/12/17 and 16/15/16, with fold sizes of 17, 17, 17, 17 and 16.

On the standard canvas the most suspicious patch center was inside a corrected axis-aligned envelope on 55/68 held positives and an original rotated rectangle on 49/68. A fixed image-center pointer hit 29/68 and 25/68; expected hits for a uniformly sampled valid token were 8.84 and 6.08. For gray padding and origin alignment, the most suspicious patch gave 54/48 and 53/50 envelope/rotated hits. The denominators include missed positive films. These point tests evaluate whether a proposed location lies in any labeled region on a positive film. They do not cover detection of every individual lesion or segmentation quality. One additional correct standard-view result on this small, reused internal cohort does not establish improvement on new acquisition sources. Production continues using the previously verified screening model and YOLO localizer.

## 复现 / Reproduction

Use the pinned study environment. The temporary loader verifies each public film and clears its input directory on exit. The weights argument points to the local research archive.

```powershell
& $py image-study/with_film_data.py -- $py image-study/cv84/box_token_study.py --output work/box-token --weights work/box-token/heads.npz
```

[预定方案 / Prespecified protocol](../../../cv84/box_token_plan.md) · [代码 / Code](../../../cv84/box_token_study.py) · [逐片分数与原图坐标 / Scores and point coordinates](predictions.csv) · [汇总及校验值 / Summary and hashes](summary.json) · [五折研究权重，需权限 / Research heads, access required](https://github.com/theAtlantic-zza/sport-horse-xray-models/blob/main/weights/box-token/box-token-fold-heads.npz)
