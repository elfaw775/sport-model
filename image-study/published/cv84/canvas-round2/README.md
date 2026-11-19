# 画布补边与整片筛查 / Canvas padding and whole-film screening

[中文](#中文) · [English](#english)

## 中文

同一张马球节 X 光片保持相同的解剖像素、缩放尺寸和方向，只编辑模型输入画布外围的留白。两套 DINOv2 Small 分类器均对 84 张片（68 阳、16 阴）给出一次折外判断，再对同一批片重复九种扰动。每次仅改变画布颜色或摆放位置。每折有 50 或 51 张拟合片、17 张校准片和 16 或 17 张留出片。同期拟合的增强候选对每张拟合片使用四个等权视图，每张片的训练权重合计仍为 1。两套分类器各用自己的折内门槛，所有画布条件沿用该门槛。

| 336 × 336 画布条件 | 既有方案判对 | 增强候选判对 |
|---|---:|---:|
| 黑色居中（工作台预处理） | 82/84 | 83/84 |
| 灰度 8 居中 | 82/84 | 83/84 |
| 灰度 16 居中 | 81/84 | 82/84 |
| 灰度 32 居中 | 81/84 | 81/84 |
| 灰度 64 居中 | 81/84 | 82/84 |
| 灰度 128 居中（压力测试） | 81/84 | 82/84 |
| 黑底，空余空间的 0% 处起始（贴边） | 83/84 | 83/84 |
| 黑底，25% 处起始 | 82/84 | 83/84 |
| 黑底，75% 处起始 | 83/84 | 83/84 |
| 黑底，100% 处起始（反向贴边） | 83/84 | 83/84 |

全部误判都是漏检。按独立影像计，两套方案各有 81 张在十种条件下始终判对，2 张随画布而改变判断，1 张始终漏检。增强后常规画布上找回 79 号阳性片，但在灰度 32 时它仍被漏掉；78 号阳性侧位片在两套方案中都会受补边颜色影响；13-RF-侧位在两套方案的全部画布上均漏检。71 张的空余空间在左右方向，另外 13 张在上下方向且全为阳性，16 张阴性都来自前一组。该样本中增强候选的常规画布正确数增加了一张，始终判断正确的影像数没有增加。推断新马或新设备上的效果仍需要独立数据。

## English

The study changes only the external padding surrounding the same resized pixels (336 × 336 canvas). The frozen DINOv2 Small encoder and existing 84-film split are shared by both approaches. Each fold fits on 50 or 51 films, calibrates on 17, and evaluates the other 16 or 17. The candidate fits on four equally weighted views per fitting film while preserving one unit of training weight per film. Its own threshold is selected on original calibration films. Both approaches keep their respective fold thresholds fixed for all 10 conditions; the variants are repeated measurements of 68 positive and 16 negative films.

The table lists correct films for the original probe (middle column) and the augmentation candidate (right). The standard centered black input yields 82/84 and 83/84 respectively. Across the ten conditions both approaches classify 81 films correctly every time, change decisions on two films and consistently miss the same positive lateral film (13-RF). No variant produced a false alarm among the 16 negative films. Positive film 78 is missed when the padding turns gray. The augmented model recovers positive film 79 on the standard canvas but still misses that film at gray level 32. Left/right padding occurs in 71 films and top/bottom padding in 13. All 16 negative films have left/right padding. This is an internal sensitivity study. The candidate's one extra correct film on the standard canvas did not increase the number of films consistently classified correctly. Independent acquisition-matched cases are needed to assess generalization.

## 文件 / Files

- [预先固定的实验方案与复现命令 / Prespecified plan and commands](../../../cv84/canvas-round2-plan.md)
- [每张片的得分、折号、判断门槛 / Per-film scores, folds and thresholds](scores.csv)
- [分条件统计、留白方向及翻判清单 / Metrics, padding-axis strata and changed decisions](summary.json)
- [五个研究候选的分类头与阈值（私有归档，需仓库权限） / Five research heads and thresholds (private archive, repository access required)](https://github.com/theAtlantic-zza/sport-horse-xray-models/blob/main/weights/canvas-round2/augmented-fold-heads.npz)
