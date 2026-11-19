# 补边敏感性实验二 / Padding sensitivity study 2

## 中文

沿用已发布的 84 张片、五折训练/校准/留出名单和 DINOv2 Small。正片内容先按生产流程一次缩放，再逐字节复制到 336 × 336 灰度画布。固定以下十种画布，只改变外补边，原图中的黑色区域也作为影像内容保留。

| 改动 | 预先约定的条件 |
|---|---|
| 居中时外围灰度（0–255） | 0、8、16、32、64、128。0 是生产基准；128 属于压力测试 |
| 黑色画布内的摆放 | 剩余空间的 0%、25%、50%、75%、100%。50% 是同一个生产基准；0% 和 100% 是极端位置 |

主比较按旧方案在原图上拟合折内分类头，锁定发布过的每折判断门槛。原图必须复现公布的全部 84 个判断和旧对照结果。同时测试一个预定的研究候选：每张拟合片贡献四个等权视图（原图、灰度 16 居中、黑底 25%、黑底 75%），权重合计为 1。冻结同一编码器；每折独立拟合分类头，只在原始居中的校准片上按原规则选择一次门槛。该门槛用于当折全部留出条件。记录各条件的错误数、相对于同一模型原图的翻判、每张片的分数、折号及门槛，并按留白方向分层。重点看原图成绩、十种条件下均正确的片数、出现过翻判的片数。每项统计以 84 张独立图像为基本单位，十个视图相关。结果不用于为本轮挑选另一个门槛、补边或线上模型。

## English

Use the published 84 films, matched five-fold fit/calibration/held partitions and frozen DINOv2 Small encoder. Resize each oriented grayscale film once and paste its unchanged pixels on a 336 × 336 canvas. Original black regions inside the film remain part of the content. Use six fixed centered padding levels (0, 8, 16, 32, 64, 128) and five positions on black (0%, 25%, 50%, 75%, 100% of available space). Centered black is shared, giving ten distinct conditions. Mid-gray 128 and flush alignment are stress tests.

The baseline head is fit on original views with its published threshold locked per fold. It must reproduce all previously published baseline decisions and earlier canvas conditions. The single research candidate trains on four prespecified equally weighted views of each fit film (original, centered gray 16, black at 25%, black at 75%). Total sample weight per film is one. Fit each head separately. Select its threshold once on original views of that fold's calibration films via the existing balanced-accuracy rule. Use that same threshold across all held conditions. Report paired errors and flips, per-film scores, fold and threshold, padding-axis strata, stability across all ten views and original-view accuracy. The ten correlated views do not expand the sample beyond 84 films. No outer-label tuning, deployment change, or external generalization claim.

## 复现 / Reproduce

```powershell
& $py image-study/with_film_data.py -- $py image-study/cv84/canvas_round2.py --output image-study/published/cv84/canvas-round2
```
