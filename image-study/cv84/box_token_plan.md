# 框监督的 DINO 局部打分 / Box-supervised DINO patch scores

## 中文

复用种子 42 的 84 张片与既有 DINO/YOLO 对照的五套拟合、校准、留出索引。研究对象是整片筛查和一个可核对的可疑点。模型只接收像素。每折独立拟合。公开 COCO 的 81 个病灶框只在该折的拟合片上用于训练；留出片的框在打分完成后才用于定位评估。校准仅需整片的阳性或阴性。

将片等比缩放并补黑边到 336 × 336。冻结 DINOv2 Small；每个 14 × 14 像素的图块对应一个最终层特征。排除中心位于补边的图块。在原片坐标中，中心落入经旋转修正的轴对齐病灶框即为训练正例；若某框没有覆盖中心，使用最近的有效图块中心代替，仅用于拟合。其余有效中心作为弱负例。各阳性片均分正样本权重，各有背景的拟合片均分负样本权重，两类总权重相同。冻结特征归一化后乘以维度平方根，用固定正则强度 C=1 训练线性逻辑回归。

局部分支以分数最高三个有效图块的平均值为整片分数，最可疑点取最高分图块中心。主要候选使用局部分数与原全局 DINO 分数的等权平均；保留单独局部分支以解释结果。各候选只用本折原图校准片选择平衡准确率门槛。校准和推断均无需框。基准须复现原来的 82/84。对居中黑边、居中灰边（灰度 32）、黑边靠起点三种条件逐片输出分数和误报、漏检。用同一套权重与门槛评估所有条件。

对全部 68 张留出阳性片评估最高分点落入任一修正外接框及原始旋转矩形的比例，含筛查漏检片，分别与图像中心、均匀随机有效图块的期望命中率对照。百分点对照只评价指向性。以原图为主要比较，同时报告与既有 DINO 相比纠正和新增的错误。方法与权重的选择均在本轮运行之前固定。

## English

Use the seed-42 matched five-fold fit/calibration/test partitions of the 84-film study. Freeze DINOv2 Small. Map its final 24 × 24 patch tokens to oriented image coordinates using the actual rounded resize dimensions and padding offsets. Exclude centers on external padding. Train a logistic patch head on corrected axis-aligned lesion boxes from fitting films only. A box without any token center contributes its nearest valid center in fitting. Other valid centers are weak background labels. Give positive fitting films equal lesion weight, and fitting films with background equal background weight. Class totals are balanced with a combined weight equal to the number of fitting films. L2-normalize each token and multiply by the square root of its dimension; use C=1.

Average the highest three valid probabilities for a local film score, and use the highest-scoring token for the proposed point. The primary candidate averages local and global film scores equally. Calibrate local and combined thresholds on original calibration films using the original balanced-accuracy rule. The published global baseline must reproduce 82/84. Evaluate centered black, centered gray 32, and black origin alignment without changing weights or thresholds. Report paired errors, sensitivity and false alarms. On all 68 held positive films score the point against the corrected bounding envelope and the original rotated rectangle, versus the fixed center and expected uniformly random valid-token hits. Test annotations are used only to evaluate the final points. No model selection on these results.
