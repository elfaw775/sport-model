# 单步奖励权重对照 / One-step reward experiment

**中文**　在现有 DINOv2 Small 整片模型基础上，比较标准监督分类头和单步奖励微调头。沿用已公布的种子 42 五折和每折的拟合、校准、留出片。动作是提示可疑或不提示，正确奖励 1、错误奖励 0。两类片分别占总权重的一半。每次判断后没有下一状态，两种动作的收益都可由训练标签计算。直接对动作概率求期望，不采样动作。

**English**　Start each fold from its fitted supervised DINOv2 Small head. Maximize class-balanced expected correct decisions on its fitting films, minus 0.025 times the squared Euclidean distance to the supervised coefficients and intercept. Use the same frozen encoder and scaler. Apply deterministic L-BFGS-B, with at most 600 iterations. Select one candidate threshold using the existing calibration-film midpoint rule. Score all outer test films once. Do not search penalty values or replace production weights from this study.

**验收 / Checks**　原方案须复现已公布的 84 个判定，最大分数差不超过 0.002。报告每套方案的漏检、误报、校准门槛、逐片翻判及拟合目标变化。每折拟合参数归档。The test-film labels are used to tally errors after fitting. This is a one-step, fully labeled reward experiment on existing radiographs, with no online environment or sequence-level reasoning.
