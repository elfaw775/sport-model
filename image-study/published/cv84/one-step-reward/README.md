# 答对奖励能减少漏检吗？ / Can a correctness reward reduce misses?

[中文](#中文) · [English](#english)

## 中文

同一批 84 张影像中，68 张有已标注病灶，16 张为阴性。原整片 DINOv2 Small 和按奖励调整过的版本，各判对 **82/84**，各漏 2 张、误报 0 张。两种方法对全部 84 张给出了逐片相同的决定。每折分类头参数确实发生了变化，训练目标也在每一折改善了。面对这批留出片，训练目标的改善没有带来额外检出。

冻结 DINOv2 编码器，从每折标准监督分类头开始。对于“提示”与“不提示”两种动作，训练标签给出了各自的收益，答对得 1，答错得 0。阳性、阴性对总收益的贡献各占一半。对动作概率直接求期望，再限制参数偏离原值的幅度。每折在 50 或 51 张拟合片上优化参数，在另 17 张上校准报警门槛，最后一次性判断余下 16 或 17 张。每张片仅计入一次测试。这里实现的是单步、两种动作收益均已知的优化。有序决策、多轮试错和医生反馈不在本次实验范围内。

程序先复算先前公布的 DINO 判定，再比较候选。复算逐片判定完全一致，最大分数差为 0.000034。开发中一次 CPU 重算未通过预定的一致性检查，另一次优化器在线搜索时停止。最终运行使用与基准一致的 CUDA 图像处理流程，以双精度优化权重并检查收敛和逐片基准，未调整目标函数、训练/测试划分或正则强度。训练使用 PyTorch 提取视觉特征、SciPy 更新轻量分类头，工作台使用 ONNX Runtime 在 CPU 推理。

### 三个公开团队给这个项目的启发

- 腾讯混元 HunyuanVision 展示了局部裁剪放大，以及在图上标点、画框等视觉操作。这里可借鉴“把关注的位置留给医生核对”。这一版阅片页可从文字提示直接跳到原图候选框，框监督实验另外测量最高分点是否真的进入标注区域。[混元官方说明](https://github.com/Tencent-Hunyuan/HunyuanVision#model-capabilities)
- 小米 MiMo-VL 同时发布监督微调与强化学习版本。其报告针对框和点设计可核验的几何奖励，例如点是否进入标注框。这启发我们把“整片是否判对”和“模型指在哪里”分开计分。本轮局部头使用病灶框监督拟合，单步奖励对照仅奖励整片答案。[官方代码与权重说明](https://github.com/XiaomiMiMo/MiMo-VL) · [MiMo-VL 技术报告 §3.1](https://arxiv.org/html/2506.03569v1#S3.SS1)
- DeepSeek-R1 把筛选过的教师输出用于监督微调较小模型。以后有了经医生核验的高质量示例，可以用它们教较小模型。项目目前有人工框，能直接检查学习效果；尚未开展教师蒸馏。[DeepSeek-R1 技术报告 §2.4](https://arxiv.org/html/2501.12948v1#S2.SS4)

既有高度对照仅用影像高度即可判对 81/84。答对奖励本身无法识别模型依据的是病灶、裁剪习惯还是设备特征。当前仍缺按马匹、设备及来源独立的外部病例验证。工作台的已验收权重继续使用。[病灶框监督与定位验证](../box-token/README.md) · [五折决定和门槛](scores.csv) · [各折权重变化](summary.json) · [运行前方案](../../../cv84/one-step-reward-plan.md) · [实验权重，需仓库权限](https://github.com/theAtlantic-zza/sport-horse-xray-models/blob/main/weights/one-step-reward/reward-fold-heads.npz)

## English

On the same 84 held-out film decisions (68 positive, 16 negative), the supervised DINOv2 Small head and the reward-updated head both produced 82 correct decisions, two misses and zero false alarms. All 84 decisions matched. Parameters changed and the penalized training objective improved in all five folds. The standard black canvas was used for this experiment.

Freeze the encoder, start from the fitted supervised head and maximize the expected reward of a correct decision, weighting both classes equally and penalizing squared distance to the original parameters. Both action rewards are known from each fitting label. Each fold fits 50 or 51 films, calibrates one threshold on 17 and predicts 16 or 17 untouched test films. This full-information one-step experiment uses PyTorch features and SciPy optimization. The workbench runs two ONNX artifacts on CPU. The baseline gate reproduced all original decisions within 0.000034 score drift. Earlier CPU and numerical runs failed reference or optimizer checks. The final run uses the reference CUDA canvas and a converged double-precision optimization with the prespecified objective.

HunyuanVision demonstrates zooming, cropping and spatial marks. MiMo-VL describes verifiable box and point rewards. DeepSeek-R1 describes supervised distillation from curated teacher outputs. The workbench lets readers inspect original pixels and numbered candidate boxes. Our research compares box-supervised localization against the labels and tests a binary correctness reward in a separate arm. Teacher distillation and online feedback learning would require their own evidence. An image-height control scored 81/84 and external, horse- and source-independent validation remains outstanding. [HunyuanVision](https://github.com/Tencent-Hunyuan/HunyuanVision#model-capabilities) · [MiMo-VL §3.1](https://arxiv.org/html/2506.03569v1#S3.SS1) · [DeepSeek-R1 §2.4](https://arxiv.org/html/2501.12948v1#S2.SS4)

[Prespecified plan](../../../cv84/one-step-reward-plan.md) · [Per-film calls](scores.csv) · [Fit details](summary.json) · [Box experiment](../box-token/README.md#english) · [Archived heads, access required](https://github.com/theAtlantic-zza/sport-horse-xray-models/blob/main/weights/one-step-reward/reward-fold-heads.npz)

### 复现 / Reproduce

Run from the repository root in the research environment with the pinned DINOv2 Small encoder and CUDA available. The helper verifies temporary film inputs and removes them on exit.

```bash
python image-study/with_film_data.py -- python image-study/cv84/one_step_reward.py --output work/one-step-reward
```
