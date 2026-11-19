# 运动马球节骨折定位 · 多款视觉模型实测

在固定历史留出组中，Qwen3‑VL‑8B 直接读片命中 2 处病灶、没有误报框，同系列 2B 为 0 处和 12 个误报框。8B 用 59 张片微调后仍命中原来两处，误报框增至 2 个。修正标签的 YOLO640 命中 3 处、误报 2 框。在这批马片上，更大的同系列模型能定位到少数病灶，工作台仍以 YOLO 提供候选框。

## 两组片的实测结果

| 模型及读片方式 | 验证命中 / 12 | 验证误报框 | 留出命中 / 12 | 留出误报框 | 留出整片判对 / 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 现役 YOLO11s · 阈值 0.15 | 3 | 15 | 3 | 9 | 8 |
| 修正标签 YOLO640 · 验证选定阈值 0.20 | 5 | 6 | 3 | 2 | 5 |
| Qwen3-VL-2B · 直接 | 1 | 15 | 0 | 12 | 9 |
| Qwen3-VL-2B · QLoRA | 0 | 13 | 1 | 7 | 8 |
| Qwen2.5-VL-7B · 直接 | 0 | 0 | 0 | 0 | 2 |
| Qwen2.5-VL-7B · QLoRA | 0 | 1 | 0 | 1 | 3 |
| Qwen3-VL-8B · 直接 | 1 | 1 | 2 | 0 | 4 |
| Qwen3-VL-8B · QLoRA | 0 | 5 | 2 | 2 | 6 |
| Qwen3.5-9B · 读取原生框字段 | 1 | 6 | 0 | 4 | 6 |
| MiMo-VL-7B-RL · 读取最终答案，上限 1024 token | 0 | 2 | 0 | 2 | 4 |
| DeepSeek-VL2-Tiny · 直接 | 0 | 23 | 0 | 22 | 10 |
| DeepSeek-VL2-Tiny · 原生定位指令 | 0 | 0 | 0 | 0 | 2 |

- 同系列 8B 在留出组新增两处命中。验证组两者均只命中一处，而且命中的位置不同。在留出组，8B 直接推理只正确识别 10 张阳性片中的 2 张，另有一张阳性片未决。零误报伴随较多漏检。
- Qwen2.5‑VL‑7B 对全部 25 张片均判阴性，使用原生像素坐标或改用 BF16 的未微调基座也得到相同结果。微调后两组各识别出一张阳性片，两个框都未达到与参考框重叠的命中标准。Qwen3.5‑9B 有 11 份回答将坐标包在标准的 `bbox_2d` 字段里。表中只取出原有坐标，验证命中一处，留出未命中。严格按统一 JSON 格式解析的结果也保留在指标文件中。
- DeepSeek 的统一提示让它把 25 张都判成阳性。45 个预测框使用 0–1 小数。事后将坐标统一放大 1000 倍重算，验证和留出仍然都未命中，最大的框重叠度分别为 0.234 和 0.307。改用官方定位标记后返回了 18 个明确阴性、7 个无法结构化的回答，仍没有预测框。两份结果均保留原输出。
- MiMo 的回答带推理段落。首次 256 token 的统一协议无法直接解析整段文本。移除已闭合的推理标签后，17 张有可读的最终答案，其余 8 张触及输出上限。表内用预先定义的 1024 token 补测，仅读取最终状态与框。这样能读出全部 12 张留出片的最终答案，模型正确指出 2 张阳性片，但两处候选均未命中。验证集中尚有 1 张无法解析。完整回答与两种解析的结果均在实验包内。

现役 YOLO 与修正标签的 YOLO640 是两份不同权重。一个后续实验是把 8B 作为第二读片器，单独展示它额外指出的区域。8B 和修正 YOLO 在留出组分别命中 2 处与 3 处，只有 1 处重合。这是从已看过的留出片得到的线索。在数据固定的前提下，可按马分组为每匹马生成折外预测，并在各训练折内部选阈值，再衡量增加了几处真病灶、每片多报几个框。

## 影像、提示与评价方式

沿用项目对 84 张片各来自一匹马的假设。59 张用于训练，13 张用于验证，12 张为历史留出片。验证集和留出集各有 10 张阳性片、12 个参考病灶。直接推理只输入完整影像像素和同一项球节骨折任务的提示。文件名、该片的参考框及阴阳性标签均留在评分端。YOLO 候选框复读实验单独使用候选区域。原片、CVAT 源文件和既有划分均沿用同一版本。[原数据和尺寸线索](../qwen3-vl-2b-2026-11-18/README.md)

同一张片上，预测框与尚未匹配的人工参考框交并比至少为 0.5 计一个命中，多余预测框计误报。整片阴阳性单独评分，无法判断与格式错误都保留在分母中。历史留出组只有两张阴性片，全判阳性即可取得 10/12 的整片准确率。另一个只读文件高度的对照可判对 12/12。该留出组已经用于多轮探索，得分只能描述这批片。

- **跨品牌直接读片**使用同一英文 JSON 提示。提示允许返回阳性、阴性和无法判断，定位框用整片的 0–1000 相对坐标。每张片独立推理，使用确定性解码，最多生成 256 个 token。各模型沿用各自的视觉编码器与图像处理方式。DeepSeek 使用原生切片，未提供参考框或病变提示位置。[通用提示与解析](../../scripts/evaluate_qwen_vl.py) · [DeepSeek 处理实现](../../scripts/evaluate_deepseek_vl.py)
- **原生定位复查**预先设置两种额外输入协议。Qwen2.5‑VL‑7B 可按其处理后图像的绝对像素输出坐标，评分前再映射回原片。DeepSeek‑VL2‑Tiny 单独使用官方的引用与定位标记，要求定位所有视图中的疑似骨折。缺失或自相矛盾的输出保留为未决。[Qwen2.5‑VL 坐标说明](https://qwenlm.github.io/blog/qwen2.5-vl/) · [DeepSeek‑VL2 原生定位示例](https://github.com/deepseek-ai/DeepSeek-VL2)
- **QLoRA**使用 59 张训练片，固定顺序跑一轮，以已有标签生成目标答案。基座为 4 位量化权重，更新最后四个语言层的查询和值投影适配器，秩为 8、学习率 0.00015，损失仅计算答案部分。每种设置的实际训练参数、耗时和产物可从实验包复算。[训练实现](../../scripts/train_qwen_lora.py)
- **检测器加第二读片模型**先用修正标签的 YOLO640 和验证集选定的 0.20 阈值生成候选，再将每个候选周围约两倍宽高的区域交给冻结的 Qwen3‑VL‑2B。所有 16 个候选都被判为阳性。留出集仍命中 3 处、误报 2 框，验证集仍命中 5 处、误报 6 框。[候选复读实现](../../scripts/review_detector_candidates.py)

## 本机运行与复算

实验机为 Apple M5 MacBook Air，系统报告 24 GiB 统一内存。模型下载后，本地 MLX 完成视觉推理和微调，未向模型托管服务发送马片。Qwen3‑VL‑8B 的 59 步微调耗时 242 秒，MLX 峰值 8.12 GB；Qwen2.5‑VL‑7B 的相同训练耗时 232 秒，峰值 8.24 GB，均按十进制 GB 记录。7B 首次以 FP16 运行时第 2 步损失出现非有限值，改用 BF16 浮点计算并保留 4 位量化权重后完成训练。新增的 BF16 基座对照仍将所有片判为阴性。复现 7B 微调及其配套推理时均加入 `--bf16-compute`。

工作台使用 FastAPI 和现役 YOLO11s ONNX。实测同一 CPU 执行器，预热后对 12 张历史片各重复 3 次，单张中位耗时 106 毫秒，包括缩放、模型和框解码，排除磁盘读图、上传和网络。对 RTX 4070 Laptop，可在装有 CUDA 版 ONNX Runtime 的环境运行 `python training/scripts/benchmark_detector_device.py --provider CUDAExecutionProvider` 复测。此脚本会验证 CUDA 执行器是否实际初始化。已有后端固定选择 CPU，在 ROG 上启动相同后端仍会走 CPU。[基准脚本](../../scripts/benchmark_detector_device.py) · [现役模型加载](../../../backend/detector.py)

华硕 RTX 4070 Laptop 机型配 8 GB GDDR6。CUDA 与主动散热适合 YOLO 多轮训练。8B 的 4 位模型权重约 5.76 GB，推理还需中间张量、注意力缓存等内存。MLX 与 CUDA 的分配方式不同，本次 Mac 的峰值只用来估算量级，ROG 的实际速度和可用批量需要实机测量。[华硕机型规格](https://rog.asus.com/np/laptops/rog-strix/rog-strix-g18-2023-series/spec/)

Qwen2.5‑VL‑7B 用于回答精确 7B 规模的问题，Qwen3‑VL‑2B 与 8B 才能做同代容量比较。截至 2026 年 11 月 18 日，Qwen 已发布支持原生视觉的 Qwen4‑Max，主干有 2.4 万亿参数。小米 MiMo‑V2.5 有 3100 亿总参数、150 亿激活参数，DeepSeek V4.1 Flash 也支持原生视觉。这三款大模型未参加本次马片实验。厂商通用视觉评测与本项目的马球节骨折定位是不同任务。[Qwen4‑Max 公告](https://qwen.ai/blog/qwen4-max) · [MiMo‑V2.5 说明](https://mimo.xiaomi.com/mimo-v2-5) · [DeepSeek 官方变更记录](https://api-docs.deepseek.com/updates/)

在仓库根目录先运行 `python training/scripts/prepare_comparison.py`，再按上一轮说明配置 Apple Silicon 的 MLX 环境。模型转换版本和权重 SHA‑256 见登记表，DeepSeek 的版本及权重哈希在单独的推理脚本中。同一张马片不需要上传给任何外部服务。[量化模型登记表](../../scripts/qwen_checkpoints.py) · [MLX 依赖版本](../qwen3-vl-2b-2026-11-18/README.md)

```sh
python training/scripts/evaluate_qwen_vl.py --checkpoint 8b --model work/model-comparison/qwen3-vl-8b/model --output work/model-comparison/repro-8b/direct
python training/scripts/train_qwen_lora.py --checkpoint 8b --model work/model-comparison/qwen3-vl-8b/model --output work/model-comparison/repro-8b/fit
python training/scripts/evaluate_qwen_vl.py --checkpoint 8b --model work/model-comparison/qwen3-vl-8b/model --adapter work/model-comparison/repro-8b/fit/adapter --output work/model-comparison/repro-8b/adapted
python training/scripts/score_qwen_vl.py work/model-comparison/repro-8b/direct/predictions.jsonl
python training/scripts/compare_qwen_sizes.py training/results/qwen3-vl-2b-2026-11-18/direct.jsonl work/model-comparison/repro-8b/direct/predictions.jsonl
```

同一个直接推理入口还支持 `7b`、`mimo-7b`、`qwen35-9b`。下载登记版本的 MiMo 时，选择两块名为 `model-0000*-of-00002.safetensors` 的权重及相应预处理、词表文件。该固定版本的仓库另留有旧的单文件权重与指向四块旧权重的索引，本次只加载已核验的两块。DeepSeek 使用 `evaluate_deepseek_vl.py --model 模型目录 --protocol json --output 新结果目录`，原生定位将协议改为 `grounding`。已归档结果可直接运行评分脚本，省去重新下载模型的时间。[逐片模型输出、适配器和运行记录](runs.zip) · [两组完整指标和逐病灶配对](metrics.json)
