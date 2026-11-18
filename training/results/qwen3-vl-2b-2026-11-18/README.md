# Qwen3-VL 对运动马球节 X 线片的判读实验

**当前工作台继续使用 YOLO。** 在历史留出的 12 匹马上，直接生成坐标的 Qwen 命中 0/12 处标注；用 59 张现有训练片做一轮 QLoRA 后命中 1/12，现役 YOLO 在工作台阈值 0.15 下为 3/12。微调生成的唯一命中出现在第 04 张双视图影像，一侧的框与参考框的 IoU 为 0.705，另一侧病灶漏检。第 03 张则显示了常见的定位偏差：框住关节的大区域，只擦到很小的标注。[查看全部 12 张的框叠图](heldout-overlays.png)

| 方法 | 历史留出阳性片判对 / 10 | 阴性片判对 / 2 | 定位命中 / 12 | 误报框 | 验证集定位命中 / 12，误报框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 直接读片 | 9 | 0 | 0 | 12 | 1，15 |
| Qwen QLoRA 微调 | 7 | 1 | 1 | 7 | 0，13 |
| 冻结 Qwen 视觉特征，训练定位头 | 5 | 2 | 1 | 8 | 2，9 |
| 现役 YOLO，阈值 0.15 | 6 | 2 | 3 | 9 | 3，15 |
| 修正框训练的 YOLO 640，阈值 0.15 | 4 | 2 | 3 | 5 | 5，11 |
| 修正框训练的 YOLO 640，验证集选阈值 0.20 | 3 | 2 | 3 | 2 | 5，6 |
| 始终输出训练集框坐标的中位数 | 10 | 0 | 1 | 11 | 1，12 |

直接读片时有一张阳性片达到 256 个输出 token 后 JSON 仍未闭合，按未判读计入分母。QLoRA 的 25 张均可解析。另行训练的整片分类头不输出坐标，验证集判对 13/13，留出判对 11/12（阳性 9/10、阴性 2/2）。本次结果仅对应所测的 2B 量化版本、训练和推理配置。[完整验证与留出指标](metrics.json)

## 数据与判分

沿用 84 匹马的原始影像、既有框标注和固定马匹划分。训练集为 59 匹（48 阳性、11 阴性、57 个病灶），验证集为 13 匹（10 阳性、3 阴性、12 个病灶），历史留出集为 12 匹（10 阳性、2 阴性、12 个病灶）。沿用上一轮已应用 CVAT 旋转角度的框坐标。推理只接收原图像素和统一任务提示；文件名、病例编号与参考框不出现在判读输入中。历史留出集已用于前几次模型比较。

整片判读分别统计阳性片命中数和阴性片命中数。未能判断、格式错误和推理错误单独计数，并保留在总分母内。病灶定位沿用检测实验的判分规则：预测框和标注框的交并比至少为 0.5，每个病灶最多匹配一次；其余预测框计为误报，未匹配标注计为漏检。模型判断整张片可疑时，仍须用位置正确的框才能获得定位命中。

## 同一视觉模型的用法

- **直接读片。** Qwen3-VL-2B-Instruct 的 MLX 4 位量化版接收一张完整影像，可以回答阳性、阴性或不能判断。候选框以全图 0–1000 坐标给出。解码固定为贪心，每张最多生成 256 个 token。25 张逐片运行总计约 54 秒。官方基座与实际量化版本分别为 [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 和 [MLX 固定版本](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit/tree/9c4f5209e57b31f4b9dfba735de3fb983739c9cc)。[任务提示与实际环境](direct-runtime.json) · [逐片原始回答](direct.jsonl)
- **QLoRA 微调。** 59 张训练片各用一次，将已有标注转换为阳性或阴性及病灶框坐标的 JSON，只对助手回答和结束标记计算损失。冻结视觉编码器与量化基座，训练最后四个语言层的查询和值投影适配器，rank 8、共 229,376 个可训练参数。单轮耗时 84 秒，峰值内存 3.75 GB。推理时使用与直接读片相同的提示、解码方式和图像面积上限。留出片单张中位推理时间约 1.09 秒。[适配器](adapter/) · [训练样本哈希与实参](qlora-training.json) · [逐步损失](qlora-loss.jsonl) · [原始回答](qlora.jsonl) · [推理配置](qlora-runtime.json)
- **冻结视觉编码器训练定位头。** 将每张片的视觉特征保留为二维网格，用固定随机投影降至 128 维。仅用训练片的框拟合网格点分类器和边框回归器。验证集按平均每片最多一个误报选择阈值，最终取 0.99。另以整片特征的均值和标准差训练一个只判断阳性或阴性的分类器，按验证集平衡准确率选出阈值 0.50。视觉头输入的面积上限为 640² 像素，直接生成的上限为 1024² 像素。[定位逐片结果](probe.jsonl) · [整片分类结果](screen.jsonl) · [冻结特征的小型模型参数及原始候选框](probe-weights.zip)

## 复算

在仓库根目录运行 `python3 training/scripts/prepare_comparison.py` 可在 `work/model-comparison/` 准备实验副本。只复算指标时使用 Python 和 Pillow 即可，例如：

```sh
python3 training/scripts/score_qwen_vl.py training/results/qwen3-vl-2b-2026-11-18/direct.jsonl
```

重做本机训练和推理时，使用 Apple Silicon、Python 3.12、`mlx-vlm==0.8.0`、`mlx==0.32.4`、`transformers==5.23.0`、`Pillow==13.0.0`、`numpy==2.5.5`、`scipy==1.18.1`。把固定版本的量化模型下载到 `work/model-comparison/qwen-vl/model`。权重 SHA-256 为 `4750d95a2162829e127a94e83ac350d498d02070aab216c4687da48804a06ffb`。以下命令写入新的实验目录。

```sh
qwen_model=work/model-comparison/qwen-vl/model
trial=work/model-comparison/qwen-reproduction
python training/scripts/evaluate_qwen_vl.py --model "$qwen_model" --output "$trial/direct"
python training/scripts/train_qwen_probe.py extract --model "$qwen_model" --output "$trial/probe"
python training/scripts/train_qwen_probe.py fit --output "$trial/probe"
python training/scripts/train_qwen_lora.py --model "$qwen_model" --output "$trial/qlora"
python training/scripts/evaluate_qwen_vl.py --model "$qwen_model" --adapter "$trial/qlora/adapter" --output "$trial/adapted"
```

## 数据中与阳性标签相关的线索

一个只检查文件尺寸的探索性对照说明了整片准确率的局限。在 11 张标为阴性的训练片中，10 张高度恰好为 768 像素。把“高 768 像素”判为阴性，其余判为阳性，验证集可判对 12/13 张，历史留出集可判对 12/12 张。这条规则只读取尺寸。它提示这批片的采集或导出版式与标签相关；即便某个模型在整片判别上得分很高，仍要检查它能否把病灶圈在正确位置。这一对照不能证明某个具体模型用了尺寸作判断。

原始 CVAT 文件 SHA-256 为 `868262739dfd91bcc0c67c05d5522d12e8f39012e9b206f6d0fcac6ccf37f382`；划分清单为 `f8637308331fc3123943407d9c2609b48f74eeecfed980b92f3c12051a05f47d`。
