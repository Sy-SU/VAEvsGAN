# MNIST 条件生成模型：VAE 与 DCGAN

## 可复现对比实验

`configs/comparison.toml` 固定三种子、50k/10k 训练验证划分和共同 MNIST 评估器。
同为 20 epoch 的 VAE/GAN 是主对照，GAN 50 epoch 是默认预算补充。

```bash
python tools/compare.py --stage all
```

可分别使用 `--stage train`、`--stage evaluate`、`--stage report`；报告阶段仅需 CPU。
`--config` 接受独立 TOML 配置，路径相对仓库根目录解析。实验中断后以同一配置重新
执行训练会恢复模型、优化器和随机数状态；已完成的训练会复用。新协议应使用新的
`study.run_dir`，避免混用结果。现有训练和生成命令仍可独立使用。

正式结果保存到 `outputs/comparison-study/`，中文报告与图表保存到
[完整实验报告](reports/comparison.md)。报告包含三种子结果、27 张对比图与原始指标表。
模型权重不放进报告目录。

```bash
# 隔离的小规模真实数据验证；输出到独立的 comparison-study-smoke 目录
python tools/compare.py --smoke --stage all
# 核查指标公式、随机数隔离、模型状态与旧权重兼容性
python tools/check_comparison.py --device cpu
python tools/check_comparison.py --device cuda
```

烟雾验证不代表正式实验结果。源 PNG 首次加载后会无损缓存到
`outputs/comparison-data/`；更换源数据时请使用新的缓存或删除该缓存后重建。

---

这是一个使用 PyTorch 实现的 MNIST 条件生成项目，目前包含条件卷积变分自编码器（Conditional VAE）和条件深度卷积生成对抗网络（Conditional DCGAN）。两种模型在训练时都使用 MNIST 标签作为条件，生成时可以输入 `0` 到 `9` 中的数字，引导模型生成对应类别的不同手写图片。

训练、测试和生成分别由 `tools/train.py`、`tools/test.py` 和 `tools/generate.py` 提供。第一个位置参数选择 `vae` 或 `gan`，三个工具通过公共模型工作流接口分发任务。

## Conda 环境

```bash
conda env create -f environment.yml
conda activate vae
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## 实验环境

### 硬件与系统

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows WSL2，Ubuntu 24.04.5 LTS |
| GPU | NVIDIA GeForce RTX 5080 |
| NVIDIA 驱动 | 581.29 |
| CUDA 版本 | 13.0 |

### 软件环境

| 软件 | 版本 |
| --- | --- |
| Python | 3.13.15 |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |

### 默认实验参数

| 参数 | 条件 VAE | 条件 DCGAN |
| --- | ---: | ---: |
| 数据集 | MNIST，训练集 60,000，测试集 10,000 | MNIST，训练集 60,000，测试集 10,000 |
| Epoch | 20 | 50 |
| Batch size | 128 | 128 |
| 潜变量/噪声维度 | 20 | 20 |
| 优化器 | Adam | Adam |
| 学习率 | `1e-3` | `2e-4` |
| Adam betas | `(0.9, 0.999)` | `(0.5, 0.999)` |
| 标签条件表示 | 10 维 one-hot | 50 维可学习 embedding |
| 条件错误标签负样本权重 | 不适用 | 0.5 |
| 随机种子 | 42 | 42 |
| DataLoader workers | 0 | 0 |

表格对应 [configs/vae.toml](configs/vae.toml) 和 [configs/gan.toml](configs/gan.toml) 的默认值。进行 VAE/GAN 视觉对比时，应保持数据划分、随机种子、batch size 和潜空间维度一致；两种模型的 epoch 和优化器参数按各自常用训练设置保留差异。

## 配置文件

训练、测试、生成和对比可视化的默认参数集中在：

```text
configs/
├── vae.toml
├── gan.toml
└── visualization.toml
```

直接修改相应 TOML 文件即可保存一组实验配置。命令行参数优先级更高，适合临时覆盖单个值；也可以用 `--config` 选择另一份配置：

```bash
python tools/train.py vae --config configs/vae.toml --epochs 30
python tools/visualize.py samples --config configs/visualization.toml --digit 7
```

模型配置按 `[train]`、`[test]` 和 `[generate]` 分组；可视化配置按 `[samples]` 和 `[latent]` 分组。`--digit` 仍由生成命令显式指定，避免不小心生成错误类别。

## 准备数据

```bash
python tools/dataset.py
```

数据脚本会生成以下目录：

```text
data/MNIST/
├── train/
│   ├── images/
│   └── labels.csv
└── test/
    ├── images/
    └── labels.csv
```

## 训练

### 训练 VAE

```bash
python tools/train.py vae \
  --data-dir data/MNIST \
  --checkpoint models/vae.pt \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --latent-dim 20 \
  --seed 42
```

默认训练 20 轮，批量大小 128，学习率 `1e-3`，潜空间维度 20。

### 训练 GAN

```bash
python tools/train.py gan \
  --data-dir data/MNIST \
  --checkpoint models/gan.pt \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 0.0002 \
  --latent-dim 20 \
  --seed 42
```

GAN 默认训练 50 轮，批量大小 128，学习率 `2e-4`，噪声维度 20。生成器和判别器使用 Adam，`betas=(0.5, 0.999)`。训练图片会从 `[0,1]` 转换到 `[-1,1]`；判别器还会把“真实图片配错误数字标签”作为负样本，避免模型忽略生成条件。每轮输出生成器/判别器损失、真假样本分数和错误标签分数，并覆盖最新检查点。

## 测试与重建

VAE 测试会输出平均总损失、BCE 重建损失和 KL 散度，并保存原图与条件重建结果：

```bash
python tools/test.py vae \
  --data-dir data/MNIST \
  --checkpoint models/vae.pt \
  --output outputs/vae/reconstructions.png \
  --device cuda
```

GAN 测试会在完整测试集上输出判别器 BCE、生成器 BCE、真实样本分数和生成样本分数，并保存每个数字一行的平衡生成网格：

```bash
python tools/test.py gan \
  --data-dir data/MNIST \
  --checkpoint models/gan.pt \
  --samples-per-class 8 \
  --output outputs/gan/test-grid.png \
  --device cuda
```


## 按数字生成

使用必填的 `--digit` 指定希望生成的数字。VAE 示例：

```bash
python tools/generate.py vae \
  --checkpoint models/vae.pt \
  --digit 7 \
  --num-samples 64 \
  --seed 42 \
  --output outputs/vae/generated-7.png \
  --device cuda
```

`--digit` 只接受 `0` 到 `9`。同一个数字会结合不同的随机潜变量生成不同写法；相同检查点、数字和 `--seed` 会得到相同结果。

GAN 使用相同的生成接口：

```bash
python tools/generate.py gan \
  --checkpoint models/gan.pt \
  --digit 7 \
  --num-samples 64 \
  --seed 42 \
  --output outputs/gan/generated-7.png \
  --device cuda
```

使用 `python tools/<任务>.py vae --help` 查看某个模型工作流的完整参数；使用 `python tools/<任务>.py --help` 查看已注册的模型。

## VAE 与 GAN 可视化对比

### 同条件随机生成

下面的命令使用相同数字、样本数和随机种子对比 VAE 与 GAN。两个检查点的潜空间维度相同时，两种模型还会复用完全相同的潜变量。

```bash
python tools/visualize.py samples \
  --vae-checkpoint models/vae.pt \
  --gan-checkpoint models/gan.pt \
  --digit 7 \
  --samples-per-class 8 \
  --seed 42 \
  --output outputs/comparison/conditional-samples-7.png \
  --device cuda
```

省略 `--digit` 会生成包含 `0–9` 的完整对比图，每个数字依次显示一行 VAE 和一行 GAN 样本。

### 静态训练过程

训练过程图由训练命令自动生成，不使用 GIF。默认配置下：

- VAE 展示 epoch `1, 3, 5, 7, 9, 12, 14, 16, 18, 20`。
- GAN 展示 epoch `1, 6, 12, 17, 23, 28, 34, 39, 45, 50`。

每张子图使用相同的潜变量和数字标签，因此变化只来自模型在不同 epoch 的参数。默认输出为：

```text
outputs/vae/train-progress.png
outputs/gan/train-progress.png
```

### 不同数字的潜空间分布

```bash
python tools/visualize.py latent \
  --data-dir data/MNIST \
  --vae-checkpoint models/vae.pt \
  --gan-checkpoint models/gan.pt \
  --samples-per-class 200 \
  --seed 42 \
  --output outputs/comparison/latent-distribution.png \
  --device cuda
```
