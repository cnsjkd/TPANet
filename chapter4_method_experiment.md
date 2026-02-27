# 4.2 模型框架与结构

本章提出了一种面向 SEED 原始 EEG 的端到端三分类情感识别框架（可学习 DE/LDS-like 前端 + Conformer 主干）。该方法以单个 trial 的原始脑电信号为输入，依次完成非重叠切窗、可学习频带滤波、DE-like 特征映射、TCN 时序平滑、Conformer 编码与分类输出，在保留传统 DE/LDS 思想可解释性的同时，将频带划分与时序平滑参数化、可学习化。图 4-1 给出了本章方法的总体框架。

## Figure 4-1 总框架图（论文排版时可重绘）

```mermaid
flowchart LR
    A[输入: 单个 trial 原始 EEG<br/>形状 C×T, C=62] --> B[非重叠切窗<br/>W×C×800, 4s@200Hz]
    B --> C[可学习 Sinc 滤波器组<br/>5 频带]
    C --> D[DE-like 特征<br/>按通道-频带取 log-variance]
    D --> E[特征序列<br/>W×310]
    E --> F[TCN 平滑模块<br/>沿窗口轴建模]
    F --> G[线性投影<br/>310→d_model]
    G --> H[Conformer 编码器×L]
    H --> I[Mask Mean Pooling]
    I --> J[分类头<br/>3 类情感]
```

## 4.2.1 方法总体框架概览

从代码实现角度看，整体框架可按 `dataset.py -> model.py -> train.py` 三个模块理解：

1. `dataset.py`：负责 SEED 原始 `.mat` 文件扫描、trial 索引构建、标签映射、trial 级切窗、可选标准化、batch padding 与长度掩码生成。
2. `model.py`：负责模型主体实现，包括 `SincFilterbank`、`DELike`、`TCNSmoother`、`LearnableDELDSLikeFrontend` 与 `EEGConformerClassifier`。
3. `train.py`：负责 LOSO 划分、训练/验证/测试循环、损失函数、优化器、早停策略、指标统计与 CSV 日志输出。

相较“离线 DE/LDS 特征 + 分类器”的传统流程，本章方法的核心变化是：将频域特征构造与时序平滑并入模型前端，并与分类目标联合优化，从而减少手工特征与任务目标之间的优化割裂。

## 4.2.2 数据输入与可学习特征前端（`dataset.py` + `model.py`）

### (1) 数据组织与 trial 级索引构建

`SEEDIVRawTrialDataset` 以 SEED 原始数据目录为输入（根目录下平铺 45 个 `.mat` 文件与 `label.mat`），并完成以下处理：

1. 扫描所有被试文件，按文件名中的日期顺序映射到 3 个 session；
2. 读取 `label.mat`，将标签从 `{-1,0,1}` 映射为 `{0,1,2}`；
3. 从每个被试 `.mat` 文件中提取 `*_eeg1 ~ *_eeg15` trial 键；
4. 校验 trial 编号完整性（必须覆盖 `1..15`）后建立 trial 索引。

该设计保证了训练阶段按 `(session_id, subject_id, trial_id)` 精确抽取样本，并为 LOSO 划分提供稳定的数据访问接口。

### (2) 非重叠切窗与变长序列表示

每个样本单位为一个 trial。原始 trial 被对齐为 `(C, T)`（`C=62`），然后按 `chunk_size=800`（4 秒 @ 200Hz）进行非重叠切窗，得到 `(W, C, 800)`：

1. `W = floor(T / 800)`；
2. 末尾不足 800 采样点的数据直接截断；
3. 不同 trial 的窗口数 `W` 可变。

在 batch 组装阶段，`collate_trials` 将不同长度 trial 补零到同一 `max_windows`，并返回 `lengths` 掩码，供后续 TCN 与 Conformer 进行 mask-aware 计算。

### (3) 可学习 Sinc 滤波器组（`SincFilterbank`）

为避免固定频带边界对个体差异适配不足，模型前端引入 SincNet 风格带通滤波器组。代码中默认初始化 5 个频带：

- `[(1,4), (4,8), (8,14), (14,31), (31,50)] Hz`

每个频带的低截止频率与带宽由可学习参数表示，并通过 `softplus` 保证频率合法，再组合为带通卷积核。第 `b` 个滤波器的形式可写为：

\[
h_b[n] = \left(2f_{2,b}\mathrm{sinc}(2f_{2,b}n) - 2f_{1,b}\mathrm{sinc}(2f_{1,b}n)\right)w[n]
\]

其中 \(w[n]\) 为汉明窗。实现中还对滤波器核做幅值归一化，以提升训练稳定性。

### (4) DE-like 特征映射（`DELike`）

对滤波输出按“通道-频带”维度计算对数方差，以近似传统 DE 特征的能量表达：

\[
\mathrm{DE\_like} = \log(\mathrm{Var}(x)+\epsilon)
\]

具体实现先去均值，再计算时间轴均方，最后取对数。对每个窗口输出 `62 × 5 = 310` 维特征，即将滤波结果映射为窗口级低维表示 `W × 310`。

### (5) 标准化策略（代码默认关闭）

数据集类支持两类标准化：

1. trial 内按通道 z-score（`per_channel_zscore=True`）；
2. 外部给定训练集统计量的按通道标准化（`channel_mean/channel_std`）。

当前 `train.py` 中默认不启用 z-score；只有显式传入 `--zscore` 时才会对每个 trial 执行按通道 z-score（并通过 `--no_zscore` 保留兼容开关）。

## 4.2.3 主体网络结构与时序增强（`model.py`）

### (1) TCN 平滑模块（`TCNSmoother`）

在 DE-like 特征序列进入主干编码器前，模型先使用 `TCNSmoother` 沿窗口轴 `W` 进行局部时序平滑。每个 TCN block 包含：

1. 深度卷积 `Conv1d(groups=dim)`（沿窗口轴建模）；
2. `GELU + Dropout`；
3. `1×1` 卷积重组通道；
4. Dropout 与残差连接。

模块末端使用 `LayerNorm`。同时，若提供 `lengths`，TCN 前后都会对 padding 区域进行显式掩码，避免补零窗口干扰特征统计。

### (2) 可学习前端封装（`LearnableDELDSLikeFrontend`）

`LearnableDELDSLikeFrontend` 将三个步骤串联为统一前端：

1. `SincFilterbank`：原始窗口 EEG 的频带滤波；
2. `DELike`：滤波输出映射为通道-频带能量特征；
3. `TCNSmoother`：窗口级特征时序平滑。

该模块输入为 `x: (B, W, C, T)`，输出为 `feat: (B, W, 310)`，是连接原始 EEG 与高层序列编码器的关键桥梁。

### (3) Conformer 主干编码器（`EEGConformerClassifier`）

分类器主干结构如下：

1. 线性投影：`310 -> d_model`；
2. `torchaudio.models.Conformer` 编码器：同时建模长程依赖（自注意力）与局部模式（卷积分支）；
3. Masked mean pooling：基于 `lengths` 对有效窗口做平均池化；
4. 分类头：`LayerNorm + Linear` 输出 3 类 logits。

代码中将 `Conformer` 的长度掩码一路传递至池化阶段，使变长 trial 的 padding 不参与最终统计。

### (4) 结构设计动机总结（论文行文可直接使用）

1. 通过可学习 Sinc 滤波器替代固定频带边界，提升频域建模的自适应性。
2. 通过 TCN 平滑模块模拟/替代传统 LDS 的时序平滑作用，增强窗口级特征鲁棒性。
3. 通过 Conformer 联合建模全局依赖与局部动态模式，适配 EEG 的多尺度时序特征。

## 4.2.4 训练目标、优化策略与评价指标（`train.py`）

### (1) 训练目标

当前实现采用三分类交叉熵损失：

\[
\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{3}y_{i,c}\log p_{i,c}
\]

其中 \(p_{i,c}\) 为样本 \(i\) 属于第 \(c\) 类的预测概率。

### (2) 数据划分与训练流程（LOSO）

`train.py` 采用严格被试独立 LOSO（Leave-One-Subject-Out）流程：

1. `list_common_subject_ids()` 获取指定 session 集合上的公共被试集合；
2. 每次选择 1 名被试作为测试被试；
3. 其余被试内部按 trial 随机划分训练/验证（`split_train_val_trial_keys()`）；
4. 训练时以验证集 `val_acc` 进行早停和选模；
5. 训练结束后仅对最佳模型进行一次测试集评估。

其中验证划分基于 trial 随机打乱，随机种子使用 `args.seed + test_subject`，可降低不同 fold 验证集划分完全一致带来的偶然性。

### (3) 优化与训练细节（代码默认值）

- 优化器：AdamW
- 学习率调度：`CosineAnnealingLR`（默认关闭，需显式传 `--cosine`）
- 混合精度：AMP（默认关闭，需显式传 `--amp`）
- 梯度裁剪：`max_norm = 5.0`
- 早停指标：`val_acc`
- 早停条件：当 `val_acc` 提升不超过 `early_stop_min_delta` 且连续 `early_stop_patience` 个 epoch 未提升时停止

### (4) 评价指标与日志输出

代码默认评价指标为 Accuracy：

\[
\mathrm{Acc} = \frac{\#\text{correct}}{\#\text{all}}
\]

单个 fold 记录 `best_val_acc`、`test_acc`、`best_epoch`、运行时长等信息；完整 LOSO 还会输出 `mean_test_acc` 与 `std`。此外，代码支持可选的前向 FLOPs 估计（`--report_flops`）以及 CSV 结果记录。

# 4.3 实验设计与结果分析（论文撰写版）

说明：本节中“数据集介绍”和“实验设置”可直接依据当前代码实现撰写；“结果分析/消融实验”的具体数值、表格与图示不在代码中，需要在实验运行后补充，因此使用注释占位。

## 4.3.1 数据集介绍

实验数据为 SEED 原始 EEG 数据（根目录为平铺 `.mat` 文件 + `label.mat` 的组织形式）。结合 `dataset.py` 与 `check_eeg_raw_data.py` 的检查逻辑，可得到以下关键信息：

1. 每个被试包含 3 个 session 文件；
2. 每个 session 文件包含 15 个 trial（键名形如 `*_eeg1 ~ *_eeg15`）；
3. 每个 trial 使用 62 通道 EEG；
4. 标签来自 `label.mat`，原始标签 `{-1,0,1}` 在代码中映射为 `{0,1,2}`。

因此，若数据完整，则总 trial 数为 `15（被试） × 3（session） × 15（trial） = 675`。

### 划分协议（代码实现对应）

采用严格 LOSO 划分：

1. 每个 fold 留 1 名被试作为测试集；
2. 剩余被试用于训练/验证；
3. 验证集由训练被试内部 trial 随机划分（默认 `val_split=0.2`）。

在默认 `val_split=0.2` 下，每个“被试-会话”约 15 个 trial 中 3 个进入验证、12 个进入训练。则每个 fold 近似为：

1. 训练集：`14 × 3 × 12 = 504` 个 trial
2. 验证集：`14 × 3 × 3 = 126` 个 trial
3. 测试集：`1 × 3 × 15 = 45` 个 trial

## 4.3.2 实验设置

### (1) 训练与运行环境（代码可见部分）

1. 深度学习框架：PyTorch
2. 设备选择：优先 `cuda`（可用时），否则 `cpu`
3. 随机种子：默认 `42`
4. 数据加载：`DataLoader + collate_trials`（支持变长窗口 padding）

<!-- TODO（论文补充）: 在正文中补充硬件环境（GPU/CPU/RAM）、操作系统、CUDA/cuDNN、PyTorch/torchaudio 版本。 -->

### (2) 预处理与输入设置（与代码一致）

1. 非重叠切窗：`chunk_size=800`（4 秒 @ 200Hz）
2. 通道数：`62`
3. trial 内按通道 z-score：默认关闭（需显式传 `--zscore` 启用）
4. batch 组装：按最大窗口数补零，并返回 `lengths` 掩码
5. 可选缓存：支持 trial 级 `.npz` 缓存（`--cache_dir`）

### (3) 主要训练超参数（`train.py` 默认值）

| 参数 | 默认值 |
|---|---|
| sessions | `[1,2,3]` |
| chunk_size | `800` |
| num_channel | `62` |
| batch_size | `8` |
| epochs | `80` |
| lr | `2e-4` |
| weight_decay | `1e-2` |
| dropout | `0.1` |
| grad_clip | `5.0` |
| d_model | `256` |
| num_heads | `4` |
| ffn_dim | `1024` |
| num_layers | `6` |
| conformer_conv_kernel | `15` |
| smoother_layers | `2` |
| num_workers | `4` |
| val_split | `0.2` |
| early_stop_patience | `10` |
| early_stop_min_delta | `0.001` |

### (4) 评价指标与结果记录方式

1. 主指标：Accuracy
2. 选模依据：验证集 `val_acc`
3. 测试策略：最佳模型在测试集评估一次，避免数据泄露
4. 结果记录：控制台日志 + CSV（包含 `best_val_acc`、`test_acc`、`best_epoch` 等）
5. 可选统计：前向 FLOPs（`--report_flops`）

<!-- TODO（论文补充）: 若论文需要类别平衡分析，可在本节补充 Macro-F1、混淆矩阵等指标定义。 -->

## 4.3.3 实验结果与分析（占位）

<!-- TODO（论文补充）: 插入主结果表（LOSO 各 fold test_acc、mean±std）。 -->
<!-- TODO（论文补充）: 插入与基线方法的对比结果表（如 DE+LDS+SVM/传统深度模型/Conformer变体等）。 -->
<!-- TODO（论文补充）: 插入主要结果图（柱状图/折线图/箱线图），并给出文字分析。 -->
<!-- TODO（论文补充）: 分析模型在不同被试上的波动原因（个体差异、噪声、session差异等）。 -->

## 4.3.4 消融实验（占位）

<!-- TODO（论文补充）: 设计并填写消融实验表。建议至少包含以下配置： -->
<!-- 1) 去掉 SincFilterbank（固定频带或直接时域投影） -->
<!-- 2) 去掉 DELike（仅卷积特征） -->
<!-- 3) 去掉 TCNSmoother -->
<!-- 4) Conformer 替换为纯 Transformer / 仅TCN -->
<!-- 5) 关闭 z-score 或调整 smoother_layers、num_layers 等关键超参数 -->
<!-- TODO（论文补充）: 对消融结果进行逐项解释，说明各模块贡献与协同作用。 -->
