# SEED-IV raw EEG 端到端（模拟 de_LDS + Conformer）

该目录已改为不依赖官方 `de_LDS` 特征文件，直接从 `eeg_raw_data` 训练：

`raw EEG -> 可学习滤波器组 -> DE-like(log-var) -> 可学习平滑(TCN) -> Conformer -> 4分类`

## 1. 数据格式
`--root` 需要指向 `eeg_raw_data`，目录结构如下：

```text
eeg_raw_data/
  1/
  2/
  3/
```

每个 session 目录包含 `*.mat`（如 `10_20131130.mat`）。
脚本从 `*_eeg1 ... *_eeg24` 字段读取 24 个 trial，每个 trial 按 `4s@200Hz=800` 点做不重叠切窗。

## 2. 依赖

```bash
pip install numpy scipy torch torchaudio
```

如果 `.mat` 是 MATLAB v7.3，额外安装：

```bash
pip install h5py
```

## 3. 训练命令

单个 LOSO fold（例如留 1 号被试测试）：

```bash
python seed_iv_2026_like_de_LDS/train.py \
  --root /path/to/eeg_raw_data \
  --test_subject 1 \
  --epochs 80 \
  --batch_size 8 \
  --cache_dir ./cache \
  --save_dir ./ckpt
```

完整 LOSO（自动遍历所有共同被试）：

```bash
python seed_iv_2026_like_de_LDS/train.py \
  --root /path/to/eeg_raw_data \
  --loso \
  --epochs 80 \
  --batch_size 8 \
  --cache_dir ./cache \
  --save_dir ./ckpt
```

## 4. 关键参数
- `--sessions 1 2 3`：使用哪些 session（默认 `1 2 3`）
- `--chunk_size 800`：窗口长度（默认 800）
- `--num_channel 62`：EEG 通道数（默认 62）
- `--no_zscore`：关闭每个 trial 的按通道 z-score
- `--d_model --num_heads --ffn_dim --num_layers`：Conformer 主干规模
- `--smoother_layers`：TCN 平滑层数
- `--cosine`：启用余弦学习率调度
- `--amp`：启用混合精度训练

## 5. 代码结构
- `dataset.py`
  - 读取 `eeg_raw_data`
  - trial 切成 `(W, C, T)`，`W` 为窗口数
  - `collate_trials` 做变长 padding + `lengths`
- `model.py`
  - `SincFilterbank`：可学习带通滤波器组（初始化为 5 个经典频段）
  - `DELike`：按 band/channel 计算 `log(var)`
  - `TCNSmoother`：沿窗口轴平滑
  - `EEGConformerClassifier`：Conformer + masked mean pooling + 4 分类头
- `train.py`
  - 支持单折与 LOSO
  - 保存最佳 checkpoint
  - 结果写入 CSV

## 6. 输出
- checkpoint：`--save_dir/seediv_e2e_conformer_testsubXX.pt`
- 结果 CSV（默认）：`TPANet-main/results_seed_iv_2026_like_de_lds.csv`
