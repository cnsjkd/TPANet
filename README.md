# SEED-IV raw EEG 端到端（模拟 de_LDS + Conformer）

本目录实现的是不依赖官方 `de_LDS` 特征的端到端训练：

`raw EEG -> 可学习滤波器组(Sinc) -> DE-like(log-var) -> TCN平滑 -> Conformer -> 4分类`

## 1. 数据要求
`--root` 默认值是 `/home/aispeech/codes/zxy/SEED-IV`，应指向 `eeg_raw_data` 根目录，结构如下：

```text
eeg_raw_data/
  1/
  2/
  3/
```

每个 session 目录下应有 15 个 `.mat`；每个 `.mat` 里应有 24 个 trial 键（`*_eeg1 ... *_eeg24`）。
当前 `/home/aispeech/codes/zxy/SEED-IV` 已核对为：
- 每个文件 24 个 trial（`*_eeg1..24`）
- trial 数据维度第一维固定为 62（即 62 通道 EEG）

数据检查脚本：

```bash
python seed_iv_2026_like_de_LDS/check_eeg_raw_data.py --root /path/to/eeg_raw_data
```

如果输出类似下面内容，则可直接训练：

```text
Expected layout counts (1/2/3): {1: 15, 2: 15, 3: 15}
Flat mat files in root: 0
Result: YES, this already looks like eeg_raw_data.
Sample key check (...): scipy: keys=24, eeg_keys=24
```

## 2. 安装依赖
已提供版本约束文件：`seed_iv_2026_like_de_LDS/requirements.txt`

```bash
cd /home/aispeech/codes/zxy/TPANet-main
pip install -r seed_iv_2026_like_de_LDS/requirements.txt
```

## 3. 运行命令
单个 fold（默认测试被试是 1）：

```bash
python seed_iv_2026_like_de_LDS/train.py
```

单个 fold（显式指定常用参数）：

```bash
python seed_iv_2026_like_de_LDS/train.py \
  --root /path/to/eeg_raw_data \
  --test_subject 1 \
  --epochs 80 \
  --batch_size 8 \
  --cache_dir ./cache \
  --save_dir ./ckpt
```

完整 LOSO：

```bash
python seed_iv_2026_like_de_LDS/train.py \
  --root /path/to/eeg_raw_data \
  --loso \
  --epochs 80 \
  --batch_size 8 \
  --cache_dir ./cache \
  --save_dir ./ckpt
```

严格版归一化（仅用训练集统计量）：

```bash
python seed_iv_2026_like_de_LDS/train.py \
  --root /path/to/eeg_raw_data \
  --loso \
  --norm_mode train_set_zscore
```

## 4. 参数说明（哪些必须）
- `--root`：可选；默认 `/home/aispeech/codes/zxy/SEED-IV`
- `--loso`：可选；加上后跑全部被试 LOSO
- `--test_subject`：可选；不加 `--loso` 时生效，默认 `1`
- `--val_split`：可选；从训练被试中按被试比例划分验证集（默认 `0.2`）
- `--early_stop_patience` / `--early_stop_min_delta`：可选；早停参数
- `--start_fold`：可选；从指定 fold 开始跑（用于断点续跑）
- `--norm_mode`：可选；`trial_zscore`（默认）/ `train_set_zscore`（严格版）/ `none`
- `--strict_norm`：可选；等价于 `--norm_mode train_set_zscore`
- `--zscore` / `--no_zscore`：兼容旧参数，分别等价于 `trial_zscore` / `none`
- 其余参数（`--epochs --batch_size --cache_dir --save_dir`）均为可选，代码有默认值或允许为空。

## 5. 评估协议（已修复数据泄露）
- 采用 subject-wise LOSO：测试被试全程不参与训练。
- 每个 fold 内仅用训练被试数据再划分 `train/val`（按 subject）。
- 模型选择和早停只看 `val_acc`。
- `test` 只在训练结束后评估一次，不参与选模。

## 6. 关键默认参数
- `--sessions 1 2 3`
- `--chunk_size 800`（4s @ 200Hz）
- `--num_channel 62`
- `--epochs 80`
- `--batch_size 8`
- `--lr 2e-4`
- `--weight_decay 1e-2`
- `--d_model 256`
- `--num_heads 4`
- `--ffn_dim 1024`
- `--num_layers 6`
- `--conformer_conv_kernel 15`
- `--smoother_layers 2`
- `--val_split 0.2`
- `--early_stop_patience 10`
- `--early_stop_min_delta 0.001`
- `--norm_mode trial_zscore`（默认）

## 7. 输出
- checkpoint（若设置 `--save_dir`）：`seediv_e2e_conformer_testsubXX_epochEEE_valVVVV.pt`
- 结果 CSV（默认）：`TPANet-main/results_seed_iv_2026_like_de_lds.csv`
- CSV 关键字段：`best_val_acc`, `test_acc`, `best_epoch`

## 8. 代码结构
- `dataset.py`：读取 raw `.mat`、trial 切窗、变长 batch 对齐
- `model.py`：可学习 DE-like 前端 + Conformer 分类器
- `train.py`：单折/LOSO 训练与记录
- `check_eeg_raw_data.py`：数据结构与键名检查
