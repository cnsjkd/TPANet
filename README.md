# SEED raw EEG 端到端（模拟 DE/LDS + Conformer）

本目录实现的是基于 **SEED 原始 EEG** 的端到端训练，不依赖离线 `de_LDS` 特征：

`raw EEG -> 可学习滤波器组(Sinc) -> DE-like(log-var) -> TCN平滑 -> Conformer -> 3分类`

## 1. 数据要求（SEED）

`--root` 默认值为 `/home/aispeech/codes/zxy/SEED`，目录应为平铺结构：

```text
SEED/
  label.mat
  1_20131027.mat
  1_20131030.mat
  ...
  15_20131105.mat
```

要求如下：

1. 共 45 个被试数据 `.mat`（15 名被试 × 3 次 session）。
2. 每个数据 `.mat` 含 15 个 trial 键：`*_eeg1 ... *_eeg15`。
3. `label.mat` 的 `label` 长度应为 15，取值为 `{-1, 0, 1}`。
4. 标签在代码中映射为 3 类：`-1->0, 0->1, 1->2`。

数据检查脚本：

```bash
python /home/aispeech/codes/zxy/TPANet-main/seed_2026_like_de_LDS/check_eeg_raw_data.py \
  --root /home/aispeech/codes/zxy/SEED
```

## 2. 运行命令

单个 fold（默认测试被试 1）：

```bash
python /home/aispeech/codes/zxy/TPANet-main/seed_2026_like_de_LDS/train.py \
  --root /home/aispeech/codes/zxy/SEED
```

完整 LOSO：

```bash
python /home/aispeech/codes/zxy/TPANet-main/seed_2026_like_de_LDS/train.py \
  --root /home/aispeech/codes/zxy/SEED \
  --loso
```

带保存路径、FLOPs统计与CSV输出：

```bash
python /home/aispeech/codes/zxy/TPANet-main/seed_2026_like_de_LDS/train.py \
  --root /home/aispeech/codes/zxy/SEED \
  --loso \
  --save_dir /home/aispeech/codes/zxy/TPANet-main/ckpt_seed_2026_like_de_LDS \
  --report_flops \
  --flops_windows 8 \
  --flops_batch_size 1 \
  --results_csv /home/aispeech/codes/zxy/TPANet-main/results_seed_2026_like_de_lds.csv
```

## 3. 关键默认设置（已按 SEED）

1. `--root=/home/aispeech/codes/zxy/SEED`
2. `--sessions 1 2 3`
3. `--chunk_size=800`
4. `--num_channel=62`
5. `--val_split=0.2`（训练被试内部按 trial 划分 train/val）
6. 分类类别数固定为 3（来自 `label.mat` 映射）

## 4. 输出

1. checkpoint（若设置 `--save_dir`）：
   `seed_e2e_conformer_testsubXX_epochEEE_valVVVV.pt`
2. 结果 CSV（默认）：
   `/home/aispeech/codes/zxy/TPANet-main/results_seed_2026_like_de_lds.csv`
3. CSV 额外记录：
   `total_params`, `trainable_params`, `forward_gflops`, `flops_windows`, `flops_batch_size`

## 5. 轻量 Spatial GCN（可选）

默认关闭。打开后会在 `DE-like` 与 `TCN` 之间加入轻量空间图卷积残差分支。

### 开关与参数

1. `--use_gcn`：启用 Spatial GCN。
2. `--gcn_hidden`：GCN 隐层维度，默认 `16`。
3. `--gcn_beta`：`I` 与学习邻接融合系数，默认 `0.2`，范围 `[0,1]`。
4. `--gcn_dropout`：GCN 分支 dropout，默认 `0.1`。

### A/B 对照建议（其余超参保持完全一致）

基线（不加 GCN）：

```bash
python /home/xiaoying/seed_2026_like_de_LDS/train.py \
  --root /home/xiaoying/SEED \
  --loso \
  --results_csv /home/xiaoying/results_seed_2026_like_de_lds_baseline.csv
```

GCN 版本：

```bash
python /home/xiaoying/seed_2026_like_de_LDS/train.py \
  --root /home/xiaoying/SEED \
  --loso \
  --use_gcn \
  --gcn_hidden 16 \
  --gcn_beta 0.2 \
  --gcn_dropout 0.1 \
  --results_csv /home/xiaoying/results_seed_2026_like_de_lds_gcn.csv
```
