# SEED Emotion Recognition Toolkit

该项目包含三个核心脚本：数据增强、切块预处理以及模型训练，下面给出了详细的运行流程、环境依赖和目录组织方式（以SEED数据集为例）。


## 项目结构

```
TPANet/
├── README.md
├── requirements.txt
├── data/
│   ├── SEED/             # 原始 .mat 数据（外部数据，不随仓库发布）
│   │   └── label.mat
│   ├── SEED_aug/         # 运行数据增强后生成的 .npz
│   └── SEED_chunks/      # 划分固定长度块后的 .npz
├── models/
│   └── bert-base-uncased/  # 已离线下载的 BERT 权重
└── src/
    └── seed_emotion/
        ├── __init__.py
        ├── data_augmentation.py
        ├── chunk_preprocessing.py
        └── model_training.py
```

脚本放置于 `src/seed_emotion/` 下，方便通过 `python -m` 方式直接调用。

## 环境配置

1. 准备 Python ≥ 3.9 的虚拟环境。  
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 若已下载 `bert-base-uncased` 到 `models/bert-base-uncased/`，运行训练脚本时会优先读取该目录；也可通过环境变量 `BERT_MODEL_DIR` 指向其他路径。
4. 若代码通过 Git 克隆，需确保已拉取 LFS 大文件，否则 `models/bert-base-uncased/pytorch_model.bin` 可能只是指针文件（约 100 多字节）：

   ```bash
   git lfs install
   git lfs pull --include="models/bert-base-uncased/pytorch_model.bin"
   ```

## 数据准备

1. 将原始 SEED 数据集（`.mat` 文件及 `label.mat`）复制到 `data/SEED/`。  
2. 运行数据增强脚本后会自动在 `data/SEED_aug/` 生成对应的 `.npz` 文件，并写出 `label.npy`。  
3. 切块脚本会把每个增强后的 `.npz` 分割成统一长度的片段，保存到 `data/SEED_chunks/`。


## 实验

以下命令需在仓库根目录 `TPANet/` 下执行。若使用命令行运行模块，建议临时设置 `PYTHONPATH`：

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

1. **数据增强**
   ```bash
   python -m seed_emotion.data_augmentation
   ```
   - 输入：`data/SEED/*.mat`
   - 输出：`data/SEED_aug/*.npz` 与 `data/SEED_aug/label.npy`

2. **划分固定长度块**
   ```bash
   python -m seed_emotion.chunk_preprocessing
   ```
   - 输入：`data/SEED_aug/*.npz`
   - 输出：`data/SEED_chunks/*.npz`

3. **模型训练与评估**
   ```bash
   python -m seed_emotion.model_training
   ```
   - 默认输入：`/home/aispeech/codes/zxy/SEED_chunks`
   - 默认 BERT：`/home/aispeech/codes/zxy/TPANet-main/TPANet-main3_LOSO/models/bert-base-uncased`
   - 评估方式：跨被试 LOSO（每次留 1 个被试作为测试集，其余被试训练）
   - 日志：`results_confusion_matrix.xlsx`（训练中边跑边写入，异常中断时已写内容不会丢）
   - 默认 `batch_size=16`（降低 OOM 风险）
   - 数据加载：按文件懒加载，不再一次性把全部样本拼接到内存
   - 可选：`--cache_files_in_mem 2` 控制同时驻留内存的 prompt cache 文件数
   - 可选：`--loso_subject 1` 仅跑单个被试的 LOSO 折

建议在具备 GPU 的环境下运行模型训练，否则训练时间会明显延长。

## 引用说明

使用该项目时，请注明代码与论文的引用信息。
