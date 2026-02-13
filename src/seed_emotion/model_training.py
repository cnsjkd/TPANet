"""
Model training pipeline for the SEED EEG emotion recognition task.

This module mirrors the original `3model.py` script, reorganised into a package
structure without changing the core algorithmic steps. It loads preprocessed EEG
chunks, reprograms them via a BERT backbone, and performs cross-subject LOSO
(Leave-One-Subject-Out) evaluation.
"""

import copy
import json
import hashlib
import math
import os
import argparse
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from openpyxl import Workbook, load_workbook
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


DEFAULT_DATA_DIR = "/home/aispeech/codes/zxy/SEED_chunks"
DEFAULT_BERT_DIR = "/home/aispeech/codes/zxy/TPANet-main/TPANet-main3_LOSO/models/bert-base-uncased"


def _is_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        if file_path.stat().st_size > 2048:
            return False
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1")


def _validate_local_bert_weights(bert_path: str) -> None:
    model_dir = Path(bert_path)
    if not model_dir.is_dir():
        return
    pytorch_bin = model_dir / "pytorch_model.bin"
    safetensors_file = model_dir / "model.safetensors"
    if _is_git_lfs_pointer(pytorch_bin):
        if safetensors_file.is_file() and safetensors_file.stat().st_size > 1024 * 1024:
            return
        raise RuntimeError(
            f"检测到无效模型权重: {pytorch_bin}\n"
            "该文件是 Git LFS 指针，不是真实的 BERT 权重。\n"
            "请下载完整权重到该目录（任选其一）：\n"
            "  - pytorch_model.bin\n"
            "  - model.safetensors\n"
            "下载完成后重新运行脚本即可。"
        )


def _is_cuda_oom_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class XlsxTextLogger:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.sheet_name = "logs"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if self.file_path.exists():
            self.workbook = load_workbook(self.file_path)
        else:
            self.workbook = Workbook()

        if self.sheet_name in self.workbook.sheetnames:
            self.worksheet = self.workbook[self.sheet_name]
        else:
            self.worksheet = self.workbook.active
            self.worksheet.title = self.sheet_name
            self.worksheet.append(["timestamp", "message"])
            self._flush()

    def write(self, text):
        if text is None:
            return
        text = str(text).replace("\r\n", "\n")
        if not text:
            return
        lines = text.split("\n")
        for line in lines:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.worksheet.append([timestamp, line])
        self._flush()

    def _flush(self):
        self.workbook.save(self.file_path)

    def close(self):
        self._flush()
        self.workbook.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EEGDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with np.load(file_path, allow_pickle=True) as npz_file:
            self.data = npz_file['data']
            self.labels = npz_file['labels'] + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_chunk = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(data_chunk, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)


class LazyEEGDataset(Dataset):
    """
    Lazy dataset that loads EEG chunks per sample from source .npz files with mmap.

    Each sample is identified by (file_id, local_idx), so we avoid concatenating all
    chunks from all files into one giant in-memory array.
    """

    def __init__(self, file_paths, file_ids, local_indices, labels, global_indices=None):
        self.file_paths = [Path(p) for p in file_paths]
        self.file_ids = np.asarray(file_ids, dtype=np.int32)
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if global_indices is None:
            self.global_indices = np.arange(len(self.labels), dtype=np.int64)
        else:
            self.global_indices = np.asarray(global_indices, dtype=np.int64)

        self._npz_handles = {}
        self._data_arrays = {}

    def __len__(self):
        return len(self.labels)

    def _get_data_array(self, file_id):
        file_id = int(file_id)
        if file_id not in self._data_arrays:
            npz_handle = np.load(self.file_paths[file_id], allow_pickle=True, mmap_mode="r")
            self._npz_handles[file_id] = npz_handle
            self._data_arrays[file_id] = npz_handle["data"]
        return self._data_arrays[file_id]

    def __getitem__(self, idx):
        file_id = int(self.file_ids[idx])
        local_idx = int(self.local_indices[idx])
        label = int(self.labels[idx])
        global_idx = int(self.global_indices[idx])

        data_chunk = self._get_data_array(file_id)[local_idx]
        return (
            torch.tensor(data_chunk, dtype=torch.float32),
            torch.tensor(label, dtype=torch.int64),
            torch.tensor(global_idx, dtype=torch.int64),
            torch.tensor(file_id, dtype=torch.int64),
            torch.tensor(local_idx, dtype=torch.int64),
        )

    def close(self):
        for handle in self._npz_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._npz_handles.clear()
        self._data_arrays.clear()

    def __del__(self):
        self.close()


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :].to(x.device)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, stride, num_channels, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = nn.Dropout(dropout)
        self.num_channels = num_channels
        self.value_embedding = nn.Conv1d(in_channels=num_channels, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, self.num_channels, self.patch_len)
        x = self.value_embedding(x)
        x = x.permute(0, 2, 1)
        return self.dropout(x)


class ReprogrammingLayer(nn.Module):
    def __init__(self, embed_dim, llm_embed_dim, num_heads=8, max_len=5000):
        super(ReprogrammingLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, llm_embed_dim)
        self.ln = nn.LayerNorm(llm_embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=llm_embed_dim, num_heads=num_heads, batch_first=True)
        self.positional_embedding = PositionalEmbedding(d_model=llm_embed_dim, max_len=max_len)

    def forward(self, target_embedding, source_embedding, value_embedding):
        batch_size = source_embedding.size(0)
        num_patches = target_embedding.size(0) // batch_size

        target_embedding = target_embedding.view(
            batch_size,
            num_patches,
            target_embedding.size(1),
            target_embedding.size(2)
        )

        target_embedding = target_embedding.mean(dim=2)
        target_embedding = self.linear(target_embedding)

        positional_encoding = self.positional_embedding(target_embedding)
        target_embedding = target_embedding + positional_encoding

        target_embedding = self.ln(target_embedding)

        attn_output, attn_weights = self.multihead_attn(
            target_embedding, source_embedding, value_embedding,
            need_weights=True,
            average_attn_weights=False,
        )

        return attn_output, attn_weights


class ClassificationHead(nn.Module):
    def __init__(self, llm_embed_dim, num_labels):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(llm_embed_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def generate_statistics(sample_trial):
    sample_trial_flat = sample_trial.view(sample_trial.size(0), -1)
    min_values = sample_trial_flat.min(dim=1)[0]
    max_values = sample_trial_flat.max(dim=1)[0]
    median_values = sample_trial_flat.median(dim=1)[0]
    trends = sample_trial_flat[:, -1] - sample_trial_flat[:, 0]
    return min_values, max_values, median_values, trends


def generate_prompts(batch_size, min_values, max_values, median_values, trends, mode="original", rng=None):
    """
    Build natural-language prompts from simple EEG statistics.

    Modes are used for reviewer-required ablations:
      - original: use true (min/max/median/trend)
      - generic: remove statistics (same capacity, less semantic guidance)
      - shuffle: keep numbers but shuffle their roles (break semantics, keep format)
      - random: keep format but replace numbers with random values in a plausible range
    """
    if rng is None:
        rng = np.random.RandomState(42)

    prompts = []
    for i in range(batch_size):
        min_value = float(min_values[i].item())
        max_value = float(max_values[i].item())
        median_value = float(median_values[i].item())
        trend_value = float(trends[i].item())
        trend = 'upward' if trend_value > 0 else 'downward'

        if mode == "generic":
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                "The EEG signal shows certain temporal dynamics and amplitude variations."
            )
        elif mode == "shuffle":
            stats = [min_value, max_value, median_value]
            rng.shuffle(stats)
            min_s, max_s, med_s = stats
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                f"Statistics: minimum value = {min_s:.2f}, maximum value = {max_s:.2f}, "
                f"median value = {med_s:.2f}. The overall trend of the data is {trend}."
            )
        elif mode == "random":
            lo, hi = (min_value, max_value) if min_value < max_value else (max_value, min_value + 1e-3)
            min_s = rng.uniform(lo, hi)
            max_s = rng.uniform(lo, hi)
            med_s = rng.uniform(lo, hi)
            trend = rng.choice(['upward', 'downward'])
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                f"Statistics: minimum value = {min_s:.2f}, maximum value = {max_s:.2f}, "
                f"median value = {med_s:.2f}. The overall trend of the data is {trend}."
            )
        else:  # original
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                f"Statistics: minimum value = {min_value:.2f}, maximum value = {max_value:.2f}, "
                f"median value = {median_value:.2f}. The overall trend of the data is {trend}."
            )

        prompts.append(prompt)
    return prompts




def save_attention_heatmap(attn_weights, prompts, save_path, token_limit=30):
    """Save a simple cross-attention heatmap (avg over heads) for the first sample in a batch.

    attn_weights: Tensor (B, num_heads, P, L)
    """
    import matplotlib.pyplot as plt

    w = attn_weights[0].mean(dim=0).detach().cpu().numpy()  # (P, L)
    w = w[:, :token_limit]

    plt.figure()
    plt.imshow(w, aspect='auto')
    plt.colorbar()
    plt.xlabel("Prompt token index")
    plt.ylabel("EEG patch index")
    plt.title("Cross-attention (avg over heads)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def profile_efficiency(
    model_components,
    device,
    batch_size=32,
    iters=200,
    warmup=50,
    prompt_mode="original",
    soft_prompt=None,
    prompt_cache_cpu=None,
    prompt_cache_manager=None,
    use_cached_prompts=False,
):
    """Measure params + inference latency (per forward) + peak GPU memory."""
    import time

    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval(); patch_embedding.eval(); reprogramming_layer.eval(); classification_head.eval()

    total_params = 0
    trainable_params = 0
    for m in [bert_model, patch_embedding, reprogramming_layer, classification_head]:
        for p in m.parameters():
            n = p.numel()
            total_params += n
            if p.requires_grad:
                trainable_params += n

    if device.type == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()

    import torch
    x = torch.randn(batch_size, 62, 1000, device=device)
    min_values, max_values, median_values, trends = generate_statistics(x)
    prompts = generate_prompts(batch_size, min_values, max_values, median_values, trends, mode=prompt_mode)

    if prompt_mode == "soft":
        if soft_prompt is None:
            raise ValueError("soft_prompt must be provided when prompt_mode='soft'")
        prompt_embeddings = soft_prompt.expand(batch_size, -1, -1)
    else:
        if use_cached_prompts and (prompt_cache_cpu is not None):
            prompt_embeddings = prompt_cache_cpu[:batch_size].to(device).float()
        elif use_cached_prompts and (prompt_cache_manager is not None):
            prompt_embeddings = prompt_cache_manager.get_first_embeddings(batch_size=batch_size, device=device)
        else:
            prompt_inputs = tokenizer(
                prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=50
            ).to(device)
            with torch.no_grad():
                prompt_embeddings = bert_model(**prompt_inputs).last_hidden_state

    with torch.no_grad():
        for _ in range(warmup):
            eeg_emb = patch_embedding(x)
            out, _ = reprogramming_layer(eeg_emb, prompt_embeddings, prompt_embeddings)
            pooled = out.mean(dim=1)
            _ = classification_head(pooled)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            eeg_emb = patch_embedding(x)
            out, _ = reprogramming_layer(eeg_emb, prompt_embeddings, prompt_embeddings)
            pooled = out.mean(dim=1)
            _ = classification_head(pooled)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

    latency_ms = (end - start) * 1000.0 / iters
    peak_mem_gb = None
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "latency_ms_per_forward": latency_ms,
        "peak_mem_gb": peak_mem_gb,
    }


def _cache_path_for_key(cache_key, prompt_mode):
    cache_dir = Path(__file__).resolve().parents[2] / "prompt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = str(cache_key)
    safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cache_key).strip("_")
    safe_key = safe_key[:80] if safe_key else "cache"
    suffix = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{safe_key}_{suffix}__{prompt_mode}__last_hidden_state_fp16.pt"


def build_or_load_prompt_cache(cache_key, all_data, device, prompt_mode="original", max_length=50, batch_size=64):
    """
    Build (or load) cached BERT last_hidden_state for all samples in a split.

    Saves token-level embeddings: (N, L, 768) in FP16 on CPU.
    """
    import torch

    cache_path = _cache_path_for_key(cache_key, prompt_mode)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    # all_data: numpy array (N, C, T)
    X = torch.tensor(all_data, dtype=torch.float32)  # CPU

    N = X.size(0)
    min_values, max_values, median_values, trends = generate_statistics(X)
    prompts = generate_prompts(N, min_values, max_values, median_values, trends, mode=prompt_mode)

    bert_model.eval()
    all_embeds = []
    with torch.no_grad():
        start_idx = 0
        current_bs = max(1, int(batch_size))
        while start_idx < N:
            local_bs = min(current_bs, N - start_idx)
            p_batch = prompts[start_idx:start_idx + local_bs]
            try:
                prompt_inputs = tokenizer(
                    p_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
                ).to(device)
                embeds = bert_model(**prompt_inputs).last_hidden_state
                all_embeds.append(embeds.detach().cpu().half())
                start_idx += local_bs
            except Exception as exc:
                if device.type == "cuda" and _is_cuda_oom_error(exc) and local_bs > 1:
                    torch.cuda.empty_cache()
                    current_bs = max(1, local_bs // 2)
                    print(f"[OOM保护] prompt cache batch size 降为 {current_bs}")
                    continue
                raise

    cache_tensor = torch.cat(all_embeds, dim=0)  # (N, L, 768), fp16, cpu
    torch.save(cache_tensor, cache_path)
    return cache_tensor


def build_or_load_prompt_cache_for_file(file_path, device, prompt_mode="original", max_length=50, batch_size=64):
    """
    Build (or load) cached BERT last_hidden_state for one source .npz file.

    This function streams data by batch from mmap, avoiding one-shot full-array loading.
    Returns cache file path, not tensor.
    """
    import torch

    file_path = Path(file_path)
    cache_key = f"file_{file_path.stem}"
    cache_path = _cache_path_for_key(cache_key, prompt_mode)
    if cache_path.exists():
        return cache_path

    with np.load(file_path, allow_pickle=True, mmap_mode="r") as npz_file:
        data = npz_file["data"]
        n_samples = int(data.shape[0])

        bert_model.eval()
        cache_tensor = None

        with torch.no_grad():
            start_idx = 0
            current_bs = max(1, int(batch_size))
            while start_idx < n_samples:
                local_bs = min(current_bs, n_samples - start_idx)
                try:
                    x_batch = torch.tensor(data[start_idx:start_idx + local_bs], dtype=torch.float32)
                    min_values, max_values, median_values, trends = generate_statistics(x_batch)
                    prompts = generate_prompts(local_bs, min_values, max_values, median_values, trends, mode=prompt_mode)

                    prompt_inputs = tokenizer(
                        prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
                    ).to(device)
                    embeds = bert_model(**prompt_inputs).last_hidden_state.detach().cpu().half()

                    if cache_tensor is None:
                        cache_tensor = torch.empty(
                            (n_samples, embeds.size(1), embeds.size(2)),
                            dtype=torch.float16,
                        )
                    cache_tensor[start_idx:start_idx + local_bs] = embeds
                    start_idx += local_bs
                except Exception as exc:
                    if device.type == "cuda" and _is_cuda_oom_error(exc) and local_bs > 1:
                        torch.cuda.empty_cache()
                        current_bs = max(1, local_bs // 2)
                        print(f"[OOM保护] prompt cache batch size 降为 {current_bs} | file={file_path.name}")
                        continue
                    raise

    if cache_tensor is None:
        raise RuntimeError(f"无法构建 prompt cache: {file_path}")
    torch.save(cache_tensor, cache_path)
    return cache_path


class PromptCacheManager:
    """
    File-level prompt cache manager with a small CPU-memory LRU.

    - Cache file is built lazily per source file.
    - Loaded cache tensors are kept with a bounded count to avoid host OOM.
    """

    def __init__(self, file_paths, device, prompt_mode="original", max_length=50, batch_size=64, max_files_in_mem=2):
        self.file_paths = [Path(p) for p in file_paths]
        self.device = device
        self.prompt_mode = prompt_mode
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.max_files_in_mem = max(1, int(max_files_in_mem))

        self._cache_paths = {}
        self._cache_lru = OrderedDict()

    def _ensure_cache_path(self, file_id):
        file_id = int(file_id)
        if file_id not in self._cache_paths:
            self._cache_paths[file_id] = build_or_load_prompt_cache_for_file(
                file_path=self.file_paths[file_id],
                device=self.device,
                prompt_mode=self.prompt_mode,
                max_length=self.max_length,
                batch_size=self.batch_size,
            )
        return self._cache_paths[file_id]

    def _get_cache_tensor(self, file_id):
        import torch

        file_id = int(file_id)
        if file_id in self._cache_lru:
            tensor = self._cache_lru.pop(file_id)
            self._cache_lru[file_id] = tensor
            return tensor

        cache_path = self._ensure_cache_path(file_id)
        tensor = torch.load(cache_path, map_location="cpu")
        self._cache_lru[file_id] = tensor

        while len(self._cache_lru) > self.max_files_in_mem:
            _, old_tensor = self._cache_lru.popitem(last=False)
            del old_tensor
        return tensor

    def get_batch_embeddings(self, batch_file_ids, batch_local_idx, device):
        if torch.is_tensor(batch_file_ids):
            file_ids = batch_file_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            file_ids = np.asarray(list(batch_file_ids), dtype=np.int64)
        if torch.is_tensor(batch_local_idx):
            local_idx = batch_local_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            local_idx = np.asarray(list(batch_local_idx), dtype=np.int64)

        n = int(file_ids.shape[0])
        if n == 0:
            raise RuntimeError("空批次无法提取 prompt embeddings。")

        # Group by file id to avoid repeatedly fetching the same cache tensor.
        order = np.argsort(file_ids, kind="stable")
        sorted_fids = file_ids[order]
        sorted_lidx = local_idx[order]

        out_cpu = None
        pos = 0
        while pos < n:
            fid = int(sorted_fids[pos])
            end = pos + 1
            while end < n and int(sorted_fids[end]) == fid:
                end += 1

            cache_tensor = self._get_cache_tensor(fid)
            idx_tensor = torch.as_tensor(sorted_lidx[pos:end], dtype=torch.long)
            gathered = cache_tensor.index_select(0, idx_tensor)

            if out_cpu is None:
                out_cpu = torch.empty(
                    (n, gathered.size(1), gathered.size(2)),
                    dtype=gathered.dtype,
                )

            target_idx = torch.as_tensor(order[pos:end], dtype=torch.long)
            out_cpu.index_copy_(0, target_idx, gathered)
            pos = end

        return out_cpu.to(device).float()

    def get_first_embeddings(self, batch_size, device):
        if not self.file_paths:
            raise RuntimeError("PromptCacheManager 没有可用文件。")
        first_tensor = self._get_cache_tensor(0)
        batch_size = max(1, int(batch_size))
        if first_tensor.size(0) >= batch_size:
            sample = first_tensor[:batch_size]
        else:
            repeat = (batch_size + first_tensor.size(0) - 1) // first_tensor.size(0)
            sample = first_tensor.repeat((repeat, 1, 1))[:batch_size]
        return sample.to(device).float()

    def clear(self):
        self._cache_lru.clear()


def train_model(
    model_components,
    dataloader,
    optimizer,
    criterion,
    device,
    num_labels,
    prompt_mode="original",
    soft_prompt=None,
    prompt_cache_cpu=None,
    prompt_cache_manager=None,
    progress_callback=None,
    progress_every_steps=0,
    progress_prefix="",
    use_cached_prompts=False,
):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.train()
    reprogramming_layer.train()
    classification_head.train()
    total_loss = 0
    effective_steps = 0

    total_steps = len(dataloader)
    for step_idx, batch in enumerate(dataloader, start=1):
        if len(batch) == 5:
            batch_eeg, batch_labels, batch_idx, batch_file_ids, batch_local_idx = batch
        elif len(batch) == 3:
            batch_eeg, batch_labels, batch_idx = batch
            batch_file_ids = None
            batch_local_idx = None
        else:
            batch_eeg, batch_labels = batch
            batch_idx = None
            batch_file_ids = None
            batch_local_idx = None
        try:
            batch_eeg = batch_eeg.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            eeg_embeddings = patch_embedding(batch_eeg)

            min_values, max_values, median_values, trends = generate_statistics(batch_eeg)
            prompts = generate_prompts(batch_eeg.size(0), min_values, max_values, median_values, trends, mode=prompt_mode)
            if prompt_mode == "soft":
                if soft_prompt is None:
                    raise ValueError("soft_prompt must be provided when prompt_mode='soft'")
                prompt_embeddings = soft_prompt.expand(batch_eeg.size(0), -1, -1)  # (B, L, 768)
            else:
                if (
                    use_cached_prompts
                    and (prompt_cache_manager is not None)
                    and (batch_file_ids is not None)
                    and (batch_local_idx is not None)
                ):
                    prompt_embeddings = prompt_cache_manager.get_batch_embeddings(
                        batch_file_ids=batch_file_ids,
                        batch_local_idx=batch_local_idx,
                        device=device,
                    )
                elif use_cached_prompts and (prompt_cache_cpu is not None) and (batch_idx is not None):
                    prompt_embeddings = prompt_cache_cpu[batch_idx].to(device).float()
                else:
                    prompt_inputs = tokenizer(
                        prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=50
                    ).to(device)
                    with torch.no_grad():
                        prompt_embeddings = bert_model(**prompt_inputs).last_hidden_state  # (B, L, 768)
            eeg_embeddings, _ = reprogramming_layer(eeg_embeddings, prompt_embeddings, prompt_embeddings)

            pooled_output = eeg_embeddings.mean(dim=1)

            logits = classification_head(pooled_output)
            loss = criterion(logits, batch_labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            effective_steps += 1
            if progress_callback is not None and progress_every_steps > 0 and (step_idx % progress_every_steps == 0):
                avg_loss = total_loss / max(1, effective_steps)
                msg = (
                    f"{progress_prefix} step {step_idx}/{total_steps}, "
                    f"effective_steps={effective_steps}, avg_loss={avg_loss:.6f}"
                )
                try:
                    progress_callback(msg)
                except Exception:
                    pass
        except Exception as exc:
            if device.type == "cuda" and _is_cuda_oom_error(exc):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print("[OOM保护] 训练批次发生CUDA OOM，已跳过该批次。")
                continue
            raise

    if effective_steps == 0:
        raise RuntimeError("训练阶段全部批次因 OOM 被跳过，请减小 --batch_size。")
    average_loss = total_loss / effective_steps
    return average_loss


def evaluate_model(
    model_components,
    dataloader,
    device,
    num_labels,
    prompt_mode="original",
    soft_prompt=None,
    save_attn_dir=None,
    max_attn_batches=1,
    prompt_cache_cpu=None,
    prompt_cache_manager=None,
    use_cached_prompts=False,
):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.eval()
    reprogramming_layer.eval()
    classification_head.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                batch_eeg, batch_labels, batch_idx, batch_file_ids, batch_local_idx = batch
            elif len(batch) == 3:
                batch_eeg, batch_labels, batch_idx = batch
                batch_file_ids = None
                batch_local_idx = None
            else:
                batch_eeg, batch_labels = batch
                batch_idx = None
                batch_file_ids = None
                batch_local_idx = None
            try:
                batch_eeg = batch_eeg.to(device)
                batch_labels = batch_labels.to(device)

                eeg_embeddings = patch_embedding(batch_eeg)

                min_values, max_values, median_values, trends = generate_statistics(batch_eeg)
                prompts = generate_prompts(batch_eeg.size(0), min_values, max_values, median_values, trends, mode=prompt_mode)
                if prompt_mode == "soft":
                    if soft_prompt is None:
                        raise ValueError("soft_prompt must be provided when prompt_mode='soft'")
                    prompt_embeddings = soft_prompt.expand(batch_eeg.size(0), -1, -1)
                else:
                    if (
                        use_cached_prompts
                        and (prompt_cache_manager is not None)
                        and (batch_file_ids is not None)
                        and (batch_local_idx is not None)
                    ):
                        prompt_embeddings = prompt_cache_manager.get_batch_embeddings(
                            batch_file_ids=batch_file_ids,
                            batch_local_idx=batch_local_idx,
                            device=device,
                        )
                    elif use_cached_prompts and (prompt_cache_cpu is not None) and (batch_idx is not None):
                        prompt_embeddings = prompt_cache_cpu[batch_idx].to(device).float()
                    else:
                        prompt_inputs = tokenizer(
                            prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=50
                        ).to(device)
                        prompt_embeddings = bert_model(**prompt_inputs).last_hidden_state
                eeg_embeddings, attn_weights = reprogramming_layer(eeg_embeddings, prompt_embeddings, prompt_embeddings)

                # Optionally save attention maps for interpretability
                if save_attn_dir is not None and max_attn_batches > 0:
                    os.makedirs(save_attn_dir, exist_ok=True)
                    # save only a few batches to avoid huge files
                    if len(os.listdir(save_attn_dir)) < max_attn_batches:
                        try:
                            save_attention_heatmap(attn_weights, prompts, os.path.join(save_attn_dir, f"attn_batch{len(os.listdir(save_attn_dir))}.png"))
                        except Exception:
                            pass

                pooled_output = eeg_embeddings.mean(dim=1)

                logits = classification_head(pooled_output)
                predictions = torch.argmax(logits, dim=1)

                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
            except Exception as exc:
                if device.type == "cuda" and _is_cuda_oom_error(exc):
                    torch.cuda.empty_cache()
                    print("[OOM保护] 评估批次发生CUDA OOM，已跳过该批次。")
                    continue
                raise

    if not all_labels:
        raise RuntimeError("评估阶段全部批次因 OOM 被跳过，请减小 --batch_size。")

    _ = classification_report(all_labels, all_predictions, zero_division=0)

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(num_labels)))

    return accuracy, f1, precision, recall, conf_matrix


def check_class_distribution(labels, dataset_name="Dataset", log_file=None):
    class_counts = Counter(labels)
    if log_file:
        print(f"{dataset_name} class distribution: {class_counts}")
        log_file.write(f"{dataset_name} class distribution: {class_counts}\n")
    else:
        print(f"{dataset_name} class distribution: {class_counts}")


def parse_subject_id(file_path: Path) -> str:
    stem = Path(file_path).stem
    return stem.split("_", 1)[0]


def subject_sort_key(subject_id: str):
    try:
        return (0, int(subject_id))
    except ValueError:
        return (1, subject_id)


def build_subject_file_map(preprocessed_files):
    subject_to_files = {}
    for file_path in preprocessed_files:
        subject_id = parse_subject_id(file_path)
        subject_to_files.setdefault(subject_id, []).append(file_path)
    for subject_id in subject_to_files:
        subject_to_files[subject_id] = sorted(subject_to_files[subject_id])
    return dict(sorted(subject_to_files.items(), key=lambda x: subject_sort_key(x[0])))


def build_split_index_from_files(file_paths):
    """
    Build split metadata without loading all EEG arrays into RAM.

    Returns:
      {
        "file_paths": List[Path],
        "file_ids": np.ndarray[int32],
        "local_indices": np.ndarray[int64],
        "labels": np.ndarray[int64],
        "subjects": np.ndarray[object],
      }
    """
    file_paths = [Path(p) for p in file_paths]
    file_ids_list = []
    local_indices_list = []
    labels_list = []
    subjects_list = []

    for file_id, file_path in enumerate(file_paths):
        with np.load(file_path, allow_pickle=True, mmap_mode="r") as npz_file:
            labels = np.asarray(npz_file["labels"], dtype=np.int64) + 1

        n_samples = int(labels.shape[0])
        subject_id = parse_subject_id(file_path)

        file_ids_list.append(np.full(shape=(n_samples,), fill_value=file_id, dtype=np.int32))
        local_indices_list.append(np.arange(n_samples, dtype=np.int64))
        labels_list.append(labels.astype(np.int64, copy=False))
        subjects_list.append(np.full(shape=(n_samples,), fill_value=subject_id, dtype=object))

    if not labels_list:
        raise RuntimeError("未加载到任何样本。")

    return {
        "file_paths": file_paths,
        "file_ids": np.concatenate(file_ids_list, axis=0),
        "local_indices": np.concatenate(local_indices_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
        "subjects": np.concatenate(subjects_list, axis=0),
    }


def create_lazy_dataset(split_index, subset_indices=None):
    if subset_indices is None:
        subset_indices = np.arange(len(split_index["labels"]), dtype=np.int64)
    subset_indices = np.asarray(subset_indices, dtype=np.int64)

    return LazyEEGDataset(
        file_paths=split_index["file_paths"],
        file_ids=split_index["file_ids"][subset_indices],
        local_indices=split_index["local_indices"][subset_indices],
        labels=split_index["labels"][subset_indices],
        global_indices=subset_indices,
    )


def load_data_from_files(file_paths):
    all_data = []
    all_labels = []
    all_groups = []

    for file_path in file_paths:
        subject_id = parse_subject_id(file_path)
        with np.load(file_path, allow_pickle=True) as npz_file:
            data = npz_file["data"]
            labels = npz_file["labels"] + 1

        if len(data) != len(labels):
            raise ValueError(f"数据与标签长度不一致: {file_path}")

        all_data.append(data.astype(np.float32, copy=False))
        all_labels.append(labels.astype(np.int64, copy=False))
        all_groups.append(np.full(shape=(len(labels),), fill_value=subject_id, dtype=object))

    if not all_data:
        raise RuntimeError("未加载到任何样本。")

    return (
        np.concatenate(all_data, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_groups, axis=0),
    )


def compute_class_weights(labels, num_labels, device):
    class_counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    class_weights = np.zeros_like(class_counts, dtype=np.float32)
    non_zero = class_counts > 0
    if np.any(non_zero):
        class_weights[non_zero] = 1.0 / class_counts[non_zero]
        class_weights[non_zero] = class_weights[non_zero] / class_weights[non_zero].sum() * np.sum(non_zero)
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=os.getenv(
            "SEED_CHUNKS_DIR",
            DEFAULT_DATA_DIR,
        ),
        help="Directory containing preprocessed SEED chunk .npz files",
    )
    ap.add_argument(
        "--bert",
        default=os.getenv("BERT_MODEL_DIR", DEFAULT_BERT_DIR),
        help="BERT model directory or model id",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE", "16")),
        help="Training/evaluation batch size (default=16, lower to reduce OOM risk).",
    )
    ap.add_argument(
        "--cache_batch_size",
        type=int,
        default=int(os.getenv("CACHE_BATCH_SIZE", "16")),
        help="Batch size used when precomputing prompt cache (default=16).",
    )
    ap.add_argument(
        "--cache_files_in_mem",
        type=int,
        default=int(os.getenv("CACHE_FILES_IN_MEM", "2")),
        help="How many file-level prompt caches can stay in CPU memory simultaneously.",
    )
    ap.add_argument(
        "--log_every_steps",
        type=int,
        default=int(os.getenv("LOG_EVERY_STEPS", "200")),
        help="Write training progress to xlsx every N steps (0 disables step-level logging).",
    )
    ap.add_argument(
        "--profile_batch_size",
        type=int,
        default=int(os.getenv("PROFILE_BATCH_SIZE", "8")),
        help="Batch size for efficiency profiling.",
    )
    ap.add_argument(
        "--profile_iters",
        type=int,
        default=int(os.getenv("PROFILE_ITERS", "50")),
        help="Iterations for efficiency profiling.",
    )
    ap.add_argument(
        "--profile_warmup",
        type=int,
        default=int(os.getenv("PROFILE_WARMUP", "10")),
        help="Warmup iterations for efficiency profiling.",
    )
    ap.add_argument(
        "--loso_subject",
        default=None,
        help="Only run LOSO for a specific subject id (e.g. 1, 2, ...).",
    )
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = Path(__file__).resolve().parents[2] / "results_confusion_matrix.xlsx"
    with XlsxTextLogger(log_path) as log_file:
        global bert_model, tokenizer
        bert_path = str(Path(args.bert).expanduser())

        def write_progress(message):
            print(message)
            log_file.write(f"{message}\n")

        _validate_local_bert_weights(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        try:
            bert_model = BertModel.from_pretrained(bert_path, weights_only=False).to(device)
        except TypeError:
            bert_model = BertModel.from_pretrained(bert_path).to(device)

        for param in bert_model.parameters():
            param.requires_grad = False

        data_root = Path(args.data_dir).expanduser()
        if not data_root.exists():
            raise FileNotFoundError(f"未找到数据目录: {data_root}")
        preprocessed_files = sorted([f for f in data_root.iterdir() if f.suffix == '.npz'])
        print(f"Found {len(preprocessed_files)} files in {data_root}")
        log_file.write(f"Found {len(preprocessed_files)} files in {data_root}\n")
        if not preprocessed_files:
            raise RuntimeError(
                f"目录 {data_root} 下未找到 .npz 文件。"
                "请通过 --data_dir 指向真实的 SEED_chunks 路径。"
            )

        subject_file_map = build_subject_file_map(preprocessed_files)
        subject_ids = list(subject_file_map.keys())
        if args.loso_subject is not None:
            wanted_subject = str(args.loso_subject)
            if wanted_subject not in subject_file_map:
                raise ValueError(f"--loso_subject={wanted_subject} 不在当前数据中。可选: {subject_ids}")
            subject_ids = [wanted_subject]

        print(f"Found {len(subject_ids)} subjects for LOSO: {subject_ids}")
        log_file.write(f"Found {len(subject_ids)} subjects for LOSO: {subject_ids}\n")

        # Prompt ablation mode: one of {"original","generic","shuffle","random","soft"}.
        prompt_mode = os.getenv("PROMPT_MODE", "original").strip().lower()
        # Prompt embedding mode:
        #   - online: run BERT forward each iteration (slow)
        #   - cached: load per-sample cached last_hidden_state (recommended)
        prompt_emb_mode = os.getenv("PROMPT_EMB_MODE", "cached").strip().lower()
        use_cached_prompts = (prompt_emb_mode == "cached")
        log_every_steps = max(0, int(args.log_every_steps))

        if use_cached_prompts and args.cache_files_in_mem <= 1:
            warn_msg = (
                "[性能提示] --cache_files_in_mem=1 在 shuffle 训练下会非常慢，"
                "建议设置为 4-8（内存允许时）。"
            )
            print(warn_msg)
            log_file.write(f"{warn_msg}\n")

        metrics_per_subject = []
        batch_size = max(1, args.batch_size)

        for held_out_subject in subject_ids:
            test_files = subject_file_map[held_out_subject]
            train_files = []
            for subject_id, file_list in subject_file_map.items():
                if subject_id != held_out_subject:
                    train_files.extend(file_list)

            if not train_files:
                raise RuntimeError(f"LOSO失败: 被试 {held_out_subject} 没有可用训练文件。")

            print(f"\nLOSO fold | held-out subject: {held_out_subject}")
            print(f"  Train files: {len(train_files)} | Test files: {len(test_files)}")
            log_file.write(f"\nLOSO fold | held-out subject: {held_out_subject}\n")
            log_file.write(f"  Train files: {len(train_files)} | Test files: {len(test_files)}\n")

            train_split_index = build_split_index_from_files(train_files)
            test_split_index = build_split_index_from_files(test_files)

            y_train_val = train_split_index["labels"]
            groups_train = train_split_index["subjects"]
            y_test = test_split_index["labels"]

            if np.any(y_train_val < 0) or np.any(y_test < 0):
                raise ValueError(f"发现负标签，无法训练。held-out subject={held_out_subject}")

            num_labels = int(max(np.max(y_train_val), np.max(y_test)) + 1)
            check_class_distribution(y_train_val, f"Train_Val Dataset (exclude subject {held_out_subject})", log_file)
            check_class_distribution(y_test, f"Test Dataset (subject {held_out_subject})", log_file)

            prompt_cache_train = None
            prompt_cache_test = None
            if use_cached_prompts and prompt_mode != "soft":
                prompt_cache_train = PromptCacheManager(
                    file_paths=train_split_index["file_paths"],
                    device=device,
                    prompt_mode=prompt_mode,
                    max_length=50,
                    batch_size=max(1, args.cache_batch_size),
                    max_files_in_mem=max(1, args.cache_files_in_mem),
                )
                prompt_cache_test = PromptCacheManager(
                    file_paths=test_split_index["file_paths"],
                    device=device,
                    prompt_mode=prompt_mode,
                    max_length=50,
                    batch_size=max(1, args.cache_batch_size),
                    max_files_in_mem=max(1, args.cache_files_in_mem),
                )

            test_dataset = create_lazy_dataset(test_split_index)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            num_train_subjects = len(np.unique(groups_train))
            n_splits = min(5, num_train_subjects)
            if n_splits < 2:
                raise RuntimeError(
                    f"训练被试数不足以做 GroupKFold: {num_train_subjects}（held-out subject={held_out_subject}）"
                )

            best_fold_val_accuracy = -1.0
            best_fold_state = None
            best_fold_soft_prompt = None
            best_model_components = None

            group_kfold = GroupKFold(n_splits=n_splits)
            split_indices = np.arange(len(y_train_val))
            fold = 1
            for train_indices, val_indices in group_kfold.split(split_indices, y_train_val, groups_train):
                print(f"  Inner Fold {fold}/{n_splits}:")
                log_file.write(f"  Inner Fold {fold}/{n_splits}:\n")

                y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]
                train_dataset = create_lazy_dataset(train_split_index, train_indices)
                val_dataset = create_lazy_dataset(train_split_index, val_indices)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                class_weights = compute_class_weights(y_train, num_labels, device)
                print(f"    Class weights: {class_weights}")
                log_file.write(f"    Class weights: {class_weights}\n")
                criterion = nn.CrossEntropyLoss(weight=class_weights)

                patch_embedding = PatchEmbedding(
                    patch_len=500, d_model=128, stride=250, num_channels=62
                ).to(device)
                reprogramming_layer = ReprogrammingLayer(
                    embed_dim=128, llm_embed_dim=768, num_heads=8, max_len=5000
                ).to(device)
                classification_head = ClassificationHead(
                    llm_embed_dim=768, num_labels=num_labels
                ).to(device)

                soft_prompt = None
                optimizer_groups = [
                    {'params': classification_head.parameters(), 'lr': 1e-5},
                    {'params': patch_embedding.parameters(), 'lr': 1e-5},
                    {'params': reprogramming_layer.parameters(), 'lr': 1e-5},
                ]
                if prompt_mode == "soft":
                    soft_prompt = nn.Parameter(torch.randn(1, 50, 768, device=device) * 0.02)
                    optimizer_groups.append({'params': [soft_prompt], 'lr': 1e-5})

                model_components = (bert_model, patch_embedding, reprogramming_layer, classification_head)
                best_model_components = model_components

                # ---- Efficiency logging -> efficiency.jsonl (one line per fold) ----
                try:
                    eff_path = Path(__file__).resolve().parents[2] / "efficiency.jsonl"
                    eff = profile_efficiency(
                        model_components, device,
                        batch_size=max(1, args.profile_batch_size),
                        iters=max(1, args.profile_iters),
                        warmup=max(0, args.profile_warmup),
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt,
                        prompt_cache_manager=prompt_cache_train,
                        use_cached_prompts=use_cached_prompts,
                    )
                    record = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "loso_subject": str(held_out_subject),
                        "prompt_mode": prompt_mode,
                        "prompt_emb_mode": prompt_emb_mode,
                        "fold_id": int(fold),
                        "num_labels": int(num_labels),
                        **eff,
                    }
                    with open(eff_path, "a", encoding="utf-8") as f_eff:
                        f_eff.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    pass

                optimizer = AdamW(optimizer_groups, weight_decay=0.01)

                best_val_accuracy = 0.0
                best_model_state = None
                best_soft_prompt = None
                no_improve_epochs = 0

                patience = 5
                num_epochs = 100

                for epoch in range(num_epochs):
                    train_loss = train_model(
                        model_components,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        num_labels,
                        prompt_mode=prompt_mode,
                        soft_prompt=soft_prompt,
                        prompt_cache_manager=prompt_cache_train,
                        progress_callback=write_progress,
                        progress_every_steps=log_every_steps,
                        progress_prefix=f"    [S{held_out_subject} F{fold} E{epoch + 1}]",
                        use_cached_prompts=use_cached_prompts,
                    )

                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        model_components,
                        val_loader,
                        device,
                        num_labels,
                        prompt_mode=prompt_mode,
                        soft_prompt=soft_prompt,
                        prompt_cache_manager=prompt_cache_train,
                        use_cached_prompts=use_cached_prompts,
                    )

                    print(
                        f"    Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, "
                        f"Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}"
                    )
                    log_file.write(
                        f"    Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, "
                        f"Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}\n"
                    )

                    if val_accuracy >= best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_model_state = {
                            'patch_embedding': copy.deepcopy(patch_embedding.state_dict()),
                            'reprogramming_layer': copy.deepcopy(reprogramming_layer.state_dict()),
                            'classification_head': copy.deepcopy(classification_head.state_dict()),
                        }
                        if soft_prompt is not None:
                            best_soft_prompt = soft_prompt.detach().clone()
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            print(f"    Early stopping on epoch {epoch + 1}")
                            log_file.write(f"    Early stopping on epoch {epoch + 1}\n")
                            break

                if best_model_state is not None:
                    patch_embedding.load_state_dict(best_model_state['patch_embedding'])
                    reprogramming_layer.load_state_dict(best_model_state['reprogramming_layer'])
                    classification_head.load_state_dict(best_model_state['classification_head'])
                    if soft_prompt is not None and best_soft_prompt is not None:
                        with torch.no_grad():
                            soft_prompt.copy_(best_soft_prompt)

                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        model_components,
                        val_loader,
                        device,
                        num_labels,
                        prompt_mode=prompt_mode,
                        soft_prompt=soft_prompt,
                        prompt_cache_manager=prompt_cache_train,
                        use_cached_prompts=use_cached_prompts,
                    )
                    print(
                        f"    Best Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}"
                    )
                    log_file.write(
                        f"    Best Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}\n"
                    )

                    if val_accuracy >= best_fold_val_accuracy:
                        best_fold_val_accuracy = val_accuracy
                        best_fold_state = {
                            'patch_embedding': copy.deepcopy(patch_embedding.state_dict()),
                            'reprogramming_layer': copy.deepcopy(reprogramming_layer.state_dict()),
                            'classification_head': copy.deepcopy(classification_head.state_dict()),
                        }
                        best_fold_soft_prompt = best_soft_prompt.detach().clone() if best_soft_prompt is not None else None

                fold += 1

            if best_fold_state is None or best_model_components is None:
                print(f"Skip subject {held_out_subject}: no valid model state.")
                log_file.write(f"Skip subject {held_out_subject}: no valid model state.\n")
                if prompt_cache_train is not None:
                    prompt_cache_train.clear()
                if prompt_cache_test is not None:
                    prompt_cache_test.clear()
                continue

            _, patch_embedding, reprogramming_layer, classification_head = best_model_components
            patch_embedding.load_state_dict(best_fold_state['patch_embedding'])
            reprogramming_layer.load_state_dict(best_fold_state['reprogramming_layer'])
            classification_head.load_state_dict(best_fold_state['classification_head'])

            soft_prompt_for_test = None
            if prompt_mode == "soft" and best_fold_soft_prompt is not None:
                soft_prompt_for_test = best_fold_soft_prompt.to(device)

            test_accuracy, test_f1, test_precision, test_recall, test_conf_matrix = evaluate_model(
                best_model_components,
                test_loader,
                device,
                num_labels,
                prompt_mode=prompt_mode,
                soft_prompt=soft_prompt_for_test,
                prompt_cache_manager=prompt_cache_test,
                use_cached_prompts=use_cached_prompts,
                save_attn_dir=str(Path(__file__).resolve().parents[2] / "attn_viz"),
                max_attn_batches=1,
            )
            print(
                f"*** LOSO Subject {held_out_subject} Test Results - Accuracy: {test_accuracy}, "
                f"F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}, "
                f"Conf_matrix: {test_conf_matrix}"
            )
            log_file.write(
                f"*** LOSO Subject {held_out_subject} Test Results - Accuracy: {test_accuracy}, "
                f"F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}, "
                f"Conf_matrix: {test_conf_matrix}\n"
            )

            metrics_per_subject.append({
                'subject_id': held_out_subject,
                'accuracy': test_accuracy,
                'f1': test_f1,
                'precision': test_precision,
                'recall': test_recall,
                'conf_matrix': test_conf_matrix,
            })
            if prompt_cache_train is not None:
                prompt_cache_train.clear()
            if prompt_cache_test is not None:
                prompt_cache_test.clear()

        if metrics_per_subject:
            avg_accuracy = np.mean([m['accuracy'] for m in metrics_per_subject])
            avg_f1 = np.mean([m['f1'] for m in metrics_per_subject])
            avg_precision = np.mean([m['precision'] for m in metrics_per_subject])
            avg_recall = np.mean([m['recall'] for m in metrics_per_subject])
            avg_conf_matrix = np.mean([m['conf_matrix'] for m in metrics_per_subject], axis=0)

            print("Average Metrics across LOSO subjects:")
            print(f"  Accuracy: {avg_accuracy:.4f}")
            print(f"  F1 Score: {avg_f1:.4f}")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  Conf_matrix:\n{avg_conf_matrix}")

            log_file.write("Average Metrics across LOSO subjects:\n")
            log_file.write(f"  Accuracy: {avg_accuracy:.4f}\n")
            log_file.write(f"  F1 Score: {avg_f1:.4f}\n")
            log_file.write(f"  Precision: {avg_precision:.4f}\n")
            log_file.write(f"  Recall: {avg_recall:.4f}\n")
            log_file.write(f"  Conf_matrix:\n{avg_conf_matrix}\n")


if __name__ == '__main__':
    main()
