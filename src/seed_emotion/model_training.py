"""
Model training pipeline for the SEED EEG emotion recognition task.

This module mirrors the original `3model.py` script, reorganised into a package
structure without changing the underlying algorithmic steps. It loads the
preprocessed EEG chunks, reprograms them via a BERT backbone, and performs
cross-validation alongside a held-out test split.
"""

import copy
import json
import hashlib
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import KFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertModel, BertTokenizer


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


def profile_efficiency(model_components, device, batch_size=32, iters=200, warmup=50, prompt_mode="original", soft_prompt=None, prompt_cache_cpu=None, use_cached_prompts=False):
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
        else:
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
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


def _cache_path_for_file(file_path, prompt_mode):
    cache_dir = Path(__file__).resolve().parents[2] / "prompt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    return cache_dir / f"{stem}__{prompt_mode}__last_hidden_state_fp16.pt"


def build_or_load_prompt_cache(file_path, all_data, device, prompt_mode="original", max_length=50, batch_size=64):
    """
    Build (or load) cached BERT last_hidden_state for ALL samples in a given chunk file.

    Saves token-level embeddings: (N, L, 768) in FP16 on CPU.
    """
    import torch

    cache_path = _cache_path_for_file(file_path, prompt_mode)
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
        for s in range(0, N, batch_size):
            p_batch = prompts[s:s + batch_size]
            prompt_inputs = tokenizer(
                p_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            embeds = bert_model(**prompt_inputs).last_hidden_state
            all_embeds.append(embeds.detach().cpu().half())

    cache_tensor = torch.cat(all_embeds, dim=0)  # (N, L, 768), fp16, cpu
    torch.save(cache_tensor, cache_path)
    return cache_tensor


def train_model(model_components, dataloader, optimizer, criterion, device, num_labels, prompt_mode="original", soft_prompt=None, prompt_cache_cpu=None, use_cached_prompts=False):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.train()
    reprogramming_layer.train()
    classification_head.train()
    total_loss = 0

    for batch in dataloader:
        if len(batch) == 3:
            batch_eeg, batch_labels, batch_idx = batch
        else:
            batch_eeg, batch_labels = batch
            batch_idx = None
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
            if use_cached_prompts and (prompt_cache_cpu is not None) and (batch_idx is not None):
                prompt_embeddings = prompt_cache_cpu[batch_idx].to(device).float()
            else:
                prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
                with torch.no_grad():
                    prompt_embeddings = bert_model(**prompt_inputs).last_hidden_state  # (B, L, 768)
        eeg_embeddings, _ = reprogramming_layer(eeg_embeddings, prompt_embeddings, prompt_embeddings)

        pooled_output = eeg_embeddings.mean(dim=1)

        logits = classification_head(pooled_output)
        loss = criterion(logits, batch_labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate_model(model_components, dataloader, device, num_labels, prompt_mode="original", soft_prompt=None, save_attn_dir=None, max_attn_batches=1, prompt_cache_cpu=None, use_cached_prompts=False):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.eval()
    reprogramming_layer.eval()
    classification_head.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                batch_eeg, batch_labels, batch_idx = batch
            else:
                batch_eeg, batch_labels = batch
                batch_idx = None
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
                if use_cached_prompts and (prompt_cache_cpu is not None) and (batch_idx is not None):
                    prompt_embeddings = prompt_cache_cpu[batch_idx].to(device).float()
                else:
                    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
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

    report = classification_report(all_labels, all_predictions, zero_division=0)

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, f1, precision, recall, conf_matrix


def check_class_distribution(labels, dataset_name="Dataset", log_file=None):
    class_counts = Counter(labels)
    if log_file:
        print(f"{dataset_name} class distribution: {class_counts}")
        log_file.write(f"{dataset_name} class distribution: {class_counts}\n")
    else:
        print(f"{dataset_name} class distribution: {class_counts}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_path = Path(__file__).resolve().parents[2] / "results_confusion_matrix.txt"
    with open(log_path, "w") as log_file:
        global bert_model, tokenizer
        default_bert_dir = Path(__file__).resolve().parents[2] / "models" / "bert-base-uncased"
        bert_env = os.getenv("BERT_MODEL_DIR")
        if bert_env:
            bert_path = bert_env
        elif default_bert_dir.exists():
            bert_path = str(default_bert_dir)
        else:
            bert_path = "bert-base-uncased"

        tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = BertModel.from_pretrained(bert_path).to(device)

        for param in bert_model.parameters():
            param.requires_grad = False

        data_root = Path(__file__).resolve().parents[2] / "data" / "SEED_chunks"
        preprocessed_files = [f for f in data_root.iterdir() if f.suffix == '.npz']

        metrics_per_fold = []

        for file_path in preprocessed_files:
            print(f"\nProcessing file: {file_path}")
            log_file.write(f"\nProcessing file: {file_path}\n")

            eeg_dataset = EEGDataset(file_path)

            all_data = eeg_dataset.data
            all_labels = eeg_dataset.labels.astype(np.int64)
            num_labels = len(np.unique(all_labels))

            if np.any(all_labels < 0) or np.any(all_labels >= num_labels):
                print(f"Error: Labels in {file_path} are out of range [0, {num_labels - 1}]")
                continue

            num_samples = len(eeg_dataset)
            print(f"Number of samples in this file: {num_samples}")

            check_class_distribution(all_labels, f"Dataset for {file_path.name}", log_file)

            X_train_val, X_test, y_train_val, y_test = train_test_split(
                all_data, all_labels, test_size=0.2, random_state=42
            )
            all_idx = np.arange(len(all_data))
            idx_train_val, idx_test = train_test_split(
                all_idx, test_size=0.2, random_state=42
            )
            check_class_distribution(y_train_val, "Train_Val Dataset", log_file)
            check_class_distribution(y_test, "Test Dataset", log_file)

            prompt_cache_cpu = None
            if use_cached_prompts and prompt_mode != "soft":
                prompt_cache_cpu = build_or_load_prompt_cache(
                    file_path=file_path,
                    all_data=all_data,
                    device=device,
                    prompt_mode=prompt_mode,
                    max_length=50,
                    batch_size=64,
                )

            batch_size = 32

            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
                torch.tensor(idx_test, dtype=torch.long),
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            fold = 1

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            data_indices = np.arange(len(X_train_val))
            for train_indices, val_indices in kf.split(data_indices):
                print(f'Fold {fold}:')
                log_file.write(f'Fold {fold}:')
                X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
                idx_train = idx_train_val[train_indices]
                idx_val = idx_train_val[val_indices]
                y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long),
                    torch.tensor(idx_train, dtype=torch.long),
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                    torch.tensor(idx_val, dtype=torch.long),
                )

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                class_sample_counts = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
                class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float32).to(device)
                class_weights = class_weights / class_weights.sum() * len(np.unique(y_train))
                print(class_weights)
                log_file.write(f"{class_weights}")

                criterion = nn.CrossEntropyLoss(weight=class_weights)

                patch_embedding = PatchEmbedding(patch_len=500, d_model=128, stride=250, num_channels=62).to(device)
                reprogramming_layer = ReprogrammingLayer(embed_dim=128, llm_embed_dim=768, num_heads=8, max_len=5000).to(device)
                classification_head = ClassificationHead(llm_embed_dim=768, num_labels=len(np.unique(y_train))).to(device)

                # Prompt ablation mode: one of {"original","generic","shuffle","random","soft"}
                # Set prompt_mode to run reviewer-required ablations without changing model capacity.
                prompt_mode = os.getenv("PROMPT_MODE", "original").strip().lower()
                # Prompt embedding mode:
                #   - online: run BERT forward each iteration (slow)
                #   - cached: load per-sample cached last_hidden_state (recommended)
                prompt_emb_mode = os.getenv("PROMPT_EMB_MODE", "cached").strip().lower()
                use_cached_prompts = (prompt_emb_mode == "cached")

                # Learnable soft prompt (capacity-controlled baseline). Only used when prompt_mode == "soft".
                soft_prompt = None
                if prompt_mode == "soft":
                    # fixed length to match tokenizer max_length used elsewhere
                    soft_prompt = nn.Parameter(torch.randn(1, 50, 768, device=device) * 0.02)


                model_components = (bert_model, patch_embedding, reprogramming_layer, classification_head)
                # ---- Efficiency logging -> efficiency.jsonl (one line per fold) ----
                try:
                    eff_path = Path(__file__).resolve().parents[2] / "efficiency.jsonl"
                    eff = profile_efficiency(
                        model_components, device,
                        batch_size=32, iters=200, warmup=50,
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt,
                        prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts,
                    )
                    record = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset_file": str(file_path),
                        "prompt_mode": prompt_mode,
                        "prompt_emb_mode": prompt_emb_mode,
                        "fold_id": int(fold),
                        "num_labels": int(len(np.unique(y_train))),
                        **eff,
                    }
                    with open(eff_path, "a", encoding="utf-8") as f_eff:
                        f_eff.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception:
                    pass


                optimizer = AdamW([
                    {'params': classification_head.parameters(), 'lr': 1e-5},
                    {'params': patch_embedding.parameters(), 'lr': 1e-5},
                    {'params': reprogramming_layer.parameters(), 'lr': 1e-5}
                ], weight_decay=0.01)

                best_val_accuracy = 0.0
                best_model_state = None
                no_improve_epochs = 0

                patience = 5
                num_epochs = 100

                for epoch in range(num_epochs):
                    train_loss = train_model(
                        model_components,
                        train_loader, optimizer, criterion, device, len(np.unique(y_train)),
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt, prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts,
                    )

                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        model_components,
                        val_loader, device, len(np.unique(y_train)),
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt, prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts,
                    )

                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        model_components,
                        val_loader, device, len(np.unique(y_train)),
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt, prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts,
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

                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        model_components,
                        val_loader, device, len(np.unique(y_train)),
                        prompt_mode=prompt_mode, soft_prompt=soft_prompt, prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts,
                    )
                    print(
                        f"    Best Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}"
                    )
                    log_file.write(
                        f"    Best Validation Accuracy: {val_accuracy}, F1: {val_f1}, "
                        f"Precision: {val_precision}, Recall: {val_recall}, Conf_matrix: {val_conf_matrix}\n"
                    )

                    fold += 1

            test_accuracy, test_f1, test_precision, test_recall, test_conf_matrix = evaluate_model(
                model_components,
                test_loader, device, len(np.unique(y_train)), prompt_mode=prompt_mode, soft_prompt=soft_prompt, prompt_cache_cpu=prompt_cache_cpu, use_cached_prompts=use_cached_prompts, save_attn_dir=str(Path(__file__).resolve().parents[2] / "attn_viz"), max_attn_batches=1
            )
            print(
                f"*** Test Dataset Results - Accuracy: {test_accuracy}, F1: {test_f1}, "
                f"Precision: {test_precision}, Recall: {test_recall}, Conf_matrix: {test_conf_matrix}"
            )
            log_file.write(
                f"*** Test Dataset Results - Accuracy: {test_accuracy}, F1: {test_f1}, "
                f"Precision: {test_precision}, Recall: {test_recall}, Conf_matrix: {test_conf_matrix}\n"
            )

            metrics_per_fold.append({
                'accuracy': test_accuracy,
                'f1': test_f1,
                'precision': test_precision,
                'recall': test_recall,
                'conf_matrix': test_conf_matrix
            })

        if metrics_per_fold:
            avg_accuracy = np.mean([m['accuracy'] for m in metrics_per_fold])
            avg_f1 = np.mean([m['f1'] for m in metrics_per_fold])
            avg_precision = np.mean([m['precision'] for m in metrics_per_fold])
            avg_recall = np.mean([m['recall'] for m in metrics_per_fold])
            avg_conf_matrix = np.mean([m['conf_matrix'] for m in metrics_per_fold])

            print("Average Metrics across all files:")
            print(f"  Accuracy: {avg_accuracy:.2f}")
            print(f"  F1 Score: {avg_f1:.2f}")
            print(f"  Precision: {avg_precision:.2f}")
            print(f"  Recall: {avg_recall:.2f}")
            print(f"  Conf_matrix: {avg_conf_matrix:.2f}")

            log_file.write("Average Metrics across all files:\n")
            log_file.write(f"  Accuracy: {avg_accuracy:.2f}\n")
            log_file.write(f"  F1 Score: {avg_f1:.2f}\n")
            log_file.write(f"  Precision: {avg_precision:.2f}\n")
            log_file.write(f"  Recall: {avg_recall:.2f}\n")
            log_file.write(f"  Conf_matrix: {avg_conf_matrix:.2f}")


if __name__ == '__main__':
    main()
