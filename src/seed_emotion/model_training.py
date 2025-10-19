"""
Model training pipeline for the SEED EEG emotion recognition task.

This module mirrors the original `3model.py` script, reorganised into a package
structure without changing the underlying algorithmic steps. It loads the
preprocessed EEG chunks, reprograms them via a BERT backbone, and performs
cross-validation alongside a held-out test split.
"""

import copy
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

        attn_output, _ = self.multihead_attn(target_embedding, source_embedding, value_embedding)

        return attn_output


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


def generate_prompts(batch_size, min_values, max_values, median_values, trends):
    prompts = []
    for i in range(batch_size):
        min_value = min_values[i].item()
        max_value = max_values[i].item()
        median_value = median_values[i].item()
        trend_value = trends[i].item()
        trend = 'upward' if trend_value > 0 else 'downward'

        prompt = (
            "Based on the provided EEG data, classify the underlying emotion. "
            f"Statistics: minimum value = {min_value:.2f}, maximum value = {max_value:.2f}, "
            f"median value = {median_value:.2f}. The overall trend of the data is {trend}."
        )

        prompts.append(prompt)
    return prompts


def train_model(model_components, dataloader, optimizer, criterion, device, num_labels):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.train()
    reprogramming_layer.train()
    classification_head.train()
    total_loss = 0

    for batch_eeg, batch_labels in dataloader:
        batch_eeg = batch_eeg.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        eeg_embeddings = patch_embedding(batch_eeg)

        min_values, max_values, median_values, trends = generate_statistics(batch_eeg)
        prompts = generate_prompts(batch_eeg.size(0), min_values, max_values, median_values, trends)
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
        prompt_embeddings = bert_model.embeddings(input_ids=prompt_inputs.input_ids)

        eeg_embeddings = reprogramming_layer(eeg_embeddings, prompt_embeddings, prompt_embeddings)

        pooled_output = eeg_embeddings.mean(dim=1)

        logits = classification_head(pooled_output)
        loss = criterion(logits, batch_labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def evaluate_model(model_components, dataloader, device, num_labels):
    bert_model, patch_embedding, reprogramming_layer, classification_head = model_components
    bert_model.eval()
    patch_embedding.eval()
    reprogramming_layer.eval()
    classification_head.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_eeg, batch_labels in dataloader:
            batch_eeg = batch_eeg.to(device)
            batch_labels = batch_labels.to(device)

            eeg_embeddings = patch_embedding(batch_eeg)

            min_values, max_values, median_values, trends = generate_statistics(batch_eeg)
            prompts = generate_prompts(batch_eeg.size(0), min_values, max_values, median_values, trends)
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
            prompt_embeddings = bert_model.embeddings(input_ids=prompt_inputs.input_ids)

            eeg_embeddings = reprogramming_layer(eeg_embeddings, prompt_embeddings, prompt_embeddings)

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
            check_class_distribution(y_train_val, "Train_Val Dataset", log_file)
            check_class_distribution(y_test, "Test Dataset", log_file)

            batch_size = 32

            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            fold = 1

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            data_indices = np.arange(len(X_train_val))
            for train_indices, val_indices in kf.split(data_indices):
                print(f'Fold {fold}:')
                log_file.write(f'Fold {fold}:')
                X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
                y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

                train_dataset = TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.long)
                )
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long)
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
                        (bert_model, patch_embedding, reprogramming_layer, classification_head),
                        train_loader, optimizer, criterion, device, len(np.unique(y_train))
                    )
                    val_accuracy, val_f1, val_precision, val_recall, val_conf_matrix = evaluate_model(
                        (bert_model, patch_embedding, reprogramming_layer, classification_head),
                        val_loader, device, len(np.unique(y_train))
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
                        (bert_model, patch_embedding, reprogramming_layer, classification_head),
                        val_loader, device, len(np.unique(y_train))
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
                (bert_model, patch_embedding, reprogramming_layer, classification_head),
                test_loader, device, len(np.unique(y_train))
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
