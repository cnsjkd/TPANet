"""
Chunk-wise preprocessing utilities for augmented SEED EEG data.

This module preserves the behaviour of the original script `2单独切块.py`,
organised for reuse inside a Python package. It slices augmented EEG arrays
into fixed-length chunks and persists them as `.npz` archives.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def _load_trial_labels(npz_file, fallback_labels=None):
    if "labels" in npz_file:
        labels = np.array(npz_file["labels"], dtype=np.int64)
        return labels
    if fallback_labels is None:
        raise ValueError("未在 .npz 中找到 labels，且未提供 label.npy 作为回退。")
    return np.array(fallback_labels, dtype=np.int64)


def preprocess_and_save_chunks_per_file(data_files, labels=None, chunk_size=2000, overlap=0, save_folder='preprocessed_chunks'):
    save_folder = Path(save_folder)
    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    for file_idx, file_path in enumerate(data_files):
        file_path = Path(file_path)
        all_data_chunks = []
        all_labels = []

        with np.load(file_path, allow_pickle=True) as npz_file:
            eeg_data = npz_file['data']
            num_trials, num_channels, num_timepoints = eeg_data.shape
            trial_labels = _load_trial_labels(npz_file, labels)
            if len(trial_labels) != num_trials:
                raise ValueError(
                    f"{file_path} 标签数量({len(trial_labels)})与 trial 数量({num_trials})不一致。"
                )

            for trial_idx in range(num_trials):
                trial_data = eeg_data[trial_idx]
                trial_label = trial_labels[trial_idx]

                effective_chunk_size = chunk_size - overlap
                num_chunks = max(1, (trial_data.shape[1] - overlap) // effective_chunk_size)

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * effective_chunk_size
                    end = start + chunk_size
                    data_chunk = trial_data[:, start:end]

                    if data_chunk.shape[1] < chunk_size:
                        padding = np.zeros((data_chunk.shape[0], chunk_size - data_chunk.shape[1]))
                        data_chunk = np.hstack((data_chunk, padding))
                    elif data_chunk.shape[1] > chunk_size:
                        data_chunk = data_chunk[:, :chunk_size]

                    all_data_chunks.append(data_chunk)
                    all_labels.append(trial_label)

        all_data_chunks = np.array(all_data_chunks, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)

        original_filename = file_path.stem
        save_path = save_folder / f'{original_filename}.npz'
        np.savez(save_path, data=all_data_chunks, labels=all_labels)

        print(f"Processed {file_path}: {len(all_labels)} chunks saved to {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.getenv("EEG_DATASET", "seed"), choices=["seed", "seed-iv"])
    ap.add_argument("--data_root", default=os.getenv("EEG_DATA_ROOT", "/home/aispeech/codes/zxy/"))
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--chunk_size", type=int, default=int(os.getenv("EEG_CHUNK_SIZE", "1000")))
    ap.add_argument("--overlap", type=int, default=int(os.getenv("EEG_CHUNK_OVERLAP", "0")))
    args = ap.parse_args()

    dataset = args.dataset.strip().lower()
    data_root = Path(args.data_root).expanduser()
    default_input = "SEED_aug" if dataset == "seed" else "SEED_IV_aug"
    default_output = "SEED_chunks" if dataset == "seed" else "SEED_IV_chunks"

    if args.input_dir:
        data_folder = Path(args.input_dir).expanduser()
    elif dataset == "seed-iv" and Path("/home/xiaoying/SEED-IV_aug").exists():
        data_folder = Path("/home/xiaoying/SEED-IV_aug")
    else:
        data_folder = data_root / default_input

    if args.output_dir:
        save_folder = Path(args.output_dir).expanduser()
    elif dataset == "seed-iv" and Path("/home/xiaoying").exists():
        save_folder = Path("/home/xiaoying/SEED-IV_chunks")
    else:
        save_folder = data_root / default_output

    label_file = data_folder / "label.npy"
    labels = None
    if label_file.exists():
        labels = np.load(label_file, allow_pickle=True)

    data_files = [data_folder / f for f in os.listdir(data_folder) if f.endswith('.npz')]

    preprocess_and_save_chunks_per_file(
        data_files,
        labels,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        save_folder=save_folder
    )


if __name__ == '__main__':
    main()
