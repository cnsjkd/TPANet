"""
Chunk-wise preprocessing utilities for augmented SEED EEG data.

This module preserves the behaviour of the original script `2单独切块.py`,
organised for reuse inside a Python package. It slices augmented EEG arrays
into fixed-length chunks and persists them as `.npz` archives.
"""

import os
from pathlib import Path

import numpy as np


def preprocess_and_save_chunks_per_file(data_files, labels, chunk_size=2000, overlap=0, save_folder='preprocessed_chunks'):
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
            trial_labels = labels

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
    data_root = Path(__file__).resolve().parents[2] / "data"
    data_folder = data_root / "SEED_aug"
    label_file = data_root / "SEED_aug" / "label.npy"

    labels = np.load(label_file, allow_pickle=True)

    data_files = [data_folder / f for f in os.listdir(data_folder) if f.endswith('.npz')]

    preprocess_and_save_chunks_per_file(
        data_files,
        labels,
        chunk_size=1000,
        overlap=0,
        save_folder=data_root / 'SEED_chunks'
    )


if __name__ == '__main__':
    main()
