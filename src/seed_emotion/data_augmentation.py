"""
Data augmentation utilities for the SEED EEG dataset.

This module reorganizes the original `aug_data.py` script into a package-friendly
layout without altering the underlying processing logic. It slices each subject's
EEG recordings into overlapping windows of multiple lengths to expand the dataset.
"""

import math
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def augment_data(p, s_path, f_name, ws, ch=62, non_overlapping_rate=0.35):
    """
    Augment EEG recordings by sliding windows of varying durations.

    Parameters mirror the original script to keep behaviour identical.
    """
    source_root = Path(p)
    save_root = Path(s_path)

    new_data = []
    for idx, t in enumerate(ws):
        step = int(math.ceil(t * non_overlapping_rate))

        x = np.array([], dtype=np.float32).reshape(0, ch, 100000)

        data = loadmat(str(source_root / f_name))
        trail_list = list(data.keys())[3:]

        for trail in trail_list:
            raw_data = data[trail]

            n_samples = int(math.floor((raw_data.shape[1] - t) / step))

            _x = []
            for n in range(n_samples):
                _x.append(raw_data[:, n * step:n * step + t])

            _x = np.transpose(_x, (1, 0, 2)).reshape(1, ch, -1)

            x = np.append(x, _x[:, :, :100000], axis=0)

        print(str(f_name) + ' -- 时间窗长 : ' + str(t) + ' | x shape :', x.shape)

        new_data.append(np.array(x))

    new_data = np.array(new_data, dtype=object)

    concatenated_data = []
    for nd in new_data:
        concatenated_data.append(nd)
    concatenated_data = np.vstack(concatenated_data)

    print(concatenated_data.shape)

    save_root.mkdir(parents=True, exist_ok=True)

    np.savez(save_root / f_name[:-4], data=concatenated_data, allow_pickle=True)
    print("The file has already been saved............")


def main():
    window_sizes = [50, 100, 150]

    data_root = Path(__file__).resolve().parents[2] / "data"
    seed_root = data_root / "SEED"
    save_path = data_root / "SEED_aug"

    labels = loadmat(str(seed_root / "label.mat"))['label'][0]
    print(labels)

    sublist_name = os.listdir(seed_root)
    print(sublist_name)

    for file_name in sublist_name:
        if file_name.endswith(".txt") or file_name == "label.mat":
            continue
        augment_data(seed_root, save_path, file_name, window_sizes)

    augmented_labels = np.tile(labels, len(window_sizes))
    np.save(save_path / "label.npy", augmented_labels)
    print("Saved tiled labels to", save_path / "label.npy")


if __name__ == "__main__":
    main()
