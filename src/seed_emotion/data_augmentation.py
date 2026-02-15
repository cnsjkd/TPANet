"""
Data augmentation utilities for the SEED EEG dataset.

This module reorganizes the original `aug_data.py` script into a package-friendly
layout without altering the underlying processing logic. It slices each subject's
EEG recordings into overlapping windows of multiple lengths to expand the dataset.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat


SEED_IV_FALLBACK_LABELS = np.array(
    [
        [
            1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1,
            1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3,
        ],
        [
            2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3,
            2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1,
        ],
        [
            1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0,
            2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0,
        ],
    ],
    dtype=np.int64,
)


def _infer_session_id(file_path: Path) -> Optional[int]:
    parent = file_path.parent.name
    if parent.isdigit():
        sid = int(parent)
        if 1 <= sid <= 3:
            return sid
    return None


def _load_seed_labels(seed_root: Path) -> np.ndarray:
    label_file = seed_root / "label.mat"
    if not label_file.exists():
        raise FileNotFoundError(f"未找到 SEED 标签文件: {label_file}")
    labels = loadmat(str(label_file))["label"].squeeze()
    return np.array(labels, dtype=np.int64)


def _load_seed_iv_labels(seed_root: Path) -> np.ndarray:
    label_file = seed_root / "label.mat"
    if label_file.exists():
        labels = loadmat(str(label_file)).get("label")
        if labels is not None:
            labels = np.array(labels, dtype=np.int64)
            if labels.ndim == 2 and labels.shape == (24, 3):
                labels = labels.T
            return labels
    return SEED_IV_FALLBACK_LABELS.copy()


def _labels_for_file(dataset: str, seed_root: Path, file_path: Path, session_id: Optional[int] = None) -> np.ndarray:
    if dataset == "seed":
        return _load_seed_labels(seed_root)

    labels = _load_seed_iv_labels(seed_root)
    if labels.ndim == 1:
        return labels

    if session_id is None:
        session_id = _infer_session_id(file_path)
    if session_id is None:
        raise ValueError(
            f"无法从路径推断 SEED-IV 的 session_id: {file_path}\n"
            "请将原始数据按 session 分目录 (1/2/3) 放置，"
            "或确保 label.mat 只包含单个 session 的 24 个标签。"
        )
    return labels[session_id - 1]


def _collect_trials(mat_data: dict) -> list[str]:
    keys = [k for k in mat_data.keys() if not k.startswith("__")]
    eeg_keys = [k for k in keys if "eeg" in k.lower()]
    if eeg_keys:
        return sorted(eeg_keys)
    return sorted(keys)


def _parse_subject_and_date(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem
    if "_" in stem:
        subj, date_str = stem.split("_", 1)
        return subj, date_str
    return stem, ""


def _infer_sessions_from_filenames(mat_files: list[Path]) -> dict[Path, int]:
    sessions = {}
    groups: dict[str, list[Path]] = {}
    for fp in mat_files:
        subj, _ = _parse_subject_and_date(fp)
        groups.setdefault(subj, []).append(fp)

    for subj, files in groups.items():
        if len(files) != 3:
            raise ValueError(
                f"受试者 {subj} 的 .mat 文件数量为 {len(files)}，无法分配 3 个 session。"
            )
        files_sorted = sorted(files, key=lambda p: _parse_subject_and_date(p)[1])
        for idx, fp in enumerate(files_sorted, start=1):
            sessions[fp] = idx
    return sessions


def _compute_fixed_len_for_file(file_path: Path, window_sizes: list[int], non_overlapping_rate: float) -> int:
    """
    Compute a per-file fixed length to mimic the original raw_codes logic:
    for each file (session), use one fixed truncation length across all window sizes.
    """
    data = loadmat(str(file_path))
    trial_list = _collect_trials(data)
    min_len = None
    for t in window_sizes:
        step = int(math.ceil(t * non_overlapping_rate))
        for trial in trial_list:
            raw_data = data[trial]
            n_samples = int(math.floor((raw_data.shape[1] - t) / step))
            if n_samples < 1:
                n_samples = 1
            concat_len = n_samples * t
            if min_len is None or concat_len < min_len:
                min_len = concat_len
    if min_len is None:
        raise ValueError(f"无法计算固定长度，未找到有效 trial: {file_path}")
    return int(min_len)


def augment_data(file_path, s_path, ws, ch=62, non_overlapping_rate=0.35, labels=None, output_stem=None, fixed_len=None):
    """
    Augment EEG recordings by sliding windows of varying durations.

    Parameters mirror the original script to keep behaviour identical.
    """
    file_path = Path(file_path)
    save_root = Path(s_path)

    new_data = []
    for idx, t in enumerate(ws):
        step = int(math.ceil(t * non_overlapping_rate))

        data = loadmat(str(file_path))
        trial_list = _collect_trials(data)

        concat_lens = []
        timepoints_list = []
        for trial in trial_list:
            raw_data = data[trial]
            timepoints_list.append(raw_data.shape[1])

            n_samples = int(math.floor((raw_data.shape[1] - t) / step))
            if n_samples < 1:
                n_samples = 1
            concat_lens.append(n_samples * t)

        if not concat_lens:
            raise ValueError(f"{file_path} 未找到有效 trial。")

        target_len = fixed_len if fixed_len is not None else min(concat_lens)
        min_tp = min(timepoints_list)
        max_tp = max(timepoints_list)

        x = np.array([], dtype=np.float32).reshape(0, ch, target_len)

        for trial in trial_list:
            raw_data = data[trial]

            n_samples = int(math.floor((raw_data.shape[1] - t) / step))
            if n_samples < 1:
                n_samples = 1

            _x = []
            for n in range(n_samples):
                _x.append(raw_data[:, n * step:n * step + t])

            _x = np.transpose(_x, (1, 0, 2)).reshape(1, ch, -1)

            if _x.shape[2] < target_len:
                pad = np.zeros((1, ch, target_len - _x.shape[2]), dtype=_x.dtype)
                _x = np.concatenate([_x, pad], axis=2)
            x = np.append(x, _x[:, :, :target_len], axis=0)

        print(
            f"{file_path.name} -- 时间窗长: {t} | timepoints min/max: {min_tp}/{max_tp} "
            f"| target_len: {target_len} | x shape: {x.shape}"
        )

        new_data.append(np.array(x))

    if not new_data:
        raise ValueError(f"{file_path} 未生成任何数据。")

    # 不同窗口长度会导致第三维不一致，统一截断到最短长度再拼接
    min_len = min(arr.shape[2] for arr in new_data)
    aligned = []
    for arr in new_data:
        if arr.shape[2] > min_len:
            arr = arr[:, :, :min_len]
        elif arr.shape[2] < min_len:
            pad = np.zeros((arr.shape[0], arr.shape[1], min_len - arr.shape[2]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=2)
        aligned.append(arr)

    concatenated_data = np.vstack(aligned)

    print(concatenated_data.shape)

    save_root.mkdir(parents=True, exist_ok=True)

    stem = output_stem or file_path.stem
    np.savez(save_root / stem, data=concatenated_data, labels=labels, allow_pickle=True)
    print("The file has already been saved............")


def _parse_window_sizes(raw: str) -> list[int]:
    if not raw:
        return [50, 100, 150]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.getenv("EEG_DATASET", "seed"), choices=["seed", "seed-iv"])
    ap.add_argument("--data_root", default=os.getenv("EEG_DATA_ROOT", "/home/aispeech/codes/zxy"))
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--window_sizes", default=os.getenv("EEG_WINDOW_SIZES", "50,100,150"))
    ap.add_argument("--channels", type=int, default=int(os.getenv("EEG_CHANNELS", "62")))
    args = ap.parse_args()

    dataset = args.dataset.strip().lower()
    data_root = Path(args.data_root).expanduser()
    default_input = "SEED" if dataset == "seed" else "SEED_IV"
    default_output = "SEED_aug" if dataset == "seed" else "SEED_IV_aug"

    if args.input_dir:
        seed_root = Path(args.input_dir).expanduser()
    elif dataset == "seed-iv" and Path("/home/xiaoying/SEED-IV").exists():
        seed_root = Path("/home/xiaoying/SEED-IV")
    else:
        seed_root = data_root / default_input

    if args.output_dir:
        save_path = Path(args.output_dir).expanduser()
    elif dataset == "seed-iv" and Path("/home/xiaoying").exists():
        save_path = Path("/home/xiaoying/SEED-IV_aug")
    else:
        save_path = data_root / default_output

    window_sizes = _parse_window_sizes(args.window_sizes)

    mat_files = sorted([p for p in seed_root.rglob("*.mat") if p.name != "label.mat"])
    if not mat_files:
        raise FileNotFoundError(f"未在 {seed_root} 找到 .mat 文件。")

    session_map = None
    if dataset == "seed-iv":
        labels = _load_seed_iv_labels(seed_root)
        if labels.ndim == 2:
            session_map = _infer_sessions_from_filenames(mat_files)

    for file_path in mat_files:
        if file_path.name.endswith(".txt"):
            continue
        session_id = session_map.get(file_path) if session_map is not None else None
        fixed_len = _compute_fixed_len_for_file(file_path, window_sizes, non_overlapping_rate=0.35) if dataset == "seed-iv" else None
        if dataset == "seed-iv":
            print(f"[SEED-IV] {file_path.name} 计算得到固定长度 fixed_len={fixed_len}")
        labels = _labels_for_file(dataset, seed_root, file_path, session_id=session_id)
        augmented_labels = np.tile(labels, len(window_sizes))
        if file_path.parent == seed_root:
            stem = file_path.stem
        else:
            stem = f"{file_path.parent.name}__{file_path.stem}"
        augment_data(
            file_path,
            save_path,
            window_sizes,
            ch=args.channels,
            labels=augmented_labels,
            output_stem=stem,
            fixed_len=fixed_len,
        )

    # For backward compatibility: save a global label.npy when labels are shared.
    if dataset == "seed":
        labels = _load_seed_labels(seed_root)
        augmented_labels = np.tile(labels, len(window_sizes))
        np.save(save_path / "label.npy", augmented_labels)
        print("Saved tiled labels to", save_path / "label.npy")


if __name__ == "__main__":
    main()
