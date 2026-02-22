"""
python /home/aispeech/codes/zxy/TPANet-main/seed_2026_like_de_LDS/check_eeg_raw_data.py \
    --root /home/aispeech/codes/zxy/SEED
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

NUM_SEED_SUBJECTS = 15
NUM_SEED_SESSIONS = 3
NUM_SEED_TRIALS = 15


def parse_subject_id(name: str) -> int | None:
    stem = Path(name).stem
    part = stem.split("_", 1)[0]
    if part.isdigit():
        return int(part)
    return None


def list_mat_files(path: Path) -> List[Path]:
    return sorted([p for p in path.glob("*.mat") if p.is_file() and p.name != "label.mat"])


def inspect_seed_layout(root: Path) -> Tuple[bool, int, Dict[int, int], bool, List[Path]]:
    mats = list_mat_files(root)
    label_exists = (root / "label.mat").exists()
    subj_ids = [parse_subject_id(p.name) for p in mats]
    subj_ids = [x for x in subj_ids if x is not None]
    cnt = Counter(subj_ids)
    is_seed_layout = (
        label_exists
        and len(mats) == NUM_SEED_SUBJECTS * NUM_SEED_SESSIONS
        and len(cnt) == NUM_SEED_SUBJECTS
        and all(v == NUM_SEED_SESSIONS for v in cnt.values())
    )
    return is_seed_layout, len(mats), dict(sorted(cnt.items())), label_exists, mats


def try_check_trial_keys(mat_path: Path) -> str:
    """Optional .mat key check if scipy/h5py is installed."""
    try:
        import scipy.io as scio  # type: ignore

        d = scio.loadmat(str(mat_path), verify_compressed_data_integrity=False)
        keys = [k for k in d.keys() if not k.startswith("__")]
        eeg_keys = [k for k in keys if re.search(r"_eeg\d+$", k)]
        ids = sorted(int(re.findall(r"(\d+)$", k)[0]) for k in eeg_keys)
        complete = ids == list(range(1, NUM_SEED_TRIALS + 1))
        shape_str = "n/a"
        if eeg_keys:
            shape_str = str(getattr(d[eeg_keys[0]], "shape", None))
        return (
            f"scipy: keys={len(keys)}, eeg_keys={len(eeg_keys)}, "
            f"trials_complete={complete}, sample_shape={shape_str}"
        )
    except Exception:
        pass

    try:
        import h5py  # type: ignore

        with h5py.File(str(mat_path), "r") as f:
            keys = list(f.keys())
            eeg_keys = [k for k in keys if re.search(r"_eeg\d+$", k)]
            ids = sorted(int(re.findall(r"(\d+)$", k)[0]) for k in eeg_keys)
            complete = ids == list(range(1, NUM_SEED_TRIALS + 1))
            return f"h5py: keys={len(keys)}, eeg_keys={len(eeg_keys)}, trials_complete={complete}"
    except Exception:
        pass

    return "skip key check (need scipy or h5py)"


def try_check_label_file(label_path: Path) -> str:
    try:
        import scipy.io as scio  # type: ignore

        d = scio.loadmat(str(label_path), verify_compressed_data_integrity=False)
        label = d.get("label")
        if label is None:
            return "scipy: label key missing"
        vals = [int(x) for x in label.reshape(-1).tolist()]
        return f"scipy: label_len={len(vals)}, unique={sorted(set(vals))}"
    except Exception:
        pass

    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore

        with h5py.File(str(label_path), "r") as f:
            if "label" not in f:
                return "h5py: label key missing"
            arr = np.array(f["label"]).reshape(-1)
            vals = [int(x) for x in arr.tolist()]
            return f"h5py: label_len={len(vals)}, unique={sorted(set(vals))}"
    except Exception:
        pass

    return "skip label check (need scipy or h5py)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a path matches SEED raw-data layout")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"Root: {root}")

    if not root.exists() or not root.is_dir():
        print("Result: NOT SEED raw-data root (path missing or not a directory)")
        return

    is_seed_ok, mat_total, by_subject, label_exists, mats = inspect_seed_layout(root)
    print(f"label.mat exists: {label_exists}")
    print(f"Flat .mat files in root (excluding label.mat): {mat_total}")
    print("Subject file counts:", by_subject)

    label_path = root / "label.mat"
    if label_exists:
        print(f"Label check ({label_path.name}): {try_check_label_file(label_path)}")
    if mats:
        print(f"Sample key check ({mats[0].name}): {try_check_trial_keys(mats[0])}")

    if is_seed_ok:
        print("Result: YES, this looks like standard SEED raw-data layout.")
        return

    print("Result: NOT standard SEED raw-data layout.")


if __name__ == "__main__":
    main()
