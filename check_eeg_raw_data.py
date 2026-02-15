"""
python /home/aispeech/codes/zxy/TPANet-main/seed_iv_2026_like_de_LDS/check_eeg_raw_data.py \
    --root /home/aispeech/codes/zxy/SEED-IV
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def parse_subject_id(name: str) -> int | None:
    stem = Path(name).stem
    part = stem.split("_", 1)[0]
    if part.isdigit():
        return int(part)
    return None


def list_mat_files(path: Path) -> List[Path]:
    return sorted([p for p in path.glob("*.mat") if p.is_file()])


def inspect_expected_layout(root: Path) -> Tuple[bool, Dict[int, int]]:
    counts: Dict[int, int] = {}
    ok = True
    for sid in (1, 2, 3):
        sess_dir = root / str(sid)
        if not sess_dir.exists() or not sess_dir.is_dir():
            ok = False
            counts[sid] = 0
            continue
        mats = list_mat_files(sess_dir)
        counts[sid] = len(mats)
        if len(mats) == 0:
            ok = False
    return ok, counts


def inspect_flat_layout(root: Path) -> Tuple[bool, int, Dict[int, int], List[Path]]:
    mats = list_mat_files(root)
    subj_ids = [parse_subject_id(p.name) for p in mats]
    subj_ids = [x for x in subj_ids if x is not None]
    cnt = Counter(subj_ids)
    is_flat_15x3 = len(mats) == 45 and len(cnt) == 15 and all(v == 3 for v in cnt.values())
    return is_flat_15x3, len(mats), dict(sorted(cnt.items())), mats


def try_check_trial_keys(mat_path: Path) -> str:
    """Optional .mat key check if scipy/h5py is installed."""
    try:
        import scipy.io as scio  # type: ignore

        d = scio.loadmat(str(mat_path), verify_compressed_data_integrity=False)
        keys = [k for k in d.keys() if not k.startswith("__")]
        eeg_keys = [k for k in keys if re.search(r"_eeg\d+$", k)]
        return f"scipy: keys={len(keys)}, eeg_keys={len(eeg_keys)}"
    except Exception:
        pass

    try:
        import h5py  # type: ignore

        with h5py.File(str(mat_path), "r") as f:
            keys = list(f.keys())
            eeg_keys = [k for k in keys if re.search(r"_eeg\d+$", k)]
            return f"h5py: keys={len(keys)}, eeg_keys={len(eeg_keys)}"
    except Exception:
        pass

    return "skip key check (need scipy or h5py)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a path matches SEED-IV eeg_raw_data layout")
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"Root: {root}")

    if not root.exists() or not root.is_dir():
        print("Result: NOT eeg_raw_data (path missing or not a directory)")
        return

    expected_ok, sess_counts = inspect_expected_layout(root)
    flat_ok, flat_total, flat_by_subject, flat_mats = inspect_flat_layout(root)

    print(f"Expected layout counts (1/2/3): {sess_counts}")
    print(f"Flat mat files in root: {flat_total}")

    if expected_ok:
        print("Result: YES, this already looks like eeg_raw_data.")
        sample = None
        for sid in (1, 2, 3):
            mats = list_mat_files(root / str(sid))
            if mats:
                sample = mats[0]
                break
        if sample is not None:
            print(f"Sample key check ({sample.name}): {try_check_trial_keys(sample)}")
        return

    if flat_ok:
        print("Result: NOT direct eeg_raw_data, but it is a FLAT 45-file raw-data layout (15 subjects x 3 files).")
        print("Subject file counts:", flat_by_subject)
        if flat_mats:
            print(f"Sample key check ({flat_mats[0].name}): {try_check_trial_keys(flat_mats[0])}")
        print("Suggestion: reorganize into root/1, root/2, root/3 before training.")
        return

    print("Result: NOT eeg_raw_data and also not standard flat 45-file layout.")


if __name__ == "__main__":
    main()
